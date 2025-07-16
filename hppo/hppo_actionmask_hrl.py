# 文件名: hppo_actionmask_hrl.py
# 版本: 已修复config bug并完成网络定义

import os
import select
import numpy as np
from abc import ABC
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from hppo.hppo_utils import *


def weight_init(m):
    """正交初始化权重"""
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class HRLBuffer:
    """
    一个通用的经验回放缓冲区，用于HRL。
    可以存储高层或低层的经验。
    """

    def __init__(self, obs_dim, act_dim, is_discrete, size, gamma, lam, device):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        if is_discrete:
            self.act_buf = np.zeros(size, dtype=np.int64)
        else:
            self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.task_id_buf = np.zeros(size, dtype=np.int64)
        self.is_discrete = is_discrete
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act, rew, val, logp, task_id=None):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        if task_id is not None:
            self.task_id_buf[self.ptr] = task_id
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self, batch_size):
        # 确保我们只使用已填充的数据
        indices = np.arange(self.ptr)
        # 标准化优势
        adv_mean = np.mean(self.adv_buf[indices])
        adv_std = np.std(self.adv_buf[indices])
        self.adv_buf[indices] = (self.adv_buf[indices] - adv_mean) / (adv_std + 1e-8)

        sampler = BatchSampler(
            sampler=SubsetRandomSampler(indices),
            batch_size=batch_size,
            drop_last=True
        )

        for batch_indices in sampler:
            data_dict = dict(
                obs=torch.as_tensor(self.obs_buf[batch_indices], dtype=torch.float32, device=self.device),
                act=torch.as_tensor(self.act_buf[batch_indices], device=self.device),
                ret=torch.as_tensor(self.ret_buf[batch_indices], dtype=torch.float32, device=self.device),
                adv=torch.as_tensor(self.adv_buf[batch_indices], dtype=torch.float32, device=self.device),
                logp=torch.as_tensor(self.logp_buf[batch_indices], dtype=torch.float32, device=self.device)
            )
            # 如果是低层，额外返回task_id
            # 检查self.task_id_buf中是否有非零元素。ptr是当前填充位置，所以检查0到ptr的范围
            if np.any(self.task_id_buf[:self.ptr]):
                data_dict['task_id'] = torch.as_tensor(self.task_id_buf[batch_indices], dtype=torch.int64,
                                                       device=self.device)
            yield data_dict

    def clear(self):
        self.ptr, self.path_start_idx = 0, 0


# ===================================================================
# Part 1: High-Level Agent (Meta-Controller)
# ===================================================================

class MetaControllerNet(nn.Module):
    """高层网络，用于选择任务。一个简单的MLP Actor-Critic。"""

    def __init__(self, state_dim, num_tasks, hidden_dim=256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, num_tasks)  # 输出N个任务的logits
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(weight_init)

    def get_value(self, obs):
        return self.critic(obs)

    def get_action_and_value(self, obs, action_mask=None, action=None):
        logits = self.actor(obs)
        if action_mask is not None:
            logits[action_mask == 0] = -1e9
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(obs)


class MetaControllerPPO:
    """高层智能体的PPO实现"""

    def __init__(self, state_dim, num_tasks, config, device):
        # [FIX] 保存整个config字典
        self.config = config
        self.gamma = config['gamma']
        self.lam = config['lam']
        self.epochs_update = config['epochs_update']
        self.eps_clip = config['eps_clip']
        self.device = device

        self.buffer = HRLBuffer(state_dim, act_dim=num_tasks, is_discrete=True, size=config['meta_buffer_size'],
                                gamma=self.gamma, lam=self.lam, device=self.device)
        self.policy = MetaControllerNet(state_dim, num_tasks, config['meta_hidden_dim']).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['meta_lr_actor'])
        self.policy_old = MetaControllerNet(state_dim, num_tasks, config['meta_hidden_dim']).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state, mask):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            mask_tensor = torch.FloatTensor(mask).to(self.device) if mask is not None else None
            action, log_prob, _, value = self.policy_old.get_action_and_value(state_tensor, action_mask=mask_tensor)
        return action.item(), value.item(), log_prob.item()

    def update(self):
        self.policy.train()
        for i in range(self.epochs_update):
            # [FIX] 使用 self.config
            for data in self.buffer.get(batch_size=self.config['meta_batch_size']):
                obs, act, ret, adv, old_logp = data['obs'], data['act'], data['ret'], data['adv'], data['logp']
                _, logp, entropy, value = self.policy.get_action_and_value(obs, action=act)
                ratio = torch.exp(logp - old_logp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv
                # [FIX] 使用 self.config
                loss_pi = -torch.min(surr1, surr2).mean() - self.config['coeff_entropy'] * entropy.mean()
                loss_v = self.MseLoss(value.squeeze(), ret)
                total_loss = loss_pi + loss_v
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
        self.policy_old.load_state_dict(self.policy.state_dict())
        # [LOGIC] 清空buffer的调用应该在训练循环之外，例如在主训练脚本中
        # self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=self.device))


# ===================================================================
# Part 2: Low-Level Agent (Task-Conditioned Controller)
# ===================================================================

class LowLevelConditionedNet(nn.Module):
    """任务条件化的低层网络。基于原有的 ActorCritic_FRCF。"""

    def __init__(self, state_dim, action_con_dim, num_tasks,
                 cnn_input_channels, cnn_input_size, cnn_out_channels,
                 mlp_hidden_dim, init_log_std):
        super().__init__()
        self.num_tasks = num_tasks
        self.cnn_input_channels = cnn_input_channels
        self.cnn_input_size = cnn_input_size

        conditioned_state_dim = state_dim + num_tasks
        cnn_input_dim = cnn_input_channels * cnn_input_size * cnn_input_size
        self.spatialization_fc = nn.Linear(conditioned_state_dim, cnn_input_dim)

        # [COMPLETED] 填充CNN和MLP Body
        self.cnn = nn.Sequential(
            nn.Conv2d(cnn_input_channels, cnn_out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn_out_channels[0], cnn_out_channels[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, cnn_input_channels, cnn_input_size, cnn_input_size)
            cnn_output_dim = self.cnn(dummy_input).shape[1]

        self.mlp_body = nn.Sequential(
            nn.Linear(cnn_output_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU()
        )
        # ---
        self.critic_head = nn.Linear(mlp_hidden_dim, 1)
        self.actor_con_head = nn.Linear(mlp_hidden_dim, action_con_dim)
        self.log_std = nn.Parameter(torch.full((action_con_dim,), init_log_std))
        self.apply(weight_init)

    def forward_body(self, obs_vector, task_id):
        # 确保obs_vector是2D的 (batch, features)
        if obs_vector.dim() == 1:
            obs_vector = obs_vector.unsqueeze(0)

        task_one_hot = F.one_hot(task_id.long(), num_classes=self.num_tasks).float().to(obs_vector.device)
        if task_one_hot.dim() == 1:
            task_one_hot = task_one_hot.unsqueeze(0)

        # 确保one-hot编码的batch size与obs_vector匹配
        if task_one_hot.shape[0] != obs_vector.shape[0]:
            task_one_hot = task_one_hot.expand(obs_vector.shape[0], -1)

        conditioned_obs = torch.cat([obs_vector, task_one_hot], dim=-1)
        x = self.spatialization_fc(conditioned_obs)
        x = F.relu(x)
        x = x.view(-1, self.cnn_input_channels, self.cnn_input_size, self.cnn_input_size)
        x = self.cnn(x)
        x = self.mlp_body(x)
        return x

    def get_action_and_value(self, obs, task_id, action_mask=None, action=None):
        body_out = self.forward_body(obs, task_id)
        mean_raw = self.actor_con_head(body_out)
        std = torch.exp(self.log_std)

        # 确保action_mask在正确的设备上
        con_mask_bounds = action_mask['continuous_mask'].to(obs.device) if action_mask else None

        if con_mask_bounds is not None:
            # 确保掩码维度与均值匹配
            if con_mask_bounds.dim() == 2:
                con_mask_bounds = con_mask_bounds.unsqueeze(0)
            clipped_mean = torch.clamp(mean_raw, con_mask_bounds[:, :, 0], con_mask_bounds[:, :, 1])
        else:
            clipped_mean = mean_raw

        dist = Normal(clipped_mean, std)
        if action is None:
            action = dist.sample()
        if con_mask_bounds is not None:
            action = torch.clamp(action, con_mask_bounds[:, :, 0], con_mask_bounds[:, :, 1])

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic_head(body_out)
        return action, log_prob, entropy, value


class LowLevelPPO:
    """任务条件化的低层智能体的PPO实现"""

    def __init__(self, state_dim, action_con_dim, num_tasks, config, device):
        # [FIX] 保存整个config字典
        self.config = config
        self.gamma = config['gamma']
        self.lam = config['lam']
        self.epochs_update = config['epochs_update']
        self.eps_clip = config['eps_clip']
        self.device = device

        self.buffer = HRLBuffer(state_dim, act_dim=action_con_dim, is_discrete=False, size=config['low_buffer_size'],
                                gamma=self.gamma, lam=self.lam, device=self.device)
        net_args = {
            'state_dim': state_dim, 'action_con_dim': action_con_dim, 'num_tasks': num_tasks,
            'cnn_input_channels': config['cnn_input_channels'], 'cnn_input_size': config['cnn_input_size'],
            'cnn_out_channels': config['cnn_out_channels'], 'mlp_hidden_dim': config['mlp_hidden_dim'],
            'init_log_std': config['init_log_std']
        }
        self.policy = LowLevelConditionedNet(**net_args).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['lr_actor'])
        self.policy_old = LowLevelConditionedNet(**net_args).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
        self.eval_mode = False

    def set_eval_mode(self, is_eval):
        """【新增】用于切换评估模式的方法"""
        self.eval_mode = is_eval

    def select_action(self, state, task_id, mask):
        if self.eval_mode:
            # 如果是评估模式，则强制使用无限制掩码
            final_mask = {
                'continuous_mask': torch.FloatTensor([[-1.0, 1.0], [-1.0, 1.0]]).to(self.device)
            }
        else:
            # 否则，正常使用传入的掩码
            final_mask = {
                'continuous_mask': torch.FloatTensor(mask['continuous_mask']).to(self.device)
            } if mask else None
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            task_id_tensor = torch.tensor([task_id], device=self.device).long()
            # mask_dict = {
            #     'continuous_mask': torch.FloatTensor(mask['continuous_mask']).to(self.device)
            # } if mask else None
            action, log_prob, _, value = self.policy_old.get_action_and_value(state_tensor, task_id_tensor,
                                                                              action_mask=final_mask)
        return action.squeeze().cpu().numpy(), value.item(), log_prob.item()

    def update(self):
        self.policy.train()
        for i in range(self.epochs_update):
            # [FIX] 使用 self.config
            for data in self.buffer.get(batch_size=self.config['low_batch_size']):
                obs, act, ret, adv, old_logp, task_id = data['obs'], data['act'], data['ret'], data['adv'], data[
                    'logp'], data['task_id']
                _, logp, entropy, value = self.policy.get_action_and_value(obs, task_id, action=act)
                ratio = torch.exp(logp - old_logp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv
                # [FIX] 使用 self.config
                loss_pi = -torch.min(surr1, surr2).mean() - self.config['coeff_entropy'] * entropy.mean()
                loss_v = self.MseLoss(value.squeeze(), ret)
                total_loss = loss_pi + loss_v
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
        self.policy_old.load_state_dict(self.policy.state_dict())
        # [LOGIC] 清空buffer的调用应该在训练循环之外
        # self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=self.device))