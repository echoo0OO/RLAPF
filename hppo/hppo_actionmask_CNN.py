import os
import select
from abc import ABC

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from hppo.hppo_utils import *


def weight_init(m):
    """正交初始化权重"""
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ 第二部分：Buffer (关键修正) +++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of observations-action pairs.
    """

    def __init__(self, obs_dim, act_dis_dim, act_dis_len, act_con_dim, size, gamma, lam, device):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)

        # 为离散和连续动作掩码分别设置缓冲区
        self.action_mask_dis_buf = np.ones((size, act_dis_dim * act_dis_len), dtype=np.float32)
        self.action_mask_con_buf = np.zeros((size, act_con_dim, 2), dtype=np.float32)

        self.act_dis_buf = np.zeros((size, act_dis_dim), dtype=np.int64)
        self.act_con_buf = np.zeros((size, act_con_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_dis_buf = np.zeros((size, act_dis_dim), dtype=np.float32)
        self.logp_con_buf = np.zeros((size, act_con_dim), dtype=np.float32)

        # self.ptr_buf is not used in the current implementation, can be removed if desired.
        # self.ptr_buf = np.zeros(size, dtype=np.int64)

        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_dix, self.max_size = 0, 0, size
        self.device = device

    def store_hybrid(self, obs, action_mask, act_dis, act_con, rew, val, logp_dis, logp_con):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs

        # 正确存储来自字典的离散和连续掩码
        self.action_mask_dis_buf[self.ptr] = action_mask['discrete_mask'].flatten()
        self.action_mask_con_buf[self.ptr] = action_mask['continuous_mask']

        self.act_dis_buf[self.ptr] = act_dis
        self.act_con_buf[self.ptr] = act_con
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_dis_buf[self.ptr] = logp_dis
        self.logp_con_buf[self.ptr] = logp_con
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_dix, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_dix = self.ptr

    def get(self, batch_size):
        # 确保我们只使用已填充的数据
        indices = np.arange(self.ptr)

        # 优势标准化
        adv_mean = np.mean(self.adv_buf[indices])
        adv_std = np.std(self.adv_buf[indices])
        self.adv_buf[indices] = (self.adv_buf[indices] - adv_mean) / (adv_std + 1e-8)

        sampler = BatchSampler(
            sampler=SubsetRandomSampler(indices),
            batch_size=batch_size,
            drop_last=True
        )

        for batch_indices in sampler:
            # --- 关键修正：修复了 `origin` 文件中的bug ---
            # 1. 不再引用不存在的 `self.action_mask_buf`。
            # 2. `yield` 的元组现在包含9个项目，顺序与 `compute_loss_pi` 的解包逻辑完全一致。
            yield (
                torch.as_tensor(self.obs_buf[batch_indices], dtype=torch.float32, device=self.device),
                torch.as_tensor(self.action_mask_dis_buf[batch_indices], dtype=torch.float32, device=self.device),
                torch.as_tensor(self.action_mask_con_buf[batch_indices], dtype=torch.float32, device=self.device),
                torch.as_tensor(self.act_dis_buf[batch_indices], dtype=torch.int64, device=self.device),
                torch.as_tensor(self.act_con_buf[batch_indices], dtype=torch.float32, device=self.device),
                torch.as_tensor(self.adv_buf[batch_indices], dtype=torch.float32, device=self.device),
                torch.as_tensor(self.ret_buf[batch_indices], dtype=torch.float32, device=self.device),
                torch.as_tensor(self.logp_dis_buf[batch_indices], dtype=torch.float32, device=self.device),
                torch.as_tensor(self.logp_con_buf[batch_indices], dtype=torch.float32, device=self.device),
            )

    def clear(self):
        self.ptr, self.path_start_dix = 0, 0


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ 第三部分：新的、基于注意力的混合Actor-Critic网络 (保持不变) +++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class ActorCritic_FRCF(nn.Module):
    """
    实现FRCF (Fully connected and Reshaped to Convolutional Feature-extraction) 网络。
    该网络将一维状态向量空间化，然后用CNN处理。
    """

    def __init__(self, state_dim, action_con_dim, action_dis_dim, action_dis_len,
                 cnn_input_channels, cnn_input_size, cnn_out_channels,
                 mlp_hidden_dim, init_log_std):
        super().__init__()
        self.action_dis_dim = action_dis_dim
        self.action_dis_len = action_dis_len
        self.cnn_input_channels = cnn_input_channels
        self.cnn_input_size = cnn_input_size

        # 1. 空间化层: FC_1 (Linear)
        # 将 state_dim 映射到 cnn_channels * size * size
        cnn_input_dim = cnn_input_channels * cnn_input_size * cnn_input_size
        self.spatialization_fc = nn.Linear(state_dim, cnn_input_dim)

        # 2. 卷积层: CNN block
        self.cnn = nn.Sequential(
            nn.Conv2d(cnn_input_channels, cnn_out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn_out_channels[0], cnn_out_channels[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # 动态计算CNN输出维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, cnn_input_channels, cnn_input_size, cnn_input_size)
            cnn_output_dim = self.cnn(dummy_input).shape[1]

        # 3. 全连接层: MLP Body
        self.mlp_body = nn.Sequential(
            nn.Linear(cnn_output_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU()
        )

        # 4. 输出头
        # Critic Head
        self.critic_head = nn.Linear(mlp_hidden_dim, 1)

        # Actor Continuous Head
        self.actor_con_head = nn.Linear(mlp_hidden_dim, action_con_dim)
        self.log_std = nn.Parameter(torch.full((action_con_dim,), init_log_std))

        # Actor Discrete Head
        self.actor_dis_head = nn.Linear(mlp_hidden_dim, action_dis_dim * action_dis_len)

        self.apply(weight_init)

    def forward_body(self, obs_vector):
        """网络主干部分的前向传播"""
        # a. 空间化和重塑
        x = self.spatialization_fc(obs_vector)
        x = F.relu(x)
        # b. 重塑为伪图像: (batch, channels, size, size)
        x = x.view(-1, self.cnn_input_channels, self.cnn_input_size, self.cnn_input_size)

        # c. CNN特征提取
        x = self.cnn(x)

        # d. MLP主体
        x = self.mlp_body(x)
        return x

    def get_value(self, obs_vector):
        """获取评价值"""
        body_out = self.forward_body(obs_vector)
        return self.critic_head(body_out)

    def get_logprob_entropy(self, obs_vector, action_dis, action_con, action_mask):
        """计算给定动作的对数概率和策略熵"""
        body_out = self.forward_body(obs_vector)

        # --- 离散动作 ---
        action_logits = self.actor_dis_head(body_out).view(-1, self.action_dis_dim, self.action_dis_len)
        discrete_mask = action_mask['discrete_mask'].view(-1, self.action_dis_dim, self.action_dis_len)
        masked_logits = action_logits.masked_fill(discrete_mask == 0, -1e9)
        dist_dis = Categorical(logits=masked_logits)

        # .squeeze(-1) 是因为环境动作通常是 (batch, 1)
        logprobs_dis = dist_dis.log_prob(action_dis.squeeze(-1).long())
        dist_entropy_dis = dist_dis.entropy().sum(dim=-1)

        # --- 连续动作 ---
        mean_raw = self.actor_con_head(body_out)
        std = torch.exp(self.log_std)
        con_mask_bounds = action_mask['continuous_mask']
        # 裁剪均值以尊重掩码边界，防止采样到无效区域
        clipped_mean = torch.clamp(mean_raw, con_mask_bounds[:, :, 0], con_mask_bounds[:, :, 1])
        dist_con = Normal(clipped_mean, std)

        logprobs_con = dist_con.log_prob(action_con).sum(dim=-1)
        dist_entropy_con = dist_con.entropy().sum(dim=-1)

        return logprobs_dis, logprobs_con, dist_entropy_dis, dist_entropy_con

    def select_action(self, obs_vector, action_mask, deterministic=False):
        """从策略中采样动作"""
        body_out = self.forward_body(obs_vector)
        state_value = self.critic_head(body_out)

        # --- 离散动作 ---
        action_logits = self.actor_dis_head(body_out).view(-1, self.action_dis_dim, self.action_dis_len)
        discrete_mask = torch.FloatTensor(action_mask['discrete_mask']).unsqueeze(0).to(obs_vector.device)
        masked_logits = action_logits.masked_fill(discrete_mask == 0, -1e9)
        dist_dis = Categorical(logits=masked_logits)
        action_dis = dist_dis.sample()
        logprob_dis = dist_dis.log_prob(action_dis).sum(dim=-1)

        # --- 连续动作 ---
        mean_raw = self.actor_con_head(body_out)
        std = torch.exp(self.log_std)
        con_mask_bounds = torch.FloatTensor(action_mask['continuous_mask']).unsqueeze(0).to(obs_vector.device)
        clipped_mean = torch.clamp(mean_raw, con_mask_bounds[:, :, 0], con_mask_bounds[:, :, 1])

        if deterministic:
            action_con = clipped_mean
        else:
            dist_con = Normal(clipped_mean, std)
            action_con = dist_con.sample()

        action_con_final = torch.clamp(action_con, con_mask_bounds[:, :, 0], con_mask_bounds[:, :, 1])
        # 使用最终裁剪的动作计算log_prob
        log_prob_con = Normal(clipped_mean, std).log_prob(action_con_final).sum(dim=-1)

        return (state_value,
                (action_dis, action_con_final),
                (logprob_dis, log_prob_con))


class PPO_FRCF:
    """PPO Agent using the FRCF Actor-Critic"""

    def __init__(self, state_dim, action_dis_dim, action_dis_len, action_con_dim,
                 lr_actor, lr_critic, lr_decay_rate, buffer_size,
                 gamma, lam, epochs_update, eps_clip, max_norm, coeff_entropy,
                 random_seed, device, lr_std, init_log_std,
                 # FRCF-specific hyperparams
                 cnn_input_channels, cnn_input_size, cnn_out_channels, mlp_hidden_dim,
                 target_kl_dis, target_kl_con, **kwargs):

        self.gamma, self.lam, self.epochs_update, self.eps_clip = gamma, lam, epochs_update, eps_clip
        self.max_norm, self.coeff_entropy, self.random_seed, self.device = max_norm, coeff_entropy, random_seed, device
        self.target_kl_dis, self.target_kl_con = target_kl_dis, target_kl_con
        self.set_random_seeds()

        self.buffer = PPOBuffer(state_dim, action_dis_dim, action_dis_len, action_con_dim, buffer_size, gamma, lam,
                                device)

        agent_args = {
            'state_dim': state_dim, 'action_con_dim': action_con_dim, 'action_dis_dim': action_dis_dim,
            'action_dis_len': action_dis_len, 'cnn_input_channels': cnn_input_channels,
            'cnn_input_size': cnn_input_size, 'cnn_out_channels': cnn_out_channels,
            'mlp_hidden_dim': mlp_hidden_dim, 'init_log_std': init_log_std
        }

        self.agent = ActorCritic_FRCF(**agent_args).to(device)
        self.agent_old = ActorCritic_FRCF(**agent_args).to(device)
        self.agent_old.load_state_dict(self.agent.state_dict())

        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=lr_actor)
        self.loss_func = nn.SmoothL1Loss(reduction='mean')

    def select_action(self, state, action_mask):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            value_tensor, (action_dis_tensor, action_con_tensor), (logp_dis_tensor, logp_con_tensor) = \
                self.agent_old.select_action(state_tensor, action_mask)

        return (value_tensor.squeeze().cpu().numpy().item(),
                (action_dis_tensor.squeeze().cpu().numpy(), action_con_tensor.squeeze().cpu().numpy()),
                (logp_dis_tensor.squeeze().cpu().numpy().item(), logp_con_tensor.squeeze().cpu().numpy().item()))

    def compute_loss(self, data):
        obs, mask_dis, mask_con, act_dis, act_con, adv, ret, logp_old_dis, logp_old_con = data

        action_mask_dict = {
            'discrete_mask': mask_dis.view(-1, self.agent.action_dis_dim, self.agent.action_dis_len),
            'continuous_mask': mask_con
        }

        # --- Actor Loss ---
        logp_dis, logp_con, entropy_dis, entropy_con = self.agent.get_logprob_entropy(obs, act_dis, act_con,
                                                                                      action_mask_dict)

        ratio_dis = torch.exp(logp_dis - logp_old_dis.squeeze(-1))
        surr1_dis = ratio_dis * adv
        surr2_dis = torch.clamp(ratio_dis, 1 - self.eps_clip, 1 + self.eps_clip) * adv
        loss_pi_dis = - (torch.min(surr1_dis, surr2_dis)).mean()

        ratio_con = torch.exp(logp_con - logp_old_con.sum(dim=-1))
        surr1_con = ratio_con * adv
        surr2_con = torch.clamp(ratio_con, 1 - self.eps_clip, 1 + self.eps_clip) * adv
        loss_pi_con = - (torch.min(surr1_con, surr2_con)).mean()

        loss_entropy = - self.coeff_entropy * (entropy_dis.mean() + entropy_con.mean())
        loss_pi = loss_pi_dis + loss_pi_con + loss_entropy

        # --- Critic Loss ---
        state_values = self.agent.get_value(obs)
        loss_v = self.loss_func(state_values, ret.unsqueeze(1))

        # KL for early stopping
        approx_kl_dis = (logp_old_dis.squeeze(-1) - logp_dis).mean().item()
        approx_kl_con = (logp_old_con.sum(dim=-1) - logp_con).mean().item()

        return loss_pi, loss_v, approx_kl_dis, approx_kl_con

    def update(self, batch_size):
        self.agent.train()

        for i in range(self.epochs_update):
            stop_update = False
            sampler = self.buffer.get(batch_size)
            for data in sampler:
                loss_pi, loss_v, kl_dis, kl_con = self.compute_loss(data)

                # Early stopping
                if (self.target_kl_dis is not None and kl_dis > 1.5 * self.target_kl_dis) or \
                        (self.target_kl_con is not None and kl_con > 1.5 * self.target_kl_con):
                    stop_update = True
                    break

                # Total loss and optimization
                total_loss = loss_pi + loss_v
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_norm)
                self.optimizer.step()

            if stop_update:
                print(f"Early stopping at epoch {i} due to high KL divergence.")
                break

        self.buffer.clear()
        self.agent_old.load_state_dict(self.agent.state_dict())

    def save(self, checkpoint_path):
        torch.save(self.agent_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.agent_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.agent.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
    def set_random_seeds(self):
        os.environ['PYTHONHASHSEED'] = str(self.random_seed)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
        torch.manual_seed(self.random_seed);
        random.seed(self.random_seed);
        np.random.seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed);
            torch.cuda.manual_seed(self.random_seed)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ 第四部分：PPO主类 (整合与完善) +++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class PPO_Abstract(ABC):
    def __init__(self, gamma, lam, epochs_update, eps_clip, max_norm, coeff_entropy, random_seed, device):
        self.gamma, self.lam, self.epochs_update, self.eps_clip, self.max_norm = gamma, lam, epochs_update, eps_clip, max_norm
        self.coeff_entropy, self.random_seed, self.device = coeff_entropy, random_seed, device
        self.agent, self.agent_old, self.buffer = None, None, None
        self.optimizer_actor, self.optimizer_critic = None, None
        self.loss_func = nn.SmoothL1Loss(reduction='mean')
        self.set_random_seeds()

    def select_action(self, state, action_mask): raise NotImplementedError

    def compute_loss_pi(self, data): raise NotImplementedError

    def compute_loss_v(self, data): raise NotImplementedError

    def update(self, batch_size): raise NotImplementedError

    def save(self, checkpoint_path): torch.save(self.agent_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.agent_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.agent.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    def set_random_seeds(self):
        os.environ['PYTHONHASHSEED'] = str(self.random_seed)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
        torch.manual_seed(self.random_seed);
        random.seed(self.random_seed);
        np.random.seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed);
            torch.cuda.manual_seed(self.random_seed)


# class PPO_Hybrid(PPO_Abstract):
#     def __init__(self,
#                  map_channels, img_h, img_w, num_sensors, sensor_dim, map_feature_dim,
#                  state_dim, action_dis_dim, action_dis_len, action_con_dim, mid_dim,
#                  lr_actor, lr_critic, lr_decay_rate, buffer_size, target_kl_dis, target_kl_con,
#                  gamma, lam, epochs_update, eps_clip, max_norm, coeff_entropy, random_seed, device,
#                  lr_std, init_log_std, **kwargs):  # Added **kwargs for compatibility
#
#         super().__init__(gamma, lam, epochs_update, eps_clip, max_norm, coeff_entropy, random_seed, device)
#
#         self.MAP_CHANNELS, self.IMG_H, self.IMG_W = map_channels, img_h, img_w
#         self.NUM_SENSORS, self.SENSOR_DIM = num_sensors, sensor_dim
#         self.action_dis_dim, self.action_dis_len = action_dis_dim, action_dis_len
#         self.target_kl_dis, self.target_kl_con = target_kl_dis, target_kl_con
#
#         self.buffer = PPOBuffer(state_dim, action_dis_dim, action_dis_len, action_con_dim, buffer_size, gamma, lam,
#                                 device)
#
#         agent_args = (map_channels, img_h, img_w, sensor_dim, map_feature_dim,
#                       action_dis_dim, action_dis_len, action_con_dim, mid_dim, init_log_std)
#         self.agent = ActorCritic_Hybrid_Attention(*agent_args).to(device)
#         self.agent.apply(weight_init)
#         self.agent_old = ActorCritic_Hybrid_Attention(*agent_args).to(device)
#         self.agent_old.load_state_dict(self.agent.state_dict())
#
#         # 分离的优化器，用于更精细的控制
#         self.optimizer_critic = torch.optim.Adam(self.agent.critic_head.parameters(), lr=lr_critic)
#         actor_shared_params = list(self.agent.map_encoder.parameters()) + list(
#             self.agent.sensor_encoder.parameters()) + list(self.agent.attention.parameters())
#         self.optimizer_actor_con = torch.optim.Adam([
#             {'params': self.agent.actor_con_head.parameters()},
#             {'params': actor_shared_params},
#             {'params': self.agent.log_std, 'lr': lr_std}], lr=lr_actor)
#         self.optimizer_actor_dis = torch.optim.Adam([
#             {'params': self.agent.actor_dis_head.parameters()},
#             {'params': actor_shared_params}], lr=lr_actor)
#
#         # 学习率调度器
#         self.lr_scheduler_critic = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_critic, lr_decay_rate)
#         self.lr_scheduler_actor_con = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_actor_con, lr_decay_rate)
#         self.lr_scheduler_actor_dis = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_actor_dis, lr_decay_rate)
#
#     def _prepare_inputs(self, obs_flat):
#         map_size = self.MAP_CHANNELS * self.IMG_H * self.IMG_W
#         sensor_size = self.NUM_SENSORS * self.SENSOR_DIM
#         assert obs_flat.shape[1] == map_size + sensor_size, "Observation dimension mismatch!"
#         map_flat, sensor_flat = obs_flat[:, :map_size], obs_flat[:, map_size:]
#         map_input = map_flat.view(-1, self.MAP_CHANNELS, self.IMG_H, self.IMG_W)
#         sensor_data = sensor_flat.view(-1, self.NUM_SENSORS, self.SENSOR_DIM)
#         return map_input, sensor_data
#
#     def select_action(self, state, action_mask):
#         with torch.no_grad():
#             map_input, sensor_data = self._prepare_inputs(torch.FloatTensor(state).unsqueeze(0).to(self.device))
#             fused_vector = self.agent_old._forward_body(map_input, sensor_data)
#
#             # 连续动作
#             mean_raw = self.agent_old.actor_con_head(fused_vector)
#             std = torch.exp(self.agent_old.log_std)
#             con_mask_bounds = torch.FloatTensor(action_mask['continuous_mask']).unsqueeze(0).to(self.device)
#             clipped_mean = torch.clamp(mean_raw, con_mask_bounds[:, :, 0], con_mask_bounds[:, :, 1])
#             dist_con = Normal(clipped_mean, std)
#             action_con = dist_con.sample()
#             action_con_final = torch.clamp(action_con, con_mask_bounds[:, :, 0], con_mask_bounds[:, :, 1])
#             log_prob_con = dist_con.log_prob(action_con_final).sum(dim=-1)
#
#             # 离散动作
#             action_logits = self.agent_old.actor_dis_head(fused_vector).view(-1, self.action_dis_dim,
#                                                                              self.action_dis_len)
#             discrete_mask = torch.FloatTensor(action_mask['discrete_mask']).unsqueeze(0).to(self.device)
#             masked_logits = action_logits.masked_fill(discrete_mask == 0, -1e9)
#             dist_dis = Categorical(logits=masked_logits)
#             action_dis = dist_dis.sample()
#             logprob_dis = dist_dis.log_prob(action_dis).sum(dim=-1)
#
#             state_value = self.agent_old.critic_head(fused_vector)
#
#         return (state_value.squeeze().cpu().numpy().item(),
#                 (action_dis.squeeze().cpu().numpy(), action_con_final.squeeze().cpu().numpy()),
#                 (logprob_dis.squeeze().cpu().numpy().item(), log_prob_con.squeeze().cpu().numpy().item()))
#
#     def compute_loss_pi(self, data):
#         # --- 关键修正：这里的解包顺序现在与 PPOBuffer.get() 完全匹配 ---
#         obs, action_mask_dis, action_mask_con, act_dis, act_con, adv, _, logp_old_dis, logp_old_con = data
#
#         map_input, sensor_data = self._prepare_inputs(obs)
#         action_mask_dict = {
#             'discrete_mask': action_mask_dis.view(-1, self.action_dis_dim, self.action_dis_len),
#             'continuous_mask': action_mask_con
#         }
#
#         logp_dis, logp_con, entropy_dis, entropy_con = self.agent.get_logprob_entropy(
#             map_input, sensor_data, act_dis, act_con, action_mask_dict
#         )
#
#         ratio_dis = torch.exp(logp_dis - logp_old_dis.sum(dim=-1))
#         surr1_dis = ratio_dis * adv
#         surr2_dis = torch.clamp(ratio_dis, 1 - self.eps_clip, 1 + self.eps_clip) * adv
#         loss_pi_dis = - (torch.min(surr1_dis, surr2_dis) + self.coeff_entropy * entropy_dis).mean()
#
#         ratio_con = torch.exp(logp_con - logp_old_con.sum(dim=-1))
#         surr1_con = ratio_con * adv
#         surr2_con = torch.clamp(ratio_con, 1 - self.eps_clip, 1 + self.eps_clip) * adv
#         loss_pi_con = - (torch.min(surr1_con, surr2_con) + self.coeff_entropy * entropy_con).mean()
#
#         approx_kl_dis = (logp_old_dis.sum(dim=-1) - logp_dis).mean().item()
#         approx_kl_con = (logp_old_con.sum(dim=-1) - logp_con).mean().item()
#
#         return loss_pi_dis, loss_pi_con, approx_kl_dis, approx_kl_con
#
#     def compute_loss_v(self, data):
#         # 解包是脆弱的，但只要 get() 的结构稳定，它就能工作。
#         # 它需要元组中的第1个元素(obs)和第7个元素(ret)。
#         obs, _, _, _, _, _, ret, _, _ = data
#         map_input, sensor_data = self._prepare_inputs(obs)
#         state_values = self.agent.get_value(map_input, sensor_data)
#         return self.loss_func(state_values, ret.unsqueeze(1))
#
#     def update(self, batch_size):
#         self.agent.train()
#         pi_losses_dis, pi_losses_con, v_losses, kl_dis_all, kl_con_all = [], [], [], [], []
#
#         for i in range(self.epochs_update):
#             stop_dis, stop_con = False, False
#             sampler = self.buffer.get(batch_size)
#             for data in sampler:
#                 loss_pi_dis, loss_pi_con, approx_kl_dis, approx_kl_con = self.compute_loss_pi(data)
#
#                 # KL 散度早停检查
#                 if self.target_kl_dis is not None and approx_kl_dis > 1.5 * self.target_kl_dis:
#                     stop_dis = True
#                 if self.target_kl_con is not None and approx_kl_con > 1.5 * self.target_kl_con:
#                     stop_con = True
#
#                 # 优化离散和连续Actor
#                 if not stop_dis:
#                     self.optimizer_actor_dis.zero_grad()
#                     loss_pi_dis.backward(retain_graph=True)  # Retain graph as shared parts are used by both
#                     nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_norm)
#                     self.optimizer_actor_dis.step()
#
#                 if not stop_con:
#                     self.optimizer_actor_con.zero_grad()
#                     loss_pi_con.backward()
#                     nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_norm)
#                     self.optimizer_actor_con.step()
#
#                 # 优化Critic
#                 loss_v = self.compute_loss_v(data)
#                 self.optimizer_critic.zero_grad()
#                 loss_v.backward()
#                 nn.utils.clip_grad_norm_(self.agent.critic_head.parameters(), self.max_norm)
#                 self.optimizer_critic.step()
#
#                 pi_losses_dis.append(loss_pi_dis.item());
#                 pi_losses_con.append(loss_pi_con.item())
#                 v_losses.append(loss_v.item());
#                 kl_dis_all.append(approx_kl_dis);
#                 kl_con_all.append(approx_kl_con)
#
#             if stop_dis and stop_con:
#                 print(f"Early stopping at epoch {i} due to high KL divergence for both action types.")
#                 break
#
#         self.buffer.clear()
#
#         # 逻辑完善：在一个完整的 update 调用（包含所有epochs）之后，再更新学习率和旧策略
#         self.lr_scheduler_actor_dis.step()
#         self.lr_scheduler_actor_con.step()
#         self.lr_scheduler_critic.step()
#         self.agent_old.load_state_dict(self.agent.state_dict())