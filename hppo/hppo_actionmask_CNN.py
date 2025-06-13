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
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        nn.init.zeros_(m.bias)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ 第一部分：新增的独立网络模块 +++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class MapEncoder(nn.Module):
    """
    使用CNN处理多通道地图输入（奖励地图、无人机位置地图等）。
    """

    def __init__(self, in_channels, img_h, img_w, feature_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # 动态计算卷积后的平坦化维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, img_h, img_w)
            dummy_output_dim = self.cnn(dummy_input).shape[1]

        self.fc = nn.Linear(dummy_output_dim, feature_dim)
        self.ln = nn.LayerNorm(feature_dim)

    def forward(self, map_input):
        conv_out = self.cnn(map_input)
        features = self.fc(conv_out)
        features = self.ln(F.relu(features))
        return features


class SensorEncoder(nn.Module):
    """
    使用共享的MLP为每个传感器生成嵌入向量。
    """

    def __init__(self, sensor_dim, embedding_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(sensor_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        self.ln = nn.LayerNorm(embedding_dim)

    def forward(self, sensor_data):
        embeddings = self.encoder(sensor_data)
        embeddings = self.ln(F.relu(embeddings))
        return embeddings


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ 第二部分：您原有的代码（Buffer和旧网络） +++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of observations-action pairs.
    """

    def __init__(self, obs_dim, act_dis_dim, act_dis_len, act_con_dim, size, gamma, lam, device):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)

        # 重命名 action_mask_buf 为 action_mask_dis_buf 以明确其用途
        self.action_mask_dis_buf = np.ones((size, act_dis_dim * act_dis_len), dtype=np.float32)
        # 新增一个缓冲区来存储连续动作的掩码 (shape: batch, action_dim, 2 for min/max)
        self.action_mask_con_buf = np.zeros((size, act_con_dim, 2), dtype=np.float32)
        # --- 修改结束 ---
        self.act_dis_buf = np.zeros((size, act_dis_dim), dtype=np.int64)
        self.act_con_buf = np.zeros((size, act_con_dim), dtype=np.float32)

        # self.action_mask_buf = np.ones((size, act_dis_dim * act_dis_len), dtype=np.float32)
        # self.act_dis_buf = np.zeros((size, act_dis_dim), dtype=np.int64)
        # self.act_con_buf = np.zeros((size, act_con_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_dis_buf = np.zeros((size, act_dis_dim), dtype=np.float32)
        self.logp_con_buf = np.zeros((size, act_con_dim), dtype=np.float32)
        self.ptr_buf = np.zeros(size, dtype=np.int64)

        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_dix, self.max_size = 0, 0, size

        self.device = device

    def store_hybrid(self, obs, action_mask, act_dis, act_con, rew, val, logp_dis, logp_con):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        # # 注意: action_mask现在是一个字典，需要扁平化存储
        # self.action_mask_buf[self.ptr] = action_mask['discrete_mask'].flatten()

        # 现在存储离散和连续的掩码
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
        obs_buf = self.obs_buf[:self.ptr]
        action_mask_buf = self.action_mask_buf[:self.ptr]

        action_mask_dis_buf = self.action_mask_dis_buf[:self.ptr]
        action_mask_con_buf = self.action_mask_con_buf[:self.ptr]

        act_dis_buf = self.act_dis_buf[:self.ptr]
        act_con_buf = self.act_con_buf[:self.ptr]
        adv_buf = self.adv_buf[:self.ptr]
        ret_buf = self.ret_buf[:self.ptr]
        logp_dis_buf = self.logp_dis_buf[:self.ptr]
        logp_con_buf = self.logp_con_buf[:self.ptr]

        # 优势标准化
        adv_buf = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)

        sampler = BatchSampler(
            sampler=SubsetRandomSampler(range(self.ptr)),
            batch_size=batch_size,
            drop_last=True
        )

        for indices in sampler:
            yield (

                # 将两个掩码都传递出去
                torch.as_tensor(action_mask_dis_buf[indices], dtype=torch.float32, device=self.device),
                torch.as_tensor(action_mask_con_buf[indices], dtype=torch.float32, device=self.device),

                torch.as_tensor(obs_buf[indices], dtype=torch.float32, device=self.device),
                torch.as_tensor(action_mask_buf[indices], dtype=torch.float32, device=self.device),
                torch.as_tensor(act_dis_buf[indices], dtype=torch.int64, device=self.device),
                torch.as_tensor(act_con_buf[indices], dtype=torch.float32, device=self.device),
                torch.as_tensor(adv_buf[indices], dtype=torch.float32, device=self.device),
                torch.as_tensor(ret_buf[indices], dtype=torch.float32, device=self.device),
                torch.as_tensor(logp_dis_buf[indices], dtype=torch.float32, device=self.device),
                torch.as_tensor(logp_con_buf[indices], dtype=torch.float32, device=self.device),
            )

    def clear(self):
        self.ptr, self.path_start_dix = 0, 0


# (保留您原有的 ActorCritic_Discrete, ActorCritic_Continuous, 和 ActorCritic_Hybrid 以供参考)
class ActorCritic_Hybrid(nn.Module):
    # ... 您原有的实现 ...
    pass


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ 第三部分：新的、基于注意力的混合Actor-Critic网络 +++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class ActorCritic_Hybrid_Attention(nn.Module):
    def __init__(self,
                 map_channels, img_h, img_w,
                 sensor_dim,
                 map_feature_dim,
                 action_dis_dim, action_dis_len, action_con_dim,
                 mid_dim,
                 init_log_std):
        super().__init__()

        # --- 1. 共享的特征提取主体 (Shared Body) ---
        self.map_encoder = MapEncoder(map_channels, img_h, img_w, map_feature_dim)

        # 假设传感器嵌入维度与地图特征维度相同，以便于注意力计算
        sensor_embedding_dim = map_feature_dim
        self.sensor_encoder = SensorEncoder(sensor_dim, sensor_embedding_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=map_feature_dim,
            num_heads=4,
            batch_first=True
        )

        # --- 2. 分离的决策头 (Separate Heads) ---
        final_fusion_dim = map_feature_dim + sensor_embedding_dim

        self.critic_head = nn.Sequential(
            nn.Linear(final_fusion_dim, mid_dim[0]), nn.Tanh(), nn.Linear(mid_dim[0], 1)
        )

        self.actor_con_head = nn.Sequential(
            nn.Linear(final_fusion_dim, mid_dim[0]), nn.Tanh(), nn.Linear(mid_dim[0], action_con_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_con_dim) + init_log_std)

        self.actor_dis_head = nn.Sequential(
            nn.Linear(final_fusion_dim, mid_dim[0]), nn.Tanh(), nn.Linear(mid_dim[0], action_dis_dim * action_dis_len)
        )

        self.action_dis_dim = action_dis_dim
        self.action_dis_len = action_dis_len

    def _forward_body(self, map_input, sensor_data):
        """主前向传播逻辑，输出融合后的特征"""
        global_context_vector = self.map_encoder(map_input)
        sensor_embeddings = self.sensor_encoder(sensor_data)
        query = global_context_vector.unsqueeze(1)
        attended_context_vector, _ = self.attention(query, sensor_embeddings, sensor_embeddings)
        attended_context_vector = attended_context_vector.squeeze(1)
        fused_vector = torch.cat([global_context_vector, attended_context_vector], dim=1)
        return fused_vector

    def get_logprob_entropy(self, map_input, sensor_data, action_dis, action_con, action_mask):
        """根据融合特征计算log_prob和熵"""
        fused_vector = self._forward_body(map_input, sensor_data)

        # 离散动作
        action_logits = self.actor_dis_head(fused_vector).view(-1, self.action_dis_dim, self.action_dis_len)
        discrete_mask = action_mask['discrete_mask'].view(-1, self.action_dis_dim, self.action_dis_len)
        masked_logits = action_logits.masked_fill(discrete_mask == 0, -1e9)
        dist_dis = Categorical(logits=masked_logits)
        logprobs_dis = dist_dis.log_prob(action_dis.squeeze().long()).sum(dim=-1)
        dist_entropy_dis = dist_dis.entropy().sum(dim=-1)

        # 连续动作
        mean_raw = self.actor_con_head(fused_vector)
        std = torch.exp(self.log_std)

        # 获取连续动作掩码的边界
        con_mask_bounds = action_mask['continuous_mask']
        # 像 select_action 中一样，裁剪均值
        clipped_mean = torch.clamp(mean_raw, con_mask_bounds[:, 0], con_mask_bounds[:, 1])
        # 使用裁剪后的均值创建分布
        dist_con = Normal(clipped_mean, std)
        # 计算缓冲区中的动作 (act_con) 在这个分布下的对数概率
        # 注意：缓冲区中的 act_con 已经被 select_action 裁剪过，所以它一定在分布的支持域内
        logprobs_con = dist_con.log_prob(action_con).sum(dim=-1)
        dist_entropy_con = dist_con.entropy().sum(dim=-1)

        return logprobs_dis, logprobs_con, dist_entropy_dis, dist_entropy_con

    def act(self, map_input, sensor_data, action_mask):
        """在选择动作时使用"""
        fused_vector = self._forward_body(map_input, sensor_data)

        state_value = self.critic_head(fused_vector)

        action_logits = self.actor_dis_head(fused_vector).view(-1, self.action_dis_dim, self.action_dis_len)
        discrete_mask = action_mask['discrete_mask'].view(-1, self.action_dis_dim, self.action_dis_len)
        masked_logits = action_logits.masked_fill(discrete_mask == 0, -1e9)
        dist_dis = Categorical(logits=masked_logits)
        action_dis = dist_dis.sample()
        logprob_dis = dist_dis.log_prob(action_dis).sum(dim=-1)

        mean = self.actor_con_head(fused_vector)
        std = torch.exp(self.log_std)
        dist_con = Normal(mean, std)
        action_con = dist_con.sample()
        logprob_con = dist_con.log_prob(action_con).sum(dim=-1)

        return state_value, action_dis, action_con, logprob_dis, logprob_con

    def get_value(self, map_input, sensor_data):
        """只获取价值，用于更新"""
        fused_vector = self._forward_body(map_input, sensor_data)
        return self.critic_head(fused_vector)


# (保留您原有的 PPO_Abstract, PPO_Discrete, PPO_Continuous)
class PPO_Abstract(ABC):  # 明确它是一个抽象基类
    def __init__(self,
                 # 这里只保留所有子类都共有的超参数
                 gamma, lam, epochs_update, eps_clip, max_norm,
                 coeff_entropy, random_seed, device
                 ):
        # 父类只负责存储通用参数
        self.gamma = gamma
        self.lam = lam
        self.epochs_update = epochs_update
        self.eps_clip = eps_clip
        self.max_norm = max_norm
        self.coeff_entropy = coeff_entropy
        self.random_seed = random_seed
        self.device = device

        # agent, buffer, optimizers等都将在子类中定义
        self.agent = None
        self.agent_old = None
        self.buffer = None
        self.optimizer_actor = None
        self.optimizer_critic = None
        self.loss_func = nn.SmoothL1Loss(reduction='mean')

        # 这个方法现在可以被正确继承和调用
        self.set_random_seeds()

    # --- 以下方法被所有子类继承，无需修改 ---
    def select_action(self, state, action_mask):
        raise NotImplementedError

    def compute_loss_pi(self, data):
        raise NotImplementedError

    def compute_loss_v(self, data):
        raise NotImplementedError

    def update(self, batch_size):
        raise NotImplementedError

    def save(self, checkpoint_path):
        torch.save(self.agent_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.agent_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.agent.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    def set_random_seeds(self):
        os.environ['PYTHONHASHSEED'] = str(self.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(self.random_seed)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
            torch.cuda.manual_seed(self.random_seed)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ 第四部分：修改后的PPO_Hybrid类 +++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class PPO_Hybrid(PPO_Abstract, ABC):
    def __init__(self,
                 # --- 新增的架构和维度参数 ---
                 # !! 用户必须根据自己的环境配置这些参数 !!
                 map_channels, img_h, img_w,
                 num_sensors, sensor_dim,
                 map_feature_dim,
                 # --- 原有参数 ---
                 state_dim,  # state_dim 现在是扁平化后的总维度
                 action_dis_dim, action_dis_len, action_con_dim, mid_dim, lr_actor, lr_critic,
                 lr_decay_rate, buffer_size, target_kl_dis, target_kl_con,
                 gamma, lam, epochs_update, v_iters, eps_clip, max_norm, coeff_entropy, random_seed, device,
                 lr_std, init_log_std, if_use_active_selection):
        # PPO_Abstract的 __init__ 可能需要微调，或者直接在这里处理
        # --- 关键修改 1：正确调用父类初始化 ---
        # 只传递父类需要的通用参数
        super().__init__(gamma=gamma, lam=lam, epochs_update=epochs_update,
                         eps_clip=eps_clip, max_norm=max_norm,
                         coeff_entropy=coeff_entropy, random_seed=random_seed,
                         device=device)
        self.MAP_CHANNELS = map_channels
        self.IMG_H = img_h
        self.IMG_W = img_w
        self.NUM_SENSORS = num_sensors
        self.SENSOR_DIM = sensor_dim
        # --- 关键修改 2：子类负责自己的所有具体实现 ---
        self.target_kl_dis = target_kl_dis
        self.target_kl_con = target_kl_con

        # 初始化Buffer
        self.buffer = PPOBuffer(state_dim, action_dis_dim, action_dis_len, action_con_dim, buffer_size, gamma, lam,
                                device)

        # 初始化注意力网络Agent
        self.agent = ActorCritic_Hybrid_Attention(
            map_channels, img_h, img_w,
            sensor_dim,
            map_feature_dim,
            action_dis_dim, action_dis_len, action_con_dim,
            mid_dim, init_log_std
        ).to(device)
        self.agent.apply(weight_init)

        self.agent_old = ActorCritic_Hybrid_Attention(
            map_channels, img_h, img_w,
            sensor_dim,
            map_feature_dim,
            action_dis_dim, action_dis_len, action_con_dim,
            mid_dim, init_log_std
        ).to(device)
        self.agent_old.load_state_dict(self.agent.state_dict())

        # 初始化优化器和学习率调度器
        self.optimizer_critic = torch.optim.Adam(self.agent.critic_head.parameters(), lr=lr_critic)
        self.optimizer_actor_con = torch.optim.Adam([
            {'params': self.agent.actor_con_head.parameters(), 'lr': lr_actor},
            {'params': self.agent.log_std, 'lr': lr_std},
        ])
        self.optimizer_actor_dis = torch.optim.Adam(self.agent.actor_dis_head.parameters(), lr=lr_actor)

        self.lr_scheduler_critic = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_critic,
                                                                          gamma=lr_decay_rate)
        self.lr_scheduler_actor_con = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_actor_con,
                                                                             gamma=lr_decay_rate)
        self.lr_scheduler_actor_dis = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_actor_dis,
                                                                             gamma=lr_decay_rate)

    def _prepare_inputs(self, obs_flat):
        """
        辅助函数，将从buffer中取出的扁平化obs向量还原为结构化输入。
        !! 这是您需要根据自己的环境来具体实现的，尺寸必须正确 !!
        """
        map_size = self.MAP_CHANNELS * self.IMG_H * self.IMG_W
        sensor_size = self.NUM_SENSORS * self.SENSOR_DIM

        assert obs_flat.shape[1] == map_size + sensor_size, "Observation dimension mismatch!"

        map_flat = obs_flat[:, :map_size]
        sensor_flat = obs_flat[:, map_size:]

        map_input = map_flat.view(-1, self.MAP_CHANNELS, self.IMG_H, self.IMG_W)
        sensor_data = sensor_flat.view(-1, self.NUM_SENSORS, self.SENSOR_DIM)

        return map_input, sensor_data

    def select_action(self, state, action_mask):
        with torch.no_grad():
            # 1. 准备输入：将扁平化的状态向量还原为网络需要的结构化输入
            map_input, sensor_data = self._prepare_inputs(torch.FloatTensor(state).unsqueeze(0).to(self.device))

            # 2. 共享前向传播：通过共享的主体网络获取融合后的特征向量
            #    这个特征向量将被 actor 和 critic 的头部共同使用
            fused_vector = self.agent_old._forward_body(map_input, sensor_data)

            # -----------------------------------------------------------------
            # 3. 处理连续动作 (这是关键修正部分)
            # -----------------------------------------------------------------
            # a. 从头部网络获取原始的动作分布均值
            mean_raw = self.agent_old.actor_con_head(fused_vector)
            # 获取动作分布的标准差
            std = torch.exp(self.agent_old.log_std)

            # b. 将 NumPy 格式的掩码转换为 PyTorch Tensor
            con_mask_bounds = torch.FloatTensor(action_mask['continuous_mask']).to(self.device)

            # c. 【核心修正】在创建分布之前，先将“均值”裁剪到掩码定义的有效范围内
            clipped_mean = torch.clamp(mean_raw, con_mask_bounds[:, 0], con_mask_bounds[:, 1])

            # d. 使用“裁剪后的均值”来创建正态分布，这使得采样更稳定
            dist_con = Normal(clipped_mean, std)

            # e. 从这个更安全的分布中采样一个动作
            action_con = dist_con.sample()

            # f. 作为最后的保险措施，再次裁剪采样出的动作，确保它绝对不会因为标准差而越界
            action_con_final = torch.clamp(action_con, con_mask_bounds[:, 0], con_mask_bounds[:, 1])

            # g. 【核心修正】计算“最终被执行的动作”的对数概率
            #    这保证了策略梯度更新的一致性
            log_prob_con = dist_con.log_prob(action_con_final).sum(dim=-1)

            # -----------------------------------------------------------------
            # 4. 处理离散动作 (这部分逻辑通常是正确的，保持完整)
            # -----------------------------------------------------------------
            # a. 从头部网络获取离散动作的原始 logits
            action_logits = self.agent_old.actor_dis_head(fused_vector).view(
                -1,
                self.agent_old.action_dis_dim,  # <-- 从 self.agent_old 获取
                self.agent_old.action_dis_len  # <-- 从 self.agent_old 获取
            )

            # b. 准备离散动作的掩码张量
            discrete_mask_tensor = torch.FloatTensor(action_mask['discrete_mask']).to(self.device).view(
                -1,
                self.agent_old.action_dis_dim,  # <-- 从 self.agent_old 获取
                self.agent_old.action_dis_len  # <-- 从 self.agent_old 获取
            )

            # c. 将掩码应用到 logits 上（将无效动作的概率设置为极小值）
            masked_logits = action_logits.masked_fill(discrete_mask_tensor == 0, -1e9)

            # d. 创建分类分布
            dist_dis = Categorical(logits=masked_logits)

            # e. 采样离散动作并计算其对数概率
            action_dis = dist_dis.sample()
            logprob_dis = dist_dis.log_prob(action_dis).sum(dim=-1)

            # -----------------------------------------------------------------
            # 5. 获取评价值
            # -----------------------------------------------------------------
            # 从 critic 头部网络获取当前状态的评价值 V(s)
            state_value = self.agent_old.critic_head(fused_vector)

            # -----------------------------------------------------------------
            # 6. 准备并返回所有结果
            # -----------------------------------------------------------------
            # 将所有 PyTorch Tensor 转换回 NumPy 数组，并调整形状以供环境和 Buffer 使用
        return (state_value.squeeze().cpu().numpy(),
                (action_dis.squeeze().cpu().numpy(), action_con_final.squeeze().cpu().numpy()),  # 返回最终的、修正后的连续动作
                (logprob_dis.squeeze().cpu().numpy(), log_prob_con.squeeze().cpu().numpy()))


    def compute_loss_pi(self, data):
        # 解包数据，现在多了 action_mask_con
        obs, action_mask_dis_flat, action_mask_con, act_dis, act_con, adv, _, logp_old_dis, logp_old_con = data

        map_input, sensor_data = self._prepare_inputs(obs)

        # 从解包的数据中重建掩码字典
        action_mask_dict = {
            'discrete_mask': action_mask_dis_flat.view(-1, self.action_dis_dim, self.action_dis_len),
            'continuous_mask': action_mask_con  # <-- 使用从buffer来的真实掩码！
        }

        # get_logprob_entropy也需要修正
        logp_dis, logp_con, dist_entropy_dis, dist_entropy_con = self.agent.get_logprob_entropy(
            map_input, sensor_data, act_dis, act_con, action_mask_dict
        )

        # (剩余的PPO损失计算逻辑与您原代码类似)
        ratio_dis = torch.exp(logp_dis - logp_old_dis.sum(dim=-1))
        ratio_con = torch.exp(logp_con - logp_old_con.sum(dim=-1))

        adv = adv.squeeze()
        clip_adv_dis = torch.clamp(ratio_dis, 1 - self.eps_clip, 1 + self.eps_clip) * adv
        clip_adv_con = torch.clamp(ratio_con, 1 - self.eps_clip, 1 + self.eps_clip) * adv

        loss_pi_dis = - (torch.min(ratio_dis * adv, clip_adv_dis) + self.coeff_entropy * dist_entropy_dis).mean()
        loss_pi_con = - (torch.min(ratio_con * adv, clip_adv_con) + self.coeff_entropy * dist_entropy_con).mean()

        approx_kl_dis = (logp_old_dis.sum(dim=-1) - logp_dis).mean().item()
        approx_kl_con = (logp_old_con.sum(dim=-1) - logp_con).mean().item()

        return loss_pi_dis, loss_pi_con, approx_kl_dis, approx_kl_con

    def compute_loss_v(self, data):
        obs, _, _, _, _, _, ret, _, _ = data
        # obs, _, _, _, _, ret, _, _ = data
        map_input, sensor_data = self._prepare_inputs(obs)
        state_values = self.agent.get_value(map_input, sensor_data)
        return self.loss_func(state_values, ret.unsqueeze(1))

    def update(self, batch_size):
        # 此处的更新逻辑与您原有的update函数非常相似
        # 只需要确保在调用compute_loss_pi和compute_loss_v时数据被正确处理即可
        # 为简洁起见，这里省略了与您原代码重复的循环和打印部分
        # 您只需将原有的update循环逻辑复制过来即可
        # ...
        # for i in range(self.epochs_update):
        #     sampler = self.buffer.get(batch_size)
        #     for data in sampler:
        #         # 您的优化器更新逻辑
        # ...
        pi_losses_dis, pi_losses_con, v_losses, kl_dis_all, kl_con_all = [], [], [], [], []

        for i in range(self.epochs_update):
            sampler = self.buffer.get(batch_size)
            for data in sampler:
                # 计算 Actor 损失
                loss_pi_dis, loss_pi_con, approx_kl_dis, approx_kl_con = self.compute_loss_pi(data)

                # KL 散度早停
                if self.target_kl_dis is not None and approx_kl_dis > 1.5 * self.target_kl_dis:
                    print(f"Early stopping at epoch {i} due to high KL divergence for discrete actions.")
                    break
                if self.target_kl_con is not None and approx_kl_con > 1.5 * self.target_kl_con:
                    print(f"Early stopping at epoch {i} due to high KL divergence for continuous actions.")
                    break

                # 优化离散动作 Actor
                self.optimizer_actor_dis.zero_grad()
                loss_pi_dis.backward()
                nn.utils.clip_grad_norm_(self.agent.actor_dis_head.parameters(), self.max_norm)
                self.optimizer_actor_dis.step()

                # 优化连续动作 Actor
                self.optimizer_actor_con.zero_grad()
                loss_pi_con.backward()
                nn.utils.clip_grad_norm_(self.agent.actor_con_head.parameters(), self.max_norm)
                self.optimizer_actor_con.step()

                # 计算 Critic 损失并优化
                loss_v = self.compute_loss_v(data)
                self.optimizer_critic.zero_grad()
                loss_v.backward()
                nn.utils.clip_grad_norm_(self.agent.critic_head.parameters(), self.max_norm)
                self.optimizer_critic.step()

                # 记录损失和KL
                pi_losses_dis.append(loss_pi_dis.item())
                pi_losses_con.append(loss_pi_con.item())
                v_losses.append(loss_v.item())
                kl_dis_all.append(approx_kl_dis)
                kl_con_all.append(approx_kl_con)

        # 更新旧策略网络
        self.agent_old.load_state_dict(self.agent.state_dict())

        # 学习率衰减
        self.lr_scheduler_actor_dis.step()
        self.lr_scheduler_actor_con.step()
        self.lr_scheduler_critic.step()