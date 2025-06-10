# 文件名: train_drone.py

import gymnasium as gym
import numpy as np
import torch
import os

# 从您的文件中导入核心类
from env.DroneNavigationEnv import DroneNavigationEnv
from hppo.hppo_actionmask_CNN import PPO_Hybrid
from visualization_utils import plot_trajectory, plot_remaining_data, plot_position_error, plot_uncertainty_radius


def flatten_observation(obs_dict, obs_space):
    """
    辅助函数：将环境返回的结构化观测字典扁平化为一个numpy向量。
    """
    return gym.spaces.utils.flatten(obs_space, obs_dict)


def train():
    """主训练函数"""

    # --- 1. 超参数配置 ---
    config = {
        # 环境参数
        "map_channels": 2,
        "img_h": 84,
        "img_w": 84,
        "num_sensors": 5,
        "sensor_dim": 4,  # est_x, est_y, radius, data
        "area_size": (1000.0, 1000.0),
        "max_steps_per_episode": 500,
        "solo_SN_data": 2e7,

        # !! 关键修正：添加缺失的动作维度参数 !!
        "action_dis_dim": 1,
        "action_dis_len": 2,
        "action_con_dim": 2,

        # 算法网络结构参数
        "map_feature_dim": 128,
        "mid_dim": [256, 256],
        "init_log_std": -0.5,

        # PPO 训练参数
        "lr_actor": 3e-4,
        "lr_critic": 1e-3,
        "lr_decay_rate": 0.99,
        "gamma": 0.99,
        "lam": 0.95,
        "eps_clip": 0.2,
        "epochs_update": 10,
        "target_kl_dis": 0.02,
        "target_kl_con": 0.02,
        "batch_size": 64,
        "buffer_size": 2048,
        "max_norm": 0.5,
        "coeff_entropy": 0.01,
        "lr_std": 3e-4,
        "if_use_active_selection": False,
        "v_iters": 80,

        # 其他
        "random_seed": 42,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    # --- 日志、绘图和保存的超参数 ---
    log_interval = 1  # 每1个episode打印一次日志
    plot_interval = 100  # 每100个episodes绘制并保存一次图表
    save_interval = 100  # 每100个episodes保存一次模型

    # --- 2. 初始化环境和智能体 ---
    env = DroneNavigationEnv(config)

    state_dim = gym.spaces.utils.flatdim(env.observation_space)
    config["state_dim"] = state_dim
    print(f"扁平化后的状态空间维度: {state_dim}")
    print(f"使用的设备: {config['device']}")

    ppo_required_keys = [
        'map_channels', 'img_h', 'img_w', 'num_sensors', 'sensor_dim',
        'map_feature_dim', 'state_dim', 'action_dis_dim', 'action_dis_len',
        'action_con_dim', 'mid_dim', 'lr_actor', 'lr_critic', 'lr_decay_rate',
        'buffer_size', 'target_kl_dis', 'target_kl_con', 'gamma', 'lam',
        'epochs_update', 'v_iters', 'eps_clip', 'max_norm', 'coeff_entropy',
        'random_seed', 'device', 'lr_std', 'init_log_std', 'if_use_active_selection'
    ]

    # 从主config中筛选出PPO需要的参数
    agent_config = {key: config[key] for key in ppo_required_keys}

    # 使用筛选后的agent_config来实例化智能体
    agent = PPO_Hybrid(**agent_config)

    # --- 3. 创建文件夹 ---
    os.makedirs("./plots", exist_ok=True)
    os.makedirs("./models", exist_ok=True)

    # --- 4. 训练循环 ---
    max_train_episodes = 5001  # 训练5000个回合

    for episode in range(max_train_episodes):
        obs_dict, info = env.reset()
        current_ep_reward = 0
        terminated, truncated = False, False

        # --- 单个 Episode 的交互循环 ---
        while not (terminated or truncated):
            # a. 扁平化观测
            flat_obs = flatten_observation(obs_dict, env.observation_space)

            # b. 智能体选择动作
            action_mask_dict = {
                "discrete_mask": obs_dict["action_mask_discrete"],
                "continuous_mask": obs_dict["action_mask_continuous"]
            }
            value, (action_dis, action_con), (logp_dis, logp_con) = agent.select_action(flat_obs, action_mask_dict)

            # c. 格式化动作用于环境
            action_dict = {"discrete": action_dis, "continuous": action_con}

            # d. 与环境交互
            next_obs_dict, reward, terminated, truncated, info = env.step(action_dict)

            # e. 存储经验
            # 注意：您的PPOBuffer实现似乎需要logp是单个numpy值或特定形状的数组，这里做了相应处理
            agent.buffer.store_hybrid(
                flat_obs,
                {"discrete_mask": obs_dict["action_mask_discrete"]},
                np.array([action_dis]),  # 确保是数组
                action_con,
                reward,
                value,
                np.array([logp_dis]),  # 确保是数组
                logp_con
            )

            obs_dict = next_obs_dict
            current_ep_reward += reward

            # f. 如果Buffer满了，则更新网络
            if agent.buffer.ptr == agent.buffer.max_size:
                last_val = 0
                if not (terminated or truncated):  # 如果不是回合的终点
                    # 计算最后一个状态的价值
                    last_val_flat_obs = flatten_observation(next_obs_dict, env.observation_space)
                    map_input, sensor_data = agent._prepare_inputs(
                        torch.FloatTensor(last_val_flat_obs).unsqueeze(0).to(agent.device))
                    with torch.no_grad():
                        last_val = agent.agent.get_value(map_input, sensor_data).cpu().numpy().flatten()[0]

                agent.buffer.finish_path(last_val)
                agent.update(config["batch_size"])
                agent.buffer.clear()

        # ------------------------------------------------------------------
        # !! 关键修改：所有日志和绘图逻辑都移到 while 循环之外 !!
        # ------------------------------------------------------------------

        # --- 5. 日志记录 (每个episode结束时打印) ---
        print(f"Episode: {episode}, Reward: {current_ep_reward:.2f}, Steps: {env.current_step}")

        # --- 6. 绘图与模型保存 (每隔一定周期) ---
        if episode > 0 and episode % plot_interval == 0:
            print(f"\n--- Episode {episode}: 生成可视化图表和保存模型 ---")

            # a. 创建本次保存的专属文件夹
            plot_dir = f"./plots/episode_{episode}"
            os.makedirs(plot_dir, exist_ok=True)

            # b. 调用绘图函数
            plot_trajectory(env.trajectory, env.sensor_true_positions, f"{plot_dir}/trajectory.png", env.area_size)
            plot_remaining_data(env.communication_log, env.num_sensors, env.solo_SN_data,
                                f"{plot_dir}/remaining_data.png")
            plot_position_error(env.localization_log, env.sensor_true_positions, f"{plot_dir}/position_error.png")
            plot_uncertainty_radius(env.localization_log, f"{plot_dir}/uncertainty_radius.png")

            print(f"图表已保存至: {plot_dir}")

        if episode > 0 and episode % save_interval == 0:
            # c. 保存模型
            save_path = f"./models/ppo_drone_episode_{episode}.pth"
            agent.save(save_path)
            print(f"模型已保存至: {save_path}\n")

    env.close()


if __name__ == "__main__":
    train()