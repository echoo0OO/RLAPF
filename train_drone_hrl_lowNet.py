# 文件名: train_drone_hrl.py
# 描述: 使用分层强化学习 (HRL) 训练无人机导航任务的主脚本

import gymnasium as gym
import numpy as np
import torch
import os
import random
import datetime

# 【HRL】从我们重构后的文件中导入HRL版本的环境和智能体
from env.DroneNavigationEnv_hrl import DroneNavigationEnv
from hppo.hppo_actionmask_hrl import MetaControllerPPO, LowLevelPPO
from visualization_utils import plot_trajectory, plot_remaining_data, plot_position_error, plot_uncertainty_radius, \
    plot_episode_rewards


def flatten_observation(obs_dict, obs_space):
    """辅助函数：将环境返回的结构化观测字典扁平化为一个numpy向量。"""
    return gym.spaces.utils.flatten(obs_space, obs_dict)


def train_hrl():
    """HRL主训练函数"""

    # --- 1. 超参数配置 ---
    # 大部分与之前类似，但为HRL增加了特定的参数
    config = {
        # 环境参数
        "num_sensors": 5, "area_size": (1000.0, 1000.0),
        "max_steps_per_episode": 1000, "solo_SN_data": 2e7,
        "max_task_commitment_steps": 5,  # 一个高层决策最多执行多少低层步
        "approach_termination_sector_coverage": 8,
        "localize_termination_radius_thresh": 15.0,
        "maneuver_optimal_radius": 60.0,

        # HRL 智能体通用参数
        "gamma": 0.99, "lam": 0.95, "eps_clip": 0.2, "epochs_update": 10,
        "coeff_entropy": 0.01, "max_norm": 0.5,
        "random_seed": 42, "device": "cuda" if torch.cuda.is_available() else "cpu",

        # 低层智能体 (LowLevelPPO) 参数
        "low_level_state_dim": 34,  # 4 + 5 * 6
        "action_con_dim": 2,
        "num_tasks": 3,  # 对应高层的三个任务
        "low_buffer_size": 4096, "low_batch_size": 128,
        "lr_actor": 3e-4, "init_log_std": -0.5,
        "cnn_input_channels": 1, "cnn_input_size": 7,
        "cnn_out_channels": [32, 64], "mlp_hidden_dim": 256,

        # 高层智能体 (MetaControllerPPO) 参数
        "meta_state_dim": 34,  # 高低层使用相同的状态表示
        "meta_buffer_size": 1024, "meta_batch_size": 64,
        "meta_lr_actor": 5e-4,
        "meta_hidden_dim": 128,
    }

    # 日志、绘图和保存的超参数
    log_interval = 1
    plot_interval = 100
    save_interval = 1000
    update_interval_meta = 512  # 每收集多少高层经验后更新一次高层网络
    update_interval_low = 2048  # 每收集多少低层经验后更新一次低层网络

    # --- 2. 初始化HRL组件 ---
    # 设置随机种子
    torch.manual_seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    random.seed(config["random_seed"])

    # 初始化低层智能体
    low_level_agent = LowLevelPPO(
        state_dim=config["low_level_state_dim"],
        action_con_dim=config["action_con_dim"],
        num_tasks=config["num_tasks"],
        config=config,
        device=config["device"]
    )

    # 初始化高层智能体
    meta_agent = MetaControllerPPO(
        state_dim=config["meta_state_dim"],
        num_tasks=config["num_tasks"],
        config=config,
        device=config["device"]
    )

    # 初始化环境，并将低层智能体的 *旧策略* 传入
    # 使用旧策略进行数据收集是PPO的标准做法
    env = DroneNavigationEnv(config, low_level_agent)

    # --- 3. 创建文件夹和日志文件 ---
    run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"./logs_hrl/{run_name}"
    plot_dir = f"./plots_hrl/{run_name}"
    model_dir = f"./models_hrl/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    # 【新增】定义日志文件路径并写入表头
    log_file_path = os.path.join(log_dir, "rewards_log.csv")
    log_file = open(log_file_path, "w+")
    log_file.write("Episode,Reward,TotalSteps\n")
    log_file.flush()  # 确保表头被立即写入

    # --- 4. HRL训练循环 ---
    max_train_episodes = 5001
    all_episode_rewards = []

    print(f"开始HRL训练，使用设备: {config['device']}")

    for episode in range(max_train_episodes):
        obs_dict, info = env.reset()
        current_ep_reward = 0
        terminated, truncated = False, False

        # --- 单个 Episode 的 HRL 交互循环 ---
        while not (terminated or truncated):
            # 1. 高层决策
            # 扁平化观测，作为高层智能体的状态输入
            meta_obs_flat = flatten_observation(obs_dict, env.observation_space)
            # 从环境中获取高层动作掩码
            high_level_mask = env.get_high_level_action_mask()

            # 高层智能体选择一个高层任务 (action)
            high_level_action, high_val, high_logp = meta_agent.select_action(meta_obs_flat, high_level_mask)
            # 2. 环境执行高层任务
            # env.step 内部会循环执行多步，并调用低层策略来生成具体动作
            # 同时，低层经验会自动被存储到 low_level_agent.buffer 中
            next_obs_dict, high_reward, terminated, truncated, info = env.step(high_level_action)

            # 3. 存储高层经验
            meta_agent.buffer.store(
                obs=meta_obs_flat,
                act=high_level_action,
                rew=high_reward,
                val=high_val,
                logp=high_logp
            )

            # 更新奖励和状态
            current_ep_reward += high_reward
            obs_dict = next_obs_dict

            # 4. 检查是否需要更新网络
            # 更新高层网络
            if meta_agent.buffer.ptr >= update_interval_meta:
                print(f"--- 更新高层网络 (Episode: {episode}) ---")
                # 计算 GAE 和 return
                last_val_flat = flatten_observation(obs_dict, env.observation_space)
                with torch.no_grad():
                    last_val = meta_agent.policy_old.get_value(
                        torch.FloatTensor(last_val_flat).to(config["device"])
                    ).item()
                meta_agent.buffer.finish_path(last_val)
                meta_agent.update()
                meta_agent.buffer.clear()

            # 更新低层网络
            if low_level_agent.buffer.ptr >= update_interval_low:
                print(f"--- 更新低层网络 (Episode: {episode}) ---")
                # 低层经验是在env.step中一步步存的，所以它的last_val也需要计算
                last_val_flat = flatten_observation(obs_dict, env.observation_space)
                # 低层的价值函数需要 task_id，我们可以用上一个高层动作作为近似
                last_task_id = torch.tensor([high_level_action], device=config["device"])
                with torch.no_grad():
                    _, _, _, last_val_tensor = low_level_agent.policy_old.get_action_and_value(
                        torch.FloatTensor(last_val_flat).unsqueeze(0).to(config["device"]),
                        last_task_id
                    )
                    last_val = last_val_tensor.item()
                low_level_agent.buffer.finish_path(last_val)
                low_level_agent.update()
                low_level_agent.buffer.clear()

        # --- Episode 结束后的处理 ---
        all_episode_rewards.append(current_ep_reward)
        if episode % log_interval == 0:
            print(f"Episode: {episode}, Reward: {current_ep_reward:.2f}, Total Steps: {env.current_step}")
            # 在 episode 结束时，如果缓冲区里还有未处理的经验，完成其 GAE 计算并清空
            if meta_agent.buffer.ptr > 0: meta_agent.buffer.finish_path(0)
            if low_level_agent.buffer.ptr > 0: low_level_agent.buffer.finish_path(0)

        # --- 绘图和保存 ---
        if episode > 0 and episode % plot_interval == 0:
            plot_dir = f"./plots_hrl/episode_{episode}"
            os.makedirs(plot_dir, exist_ok=True)
            plot_trajectory(env.trajectory, env.sensor_true_positions, f"{plot_dir}/trajectory.png", env.area_size)
            plot_remaining_data(env.communication_log, env.num_sensors, env.solo_SN_data,
                                f"{plot_dir}/remaining_data.png")
            plot_position_error(env.localization_log, env.sensor_true_positions, f"{plot_dir}/position_error.png")
            plot_uncertainty_radius(env.localization_log, f"{plot_dir}/uncertainty_radius.png")
            print(f"图表已保存至: {plot_dir}")

        if episode > 0 and episode % save_interval == 0:
            # 【修复】调用PPO对象的save方法，它会保存稳定的旧策略
            meta_agent.save(f"./models_hrl/meta_agent_episode_{episode}.pth")
            low_level_agent.save(f"./models_hrl/low_level_agent_episode_{episode}.pth")
            print(f"HRL模型已保存至 ./models_hrl/\n")

    print("\n--- 训练完成: 正在生成最终奖励曲线 ---")
    plot_episode_rewards(all_episode_rewards, "./plots_hrl/total_rewards_over_training.png", moving_avg_window=20)
    print("最终奖励曲线图已保存至 ./plots_hrl/total_rewards_over_training.png")

    env.close()


if __name__ == "__main__":
    train_hrl()