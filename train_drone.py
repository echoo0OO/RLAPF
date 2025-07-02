import gymnasium as gym
import numpy as np
import torch
import os

from env.DroneNavigationEnv import DroneNavigationEnv
from hppo.hppo_actionmask_CNN import PPO_FRCF
from visualization_utils import plot_trajectory, plot_remaining_data, plot_position_error, plot_uncertainty_radius, \
    plot_episode_rewards


def flatten_observation(obs_dict, obs_space):
    """辅助函数：将环境返回的结构化观测字典扁平化为一个numpy向量。"""
    return gym.spaces.utils.flatten(obs_space, obs_dict)


def train():
    """主训练函数"""

    # --- 1. 超参数配置 (保持不变) ---
    config = {
        "num_sensors": 5, "area_size": (1000.0, 1000.0),
        "max_steps_per_episode": 1000, "solo_SN_data": 2e7,
        "action_dis_dim": 1, "action_dis_len": 2, "action_con_dim": 2,
        "cnn_input_channels": 1, "cnn_input_size": 7,
        "cnn_out_channels": [32, 64], "mlp_hidden_dim": 256,
        "lr_actor": 3e-4, "lr_critic": 1e-3, "lr_decay_rate": 0.99,
        "gamma": 0.99, "lam": 0.95, "eps_clip": 0.2, "epochs_update": 10,
        "target_kl_dis": 0.02, "target_kl_con": 0.02,
        "batch_size": 64, "buffer_size": 2048, "max_norm": 0.5,
        "coeff_entropy": 0.015,"lr_std": 3e-4, "init_log_std": -0.5,
        "random_seed": 42, "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    # 日志、绘图和保存的超参数
    log_interval = 1
    plot_interval = 100
    save_interval = 100

    # --- 2. 初始化环境和智能体 ---
    env = DroneNavigationEnv(config)
    state_dim = gym.spaces.utils.flatdim(env.observation_space)
    config["state_dim"] = state_dim
    print(f"扁平化后的状态空间维度: {state_dim}")
    print(f"使用的设备: {config['device']}")
    agent = PPO_FRCF(**config)

    # --- 3. 创建文件夹 ---
    os.makedirs("./plots", exist_ok=True)
    os.makedirs("./models", exist_ok=True)

    # --- 4. 训练循环 ---
    max_train_episodes = 2001
    all_episode_rewards = []

    for episode in range(max_train_episodes):
        obs_dict, info = env.reset()
        current_ep_reward = 0
        terminated, truncated = False, False

        # --- 单个 Episode 的交互循环 ---
        while not (terminated or truncated):

            # 扁平化观测
            flat_obs = flatten_observation(obs_dict, env.observation_space)

            # 检查当前是否为决策模式
            is_deciding_mode = info.get('current_mode') == 'DECIDING'

            mask_d, mask_c = env._get_action_mask()
            action_mask = {"discrete_mask": mask_d, "continuous_mask": mask_c}

            # 智能体选择动作
            # action_mask = {"discrete_mask": np.ones((1, 2)), "continuous_mask": np.array([[-1.0, 1.0]] * 2)}
            value, (action_dis, action_con), (logp_dis, logp_con) = agent.select_action(flat_obs, action_mask)
            action_dict = {"discrete": action_dis, "continuous": action_con}

            # 与环境交互
            next_obs_dict, reward, terminated, truncated, next_info = env.step(action_dict)

            # 只在智能体做出有效决策时，才存储经验
            if is_deciding_mode:
                agent.buffer.store_hybrid(
                    flat_obs, action_mask, np.array([action_dis]), action_con,
                    reward, value, np.array([logp_dis]), logp_con
                )

            # 【BUG修复 1】: 必须用环境返回的最新信息来更新本地的 info 变量
            obs_dict = next_obs_dict
            info = next_info
            current_ep_reward += reward

            # 【逻辑修复 2】: PPO在收集到足够数据后更新
            # 检查Buffer是否已满
            if agent.buffer.ptr == agent.buffer.max_size:
                # 计算最后一个状态的价值，用于GAE计算
                last_val = 0
                # 注意：这里我们不能用 terminated 或 truncated，因为它们是针对当前步的。
                # 实际上，只要缓冲区满了，下一步的状态就是存在的。
                last_val_flat_obs = flatten_observation(next_obs_dict, env.observation_space)
                with torch.no_grad():
                    last_val = agent.agent_old.get_value(
                        torch.FloatTensor(last_val_flat_obs).unsqueeze(0).to(agent.device)
                    ).cpu().numpy().flatten()[0]

                # 完成路径（计算Advantage）并更新网络
                agent.buffer.finish_path(last_val)
                agent.update(config["batch_size"])

        # --- Episode 结束后的处理 ---

        # 【逻辑修复 2 的补充】: 在回合结束时，如果缓冲区里还有未处理的经验，
        # 必须调用 finish_path 来完成对这段轨迹的优势函数计算。
        # 否则这些数据在下一个回合开始时会造成污染。
        # 这里 last_val 应该为 0，因为回合已经结束，没有未来价值了。
        if agent.buffer.path_start_dix < agent.buffer.ptr:
            agent.buffer.finish_path(0)

        all_episode_rewards.append(current_ep_reward)
        if episode % log_interval == 0:
            print(f"Episode: {episode}, Reward: {current_ep_reward:.2f}, Steps: {env.current_step}")

        # --- 绘图和保存 (逻辑不变) ---
        if episode > 0 and episode % plot_interval == 0:
            plot_dir = f"./plots/episode_{episode}"
            os.makedirs(plot_dir, exist_ok=True)
            plot_trajectory(env.trajectory, env.sensor_true_positions, f"{plot_dir}/trajectory.png", env.area_size)
            plot_remaining_data(env.communication_log, env.num_sensors, env.solo_SN_data,
                                f"{plot_dir}/remaining_data.png")
            plot_position_error(env.localization_log, env.sensor_true_positions, f"{plot_dir}/position_error.png")
            plot_uncertainty_radius(env.localization_log, f"{plot_dir}/uncertainty_radius.png")
            print(f"图表已保存至: {plot_dir}")
        if episode > 0 and episode % save_interval == 0:
            save_path = f"./models/ppo_drone_episode_{episode}.pth"
            agent.save(save_path)
            print(f"模型已保存至: {save_path}\n")

    print("\n--- 训练完成: 正在生成最终奖励曲线 ---")
    plot_episode_rewards(all_episode_rewards, "./plots/total_rewards_over_training.png", moving_avg_window=20)
    print("最终奖励曲线图已保存至 ./plots/total_rewards_over_training.png")

    env.close()


if __name__ == "__main__":
    train()
