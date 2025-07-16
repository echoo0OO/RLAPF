# 文件名: evaluate_drone_hrl.py (简洁、健壮的最终版)
# 描述: 加载模型，并通过切换智能体的内部模式，在无高低层掩码下评估。

import gymnasium as gym
import numpy as np
import torch
import os
import random

from env.DroneNavigationEnv_hrl import DroneNavigationEnv
from hppo.hppo_actionmask_hrl import MetaControllerPPO, LowLevelPPO
from visualization_utils import plot_trajectory, plot_remaining_data, plot_position_error, plot_uncertainty_radius


def flatten_observation(obs_dict, obs_space):
    return gym.spaces.utils.flatten(obs_space, obs_dict)


def evaluate_hrl(config, meta_model_path, low_level_model_path, num_eval_episodes=10):
    print("--- 开始模型评估 (无高层 & 低层掩码) ---")
    print(f"高层模型: {meta_model_path}")
    print(f"低层模型: {low_level_model_path}")

    # --- 1. 初始化HRL组件 ---
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

    # --- 2. 加载模型权重 ---
    meta_agent.load(meta_model_path)
    low_level_agent.load(low_level_model_path)

    # 【核心步骤 1】将低层智能体切换到评估模式
    low_level_agent.set_eval_mode(True)

    meta_agent.policy_old.eval()
    low_level_agent.policy_old.eval()

    # 初始化环境，传入已经切换到评估模式的低层智能体
    env = DroneNavigationEnv(config, low_level_agent)

    # --- 3. 评估循环 ---
    all_episode_rewards = []
    all_episode_steps = []

    eval_plot_dir = "./plots_hrl_eval_full_freedom"
    os.makedirs(eval_plot_dir, exist_ok=True)

    for episode in range(num_eval_episodes):
        obs_dict, info = env.reset()
        current_ep_reward = 0
        terminated, truncated = False, False
        step = 0
        while not (terminated or truncated):
            step += 1
            print(f"step : {step}")
            # 1. 高层决策 (无掩码)
            meta_obs_flat = flatten_observation(obs_dict, env.observation_space)
            unrestricted_high_level_mask = np.ones(env.NUM_HIGH_LEVEL_TASKS, dtype=np.int8)
            high_level_action, _, _ = meta_agent.select_action(meta_obs_flat, unrestricted_high_level_mask)

            # 2. 【核心步骤 2】正常调用 env.step()
            #    env.step 内部会调用 low_level_agent.select_action。
            #    由于我们已经设置了 eval_mode=True，它会自动忽略所有低层掩码。
            next_obs_dict, high_reward, terminated, truncated, info = env.step(high_level_action)

            current_ep_reward += high_reward
            obs_dict = next_obs_dict

        # --- 单个Episode结束后的处理 ---
        all_episode_rewards.append(current_ep_reward)
        all_episode_steps.append(env.current_step)
        print(
            f"评估 Episode: {episode + 1}/{num_eval_episodes}, 奖励: {current_ep_reward:.2f}, 总步数: {env.current_step}")

        # 绘图...
        episode_plot_dir = f"{eval_plot_dir}/episode_{episode + 1}"
        os.makedirs(episode_plot_dir, exist_ok=True)
        plot_trajectory(env.trajectory, env.sensor_true_positions, f"{episode_plot_dir}/trajectory.png", env.area_size)
        # ... (其他绘图函数)

    env.close()

    # --- 4. 打印最终结果 ---
    mean_reward = np.mean(all_episode_rewards)
    std_reward = np.std(all_episode_rewards)
    mean_steps = np.mean(all_episode_steps)

    print("\n--- 评估完成 (无任何掩码) ---")
    print(f"平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"平均步数: {mean_steps:.2f}")


if __name__ == "__main__":
    # --- 配置评估参数 ---
    # 1. 定义与训练时完全相同的config字典
    config = {
        # ... (从您的 train_drone_hrl.py 复制完整的config字典过来) ...
        "num_sensors": 5, "area_size": (1000.0, 1000.0), "max_steps_per_episode": 1000,
        "solo_SN_data": 2e7, "max_task_commitment_steps": 15,
        "approach_termination_sector_coverage": 11, "localize_termination_radius_thresh": 15.0,
        "maneuver_optimal_radius": 60.0, "sector_coverage_distance_thresh": 200.0, "gamma": 0.99,
        "lam": 0.95, "eps_clip": 0.2, "epochs_update": 10, "coeff_entropy": 0.01,
        "max_norm": 0.5, "random_seed": 42, "device": "cuda" if torch.cuda.is_available() else "cpu",
        "low_level_state_dim": 34, "action_con_dim": 2, "num_tasks": 3, "low_buffer_size": 4096,
        "low_batch_size": 128, "lr_actor": 3e-4, "init_log_std": -0.5, "cnn_input_channels": 1,
        "cnn_input_size": 7, "cnn_out_channels": [32, 64], "mlp_hidden_dim": 256,
        "meta_state_dim": 34, "meta_buffer_size": 1024, "meta_batch_size": 64,
        "meta_lr_actor": 5e-4, "meta_hidden_dim": 128,
    }

    # 2. 指定要评估的模型
    META_MODEL_PATH = "./models_hrl/meta_agent_episode_5000.pth"
    LOW_LEVEL_MODEL_PATH = "./models_hrl/low_level_agent_episode_5000.pth"

    # 3. 运行
    evaluate_hrl(config, META_MODEL_PATH, LOW_LEVEL_MODEL_PATH, num_eval_episodes=10)