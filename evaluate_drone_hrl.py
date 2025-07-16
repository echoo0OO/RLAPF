# 文件名: evaluate_drone_hrl.py
# 描述: 加载已训练的HRL模型，并在无动作掩码的环境中进行性能评估。

import gymnasium as gym
import numpy as np
import torch
import os
import random

# 从您的项目中导入所需模块
from env.DroneNavigationEnv_hrl import DroneNavigationEnv
from hppo.hppo_actionmask_hrl import MetaControllerPPO, LowLevelPPO
from visualization_utils import plot_trajectory, plot_remaining_data, plot_position_error, plot_uncertainty_radius


def flatten_observation(obs_dict, obs_space):
    """辅助函数：将环境返回的结构化观测字典扁平化为一个numpy向量。"""
    return gym.spaces.utils.flatten(obs_space, obs_dict)


def evaluate_hrl(config, meta_model_path, low_level_model_path, num_eval_episodes=10):
    """
    HRL模型评估主函数。

    Args:
        config (dict): 与训练时完全相同的配置字典。
        meta_model_path (str): 保存的高层模型权重文件路径。
        low_level_model_path (str): 保存的低层模型权重文件路径。
        num_eval_episodes (int): 要运行的评估episode数量。
    """

    print("--- 开始模型评估 ---")
    print(f"高层模型: {meta_model_path}")
    print(f"低层模型: {low_level_model_path}")
    print(f"评估轮数: {num_eval_episodes}")

    # --- 1. 初始化HRL组件 (与训练时类似) ---
    # 设置随机种子以保证评估的可复现性
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

    # --- 2. 加载已训练的模型权重 ---
    # 确保文件存在
    if not os.path.exists(meta_model_path) or not os.path.exists(low_level_model_path):
        print("错误: 找不到指定的模型文件！")
        return

    # 调用智能体的 load 方法
    meta_agent.load(meta_model_path)
    low_level_agent.load(low_level_model_path)

    # 将网络设置为评估模式（这会禁用Dropout等层，对于PPO不是必须，但良好实践）
    meta_agent.policy_old.eval()
    low_level_agent.policy_old.eval()

    # 初始化环境，并将低层智能体传入
    env = DroneNavigationEnv(config, low_level_agent)

    # --- 3. 评估循环 ---
    all_episode_rewards = []
    all_episode_steps = []

    # 创建保存评估结果的文件夹
    eval_plot_dir = "./plots_hrl_eval"
    os.makedirs(eval_plot_dir, exist_ok=True)

    for episode in range(num_eval_episodes):
        obs_dict, info = env.reset()
        current_ep_reward = 0
        terminated, truncated = False, False

        while not (terminated or truncated):
            # 1. 高层决策 (无掩码)
            meta_obs_flat = flatten_observation(obs_dict, env.observation_space)

            # 【核心区别】不使用 env.get_high_level_action_mask()
            # 而是允许智能体自由选择任何高层动作
            # 我们通过给 select_action 传递一个全为1的掩码或者None来实现
            unrestricted_mask = np.ones(env.NUM_HIGH_LEVEL_TASKS, dtype=np.int8)

            high_level_action, _, _ = meta_agent.select_action(meta_obs_flat, unrestricted_mask)

            # 2. 环境执行高层任务
            # 注意：在评估时，env.step内部仍然会调用低层策略来生成动作，
            # 但我们不会存储这些经验，也不会进行网络更新。
            next_obs_dict, high_reward, terminated, truncated, info = env.step(high_level_action)

            # 更新奖励和状态
            current_ep_reward += high_reward
            obs_dict = next_obs_dict

            # 【核心区别】评估过程中不进行任何学习/更新
            # meta_agent.buffer.store(...)  <- 跳过
            # low_level_agent.buffer.store(...) <- 跳过
            # meta_agent.update() <- 跳过
            # low_level_agent.update() <- 跳过

        # --- 单个Episode结束后的处理 ---
        all_episode_rewards.append(current_ep_reward)
        all_episode_steps.append(env.current_step)
        print(f"评估 Episode: {episode + 1}/{num_eval_episodes}, "
              f"奖励: {current_ep_reward:.2f}, "
              f"总步数: {env.current_step}")

        # 为每个评估episode保存一份详细的图表
        episode_plot_dir = f"{eval_plot_dir}/episode_{episode + 1}"
        os.makedirs(episode_plot_dir, exist_ok=True)
        plot_trajectory(env.trajectory, env.sensor_true_positions, f"{episode_plot_dir}/trajectory.png", env.area_size)
        plot_remaining_data(env.communication_log, env.num_sensors, env.solo_SN_data,
                            f"{episode_plot_dir}/remaining_data.png")
        plot_position_error(env.localization_log, env.sensor_true_positions, f"{episode_plot_dir}/position_error.png")
        plot_uncertainty_radius(env.localization_log, f"{episode_plot_dir}/uncertainty_radius.png")

    env.close()

    # --- 4. 打印最终评估结果 ---
    mean_reward = np.mean(all_episode_rewards)
    std_reward = np.std(all_episode_rewards)
    mean_steps = np.mean(all_episode_steps)

    print("\n--- 评估完成 ---")
    print(f"平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"平均步数: {mean_steps:.2f}")
    print(f"详细图表已保存至: {eval_plot_dir}")


if __name__ == "__main__":
    # --- 配置评估参数 ---

    # 1. 定义与训练时完全相同的config字典
    #    (您可以直接从 train_drone_hrl.py 复制过来)
    config = {
        # 环境参数
        "num_sensors": 5, "area_size": (1000.0, 1000.0),
        "max_steps_per_episode": 1000, "solo_SN_data": 2e7,
        "max_task_commitment_steps": 15,  # 使用您训练时的承诺步数
        "approach_termination_sector_coverage": 11,
        "localize_termination_radius_thresh": 15.0,
        "maneuver_optimal_radius": 60.0,
        "sector_coverage_distance_thresh": 200.0,  # 使用您训练时的值

        # HRL 智能体通用参数 (评估时大部分不使用，但为了对象初始化需要保留)
        "gamma": 0.99, "lam": 0.95, "eps_clip": 0.2, "epochs_update": 10,
        "coeff_entropy": 0.01, "max_norm": 0.5,
        "random_seed": 123,  # 可以使用不同的种子来测试泛化性
        "device": "cuda" if torch.cuda.is_available() else "cpu",

        # 低层智能体 (LowLevelPPO) 参数
        "low_level_state_dim": 34, "action_con_dim": 2, "num_tasks": 3,
        "low_buffer_size": 4096, "low_batch_size": 128,
        "lr_actor": 3e-4, "init_log_std": -0.5,
        "cnn_input_channels": 1, "cnn_input_size": 7,
        "cnn_out_channels": [32, 64], "mlp_hidden_dim": 256,

        # 高层智能体 (MetaControllerPPO) 参数
        "meta_state_dim": 34, "meta_buffer_size": 1024, "meta_batch_size": 64,
        "meta_lr_actor": 5e-4, "meta_hidden_dim": 128,
    }

    # 2. 指定您要评估的模型文件路径
    #    请确保文件名和路径正确！
    META_MODEL_PATH = "./models_hrl/meta_agent_episode_5000.pth"  # 假设您保存的模型是这个名字
    LOW_LEVEL_MODEL_PATH = "./models_hrl/low_level_agent_episode_5000.pth"

    # 3. 设置评估轮数
    NUM_EVAL_EPISODES = 20

    # 4. 运行评估
    evaluate_hrl(config, META_MODEL_PATH, LOW_LEVEL_MODEL_PATH, NUM_EVAL_EPISODES)