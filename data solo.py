# 文件名: evaluate_drone_hrl.py
# 描述: 加载已训练的HRL模型，并在有动作掩码的环境中进行性能评估，仅保存数值结果和最终趋势图。

import gymnasium as gym
import numpy as np
import torch
import os
import random

# 从您的项目中导入所需模块
from env.DroneNavigationEnv_hrl_stable_r import DroneNavigationEnv
from hppo.hppo_actionmask_hrl import MetaControllerPPO, LowLevelPPO


# 移除了所有plot函数，因为不再在每个episode绘制详细图
# from visualization_utils import plot_trajectory, plot_remaining_data, plot_position_error, plot_uncertainty_radius


def flatten_observation(obs_dict, obs_space):
    """辅助函数：将环境返回的结构化观测字典扁平化为一个numpy向量。"""
    return gym.spaces.utils.flatten(obs_space, obs_dict)


def evaluate_hrl(config, meta_model_path, low_level_model_path, num_eval_episodes=10, data_size_label=""):
    """
    HRL模型评估主函数。

    Args:
        config (dict): 与训练时完全相同的配置字典。
        meta_model_path (str): 保存的高层模型权重文件路径。
        low_level_model_path (str): 保存的低层模型权重文件路径。
        num_eval_episodes (int): 要运行的评估episode数量。
        data_size_label (str): 用于图表保存路径的标签，表示当前数据量。
    """

    print("--- 开始模型评估 ---")
    print(f"高层模型: {meta_model_path}")
    print(f"低层模型: {low_level_model_path}")
    print(f"评估轮数: {num_eval_episodes}")
    print(f"目标数据量: {config['solo_SN_data'] / 1e6:.1f}M")  # Print data size

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
        return None, None, None, None  # 返回更多的None以匹配新的返回值

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
    all_episode_position_errors = []  # 存储每个episode的平均定位误差
    all_episode_uncertainty_radii = []  # 存储每个episode的平均不确定性半径

    # 不再创建 eval_plot_dir 的子文件夹，因为不在这里保存单个episode的图

    for episode in range(num_eval_episodes):
        obs_dict, info = env.reset()
        current_ep_reward = 0
        terminated, truncated = False, False

        # 在每个episode开始时清空环境内部的日志，确保只记录当前episode的数据
        # 这通常在env.reset()内部完成，但再次确认以防万一。
        # 如果你的env.reset()不完全清空，可能需要在这里手动清空
        # env.localization_log = []
        # env.trajectory = []
        # env.communication_log = []

        while not (terminated or truncated):
            # 1. 高层决策 (现在使用动作掩码)
            meta_obs_flat = flatten_observation(obs_dict, env.observation_space)

            # 获取高层动作掩码
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

        # 收集并计算本episode的平均定位误差和平均不确定性半径
        # env.localization_log 是一个列表的列表，每个子列表是 [time, sensor_id, true_x, true_y, est_x, est_y, est_radius]
        current_episode_errors = []
        current_episode_radii = []

        # 确保 env.localization_log 存在且不为空
        if hasattr(env, 'localization_log') and env.localization_log:
            for entry in env.localization_log:
                # 检查 entry 是否有足够的元素
                if len(entry) >= 7:
                    true_pos = np.array([entry[2], entry[3]])
                    est_pos = np.array([entry[4], entry[5]])
                    error = np.linalg.norm(true_pos - est_pos)
                    radius = entry[6]
                    current_episode_errors.append(error)
                    current_episode_radii.append(radius)

            if current_episode_errors:
                all_episode_position_errors.append(np.mean(current_episode_errors))
            else:
                all_episode_position_errors.append(0.0)  # 如果没有定位日志，则为0

            if current_episode_radii:
                all_episode_uncertainty_radii.append(np.mean(current_episode_radii))
            else:
                all_episode_uncertainty_radii.append(0.0)  # 如果没有定位日志，则为0
        else:
            # 如果没有 localization_log 或者为空，则记录0
            all_episode_position_errors.append(0.0)
            all_episode_uncertainty_radii.append(0.0)

        print(f"评估 Episode: {episode + 1}/{num_eval_episodes}, "
              f"奖励: {current_ep_reward:.2f}, "
              f"总步数: {env.current_step}, "
              f"平均定位误差: {all_episode_position_errors[-1]:.2f}, "
              f"平均不确定性半径: {all_episode_uncertainty_radii[-1]:.2f}")

        # <-- 移除了单个episode的绘图代码块 -->
        # os.makedirs(episode_plot_dir, exist_ok=True)
        # plot_remaining_data(...)
        # plot_position_error(...)
        # plot_uncertainty_radius(...)

    env.close()

    # --- 4. 打印最终评估结果 ---
    mean_reward = np.mean(all_episode_rewards) if all_episode_rewards else 0.0
    std_reward = np.std(all_episode_rewards) if all_episode_rewards else 0.0
    mean_steps = np.mean(all_episode_steps) if all_episode_steps else 0.0

    mean_pos_error = np.mean(all_episode_position_errors) if all_episode_position_errors else 0.0
    std_pos_error = np.std(all_episode_position_errors) if all_episode_position_errors else 0.0

    mean_unc_radius = np.mean(all_episode_uncertainty_radii) if all_episode_uncertainty_radii else 0.0
    std_unc_radius = np.std(all_episode_uncertainty_radii) if all_episode_uncertainty_radii else 0.0

    print("\n--- 评估完成 ---")
    print(f"平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"平均步数: {mean_steps:.2f}")
    print(f"平均定位误差: {mean_pos_error:.2f} ± {std_pos_error:.2f}")
    print(f"平均不确定性半径: {mean_unc_radius:.2f} ± {std_unc_radius:.2f}")
    # 不再打印“详细图表已保存至”

    # 返回所有需要的数据
    return mean_steps, mean_reward, mean_pos_error, mean_unc_radius


if __name__ == "__main__":
    # --- 配置评估参数 ---

    # 1. 定义与训练时完全相同的config字典
    #    (您可以直接从 train_drone_hrl.py 复制过来)
    base_config = {  # Renamed to base_config
        # 环境参数
        "num_sensors": 5, "area_size": (1000.0, 1000.0),
        "max_steps_per_episode": 1000,
        "max_task_commitment_steps": 5,  # 使用您训练时的承诺步数
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

    # Define the range of solo_SN_data (in Bytes)
    data_volumes_MB = np.arange(20, 81, 10)  # From 20MB to 80MB in 10MB increments
    data_volumes_bytes = data_volumes_MB * 1e6  # Convert MB to Bytes

    results_steps = []
    results_rewards = []
    results_pos_errors = []  # 存储每个数据量的平均定位误差
    results_unc_radii = []  # 存储每个数据量的平均不确定性半径

    print("\n--- 开始多数据量评估 ---")
    # 创建一个用于保存最终趋势图的目录（如果不存在）
    final_plots_dir = "./plots_hrl_eval_summary"
    os.makedirs(final_plots_dir, exist_ok=True)

    for data_mb, data_bytes in zip(data_volumes_MB, data_volumes_bytes):
        print(f"\n--- 评估数据量: {data_mb}MB ---")
        current_config = base_config.copy()  # Create a mutable copy
        current_config["solo_SN_data"] = data_bytes

        # data_size_label 在这里主要用于打印输出，不再用于创建子文件夹
        data_size_label = f"{int(data_mb)}MB"

        # 更新 evaluate_hrl 的调用以接收新的返回值
        mean_steps, mean_reward, mean_pos_error, mean_unc_radius = evaluate_hrl(
            current_config,
            META_MODEL_PATH,
            LOW_LEVEL_MODEL_PATH,
            NUM_EVAL_EPISODES,
            data_size_label
        )

        if mean_steps is not None:
            results_steps.append((data_mb, mean_steps))
            results_rewards.append((data_mb, mean_reward))
            results_pos_errors.append((data_mb, mean_pos_error))  # 存储平均定位误差
            results_unc_radii.append((data_mb, mean_unc_radius))  # 存储平均不确定性半径

    print("\n--- 多数据量评估结果汇总 ---")
    print("数据量 (MB) | 平均步数 | 平均奖励 | 平均定位误差 | 平均不确定性半径")
    print("------------------------------------------------------------------")
    for i in range(len(results_steps)):
        data_mb = results_steps[i][0]
        steps = results_steps[i][1]
        reward = results_rewards[i][1]
        pos_error = results_pos_errors[i][1]
        unc_radius = results_unc_radii[i][1]
        print(f"{data_mb:<13} | {steps:<10.2f} | {reward:<10.2f} | {pos_error:<13.2f} | {unc_radius:.2f}")

    # Optional: Plotting the results
    try:
        import matplotlib.pyplot as plt

        data_mbs = [res[0] for res in results_steps]
        steps = [res[1] for res in results_steps]
        rewards = [res[1] for res in results_rewards]
        pos_errors = [res[1] for res in results_pos_errors]
        unc_radii = [res[1] for res in results_unc_radii]

        # 绘制平均步数
        plt.figure(figsize=(10, 6))
        plt.plot(data_mbs, steps, marker='o', linestyle='-')
        plt.title('Task Completion Steps vs. Target Data Volume (with Action Mask)')
        plt.xlabel('Target Data Volume (MB)')
        plt.ylabel('Average Task Completion Steps')
        plt.grid(True)
        plt.xticks(data_mbs)
        plt.tight_layout()
        plt.savefig(os.path.join(final_plots_dir, "steps_vs_data_volume_with_mask.png"))
        plt.show()

        # 绘制平均奖励
        plt.figure(figsize=(10, 6))
        plt.plot(data_mbs, rewards, marker='o', linestyle='-')
        plt.title('Average Reward vs. Target Data Volume (with Action Mask)')
        plt.xlabel('Target Data Volume (MB)')
        plt.ylabel('Average Reward')
        plt.grid(True)
        plt.xticks(data_mbs)
        plt.tight_layout()
        plt.savefig(os.path.join(final_plots_dir, "rewards_vs_data_volume_with_mask.png"))
        plt.show()

        # 绘制平均定位误差
        plt.figure(figsize=(10, 6))
        plt.plot(data_mbs, pos_errors, marker='o', linestyle='-')
        plt.title('Average Position Error vs. Target Data Volume (with Action Mask)')
        plt.xlabel('Target Data Volume (MB)')
        plt.ylabel('Average Position Error')
        plt.grid(True)
        plt.xticks(data_mbs)
        plt.tight_layout()
        plt.savefig(os.path.join(final_plots_dir, "position_error_vs_data_volume_with_mask.png"))
        plt.show()

        # 绘制平均不确定性半径
        plt.figure(figsize=(10, 6))
        plt.plot(data_mbs, unc_radii, marker='o', linestyle='-')
        plt.title('Average Uncertainty Radius vs. Target Data Volume (with Action Mask)')
        plt.xlabel('Target Data Volume (MB)')
        plt.ylabel('Average Uncertainty Radius')
        plt.grid(True)
        plt.xticks(data_mbs)
        plt.tight_layout()
        plt.savefig(os.path.join(final_plots_dir, "uncertainty_radius_vs_data_volume_with_mask.png"))
        plt.show()

    except ImportError:
        print("Matplotlib not found. Cannot plot results. Please install it with 'pip install matplotlib'.")