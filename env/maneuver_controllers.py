# 文件名: maneuver_controllers.py
# 版本: 已修复测试脚本的AttributeError

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


class InformationDrivenManeuverController:
    # ... (类的内部实现，即 __init__, _find_info_gap_angle, calculate_move_vector 保持不变) ...
    """
    一个基于信息增益，用于生成自适应定位机动轨迹的控制器。
    【新功能】当有多个信息最稀疏的扇区时，会选择离无人机当前位置最近的那个扇区作为目标。
    """

    def __init__(self, total_speed: float, optimal_radius: float, time_slot: float, num_sectors: int = 12):
        self.total_speed = total_speed
        self.optimal_radius = optimal_radius
        self.time_slot = time_slot
        self.num_sectors = num_sectors

    def _find_info_gap_angle(self, drone_pos: np.ndarray, sensor_pos: np.ndarray, history_points: list) -> float:
        """
        找到信息量最少的方向，并处理多个候选。
        返回一个角度（弧度），而不是方向向量。
        """

        if not history_points:
            return np.random.uniform(0, 2 * np.pi)
        vectors = np.array(history_points) - sensor_pos
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        sector_width = 2 * np.pi / self.num_sectors
        sector_indices = np.floor((angles + np.pi) / sector_width).astype(int)
        sector_indices = np.clip(sector_indices, 0, self.num_sectors - 1)
        sector_counts = np.bincount(sector_indices, minlength=self.num_sectors)
        min_count = np.min(sector_counts)
        candidate_sector_indices = np.where(sector_counts == min_count)[0]
        if len(candidate_sector_indices) == 1:
            min_count_sector_idx = candidate_sector_indices[0]
            gap_angle = (min_count_sector_idx + 0.5) * sector_width - np.pi
            return gap_angle
        else:
            best_sector_idx = -1;
            min_dist_sq = np.inf
            drone_vec_rel_sensor = drone_pos - sensor_pos
            drone_angle_rel_sensor = np.arctan2(drone_vec_rel_sensor[1], drone_vec_rel_sensor[0])
            for sector_idx in candidate_sector_indices:
                sector_angle = (sector_idx + 0.5) * sector_width - np.pi
                angle_diff = sector_angle - drone_angle_rel_sensor
                angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
                dist_sq = angle_diff ** 2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq;
                    best_sector_idx = sector_idx
            best_gap_angle = (best_sector_idx + 0.5) * sector_width - np.pi
            return best_gap_angle

    def calculate_move_vector(self, drone_pos: np.ndarray, sensor_est_pos: np.ndarray,
                              history_points: list) -> np.ndarray:
        """
        计算移动向量，其唯一目标是飞向信息空洞方向上的最优轨道点。
        """
        # 1. 动态计算当前距离，并将其用作“最优半径”
        dynamic_optimal_radius = np.linalg.norm(drone_pos - sensor_est_pos)
        # 增加一个最小半径保护，避免无人机离目标太近时半径为0
        if dynamic_optimal_radius < 1.0:
            dynamic_optimal_radius = 1.0
        optimal_radius = min(dynamic_optimal_radius, 100)
        gap_angle = self._find_info_gap_angle(drone_pos, sensor_est_pos, history_points)
        gap_direction = np.array([np.cos(gap_angle), np.sin(gap_angle)])
        target_point = sensor_est_pos + gap_direction * optimal_radius
        vector_to_target = target_point - drone_pos
        dist_to_target = np.linalg.norm(vector_to_target)
        step_distance = self.total_speed * self.time_slot
        if step_distance > dist_to_target:
            move_vector = vector_to_target
        else:
            norm_vector_to_target = vector_to_target / (dist_to_target + 1e-6)
            move_vector = norm_vector_to_target * step_distance
        return move_vector


# ==============================================================================
# 当此文件作为主脚本运行时，执行以下可视化和测试代码
# ==============================================================================
if __name__ == "__main__":
    print("Running standalone test for InformationDrivenManeuverController with GDOP plot...")


    # --- 辅助函数：模拟GDOP计算 ---
    def calculate_gdop(measurement_points, target_pos):
        if len(measurement_points) < 2: return 50.0
        H = []
        for p in measurement_points:
            diff = p - target_pos;
            dist = np.linalg.norm(diff)
            if dist < 1e-6: continue
            H.append(diff / dist)
        H = np.array(H)
        if H.shape[0] < 2: return 50.0
        try:
            H_T_H = H.T @ H
            if np.linalg.det(H_T_H) < 1e-9: return 50.0
            H_T_H_inv = np.linalg.inv(H_T_H)
            return np.sqrt(np.trace(H_T_H_inv))
        except np.linalg.LinAlgError:
            return 50.0


    # --- 场景和控制器参数设定 ---
    SIM_TOTAL_SPEED = 20.0;
    SIM_OPTIMAL_RADIUS = 40.0;
    SIM_TIME_SLOT = 1.0
    sensor_est_pos = np.array([500.0, 500.0])

    # --- 初始化控制器 ---
    controller = InformationDrivenManeuverController(
        total_speed=SIM_TOTAL_SPEED, optimal_radius=SIM_OPTIMAL_RADIUS,
        time_slot=SIM_TIME_SLOT, num_sectors=12
    )

    # --- 模拟设置 ---
    initial_history_points = [
        sensor_est_pos + np.array([50, 10]),
        sensor_est_pos + np.array([60, -20]),
        sensor_est_pos + np.array([40, -30]),
    ]
    drone_pos = sensor_est_pos + np.array([0.0, 120.0])
    trajectory = [drone_pos.copy()]
    gdop_history = [calculate_gdop(initial_history_points, sensor_est_pos)]

    # --- 模拟循环 ---
    max_steps = 100
    print(f"Starting simulation for {max_steps} steps...")
    # 使用 all_history_points 来累积所有点，以避免修改 initial_history_points
    all_history_points = initial_history_points.copy()
    for step in range(max_steps):
        move_vector = controller.calculate_move_vector(drone_pos, sensor_est_pos, all_history_points)
        if np.linalg.norm(move_vector) < 1e-6:
            print(f"Maneuver stopped at step {step + 1}.")
            break
        drone_pos += move_vector
        trajectory.append(drone_pos.copy())
        all_history_points.append(drone_pos.copy())
        gdop_history.append(calculate_gdop(all_history_points, sensor_est_pos))
    print("Simulation finished.")

    # --- 在循环结束后，绘制包含两个子图的最终结果 ---
    print("Generating final plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle("Information-Driven Maneuver Analysis", fontsize=20)

    # --- 左图: 轨迹图 ---
    ax1.set_title(f"Final Trajectory ({len(trajectory) - 1} steps)", fontsize=16)
    hist_pts_arr = np.array(all_history_points)
    ax1.scatter(hist_pts_arr[:, 0], hist_pts_arr[:, 1], c='gray', s=30, alpha=0.5, label='All History Points')
    ax1.scatter(sensor_est_pos[0], sensor_est_pos[1], c='blue', s=100, marker='x', zorder=5, label='Sensor Est. Pos')
    optimal_circle = plt.Circle(sensor_est_pos, SIM_OPTIMAL_RADIUS, color='blue', fill=False, linestyle='--', alpha=0.5,
                                label='Optimal Radius')
    ax1.add_patch(optimal_circle)
    traj_arr = np.array(trajectory)
    ax1.plot(traj_arr[:, 0], traj_arr[:, 1], 'g-', lw=2, label='UAV Trajectory')
    ax1.scatter(traj_arr[0, 0], traj_arr[0, 1], c='cyan', edgecolors='black', s=150, marker='^', zorder=5,
                label='Start Pos')
    ax1.scatter(traj_arr[-1, 0], traj_arr[-1, 1], c='orange', edgecolors='black', s=150, marker='s', zorder=5,
                label='End Pos')

    # 【核心修正】正确调用新方法来获取最终方向
    # 1. 调用 _find_info_gap_angle 获取角度
    #    需要传入无人机在循环结束时的最终位置 traj_arr[-1]
    final_gap_angle = controller._find_info_gap_angle(traj_arr[-1], sensor_est_pos, all_history_points)
    # 2. 从角度计算出方向向量
    final_gap_dir = np.array([np.cos(final_gap_angle), np.sin(final_gap_angle)])

    arrow_start = sensor_est_pos
    arrow_end = sensor_est_pos + final_gap_dir * (SIM_OPTIMAL_RADIUS + 20)
    gap_arrow = FancyArrowPatch(arrow_start, arrow_end, arrowstyle='->', color='red', mutation_scale=20, lw=2,
                                label='Final Info Gap Direction')
    ax1.add_patch(gap_arrow)
    ax1.set_xlabel("X (m)");
    ax1.set_ylabel("Y (m)")
    ax1.legend();
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.axis('equal');
    ax1.set_xlim(sensor_est_pos[0] - 150, sensor_est_pos[0] + 150)
    ax1.set_ylim(sensor_est_pos[1] - 150, sensor_est_pos[1] + 150)

    # --- 右图: GDOP变化图 ---
    ax2.set_title("GDOP vs. Number of Measurements", fontsize=16)
    num_measurements = np.arange(len(all_history_points) - len(trajectory) + 1, len(all_history_points) + 1)
    ax2.plot(num_measurements, gdop_history, 'r-o', lw=2, markersize=5, label='GDOP value')
    ax2.set_xlabel("Number of Measurement Points");
    ax2.set_ylabel("GDOP (Geometric Dilution of Precision)")
    ax2.set_xticks(np.arange(min(num_measurements), max(num_measurements) + 1, 5))
    ax2.grid(True, linestyle='--', alpha=0.7);
    ax2.legend()
    for i, gdop in enumerate(gdop_history):
        if i == 0 or i == len(gdop_history) - 1 or (i + 1) % 10 == 0:
            ax2.text(num_measurements[i], gdop, f'{gdop:.2f}', ha='center', va='bottom')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    print("Plot displayed.")