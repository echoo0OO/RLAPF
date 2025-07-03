

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


class InformationDrivenManeuverController:
    """
    一个基于信息增益，用于生成自适应定位机动轨迹的控制器。

    该控制器通过分析历史测距点的角度分布，找到信息最稀疏的“空洞”区域，
    并引导无人机飞向该区域进行测量，从而以最少的步骤最高效地改善几何构型(GDOP)。
    """

    def __init__(self, total_speed: float, optimal_radius: float, time_slot: float, num_sectors: int = 12):
        """
        初始化控制器。

        Args:
            total_speed (float): 无人机在机动过程中的总速度 (m/s)。
            optimal_radius (float): 无人机期望保持的绕飞半径 (m)。
            time_slot (float): 环境的仿真步长 (s)。
            num_sectors (int): 用于分析角度分布的扇区数量。
        """
        self.total_speed = total_speed
        self.optimal_radius = optimal_radius
        self.time_slot = time_slot
        self.num_sectors = num_sectors

    def _find_info_gap_direction(self, sensor_pos: np.ndarray, history_points: list) -> np.ndarray:
        """找到信息量最少的方向（测距点最少的扇区）"""
        if not history_points:
            random_angle = np.random.uniform(0, 2 * np.pi)
            return np.array([np.cos(random_angle), np.sin(random_angle)])

        vectors = np.array(history_points) - sensor_pos
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        sector_width = 2 * np.pi / self.num_sectors
        sector_indices = np.floor((angles + np.pi) / sector_width).astype(int)
        sector_indices = np.clip(sector_indices, 0, self.num_sectors - 1)
        sector_counts = np.bincount(sector_indices, minlength=self.num_sectors)
        min_count_sector_idx = np.argmin(sector_counts)
        gap_angle = (min_count_sector_idx + 0.5) * sector_width - np.pi
        return np.array([np.cos(gap_angle), np.sin(gap_angle)])

    def calculate_move_vector(self, drone_pos: np.ndarray, sensor_est_pos: np.ndarray,
                              history_points: list) -> np.ndarray:
        """
        计算移动向量，其唯一目标是飞向信息空洞方向上的最优轨道点。
        """
        dist_to_center = np.linalg.norm(sensor_est_pos - drone_pos)

        # # 简单的退出条件
        # if dist_to_center < self.optimal_radius * 0.5:  # 如果进入最优半径的一半以内，可以认为任务完成
        #     # 这里的退出条件可以根据需求调整，比如用GDOP阈值
        #     return np.array([0.0, 0.0])

        # 1. 找到信息空洞的方向 (这部分逻辑不变)
        gap_direction = self._find_info_gap_direction(sensor_est_pos, history_points)

        # 2. 确定唯一的目标点
        target_point = sensor_est_pos + gap_direction * self.optimal_radius

        # 3. 计算飞向该目标点的向量
        vector_to_target = target_point - drone_pos

        # 4. 将总速度施加到这个方向上
        norm_vector_to_target = vector_to_target / (np.linalg.norm(vector_to_target) + 1e-6)
        move_vector = norm_vector_to_target * self.total_speed * self.time_slot

        return move_vector


# ==============================================================================
# 当此文件作为主脚本运行时，执行以下可视化和测试代码
# ==============================================================================
if __name__ == "__main__":
    print("Running standalone test for InformationDrivenManeuverController with GDOP plot...")


    # --- 辅助函数：模拟GDOP计算 ---
    def calculate_gdop(measurement_points, target_pos):
        # 至少需要2个点才能计算2D GDOP
        if len(measurement_points) < 2: return 50.0  # 返回一个较大的默认值

        H = []
        for p in measurement_points:
            diff = p - target_pos
            dist = np.linalg.norm(diff)
            if dist < 1e-6: continue
            H.append(diff / dist)

        H = np.array(H)
        if H.shape[0] < 2: return 50.0

        try:
            # 计算几何矩阵 (H^T * H) 的逆
            H_T_H = H.T @ H
            # 检查矩阵是否奇异
            if np.linalg.det(H_T_H) < 1e-9:
                return 50.0

            H_T_H_inv = np.linalg.inv(H_T_H)
            # GDOP 是逆矩阵对角线元素之和的平方根
            return np.sqrt(np.trace(H_T_H_inv))
        except np.linalg.LinAlgError:
            # 如果计算出错，返回一个较大的默认值
            return 50.0


    # --- 场景和控制器参数设定 ---
    SIM_TOTAL_SPEED = 20.0
    SIM_OPTIMAL_RADIUS = 40.0
    SIM_TIME_SLOT = 1.0

    sensor_est_pos = np.array([500.0, 500.0])

    # --- 初始化控制器 ---
    controller = InformationDrivenManeuverController(
        total_speed=SIM_TOTAL_SPEED,
        optimal_radius=SIM_OPTIMAL_RADIUS,
        time_slot=SIM_TIME_SLOT,
        num_sectors=12
    )

    # --- 模拟设置 ---
    initial_history_points = [
        sensor_est_pos + np.array([50, 10]),
        sensor_est_pos + np.array([60, -20]),
        sensor_est_pos + np.array([40, -30]),
    ]
    drone_pos = sensor_est_pos + np.array([0.0, 120.0])

    # 准备记录轨迹和GDOP历史
    trajectory = [drone_pos.copy()]
    gdop_history = [calculate_gdop(initial_history_points, sensor_est_pos)]

    # --- 模拟循环 ---
    max_steps = 100
    print(f"Starting simulation for {max_steps} steps...")
    for step in range(max_steps):
        move_vector = controller.calculate_move_vector(drone_pos, sensor_est_pos, initial_history_points)

        if np.linalg.norm(move_vector) < 1e-6:
            print(f"Maneuver stopped at step {step + 1}.")
            break

        drone_pos += move_vector
        trajectory.append(drone_pos.copy())
        initial_history_points.append(drone_pos.copy())

        # 记录当前步的GDOP
        gdop_history.append(calculate_gdop(initial_history_points, sensor_est_pos))

    print("Simulation finished.")

    # --- 在循环结束后，绘制包含两个子图的最终结果 ---
    print("Generating final plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))  # 创建1行2列的子图
    fig.suptitle("Information-Driven Maneuver Analysis", fontsize=20)

    # --- 左图: 轨迹图 ---
    ax1.set_title(f"Final Trajectory ({len(trajectory) - 1} steps)", fontsize=16)
    hist_pts_arr = np.array(initial_history_points)
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
    final_gap_dir = controller._find_info_gap_direction(sensor_est_pos, initial_history_points)
    arrow_start = sensor_est_pos
    arrow_end = sensor_est_pos + final_gap_dir * (SIM_OPTIMAL_RADIUS + 20)
    gap_arrow = FancyArrowPatch(arrow_start, arrow_end,
                                arrowstyle='->', color='red', mutation_scale=20, lw=2, label='Final Info Gap Direction')
    ax1.add_patch(gap_arrow)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.axis('equal')
    ax1.set_xlim(sensor_est_pos[0] - 150, sensor_est_pos[0] + 150)
    ax1.set_ylim(sensor_est_pos[1] - 150, sensor_est_pos[1] + 150)

    # --- 右图: GDOP变化图 ---
    ax2.set_title("GDOP vs. Number of Measurements", fontsize=16)
    # X轴是测量点的数量 (从初始点数开始)
    num_measurements = np.arange(len(initial_history_points) - len(trajectory) + 1, len(initial_history_points) + 1)
    ax2.plot(num_measurements, gdop_history, 'r-o', lw=2, markersize=5, label='GDOP value')
    ax2.set_xlabel("Number of Measurement Points")
    ax2.set_ylabel("GDOP (Geometric Dilution of Precision)")
    ax2.set_xticks(np.arange(min(num_measurements), max(num_measurements) + 1, 5))  # 每5个点一个刻度
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    # 在关键点上标注数值
    for i, gdop in enumerate(gdop_history):
        if i == 0 or i == len(gdop_history) - 1 or (i + 1) % 10 == 0:  # 标注开始、结束和每10个点
            ax2.text(num_measurements[i], gdop, f'{gdop:.2f}', ha='center', va='bottom')

    # 调整布局防止重叠
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局为总标题留出空间

    # 显示最终的静态图
    plt.show()
    print("Plot displayed.")