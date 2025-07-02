import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


class SpiralManeuverController:
    """
    一个用于生成自适应螺旋定位机动轨迹的控制器。

    该控制器计算单步的移动向量，使得无人机能够沿着一条从外向内收缩的
    螺旋线飞行。在螺旋线外圈时，主要进行径向移动（快速靠近）；在内圈时，
    主要进行切向移动（绕圈飞行），从而在保持总速度恒定的前提下，高效地
    平衡“接近”和“获取几何多样性”两个目标。
    """

    def __init__(self, total_speed: float, start_radius: float, min_radius: float, time_slot: float):
        """
        初始化螺旋机动控制器。

        Args:
            total_speed (float): 无人机在机动过程中的总速度 (m/s)。
            start_radius (float): 触发机动时，无人机距离目标的期望半径 (m)。
            min_radius (float): 螺旋线结束的最小半径 (m)。当无人机进入此半径内，机动结束。
            time_slot (float): 环境的仿真步长 (s)。
        """
        self.total_speed = total_speed
        self.start_radius = start_radius
        self.min_radius = min_radius
        self.time_slot = time_slot

    def calculate_move_vector(self, drone_pos: np.ndarray, sensor_est_pos: np.ndarray) -> np.ndarray:
        """
        根据当前状态，计算下一步的移动向量。

        Args:
            drone_pos (np.ndarray): 无人机当前2D位置。
            sensor_est_pos (np.ndarray): 目标传感器估计的2D位置（螺旋圆心）。

        Returns:
            np.ndarray: 无人机在下一个时间步应该移动的2D向量。
        """
        # 1. 计算径向和切向的单位向量
        vector_to_center = sensor_est_pos - drone_pos
        dist_to_center = np.linalg.norm(vector_to_center)

        # 如果已经非常接近，则悬停（返回零向量）
        if dist_to_center < self.min_radius:
            return np.array([0.0, 0.0])

        norm_radial_vector = vector_to_center / (dist_to_center + 1e-6)
        norm_tangent_vector = np.array([-norm_radial_vector[1], norm_radial_vector[0]])

        # 2. 根据归一化距离动态计算权重
        #    progress: 描述了从外圈到内圈的进度，从1 (最远) -> 0 (最近)
        progress = (dist_to_center - self.min_radius) / (self.start_radius - self.min_radius)
        #    将进度裁剪到[0, 1]区间，以处理无人机在起始半径之外或最小半径之内的情况
        progress = np.clip(progress, 0, 1)

        #    当在远处时(progress ≈ 1), 径向权重高, 快速靠近
        #    当在近处时(progress ≈ 0), 切向权重高, 快速绕圈
        #weight_radial = progress
        weight_radial = progress
        weight_tangent = 1.0 - progress

        # 3. 合成最终的飞行方向向量并归一化
        final_direction = (norm_radial_vector * weight_radial) + (norm_tangent_vector * weight_tangent)
        norm_final_direction = final_direction / (np.linalg.norm(final_direction) + 1e-6)

        # 4. 将总速度施加到最终方向上，得到移动向量
        move_vector = norm_final_direction * self.total_speed * self.time_slot

        return move_vector


# ==============================================================================
# 当此文件作为主脚本运行时，执行以下可视化和测试代码
# ==============================================================================
if __name__ == "__main__":

    print("正在运行 `maneuver_controllers.py` 的独立可视化测试...")


    # --- 辅助函数：模拟GDOP计算 ---
    def calculate_gdop(measurement_points, target_pos):
        if len(measurement_points) < 2: return 99.0
        H = []
        for p in measurement_points:
            diff = p - target_pos
            dist = np.linalg.norm(diff)
            if dist < 1e-6: continue
            H.append(diff / dist)
        H = np.array(H)
        if H.shape[0] < 2: return 99.0
        try:
            H_T_H_inv = np.linalg.inv(H.T @ H)
            return np.sqrt(np.trace(H_T_H_inv))
        except np.linalg.LinAlgError:
            return 99.0


    # --- 场景和控制器参数设定 ---
    SIM_TOTAL_SPEED = 30.0
    SIM_START_RADIUS = 100.0
    SIM_MIN_RADIUS = 0.0
    SIM_TIME_SLOT = 1.0

    sensor_true_pos = np.array([505.0, 495.0])
    sensor_est_pos = np.array([500.0, 500.0])

    # --- 初始化控制器 ---
    controller = SpiralManeuverController(
        total_speed=SIM_TOTAL_SPEED,
        start_radius=SIM_START_RADIUS,
        min_radius=SIM_MIN_RADIUS,
        time_slot=SIM_TIME_SLOT
    )

    # --- 模拟循环 ---
    # 将无人机放置在起始半径处
    drone_pos = sensor_est_pos + np.array([SIM_START_RADIUS, 0.0])

    trajectory = [drone_pos.copy()]
    ranging_points = [drone_pos.copy()]
    gdop_history = [calculate_gdop(ranging_points, sensor_est_pos)]

    max_steps = 100  # 防止无限循环
    for step in range(max_steps):
        # 核心：调用控制器计算下一步移动
        move_vector = controller.calculate_move_vector(drone_pos, sensor_est_pos)

        # 如果移动向量为零（表示已到达最小半径），则停止
        if np.linalg.norm(move_vector) < 1e-6:
            print(f"机动在第 {step + 1} 步完成，已到达最小半径。")
            break

        # 更新状态
        drone_pos += move_vector
        trajectory.append(drone_pos.copy())
        ranging_points.append(drone_pos.copy())
        gdop_history.append(calculate_gdop(ranging_points, sensor_est_pos))

    # --- 绘图 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 左图: 轨迹
    ax1.set_title("自适应螺旋定位轨迹 (控制器测试)", fontsize=16)
    ax1.plot(np.array(trajectory)[:, 0], np.array(trajectory)[:, 1], 'g-', lw=2, label='无人机飞行轨迹')
    ax1.scatter(sensor_true_pos[0], sensor_true_pos[1], c='red', s=150, marker='*', zorder=5, label='传感器真实位置')
    ax1.scatter(sensor_est_pos[0], sensor_est_pos[1], c='blue', s=100, marker='x', zorder=5,
                label='传感器估计位置 (圆心)')
    # 绘制起始点和结束点
    ax1.scatter(trajectory[0][0], trajectory[0][1], c='lime', edgecolors='black', s=120, marker='o', zorder=5,
                label='开始点')
    ax1.scatter(trajectory[-1][0], trajectory[-1][1], c='orange', edgecolors='black', s=120, marker='s', zorder=5,
                label='结束点')

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.axis('equal')

    # 右图: GDOP
    ax2.set_title("GDOP随定位点增加而改善", fontsize=16)
    steps_axis = np.arange(1, len(gdop_history) + 1)
    bars = ax2.bar(steps_axis, gdop_history, color='mediumseagreen')
    ax2.set_xlabel("定位点数量", fontsize=12)
    ax2.set_ylabel("GDOP值", fontsize=12)
    ax2.set_xticks(steps_axis)
    ax2.set_ylim(0, max(gdop_history) * 1.1 if len(gdop_history) > 1 and max(gdop_history) < 90 else 10)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        if yval < 90:
            ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.1, f'{yval:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()