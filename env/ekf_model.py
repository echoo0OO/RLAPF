
# env/ekf_model.py
# Implements a standard Extended Kalman Filter (EKF) for sensor localization.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from typing import Optional


class EKF_Model:
    """
    使用标准的扩展卡尔曼滤波器（EKF）实现传感器位置的不确定性模型。
    这是一个经过工业界和学术界广泛验证的、非常鲁棒的定位算法。
    """

    def __init__(self, num_sensors: int, confidence_level: float = 0.99):
        """
        初始化EKF模型。

        Args:
            num_sensors: 要跟踪的传感器数量。
            confidence_level: 用于计算不确定性椭圆的置信水平。
        """
        self.num_sensors = num_sensors
        self.confidence_level = confidence_level

        # 状态向量 x_k: [x, y] 位置
        self.estimated_positions = np.zeros((num_sensors, 2))

        # 状态协方差矩阵 P_k
        self.covariance_matrices = np.array([np.eye(2) * 100 ** 2 for _ in range(num_sensors)])

        # 不确定性半径，用于快速检查和可视化
        self.uncertainty_radii = np.full(num_sensors, 100.0)
        self.drone_height = 60.0  # 固定的无人机与地面传感器的高度差
        self.g0 = 1.125e-5  # 测量噪声方差系数, 从环境中移入

        # 过程噪声协方差矩阵 Q
        # 代表了我们对状态转移模型（x_k = x_{k-1}）的不确定性。
        # 对于静止传感器，这个值应该很小。
        q_val = 1e-4  # 推荐使用一个非常小的值
        self.process_noise_q = np.eye(2) * q_val

        # EKF需要用到的其他变量
        self.ranging_points = [[] for _ in range(num_sensors)]  # 仍然保留，用于可能的上层决策

    def initialize_states(self, true_positions: np.ndarray, initial_radius: float, np_random: np.random.Generator):
        """初始化所有传感器的状态。"""
        self.estimated_positions = np.array([
            true_pos + np_random.uniform(-initial_radius / 2, initial_radius / 2, 2)
            for true_pos in true_positions
        ])
        self.covariance_matrices = np.array([np.eye(2) * initial_radius ** 2 for _ in range(self.num_sensors)])
        self.uncertainty_radii = np.full(self.num_sensors, initial_radius)
        self.ranging_points = [[] for _ in range(self.num_sensors)]

    def add_ranging_point(self, sensor_id: int, drone_position: np.ndarray):
        """【新增】只向指定传感器的历史记录中添加一个新的测距点（无人机位置）。"""
        self.ranging_points[sensor_id].append(drone_position)
        if len(self.ranging_points[sensor_id]) > 50:
            self.ranging_points[sensor_id].pop(0)

    def update(self, sensor_id: int, drone_position: np.ndarray,
               measured_distance: float, measurement_variance: float):
        """
        执行一次完整的EKF预测和更新步骤。

        Args:
            sensor_id (int): 要更新的传感器ID。
            drone_position (np.ndarray): 进行测量的无人机2D位置。
            measured_distance (float): 实际的测距值 (z_k)。
            measurement_variance (float): 测距噪声的方差 (R_k)。
        """
        # --- 1. 预测 (Prediction) ---

        # 状态预测：对于静止传感器，预测位置就是上一时刻的位置。
        x_prior = self.estimated_positions[sensor_id]

        # 协方差预测：P_k|k-1 = F * P_k-1|k-1 * F^T + Q
        # 对于静止模型，状态转移矩阵 F 是单位矩阵 I。
        # 所以 P_prior = P_post_previous + Q
        P_prior = self.covariance_matrices[sensor_id] + self.process_noise_q

        # --- 2. 更新 (Update) ---

        # a. 计算测量预测值 h(x_prior) 和 测量雅可比矩阵 H
        diff_vec = x_prior - drone_position
        estimated_dist = np.linalg.norm(diff_vec)

        if estimated_dist < 1e-6:  # 避免除以零
            return

        # H 是测量函数 h(x) 对状态 x 的偏导数
        H = (diff_vec / estimated_dist).reshape(1, 2)

        # b. 计算测量残差 (Innovation)
        # y = z - h(x_prior)
        residual = measured_distance - estimated_dist

        # c. 计算残差的协方差 (Innovation Covariance)
        # S = H * P_prior * H^T + R
        R_k = np.array([[measurement_variance]])
        S = H @ P_prior @ H.T + R_k
        if S[0, 0] < 1e-9:  # 避免除以零
            return

        # d. 计算最优卡尔曼增益 (Optimal Kalman Gain)
        # K = P_prior * H^T * S^-1
        K = (P_prior @ H.T) / S[0, 0]  # K 是一个 (2, 1) 的列向量

        # e. 更新状态估计
        # x_post = x_prior + K * y
        x_post = x_prior + (K * residual).flatten()

        # f. 更新协方差矩阵 (使用数值上最稳定的 Joseph form)
        # P_post = (I - K * H) * P_prior
        I = np.eye(2)
        P_post = (I - K @ H) @ P_prior

        # --- 3. 保存更新后的状态 ---
        self.estimated_positions[sensor_id] = x_post
        self.covariance_matrices[sensor_id] = P_post
        self.uncertainty_radii[sensor_id] = self.calculate_confidence_radius(P_post)


    def calculate_confidence_radius(self, covariance_matrix: np.ndarray) -> float:
        """计算置信椭圆的长轴半径。"""
        # ... (此函数与之前完全相同) ...
        eigenvalues = np.linalg.eigvals(covariance_matrix)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        chi2_val = chi2.ppf(self.confidence_level, df=2)
        return np.sqrt(chi2_val * max_eigenvalue)

    # ... (其他辅助函数如 get_confidence_ellipse, get_sector_coverage 等可以保留，因为上层可能需要)
    def get_sector_coverage(self, sensor_id: int, num_sectors: int = 12) -> int:
        # ... (此函数与之前完全相同) ...
        history_points = self.ranging_points[sensor_id]
        if len(history_points) < 1: return 0
        sensor_pos = self.estimated_positions[sensor_id]
        vectors = np.array(history_points) - sensor_pos
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        sector_width = 2 * np.pi / num_sectors
        sector_indices = np.floor((angles + np.pi) / sector_width).astype(int)
        sector_indices = np.clip(sector_indices, 0, num_sectors - 1)
        return len(np.unique(sector_indices))


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Testing EKF Model ---")

    num_sensors = 1
    true_sensor_pos = np.array([500.0, 500.0])

    # Instantiate the EKF model
    model = EKF_Model(num_sensors=num_sensors)

    # Initialize state with a large uncertainty
    np_random = np.random.default_rng()
    model.initialize_states(np.array([true_sensor_pos]), 100.0, np_random)
    initial_estimate = model.estimated_positions[0]

    print("\n--- Initial State ---")
    print(f"True Position: {true_sensor_pos}")
    print(f"Initial Estimate: {initial_estimate}")
    print(f"Initial Uncertainty Radius: {model.uncertainty_radii[0]:.2f}m")

    # Simulate UAV Measurements from ideal locations
    uav_positions = [
        np.array([500, 0]),
        np.array([1000, 500]),
        np.array([500, 1000]),
        np.array([0, 500]),
        np.array([800, 800])
    ]
    ranging_noise_std = 2.0

    print("\n--- Starting EKF Updates ---")
    for i, uav_pos in enumerate(uav_positions):
        measurement_variance = ranging_noise_std ** 2
        true_dist = np.linalg.norm(uav_pos - true_sensor_pos)
        # Simulate a noisy measurement
        measured_dist = true_dist + np.random.normal(0, ranging_noise_std)

        # Perform the EKF update
        model.update(0, uav_pos, measured_dist, measurement_variance)

        print(f"\n--- Update #{i + 1} (UAV at: {uav_pos}) ---")
        print(f"Estimated Position: {model.estimated_positions[0]}")
        print(f"Uncertainty Radius: {model.uncertainty_radii[0]:.2f}m")
        print(f"Current Error: {np.linalg.norm(model.estimated_positions[0] - true_sensor_pos):.2f}m")

    print("\n--- Final Result ---")
    final_error = np.linalg.norm(model.estimated_positions[0] - true_sensor_pos)
    print(f"Final Estimated Position: {model.estimated_positions[0]}")
    print(f"Final Uncertainty Radius: {model.uncertainty_radii[0]:.2f}m")
    print(f"Final Positioning Error: {final_error:.2f}m")