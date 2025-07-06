# env/uncertain_model.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from typing import Optional, List

# Matplotlib 配置
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['STZhongsong']
mpl.rcParams['axes.unicode_minus'] = False


class UncertaintyModel:
    """
    【G-MGF 版本】
    实现 GDOP-Assisted Manifold Gradient Filtering (G-MGF) 算法。
    GDOP的计算现在是模型内部的功能，并直接用于动态调整MGF的更新步长(eta)。
    """

    def __init__(self, num_sensors: int, confidence_level: float = 0.99):
        self.num_sensors = num_sensors
        self.confidence_level = confidence_level
        self.estimated_positions = np.zeros((num_sensors, 2))
        self.covariance_matrices = np.array([np.eye(2) * 100 ** 2 for _ in range(num_sensors)])
        self.uncertainty_radii = np.full(num_sensors, 100.0)
        q_val = 1e-2
        self.process_noise_q = np.eye(2) * q_val
        self.ranging_points = [[] for _ in range(num_sensors)]

        # G-MGF specific parameters
        self.drone_height = 60.0  # 固定的无人机与地面传感器的高度差
        self.g0 = 1.125e-5  # 测量噪声方差系数, 从环境中移入
        self.gdop_sup = 10.0  # GDOP的上界(可调参数)，用于归一化eta

    def initialize_states(self, true_positions: np.ndarray, initial_radius: float, np_random: np.random.Generator):
        """初始化传感器状态。"""
        self.estimated_positions = np.array([
            true_pos + np_random.uniform(-initial_radius / 2, initial_radius / 2, 2)
            for true_pos in true_positions
        ])
        self.covariance_matrices = np.array([np.eye(2) * initial_radius ** 2 for _ in range(self.num_sensors)])
        self.uncertainty_radii = np.full(self.num_sensors, initial_radius)
        self.ranging_points = [[] for _ in range(self.num_sensors)]

    def add_ranging_point(self, sensor_id: int, drone_position: np.ndarray):
        """
        只向指定传感器的历史记录中添加一个新的测距点（无人机位置）。
        """
        self.ranging_points[sensor_id].append(drone_position)
        # 保持历史记录的长度，防止无限增长
        if len(self.ranging_points[sensor_id]) > 50:
            self.ranging_points[sensor_id].pop(0)

    def get_sector_coverage(self, sensor_id: int, num_sectors: int = 12) -> int:
        """
        计算一个传感器的历史点覆盖了多少个不同的扇区。

        Args:
            sensor_id (int): 传感器的ID。
            num_sectors (int): 总的扇区数量。

        Returns:
            int: 被覆盖的扇区的数量。
        """
        history_points = self.ranging_points[sensor_id]
        if len(history_points) < 1:
            return 0

        sensor_pos = self.estimated_positions[sensor_id]
        vectors = np.array(history_points) - sensor_pos
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        sector_width = 2 * np.pi / num_sectors
        sector_indices = np.floor((angles + np.pi) / sector_width).astype(int)
        sector_indices = np.clip(sector_indices, 0, num_sectors - 1)

        # 使用np.unique找到所有出现过的扇区索引，然后计算其数量
        num_covered_sectors = len(np.unique(sector_indices))

        return num_covered_sectors

    def _select_points_for_gdop(self, sensor_id: int, num_sectors: int = 12) -> List[np.ndarray]:
        """
        【新】根据角度分区选择用于计算GDOP的历史测距点。
        """
        sensor_pos_2d = self.estimated_positions[sensor_id]
        history_points_2d = np.array(self.ranging_points[sensor_id])

        if history_points_2d.shape[0] == 0:
            return []

        vectors = history_points_2d - sensor_pos_2d
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        distances = np.linalg.norm(vectors, axis=1)

        sector_width = 2 * np.pi / num_sectors
        sector_indices = np.floor((angles + np.pi) / sector_width).astype(int)
        sector_indices = np.clip(sector_indices, 0, num_sectors - 1)

        selected_points = []
        for i in range(num_sectors):
            points_in_sector_indices = np.where(sector_indices == i)[0]
            if len(points_in_sector_indices) > 0:
                closest_point_global_idx = points_in_sector_indices[np.argmin(distances[points_in_sector_indices])]
                selected_points.append(history_points_2d[closest_point_global_idx])

        return selected_points

    def calculate_current_gdop(self, sensor_id: int) -> float:
        """
        【新】计算单个传感器当前的GDOP值。
        """
        # 1. 使用角度分区策略选择历史测量点
        measurement_points_2d = self._select_points_for_gdop(sensor_id)

        # 2D定位至少需要2个点才有意义
        if len(measurement_points_2d) < 2:
            return self.gdop_sup  # 返回一个默认的大值

        # 2. 将点和目标提升到3D
        sensor_pos_3d = np.append(self.estimated_positions[sensor_id], 0.0)
        measurement_points_3d = [np.append(p, self.drone_height) for p in measurement_points_2d]

        # 3. 构建几何矩阵 G 和权重矩阵 W
        G_list = []
        weights = []
        for point_3d in measurement_points_3d:
            diff_vector = point_3d - sensor_pos_3d
            dist_3d = np.linalg.norm(diff_vector)
            if dist_3d < 1e-6: continue

            # G矩阵的行向量是归一化的方向向量（只取水平分量）
            unit_vector = diff_vector / dist_3d
            G_list.append(unit_vector[:2])  # PDOP (水平)

            # 权重是测量方差的倒数
            variance = self.g0 * (dist_3d ** 2)
            weights.append(1.0 / variance if variance > 1e-9 else 1e9)

        if len(G_list) < 2:
            return self.gdop_sup

        G_matrix = np.array(G_list)
        W_matrix = np.diag(weights)

        # 4. 计算PDOP
        try:
            GtWG = G_matrix.T @ W_matrix @ G_matrix
            if np.linalg.det(GtWG) < 1e-10: return self.gdop_sup
            GtWG_inv = np.linalg.inv(GtWG)
            pdop = np.sqrt(np.trace(GtWG_inv))
            return pdop
        except np.linalg.LinAlgError:
            return self.gdop_sup

    def g_mgf_update(self, sensor_id: int, drone_position: np.ndarray,
                     measured_distance: float, measurement_variance: float):
        """
        【升级版】执行单步 G-MGF 更新。
        """
        # --- 0. 计算动态步长 eta ---
        # 首先加入新的测距点，然后计算GDOP
        current_gdop = self.calculate_current_gdop(sensor_id)
        # 根据论文公式 η = GDOP_k / GDOP_sup
        step_size_eta = min(current_gdop / self.gdop_sup, 1.0)  # 归一化并限制最大为1

        # --- 1. 状态预测 ---
        x_prior = self.estimated_positions[sensor_id]
        P_prior = self.covariance_matrices[sensor_id] + self.process_noise_q
        R = measurement_variance
        if R < 1e-9: R = 1e-9

        # --- 2. 计算MGF组件 ---
        diff_vec = x_prior - drone_position
        estimated_dist = np.linalg.norm(diff_vec)
        if estimated_dist < 1e-6: return

        H = (diff_vec / estimated_dist).reshape(1, 2)
        e_z = estimated_dist - measured_distance
        grad_L = H.T * (1 / R) * e_z

        try:
            P_prior_inv = np.linalg.inv(P_prior)
        except np.linalg.LinAlgError:
            P_prior_inv = np.linalg.pinv(P_prior)

        G = (H.T * (1 / R)) @ H + P_prior_inv

        # --- 3. 执行MGF更新 ---
        try:
            G_inv = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            return

        natural_grad = G_inv @ grad_L
        # 使用动态步长eta进行更新
        x_post = x_prior - step_size_eta * natural_grad.flatten()
        P_post = G_inv

        # --- 4. 保存结果 ---
        self.estimated_positions[sensor_id] = x_post
        self.covariance_matrices[sensor_id] = P_post
        self.uncertainty_radii[sensor_id] = self.calculate_confidence_radius(P_post)

    # --- 可视化和辅助函数 (保持不变) ---
    def calculate_confidence_radius(self, covariance_matrix: np.ndarray) -> float:
        # ... (代码不变) ...
        eigenvalues = np.linalg.eigvals(covariance_matrix)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        chi2_val = chi2.ppf(self.confidence_level, df=2)
        return np.sqrt(chi2_val * max_eigenvalue)

    def get_confidence_ellipse(self, sensor_id: int, n_points: int = 100) -> np.ndarray:
        # ... (代码不变) ...
        center = self.estimated_positions[sensor_id]
        cov = self.covariance_matrices[sensor_id]
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        chi2_val = chi2.ppf(self.confidence_level, df=2)
        a, b = np.sqrt(chi2_val * eigenvalues)
        t = np.linspace(0, 2 * np.pi, n_points)
        ellipse_x, ellipse_y = a * np.cos(t), b * np.sin(t)
        R_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        ellipse_points = np.dot(R_mat, np.vstack([ellipse_x, ellipse_y]))
        return ellipse_points.T + center

    def visualize_uncertainty(self, sensor_id: int, true_position: Optional[np.ndarray] = None):
        # ... (代码不变) ...
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        if self.ranging_points and self.ranging_points[sensor_id]:
            points = np.array(self.ranging_points[sensor_id])
            ax.scatter(points[:, 0], points[:, 1], c='gray', s=30, alpha=0.6, label='历史测距点')
        est_pos = self.estimated_positions[sensor_id]
        ax.scatter(est_pos[0], est_pos[1], c='blue', s=100, marker='x', label='估计位置 (G-MGF)')
        ellipse_points = self.get_confidence_ellipse(sensor_id)
        ax.plot(ellipse_points[:, 0], ellipse_points[:, 1], 'b-', alpha=0.7,
                label=f'{int(self.confidence_level * 100)}% 置信椭圆')
        if true_position is not None:
            ax.scatter(true_position[0], true_position[1], c='red', s=100, marker='*', label='真实位置', zorder=5)
        ax.set_xlabel('X坐标 (m)')
        ax.set_ylabel('Y坐标 (m)')
        ax.set_title(f'传感器 {sensor_id} 位置不确定性 (G-MGF 算法)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.axis('equal')
        plt.show()


# __main__ 测试块可以保留用于独立调试


if __name__ == '__main__':
    # 添加一个测试用的主程序块，方便单独运行和调试此文件
    print("--- 测试 MGF 模型和可视化功能 ---")

    # 1. 初始化模型
    num_sensors = 1
    model = UncertaintyModel(num_sensors=num_sensors)

    # 2. 设置模拟环境
    np_random = np.random.default_rng()
    true_sensor_pos = np.array([[500.0, 500.0]])
    model.initialize_states(true_sensor_pos, initial_radius=100.0, np_random=np_random)

    print("\n--- 初始状态 ---")
    print(f"真实位置: {true_sensor_pos[0]}")
    print(f"初始估计位置: {model.estimated_positions[0]}")
    print(f"初始不确定性半径: {model.uncertainty_radii[0]:.2f}m")

    # 3. 模拟几次UAV测量和更新
    uav_positions = [
        np.array([500, 100]),
        np.array([900, 500]),
        np.array([500, 900]),
        np.array([100, 500]),
        np.array([800, 800])
    ]
    ranging_noise_std = 1.0  # 测距噪声标准差
    measurement_variance = ranging_noise_std ** 2

    print("\n--- 开始 MGF 更新 ---")
    for i, uav_pos in enumerate(uav_positions):
        # 模拟测量过程
        true_dist = np.linalg.norm(uav_pos - true_sensor_pos[0])
        measured_dist = true_dist + np.random.normal(0, ranging_noise_std)

        # 执行MGF更新
        model.g_mgf_update(0, uav_pos, measured_dist, measurement_variance)

        print(f"\n--- 更新 #{i + 1} (无人机位置: {uav_pos}) ---")
        print(f"估计位置: {model.estimated_positions[0]}")
        print(f"不确定性半径: {model.uncertainty_radii[0]:.2f}m")

        # --- 新增修改 ---
        # 从第二个测距点开始 (i 从 0 开始计数, i=1 代表第二个点),
        # 每一次更新后都进行可视化，展示不确定性的变化过程。
        if i >= 1:
            print(f"--- 正在生成第 {i + 1} 个点的更新图 ---")
            model.visualize_uncertainty(0, true_sensor_pos[0])

    print("\n--- 所有更新完成 ---")
    final_error = np.linalg.norm(model.estimated_positions[0] - true_sensor_pos[0])
    print(f"最终估计位置: {model.estimated_positions[0]}")
    print(f"最终不确定性半径: {model.uncertainty_radii[0]:.2f}m")
    print(f"最终定位误差: {final_error:.2f}m")