import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from typing import List, Tuple, Dict, Optional
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题

class UncertaintyModel:
    """传感器位置不确定性模型"""
    #ranging_noise_std: float

    # 在 uncertain_model.py 的 UncertaintyModel 类中添加此方法
    def initialize_states(self, true_positions, initial_radius, np_random):
        self.true_positions = true_positions  # 仅为模拟目的存储
        self.estimated_positions = np.array([
            true_pos + np_random.uniform(-initial_radius / 2, initial_radius / 2, 2)
            for true_pos in self.true_positions
        ])
        # 【修改】初始化协方差 P0
        self.covariance_matrices = np.array([np.eye(2) * initial_radius ** 2 for _ in range(self.num_sensors)])
        self.uncertainty_radii = np.full(self.num_sensors, initial_radius)
        # 不再需要 ranging_points 和 ranging_measurements 历史记录

    def __init__(self, num_sensors: int, confidence_level: float = 0.99): # 置信度由0.95改到0.99
        """
        初始化不确定性模型
        
        Args:
            num_sensors: 传感器数量
            confidence_level: 置信水平
        """
        self.num_sensors = num_sensors
        self.confidence_level = confidence_level
        
        # 传感器估计位置和协方差矩阵
        # EKF状态: 估计位置和协方差矩阵
        self.estimated_positions = np.zeros((num_sensors, 2))
        # 【修改】P矩阵是EKF的核心
        self.covariance_matrices = np.array([np.eye(2) * 100**2 for _ in range(num_sensors)])
        self.uncertainty_radii = np.full(num_sensors, 100.0)
        
        # 测距点历史记录
        self.ranging_points = [[] for _ in range(num_sensors)]
        #self.ranging_distances = [[] for _ in range(num_sensors)]
        # 【修改】现在存储 (measured_distance, variance) 的元组
        self.ranging_measurements = [[] for _ in range(self.num_sensors)]
        
        # 定位误差模型参数
        #self.ranging_noise_std = 5.0  # 测距噪声标准差

    # 2. 【核心修改】用EKF更新步骤替换所有旧逻辑
    def ekf_update_step(self, sensor_id: int, drone_position: np.ndarray,
                        measured_distance: float, measurement_variance: float):
        """
        使用单个测量对传感器的状态进行EKF更新。
        """
        # --- 1. 获取当前状态 (预测步骤，由于状态静止，预测=当前) ---
        x_priori = self.estimated_positions[sensor_id]
        P_priori = self.covariance_matrices[sensor_id]
        # --- 2. 计算更新所需项 ---
        # a. 计算雅可比矩阵 H (在EKF中通常用H表示)
        diff_vec = drone_position - x_priori
        estimated_dist = np.linalg.norm(diff_vec)
        if estimated_dist < 1e-6: return  # 距离太近，更新不稳定
        H = (-diff_vec / estimated_dist).reshape(1, 2)  # 必须是 1x2 的行向量
        # b. 计算卡尔曼增益 K
        # K = P' * H^T * (H * P' * H^T + R)^-1
        H_P_Ht = H @ P_priori @ H.T
        # R 是测量噪声方差，是一个 1x1 标量
        R = measurement_variance
        # (H * P' * H^T + R) 是一个 1x1 矩阵，求逆就是取倒数
        S = H_P_Ht[0, 0] + R
        if S < 1e-9: return  # 避免除以零
        K = (P_priori @ H.T) / S  # K 是一个 2x1 的列向量
        # --- 3. 更新状态和协方差 ---
        # a. 计算残差 y
        residual = measured_distance - estimated_dist
        # b. 更新状态估计 x_post = x_priori + K * y
        x_post = x_priori + (K * residual).flatten()
        # c. 更新协方差 P_post = (I - K * H) * P_priori
        I_KH = np.eye(2) - K @ H
        P_post = I_KH @ P_priori
        # --- 4. 保存更新后的结果 ---
        self.estimated_positions[sensor_id] = x_post
        self.covariance_matrices[sensor_id] = P_post
        self.uncertainty_radii[sensor_id] = self.calculate_confidence_radius(P_post)
    def calculate_confidence_radius(self, covariance_matrix: np.ndarray) -> float:
        """
        计算置信椭圆长轴作为不确定性半径
        
        Args:
            covariance_matrix: 2x2协方差矩阵
            
        Returns:
            不确定性半径
        """
        # 计算特征值
        eigenvalues = np.linalg.eigvals(covariance_matrix)
        max_eigenvalue = np.max(eigenvalues)
        
        # 计算置信椭圆半径
        chi2_val = chi2.ppf(self.confidence_level, df=2)
        radius = np.sqrt(chi2_val * max_eigenvalue)
        
        return radius
    
    def get_confidence_ellipse(self, sensor_id: int, n_points: int = 100) -> np.ndarray:
        """
        获取置信椭圆边界点
        
        Args:
            sensor_id: 传感器ID
            n_points: 椭圆边界点数量
            
        Returns:
            椭圆边界点坐标 shape: (n_points, 2)
        """
        center = self.estimated_positions[sensor_id]
        cov = self.covariance_matrices[sensor_id]
        
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # 计算椭圆参数
        chi2_val = chi2.ppf(self.confidence_level, df=2)
        a = np.sqrt(chi2_val * eigenvalues[1])  # 长轴
        b = np.sqrt(chi2_val * eigenvalues[0])  # 短轴
        
        # 旋转角度
        angle = np.arctan2(eigenvectors[1, 1], eigenvectors[1, 0])
        
        # 生成椭圆边界点
        t = np.linspace(0, 2*np.pi, n_points)
        ellipse_x = a * np.cos(t)
        ellipse_y = b * np.sin(t)
        
        # 旋转和平移
        cos_angle, sin_angle = np.cos(angle), np.sin(angle)
        x = center[0] + ellipse_x * cos_angle - ellipse_y * sin_angle
        y = center[1] + ellipse_x * sin_angle + ellipse_y * cos_angle
        
        return np.column_stack([x, y])
    
    def is_converged(self, threshold: float = 1.0, stable_updates: int = 3) -> bool:
        """
        检查不确定性是否收敛
        
        Args:
            threshold: 收敛阈值（米）
            stable_updates: 需要稳定的更新次数
            
        Returns:
            是否收敛
        """
        # 简化实现：检查所有传感器的不确定性半径是否小于阈值
        return np.all(self.uncertainty_radii < threshold)
    
    def get_worst_case_distance(self, sensor_id: int, drone_position: np.ndarray) -> float:
        """
        计算最坏情况下的距离（传感器在距离无人机最远的位置）
        
        Args:
            sensor_id: 传感器ID
            drone_position: 无人机位置
            
        Returns:
            最坏情况距离
        """
        estimated_pos = self.estimated_positions[sensor_id]
        uncertainty_radius = self.uncertainty_radii[sensor_id]
        
        # 计算从估计位置到无人机的距离
        base_distance = np.linalg.norm(drone_position - estimated_pos)
        
        # 最坏情况：传感器在远离无人机的方向上偏移不确定性半径
        worst_case_distance_horizon = base_distance + uncertainty_radius
        worst_case_distance = np.sqrt(worst_case_distance_horizon**2 + 60**2) #高度差60m

        
        return worst_case_distance
    
    def visualize_uncertainty(self, sensor_id: int, true_position: Optional[np.ndarray] = None):
        """可视化单个传感器的不确定性"""
        plt.figure(figsize=(8, 8))
        
        # 绘制测距点
        if self.ranging_points[sensor_id]:
            points = np.array(self.ranging_points[sensor_id])
            plt.scatter(points[:, 0], points[:, 1], c='gray', s=30, alpha=0.6, label='测距点')
        
        # 绘制估计位置
        est_pos = self.estimated_positions[sensor_id]
        plt.scatter(est_pos[0], est_pos[1], c='blue', s=100, marker='x', label='估计位置')
        
        # 绘制置信椭圆
        ellipse_points = self.get_confidence_ellipse(sensor_id)
        plt.plot(ellipse_points[:, 0], ellipse_points[:, 1], 'b-', alpha=0.7, label='置信椭圆')
        
        # 绘制真实位置（如果提供）
        if true_position is not None:
            plt.scatter(true_position[0], true_position[1], c='red', s=100, marker='o', label='真实位置')
        
        plt.xlabel('X坐标 (m)')
        plt.ylabel('Y坐标 (m)')
        plt.title(f'传感器{sensor_id}位置不确定性')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.show()

if __name__ == "__main__":
    # 测试代码
    model = UncertaintyModel(num_sensors=1)
    
    # 模拟一些测距测量
    true_sensor_pos = np.array([500, 500])
    model.estimated_positions[0] = true_sensor_pos + np.random.normal(0, 20, 2)
    
    # 添加测距测量
    for i in range(10):
        drone_pos = np.random.uniform(0, 1000, 2)
        true_distance = np.linalg.norm(drone_pos - true_sensor_pos)
        measured_distance = true_distance + np.random.normal(0, model.ranging_noise_std)
        model.add_ranging_measurement(0, drone_pos, measured_distance)
    
    # 更新估计
    model.update_sensor_estimate(0)
    
    print(f"真实位置: {true_sensor_pos}")
    print(f"估计位置: {model.estimated_positions[0]}")
    print(f"不确定性半径: {model.uncertainty_radii[0]:.2f}m")
    
    # 可视化
    model.visualize_uncertainty(0, true_sensor_pos) 