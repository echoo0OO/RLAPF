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
        self.covariance_matrices = np.array([np.eye(2) * initial_radius ** 2 for _ in range(self.num_sensors)])
        self.uncertainty_radii = np.full(self.num_sensors, initial_radius)
        self.ranging_points = [[] for _ in range(self.num_sensors)]
        #self.ranging_distances = [[] for _ in range(self.num_sensors)]
        # 【修改】
        self.ranging_measurements = [[] for _ in range(self.num_sensors)]

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
        self.estimated_positions = np.zeros((num_sensors, 2))
        self.covariance_matrices = np.array([np.eye(2) * 100**2 for _ in range(num_sensors)])
        self.uncertainty_radii = np.full(num_sensors, 100.0)
        
        # 测距点历史记录
        self.ranging_points = [[] for _ in range(num_sensors)]
        #self.ranging_distances = [[] for _ in range(num_sensors)]
        # 【修改】现在存储 (measured_distance, variance) 的元组
        self.ranging_measurements = [[] for _ in range(self.num_sensors)]
        
        # 定位误差模型参数
        #self.ranging_noise_std = 5.0  # 测距噪声标准差
        
    def add_ranging_measurement(self, sensor_id: int, drone_position: np.ndarray, 
                              measured_distance: float, measurement_variance: float):
        """
        添加测距测量数据
        
        Args:
            sensor_id: 传感器ID
            drone_position: 无人机位置
            measured_distance: 测量距离
        """
        self.ranging_points[sensor_id].append(drone_position.copy())
        #self.ranging_distances[sensor_id].append(measured_distance)
        # 【修改】
        self.ranging_measurements[sensor_id].append((measured_distance, measurement_variance))
        
        # 限制历史记录长度
        max_history = 50
        if len(self.ranging_points[sensor_id]) > max_history:
            self.ranging_points[sensor_id].pop(0)
            #self.ranging_distances[sensor_id].pop(0)
            self.ranging_measurements[sensor_id].pop(0)

    def nonlinear_least_squares_estimation(self, sensor_id: int, max_iterations=10, tolerance=1e-6):
        """
        非线性最小二乘法 (Gauss-Newton) 位置估计。
        这能提供更准确的估计和协方差。
        """
        points_2d = np.array(self.ranging_points[sensor_id])
        measurements = self.ranging_measurements[sensor_id]

        # 1. 使用上一次的估计作为初始猜测
        current_estimate = self.estimated_positions[sensor_id].copy()

        num_measurements = len(points_2d)
        if num_measurements < 3:
            return self.estimated_positions[sensor_id], self.covariance_matrices[sensor_id]

        # 准备权重矩阵的逆（即方差矩阵）
        variances = np.array([m[1] for m in measurements])
        # 权重是方差的倒数
        W = np.diag(1.0 / variances)

        # 2. 迭代优化
        for i in range(max_iterations):
            # a. 构建雅可比矩阵 J 和残差向量 r
            J = np.zeros((num_measurements, 2))
            r = np.zeros(num_measurements)

            for k in range(num_measurements):
                drone_pos = points_2d[k]
                measured_dist = measurements[k][0]

                # 计算当前估计下的预测距离
                diff_vec = drone_pos - current_estimate
                estimated_dist = np.linalg.norm(diff_vec)
                if estimated_dist < 1e-6: estimated_dist = 1e-6  # 避免除以零

                # 计算残差 (观测值 - 预测值)
                r[k] = measured_dist - estimated_dist

                # 计算雅可比矩阵的一行 (残差对x, y的偏导)
                J[k, 0] = -diff_vec[0] / estimated_dist
                J[k, 1] = -diff_vec[1] / estimated_dist

            # b. 求解线性子问题 (J^T * W * J) * dx = J^T * W * r
            try:
                JtWJ = J.T @ W @ J
                JtWr = J.T @ W @ r

                # 检查矩阵条件，防止数值不稳定
                if np.linalg.det(JtWJ) < 1e-12 or np.linalg.cond(JtWJ) > 1e8:
                    # 如果矩阵病态，说明几何构型很差，无法继续优化，返回当前结果
                    print(f"警告: 传感器{sensor_id}的JtWJ矩阵病态，迭代提前终止。")
                    # 此时的协方差更可信
                    final_covariance = np.linalg.inv(JtWJ)
                    return current_estimate, final_covariance

                delta_x = np.linalg.solve(JtWJ, JtWr)
            except np.linalg.LinAlgError:
                print(f"警告: 传感器{sensor_id}的NLS求解失败，迭代提前终止。")
                return self.estimated_positions[sensor_id], self.covariance_matrices[sensor_id]

            # c. 更新估计值
            current_estimate += delta_x

            # d. 检查收敛
            if np.linalg.norm(delta_x) < tolerance:
                # print(f"传感器{sensor_id} NLS在 {i+1} 次迭代后收敛。")
                break

        # 3. 计算最终的协方差矩阵
        # 理论上，参数的协方差矩阵是 (J^T * W * J)^-1
        try:
            final_covariance = np.linalg.inv(J.T @ W @ J)
        except np.linalg.LinAlgError:
            # 如果最后一步还是出错，返回上一次的协方差
            return current_estimate, self.covariance_matrices[sensor_id]

        return current_estimate, final_covariance
    
    def update_sensor_estimate(self, sensor_id: int):
        """更新传感器位置估计 (使用NLS)"""
        if len(self.ranging_points[sensor_id]) < 3:
            return

        # 使用非线性最小二乘法进行更新
        new_pos, new_cov = self.nonlinear_least_squares_estimation(sensor_id)

        # 添加一个检查，防止更新结果跳跃过大
        if np.linalg.norm(new_pos - self.estimated_positions[sensor_id]) < 200: # 如果更新跳跃小于200米
            self.estimated_positions[sensor_id] = new_pos
            self.covariance_matrices[sensor_id] = new_cov
            self.uncertainty_radii[sensor_id] = self.calculate_confidence_radius(new_cov)
        else:
            print(f"警告：传感器{sensor_id}的NLS更新跳跃过大，已忽略本次更新。")
    
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