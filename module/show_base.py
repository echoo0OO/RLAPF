import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


class DataPointPotential:
    """
    数据点势场类，管理单个数据点的势场计算
    """

    def __init__(self, position, base_potential_params, refined_potential_params):
        """
        初始化数据点势场

        参数:
        position -- 数据点位置 [x, y, z]
        base_potential_params -- 基础势场参数字典
        refined_potential_params -- 精调势场参数字典
        """
        self.position = np.array(position)
        self.collected = False  # 标记是否已被采集

        # 基础势场参数
        self.base_amplitude = base_potential_params['amplitude']
        self.base_range = base_potential_params['range']
        self.base_k = base_potential_params['k']
        self.base_type = base_potential_params.get('type', 'gaussian')  # 默认高斯形式

        # 精调势场参数
        self.refined_amplitude = refined_potential_params['amplitude']
        self.refined_k = refined_potential_params['k']
        self.uncertainty_radius = refined_potential_params.get('initial_uncertainty', 1.0)
        self.min_uncertainty = refined_potential_params.get('min_uncertainty', 0.1)

    def update_uncertainty(self, new_radius):
        """更新不确定性半径"""
        self.uncertainty_radius = max(self.min_uncertainty, new_radius)

    def mark_collected(self):
        """标记数据点已被采集"""
        self.collected = True

    def base_potential(self, point):
        """
        计算基础势场值

        参数:
        point -- 空间点坐标 [x, y, z]
        """
        d = np.linalg.norm(np.array(point) - self.position)

        if self.base_type == 'gaussian':
            # 高斯形式
            return self.base_amplitude * np.exp(-d ** 2 / (2 * self.base_k * self.base_range ** 2))
        else:
            # 线性形式
            return self.base_amplitude * max(0, 1 - d / self.base_range)

    def refined_potential(self, point):
        """
        计算精调势场值

        参数:
        point -- 空间点坐标 [x, y, z]
        """
        d = np.linalg.norm(np.array(point) - self.position)
        return self.refined_amplitude * np.exp(-d ** 2 / (2 * self.refined_k * self.uncertainty_radius ** 2))

    def total_potential(self, point, combination='add'):
        """
        计算该数据点的总势场值

        参数:
        point -- 空间点坐标 [x, y, z]
        combination -- 势场组合方式 ('add', 'max', or 'weighted')
        """
        if self.collected:
            return 0.0

        base_val = self.base_potential(point)
        refined_val = self.refined_potential(point)

        if combination == 'add':
            return base_val + refined_val
        elif combination == 'max':
            return max(base_val, refined_val)
        elif combination == 'weighted':
            # 加权组合 - 权重基于不确定性
            alpha = 0.5  # 基础权重
            # 不确定性越大，基础势场权重越高
            alpha = min(1.0, max(0.0, alpha * (self.uncertainty_radius / self.base_range)))
            return (1 - alpha) * refined_val + alpha * base_val
        else:
            return refined_val  # 默认返回精调势场


class MultiPointPotentialField:
    """
    多点势场系统，管理多个数据点的势场叠加
    """

    def __init__(self, points, base_params, refined_params):
        """
        初始化多点势场系统

        参数:
        points -- 数据点位置列表 [[x1,y1,z1], [x2,y2,z2], ...]
        base_params -- 基础势场参数字典
        refined_params -- 精调势场参数字典
        """
        self.data_points = [
            DataPointPotential(pos, base_params, refined_params)
            for pos in points
        ]
        self.combination_method = 'add'  # 默认加法组合

    def set_combination_method(self, method):
        """设置势场组合方法 ('add', 'max', 'weighted')"""
        self.combination_method = method

    def update_uncertainties(self, uncertainties):
        """更新所有数据点的不确定性半径"""
        for i, radius in enumerate(uncertainties):
            if i < len(self.data_points):
                self.data_points[i].update_uncertainty(radius)

    def mark_point_collected(self, index):
        """标记指定数据点已被采集"""
        if index < len(self.data_points):
            self.data_points[index].mark_collected()

    def total_field(self, point):
        """
        计算给定空间点的总势场值（所有数据点势场之和）

        参数:
        point -- 空间点坐标 [x, y, z]
        """
        total = 0.0
        for dp in self.data_points:
            total += dp.total_potential(point, self.combination_method)
        return total

    def visualize_2d(self, x_range=(-10, 10), y_range=(-10, 10), resolution=100):
        """
        可视化二维切片（固定z=0）的势场分布

        参数:
        x_range -- x轴范围 (min, max)
        y_range -- y轴范围 (min, max)
        resolution -- 网格分辨率
        """
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        # 计算势场值
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = [X[i, j], Y[i, j], 0]  # 固定z=0
                Z[i, j] = self.total_field(point)

        # 绘制等高线图
        plt.figure(figsize=(12, 10))
        contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar(contour, label='Potential Field Strength')

        # 标记数据点位置
        for i, dp in enumerate(self.data_points):
            color = 'red' if dp.collected else 'white'
            plt.scatter(dp.position[0], dp.position[1], s=100, c=color, edgecolors='black')
            plt.text(dp.position[0], dp.position[1], f'P{i}',
                     fontsize=12, ha='center', va='center', color='black')

        plt.title('2D Potential Field Visualization')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(alpha=0.3)
        plt.show()

        return X, Y, Z

    def visualize_3d(self, x_range=(-10, 10), y_range=(-10, 10), resolution=50):
        """
        可视化三维势场（固定z=0）

        参数:
        x_range -- x轴范围 (min, max)
        y_range -- y轴范围 (min, max)
        resolution -- 网格分辨率
        """
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        # 计算势场值
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = [X[i, j], Y[i, j], 0]  # 固定z=0
                Z[i, j] = self.total_field(point)

        # 创建3D图
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制曲面
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
        fig.colorbar(surf, ax=ax, shrink=0.5, label='Potential Field Strength')

        # 标记数据点位置
        for i, dp in enumerate(self.data_points):
            z_val = self.total_field([dp.position[0], dp.position[1], 0])
            color = 'red' if dp.collected else 'yellow'
            ax.scatter(dp.position[0], dp.position[1], z_val,
                       s=100, c=color, edgecolors='black')
            ax.text(dp.position[0], dp.position[1], z_val, f'P{i}',
                    fontsize=12, ha='center', va='bottom')

        ax.set_title('3D Potential Field Visualization')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Potential')

        plt.show()

        return X, Y, Z


# 示例使用
if __name__ == "__main__":
    # 定义数据点位置
    points = [
        [-5, -5, 0],
        [0, 0, 0],
        [5, 5, 0],
        [5, -5, 0]
    ]

    # 基础势场参数
    base_params = {
        'amplitude': 1.0,  # 基础势场强度
        'range': 15.0,  # 基础势场范围
        'k': 1.0,  # 高斯衰减参数
        'type': 'gaussian'  # 势场类型 (gaussian 或 linear)
    }

    # 精调势场参数
    refined_params = {
        'amplitude': 3.0,  # 精调势场强度
        'k': 0.5,  # 精调势场衰减参数
        'initial_uncertainty': 2.0,  # 初始不确定性半径
        'min_uncertainty': 0.1  # 最小不确定性半径
    }

    # 创建势场系统
    field_system = MultiPointPotentialField(points, base_params, refined_params)

    print("初始状态 - 所有点未采集，中等不确定性")
    field_system.visualize_2d()
    field_system.visualize_3d()

    # 更新不确定性 - 减少某些点的不确定性
    print("\n更新后 - 点0和点2不确定性降低")
    uncertainties = [0.5, 2.0, 0.3, 2.0]  # 更新不确定性半径
    field_system.update_uncertainties(uncertainties)
    field_system.visualize_2d()
    field_system.visualize_3d()

    # 标记部分点已被采集
    print("\n更新后 - 点1和点3已被采集")
    field_system.mark_point_collected(1)
    field_system.mark_point_collected(3)
    field_system.visualize_2d()
    field_system.visualize_3d()

    # 使用不同的组合方法
    print("\n使用最大值组合方法")
    field_system.set_combination_method('max')
    field_system.visualize_2d()

    print("\n使用加权组合方法")
    field_system.set_combination_method('weighted')
    field_system.visualize_2d()