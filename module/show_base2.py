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
        计算基础势场值 - 广域高斯分布

        参数:
        point -- 空间点坐标 [x, y, z]
        """
        d = np.linalg.norm(np.array(point) - self.position)
        return self.base_amplitude * np.exp(-d ** 2 / (2 * self.base_k * self.base_range ** 2))

    def refined_potential(self, point):
        """
        计算精调势场值 - 局部精确高斯分布

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
            # 不确定性越大，基础势场权重越高
            alpha = min(1.0, max(0.0, (self.uncertainty_radius / self.base_range)))
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
        self.only_use_collected_points = False  # 是否仅使用已采集点的精调势场

    def set_combination_method(self, method):
        """设置势场组合方法 ('add', 'max', 'weighted')"""
        self.combination_method = method

    def set_use_collected_points(self, use_collected):
        """设置是否使用已采集点的精调势场"""
        self.only_use_collected_points = use_collected

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
            # 如果设置仅使用已采集点，且该点未被采集，则跳过精调势场
            if self.only_use_collected_points and not dp.collected:
                continue

            total += dp.total_potential(point, self.combination_method)
        return total

    def visualize_2d(self, x_range=(-10, 10), y_range=(-10, 10), resolution=100, z_value=0):
        """
        可视化二维切片（固定z=z_value）的势场分布

        参数:
        x_range -- x轴范围 (min, max)
        y_range -- y轴范围 (min, max)
        resolution -- 网格分辨率
        z_value -- 固定z坐标值
        """
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        # 计算势场值
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = [X[i, j], Y[i, j], z_value]
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

    def visualize_3d(self, x_range=(-10, 10), y_range=(-10, 10), resolution=50, z_value=0):
        """
        可视化三维势场（固定z=z_value）

        参数:
        x_range -- x轴范围 (min, max)
        y_range -- y轴范围 (min, max)
        resolution -- 网格分辨率
        z_value -- 固定z坐标值
        """
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        # 计算势场值
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = [X[i, j], Y[i, j], z_value]
                Z[i, j] = self.total_field(point)

        # 创建3D图
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制曲面
        surf = ax.plot_surface(X, Y, Z, cmap='viridis',
                               rstride=1, cstride=1,
                               edgecolor='none',
                               alpha=0.8)
        fig.colorbar(surf, ax=ax, shrink=0.5, label='Potential Field Strength')

        # 标记数据点位置
        for i, dp in enumerate(self.data_points):
            z_val = self.total_field([dp.position[0], dp.position[1], z_value])
            color = 'red' if dp.collected else 'yellow'
            ax.scatter(dp.position[0], dp.position[1], z_val,
                       s=100, c=color, edgecolors='black', depthshade=True)
            ax.text(dp.position[0], dp.position[1], z_val * 1.05, f'P{i}',
                    fontsize=12, ha='center', va='bottom', color='black')

        ax.set_title('3D Potential Field Visualization')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Potential')

        # 设置视角以获得更好的观察角度
        ax.view_init(elev=30, azim=-60)

        plt.show()

        return X, Y, Z

    def visualize_field_components(self, point_index, x_range=(-10, 10), y_range=(-10, 10), resolution=100, z_value=0):
        """
        可视化单个数据点的势场分量（包含3D视图）
        """
        if point_index >= len(self.data_points):
            print(f"错误：不存在索引为 {point_index} 的数据点")
            return

        dp = self.data_points[point_index]
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        # 计算各分量
        Z_base = np.zeros_like(X)
        Z_refined = np.zeros_like(X)
        Z_total = np.zeros_like(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                p = [X[i, j], Y[i, j], z_value]
                Z_base[i, j] = dp.base_potential(p)
                Z_refined[i, j] = dp.refined_potential(p)
                Z_total[i, j] = dp.total_potential(p, self.combination_method)

        # 创建2D分量图
        fig_2d, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 基础势场
        contour1 = axes[0].contourf(X, Y, Z_base, levels=50, cmap='viridis')
        axes[0].set_title(f'Point {point_index} Base Potential\nRange={dp.base_range}, Amp={dp.base_amplitude}')
        axes[0].scatter(dp.position[0], dp.position[1], s=100, c='red')
        fig_2d.colorbar(contour1, ax=axes[0], label='Intensity')

        # 精调势场
        contour2 = axes[1].contourf(X, Y, Z_refined, levels=50, cmap='viridis')
        title = f'Point {point_index} Refined Potential\nσ={dp.uncertainty_radius:.2f}, Amp={dp.refined_amplitude}'
        axes[1].set_title(title)
        axes[1].scatter(dp.position[0], dp.position[1], s=100, c='red')
        fig_2d.colorbar(contour2, ax=axes[1], label='Intensity')

        # 总势场
        contour3 = axes[2].contourf(X, Y, Z_total, levels=50, cmap='viridis')
        axes[2].set_title(f'Point {point_index} Total Potential\n({self.combination_method} combination)')
        axes[2].scatter(dp.position[0], dp.position[1], s=100, c='red')
        fig_2d.colorbar(contour3, ax=axes[2], label='Intensity')

        plt.tight_layout()
        plt.show()

        # 创建3D分量图
        fig_3d = plt.figure(figsize=(18, 6))

        # 基础势场3D
        ax1 = fig_3d.add_subplot(131, projection='3d')
        surf1 = ax1.plot_surface(X, Y, Z_base, cmap='viridis', alpha=0.8)
        ax1.set_title(f'Base Potential (Point {point_index})')
        ax1.scatter(dp.position[0], dp.position[1], np.max(Z_base), s=100, c='red')
        fig_3d.colorbar(surf1, ax=ax1, shrink=0.6, label='Intensity')

        # 精调势场3D
        ax2 = fig_3d.add_subplot(132, projection='3d')
        surf2 = ax2.plot_surface(X, Y, Z_refined, cmap='viridis', alpha=0.8)
        ax2.set_title(f'Refined Potential (Point {point_index})')
        ax2.scatter(dp.position[0], dp.position[1], np.max(Z_refined), s=100, c='red')
        fig_3d.colorbar(surf2, ax=ax2, shrink=0.6, label='Intensity')

        # 总势场3D
        ax3 = fig_3d.add_subplot(133, projection='3d')
        surf3 = ax3.plot_surface(X, Y, Z_total, cmap='viridis', alpha=0.8)
        ax3.set_title(f'Total Potential (Point {point_index})')
        ax3.scatter(dp.position[0], dp.position[1], np.max(Z_total), s=100, c='red')
        fig_3d.colorbar(surf3, ax=ax3, shrink=0.6, label='Intensity')

        # 设置统一视角
        for ax in [ax1, ax2, ax3]:
            ax.view_init(elev=30, azim=-60)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Potential')

        plt.tight_layout()
        plt.show()


# 示例使用 - 多个传感器位置的广域高斯叠加
if __name__ == "__main__":
    # 定义数据点位置 - 模拟传感器位置
    points = [
        [-7, -7, 0],  # 左下角传感器
        [7, -7, 0],  # 右下角传感器
        [0, 0, 0],  # 中心传感器
        [-7, 7, 0],  # 左上角传感器
        [7, 7, 0]  # 右上角传感器
    ]

    # 基础势场参数 - 广域高斯
    base_params = {
        'amplitude': 0.8,  # 基础势场强度
        'range': 12.0,  # 基础势场范围 - 覆盖大部分区域
        'k': 1.0,  # 高斯衰减参数
    }

    # 精调势场参数 - 局部精确高斯
    refined_params = {
        'amplitude': 3.0,  # 精调势场强度 - 比基础强
        'k': 0.3,  # 精调势场衰减参数 - 更陡峭
        'initial_uncertainty': 2.0,  # 初始不确定性半径
        'min_uncertainty': 0.2  # 最小不确定性半径
    }

    # 创建势场系统
    field_system = MultiPointPotentialField(points, base_params, refined_params)

    # 1. 初始状态可视化
    print("初始状态 - 所有传感器未采集")
    print("2D视图:")
    field_system.visualize_2d(x_range=(-10, 10), y_range=(-10, 10))
    print("3D视图:")
    field_system.visualize_3d(x_range=(-10, 10), y_range=(-10, 10))

    # 可视化中心点的势场分量
    print("\n中心传感器(P2)的势场分量:")
    field_system.visualize_field_components(2, x_range=(-10, 10), y_range=(-10, 10))

    # 2. 更新不确定性 - 减少中心点的不确定性
    print("\n更新后 - 中心传感器(P2)不确定性降低")
    uncertainties = [2.0, 2.0, 0.5, 2.0, 2.0]
    field_system.update_uncertainties(uncertainties)
    print("2D视图:")
    field_system.visualize_2d(x_range=(-10, 10), y_range=(-10, 10))
    print("3D视图:")
    field_system.visualize_3d(x_range=(-10, 10), y_range=(-10, 10))

    # 3. 标记部分传感器已被采集
    print("\n更新后 - 左上角(P3)和右上角(P4)传感器已被采集")
    field_system.mark_point_collected(3)
    field_system.mark_point_collected(4)
    print("2D视图:")
    field_system.visualize_2d(x_range=(-10, 10), y_range=(-10, 10))
    print("3D视图:")
    field_system.visualize_3d(x_range=(-10, 10), y_range=(-10, 10))

    # 4. 使用加权组合方法
    print("\n使用加权组合方法")
    field_system.set_combination_method('weighted')
    print("2D视图:")
    field_system.visualize_2d(x_range=(-10, 10), y_range=(-10, 10))
    print("3D视图:")
    field_system.visualize_3d(x_range=(-10, 10), y_range=(-10, 10))

    # 5. 仅使用已采集点的精调势场
    print("\n仅使用已采集点的精调势场")
    field_system.set_use_collected_points(True)
    print("2D视图:")
    field_system.visualize_2d(x_range=(-10, 10), y_range=(-10, 10))
    print("3D视图:")
    field_system.visualize_3d(x_range=(-10, 10), y_range=(-10, 10))