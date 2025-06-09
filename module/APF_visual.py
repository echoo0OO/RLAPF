import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def attractive_potential(x, y, goal_x=0, goal_y=0, zeta=1.0, d_star=1.0):
    """
    计算引力势场

    参数:
    x, y: 二维平面上的坐标点
    goal_x, goal_y: 目标点坐标
    zeta: 引力系数
    d_star: 距离阈值，区分二次势和线性势的切换点

    返回:
    U_att: 引力势场值
    """
    # 计算到目标的距离
    dx = x - goal_x
    dy = y - goal_y
    d = np.sqrt(dx ** 2 + dy ** 2)

    # 计算引力势场 (使用分段函数)
    U_att = np.zeros_like(d)

    # 当距离小于等于阈值时，使用二次函数
    mask_quad = d <= d_star
    U_att[mask_quad] = 0.5 * zeta * d[mask_quad] ** 2

    # 当距离大于阈值时，使用线性函数
    mask_lin = d > d_star
    U_att[mask_lin] = d_star * zeta * d[mask_lin] - 0.5 * zeta * d_star ** 2

    return U_att


def visualize_attractive_potential(goal_x=0, goal_y=0, zeta=1.0, d_star=1.0, x_range=(-5, 5), y_range=(-5, 5),
                                   resolution=0.1):
    """
    可视化引力势场

    参数:
    goal_x, goal_y: 目标点坐标
    zeta: 引力系数
    d_star: 距离阈值
    x_range, y_range: 可视化的坐标范围
    resolution: 网格分辨率
    """
    # 创建网格数据
    x = np.arange(x_range[0], x_range[1], resolution)
    y = np.arange(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)

    # 计算引力势场
    Z = attractive_potential(X, Y, goal_x, goal_y, zeta, d_star)

    # 创建3D图形
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制3D表面
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)

    # 添加颜色条
    fig.colorbar(surf, shrink=0.5, aspect=5, label='引力势场值')

    # 标记目标点
    ax.scatter([goal_x], [goal_y], [0], color='red', s=100, marker='*', label='目标点')

    # 设置坐标轴标签和标题
    ax.set_xlabel('X坐标')
    ax.set_ylabel('Y坐标')
    ax.set_zlabel('引力势场值')
    ax.set_title(f'引力势场函数 U_att(q) (目标点: ({goal_x}, {goal_y}), ζ={zeta}, d*={d_star})')

    # 设置视角
    ax.view_init(elev=30, azim=45)

    # 添加图例
    ax.legend()

    plt.tight_layout()
    plt.show()


def visualize_attractive_potential_contour(goal_x=0, goal_y=0, zeta=1.0, d_star=1.0, x_range=(-5, 5), y_range=(-5, 5),
                                           resolution=0.1):
    """
    可视化引力势场的等高线图

    参数:
    goal_x, goal_y: 目标点坐标
    zeta: 引力系数
    d_star: 距离阈值
    x_range, y_range: 可视化的坐标范围
    resolution: 网格分辨率
    """
    # 创建网格数据
    x = np.arange(x_range[0], x_range[1], resolution)
    y = np.arange(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)

    # 计算引力势场
    Z = attractive_potential(X, Y, goal_x, goal_y, zeta, d_star)

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制等高线
    contour = ax.contourf(X, Y, Z, 50, cmap=cm.coolwarm)

    # 添加颜色条
    cbar = fig.colorbar(contour, ax=ax, label='引力势场值')

    # 标记目标点
    ax.scatter([goal_x], [goal_y], color='red', s=100, marker='*', label='目标点')

    # 计算并绘制梯度向量场 (引力方向)
    dx = X - goal_x
    dy = Y - goal_y
    d = np.sqrt(dx ** 2 + dy ** 2)

    # 避免除以零
    mask = d > 0
    dx_norm = np.zeros_like(dx)
    dy_norm = np.zeros_like(dy)
    dx_norm[mask] = dx[mask] / d[mask]
    dy_norm[mask] = dy[mask] / d[mask]

    # 绘制向量场
    ax.quiver(X[::5, ::5], Y[::5, ::5], dx_norm[::5, ::5], dy_norm[::5, ::5],
              color='blue', scale=30, label='引力方向')

    # 设置坐标轴标签和标题
    ax.set_xlabel('X坐标')
    ax.set_ylabel('Y坐标')
    ax.set_title(f'引力势场等高线图 (目标点: ({goal_x}, {goal_y}), ζ={zeta}, d*={d_star})')

    # 添加图例
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 可视化引力势场的3D图像
    visualize_attractive_potential(goal_x=0, goal_y=0, zeta=1.0, d_star=1.0)

    # 可视化引力势场的等高线图
    visualize_attractive_potential_contour(goal_x=0, goal_y=0, zeta=1.0, d_star=1.0)