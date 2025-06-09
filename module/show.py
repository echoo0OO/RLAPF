import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# 参数设置
A_i = 1.0  # 峰值强度
K = 1.0  # 缩放常数
sigma_values = [50, 100, 200]  # 不确定半径的三种取值

# 创建二维网格（仅x和y坐标）
x = np.linspace(-500, 500, 100)
y = np.linspace(-500, 500, 100)
X, Y = np.meshgrid(x, y)

# 计算距离（仅基于x和y坐标）
d_squared = X ** 2 + Y ** 2

# 创建三维图
fig = plt.figure(figsize=(18, 6))

for i, sigma in enumerate(sigma_values):
    # 计算引力场强度（Z值）
    Z = A_i * np.exp(-d_squared / (2 * K * sigma ** 2))

    # 创建三维子图
    ax = fig.add_subplot(1, 3, i + 1, projection='3d')

    # 绘制曲面
    surf = ax.plot_surface(
        X, Y, Z,
        cmap=cm.viridis,  # 使用彩色映射表示高度
        edgecolor='none',  # 无网格线，更平滑
        rstride=2,  # 减少行步长以提高渲染速度
        cstride=2,  # 减少列步长以提高渲染速度
        alpha=0.9  # 略微透明以便观察
    )

    # 设置标题和标签
    ax.set_title(f'σ = {sigma}', fontsize=16, pad=15)
    ax.set_xlabel('X', fontsize=12, labelpad=10)
    ax.set_ylabel('Y', fontsize=12, labelpad=10)
    ax.set_zlabel('Mask_Data', fontsize=12, labelpad=10)

    # 设置视角以便更好观察
    ax.view_init(elev=30, azim=-45)  # 仰角30度，方位角-45度

    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label='Intensity')

# 调整布局并显示
plt.tight_layout(pad=3.0)
plt.show()