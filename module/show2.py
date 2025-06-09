import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# 参数设置
A_i = 1.0
K = 1.0
zeta = 0.05  # 不确定性缩放因子
sigma_values = [1, 50, 100]

# 创建网格
x = np.linspace(-500, 500, 200)
y = np.linspace(-500, 500, 200)
X, Y = np.meshgrid(x, y)
d_squared = X ** 2 + Y ** 2

# 创建三维图
fig = plt.figure(figsize=(18, 12))

# 绘制原始公式
ax1 = fig.add_subplot(231, projection='3d')
ax2 = fig.add_subplot(232, projection='3d')
ax3 = fig.add_subplot(233, projection='3d')

# 绘制新公式
ax4 = fig.add_subplot(234, projection='3d')
ax5 = fig.add_subplot(235, projection='3d')
ax6 = fig.add_subplot(236, projection='3d')

for i, sigma in enumerate(sigma_values):
    # 原始公式
    Z_original = A_i * np.exp(-d_squared / (2 * K * sigma ** 2))

    # 新公式
    peak_scaling = A_i / (1 + zeta * sigma)
    Z_new = peak_scaling * np.exp(-d_squared / (2 * K * sigma ** 2))

    # 绘制原始公式
    ax = [ax1, ax2, ax3][i]
    surf = ax.plot_surface(X, Y, Z_original, cmap=cm.viridis, alpha=0.8)
    ax.set_title(f'原始公式: σ = {sigma}', fontsize=12)
    ax.set_zlim(0, 1.2)

    # 绘制新公式
    ax = [ax4, ax5, ax6][i]
    surf = ax.plot_surface(X, Y, Z_new, cmap=cm.plasma, alpha=0.8)
    title = f'新公式: σ = {sigma}, ζ = {zeta}\n峰值 = {peak_scaling:.2f}'
    ax.set_title(title, fontsize=12)
    ax.set_zlim(0, 1.2)

# 添加标签
for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Mask_Data')

plt.tight_layout()
plt.show()