import numpy as np
import matplotlib.pyplot as plt

# 参数设置
A_i = 1.0
sigma_ref = 1.0
K = 1.0
beta = 0.3  # 最小范围参数
sigma_values = [0.1, 0.5, 1.0, 2.0]  # 不确定性变化

x = np.linspace(-3, 3, 500)
results = []

plt.figure(figsize=(10, 6))
for sigma in sigma_values:
    # 计算场强分布
    peak = A_i * sigma_ref / sigma
    width = sigma + beta
    Z = peak * np.exp(-x ** 2 / (2 * K * width ** 2))

    # 计算半值距离
    d_half = width * np.sqrt(2 * K * np.log(2))

    # 存储结果
    results.append({
        'sigma': sigma,
        'profile': Z,
        'd_half': d_half,
        'peak': peak
    })

    # 绘制曲线
    plt.plot(x, Z, label=f'σ={sigma}, d_half={d_half:.2f}, peak={peak:.2f}')

# 添加标注
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.title(f'自适应高斯场 (β={beta}, K={K})')
plt.xlabel('Distance from Target')
plt.ylabel('Reward Intensity')
plt.legend()
plt.grid(True, alpha=0.2)
plt.ylim(0, 12)
plt.tight_layout()
plt.show()