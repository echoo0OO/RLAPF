import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题



# 目标函数（加权残差平方和）
def objective_function(x, anchors, measurements, weights):
    d_est = np.linalg.norm(anchors - x, axis=1)
    residuals = measurements - d_est
    return np.sum(weights * residuals ** 2)


# 计算雅可比矩阵
def compute_jacobian(x, anchors):
    delta = anchors - x
    d = np.linalg.norm(delta, axis=1)
    d[d < 1e-3] = 1e-3  # 正则化处理
    return -delta / d[:, np.newaxis]


# 更新估计模型
def update_uncertain_model(x_est, r_est, anchors, measurements, g0, num_rounds):
    confidence_level = 0.99
    current_d_est = np.linalg.norm(anchors - x_est, axis=1)
    weights = 1 / (g0 * current_d_est ** 2)

    # 优化求解
    result = minimize(objective_function, x_est, args=(anchors, measurements, weights),
                      bounds=[(x_est[0] - 2 * r_est, x_est[0] + 2 * r_est),
                              (x_est[1] - 2 * r_est, x_est[1] + 2 * r_est)])
    x_new = result.x

    # 不确定性量化
    J = compute_jacobian(x_new, anchors)
    info_matrix = J.T @ np.diag(weights) @ J
    cov_matrix = np.linalg.inv(info_matrix)

    eigvals, _ = np.linalg.eig(cov_matrix)
    semi_axes = np.sqrt(eigvals) * np.sqrt(chi2.ppf(confidence_level, 2))
    r_new = np.max(semi_axes)

    # 更新估计
    pos_update_gain = 1 / (1 + 0.1 * num_rounds)
    x_est_new = x_est + pos_update_gain * (x_new - x_est)
    radius_update_gain = 0.3
    r_est_new = r_est + radius_update_gain * (r_new - r_est)
    return x_est_new, r_est_new


# 主程序
def main():
    # 参数设置
    np.random.seed(42)
    true_pos = np.array([50.0, 50.0])  # 真实位置
    init_est = np.array([40.0, 60.0])  # 初始估计位置
    init_radius = 20.0  # 初始估计半径
    area_size = 100  # 区域大小
    num_rounds = 15  # 总程数
    anchors_per_round = 5  # 每程新增锚点数
    alpha = 1.125e-10  # 噪声方差系数
    g0 = alpha  # 权重参数

    # 历史记录
    hist_pos = np.zeros((num_rounds + 1, 2))
    hist_radius = np.zeros(num_rounds + 1)
    hist_error = np.zeros(num_rounds + 1)

    # 初始化
    x_est = init_est.copy()
    r_est = init_radius
    hist_pos[0] = x_est
    hist_radius[0] = r_est
    hist_error[0] = np.linalg.norm(x_est - true_pos)

    # 创建累积锚点集合
    all_anchors = np.empty((0, 2))
    all_measurements = np.array([])

    # 多程定位主循环
    for round in range(1, num_rounds + 1):
        # 生成新的随机测距点
        new_anchors = true_pos + area_size * (np.random.rand(anchors_per_round, 2) - 0.5)

        # 计算真实距离并添加噪声
        true_d = np.linalg.norm(new_anchors - true_pos, axis=1)
        noise = np.sqrt(alpha * true_d ** 2) * np.random.randn(anchors_per_round)
        new_measurements = true_d + noise

        # 累积锚点和测量值
        all_anchors = np.vstack((all_anchors, new_anchors))
        all_measurements = np.hstack((all_measurements, new_measurements))

        # 更新估计模型
        x_est, r_est = update_uncertain_model(x_est, r_est, all_anchors, all_measurements, g0, round)

        # 记录结果
        hist_pos[round] = x_est
        hist_radius[round] = r_est
        hist_error[round] = np.linalg.norm(x_est - true_pos)

        # 打印进度
        print(f"Round {round}: Position={x_est}, Error={hist_error[round]:.4f}, Radius={r_est:.4f}")

    # 可视化估计半径与真实误差
    plt.figure(figsize=(10, 6))
    rounds = np.arange(num_rounds + 1)

    # 绘制估计半径和真实误差
    plt.plot(rounds, hist_radius, 'b-o', label='估计半径', linewidth=2)
    plt.plot(rounds, hist_error, 'r-s', label='真实误差', linewidth=2)

    # 绘制误差条（估计半径 ± 标准差）
    plt.fill_between(rounds,
                     hist_radius - 0.5 * hist_radius,
                     hist_radius + 0.5 * hist_radius,
                     color='blue', alpha=0.2)

    # 标记收敛点
    convergence_round = np.argmin(np.abs(hist_radius - hist_error))
    plt.plot(convergence_round, hist_radius[convergence_round],
             'g*', markersize=15, label=f'收敛点 (程={convergence_round})')

    # 添加标签和标题
    plt.title('多程定位性能分析', fontsize=14)
    plt.xlabel('程数', fontsize=12)
    plt.ylabel('距离 (m)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.xticks(rounds)

    # 添加统计信息
    stats_text = (f"初始误差: {hist_error[0]:.2f}m\n"
                  f"最终误差: {hist_error[-1]:.2f}m\n"
                  f"收敛程数: {convergence_round}\n"
                  f"半径缩减: {hist_radius[0] / hist_radius[-1]:.1f}倍")
    plt.annotate(stats_text, xy=(0.7, 0.7), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='gray', alpha=0.8))

    plt.tight_layout()
    plt.savefig('radius_error_comparison.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()