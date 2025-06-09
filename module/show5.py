import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Dict, Optional
from scipy.spatial.distance import cdist
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题


# --- GDOPCalculator Class Definition ---
class GDOPCalculator:
    """GDOP（几何精度因子）计算器"""

    def __init__(self, area_size: Tuple[float, float] = (1000.0, 1000.0),
                 grid_resolution: float = 100.0):
        self.area_size = area_size
        self.grid_resolution = grid_resolution
        self.x_grid = np.arange(0, area_size[0] + grid_resolution, grid_resolution)
        self.y_grid = np.arange(0, area_size[1] + grid_resolution, grid_resolution)
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)
        self.gdop_heatmap = np.zeros_like(self.X)  # This heatmap is not directly used in "Situation A"

    def calculate_weighted_gdop_at_target(self,
                                          target_point: np.ndarray,
                                          sensor_locations: List[np.ndarray],
                                          sensor_weights: np.ndarray) -> float:
        """
        计算指定目标点使用给定传感器网络和权重的加权GDOP值。
        (Calculates weighted GDOP for a specific target_point using a given set of sensor_locations and their weights)

        Args:
            target_point: 计算GDOP的目标点坐标 [x, y] (The specific target point for which GDOP is calculated)
            sensor_locations: 传感器位置列表 (List of sensor locations)
            sensor_weights: 每个传感器的权重数组 (Array of weights for each sensor)

        Returns:
            加权GDOP值 (Weighted GDOP value)
        """
        if not isinstance(target_point, np.ndarray):
            target_point = np.array(target_point)

        valid_sensors = []
        valid_sensor_weights = []
        if sensor_locations and sensor_weights is not None and len(sensor_locations) == len(sensor_weights):
            for i, sensor_loc in enumerate(sensor_locations):
                if isinstance(sensor_loc, np.ndarray):
                    valid_sensors.append(sensor_loc)
                    valid_sensor_weights.append(sensor_weights[i])
                else:  # Fallback if not numpy array
                    valid_sensors.append(np.array(sensor_loc))
                    valid_sensor_weights.append(sensor_weights[i])

        if not valid_sensors or len(valid_sensors) < 2:
            return float('inf')

        G_list_for_target = []
        current_weights_for_G = []

        for i, sensor_loc in enumerate(valid_sensors):
            dx = sensor_loc[0] - target_point[0]
            dy = sensor_loc[1] - target_point[1]
            distance_to_target = np.sqrt(dx ** 2 + dy ** 2)

            if distance_to_target < 1e-6:
                continue

            G_list_for_target.append([dx / distance_to_target, dy / distance_to_target])
            current_weights_for_G.append(valid_sensor_weights[i])

        if len(G_list_for_target) < 2:
            return float('inf')

        G_matrix = np.array(G_list_for_target)
        W_matrix = np.diag(current_weights_for_G)

        try:
            GtWG = G_matrix.T @ W_matrix @ G_matrix
            if GtWG.shape != (2, 2) or np.linalg.det(GtWG) < 1e-10:
                return float('inf')

            GtWG_inv = np.linalg.inv(GtWG)
            wdop = np.sqrt(np.trace(GtWG_inv))
            return wdop
        except np.linalg.LinAlgError:
            return float('inf')


# --- 2D Reward Function Definition (Gaussian) ---
def reward_function_2d(x_coords_mesh, y_coords_mesh, K_val, sigma_val, xc_val, yc_val):
    coeff = K_val / (2 * np.pi * sigma_val ** 2)
    exponent = -((x_coords_mesh - xc_val) ** 2 + (y_coords_mesh - yc_val) ** 2) / (2 * sigma_val ** 2)
    return coeff * np.exp(exponent)


# --- Main Simulation and Plotting ---

# 1. Simulation Area and Parameters
sim_area_width = 1000.0
sim_area_height = 1000.0
plot_grid_points = 50  # Reduced for faster computation in "Situation A"
x_plot_coords = np.linspace(0, sim_area_width, plot_grid_points)
y_plot_coords = np.linspace(0, sim_area_height, plot_grid_points)
X_plot_mesh, Y_plot_mesh = np.meshgrid(x_plot_coords, y_plot_coords)

# Reward function parameters (center of the Gaussian reward distribution)
reward_gaussian_center_x = sim_area_width / 2
reward_gaussian_center_y = sim_area_height / 2
K_total_reward_val = 100000.0

sigmas_for_reward_func = {
    "large_sigma": sim_area_width / 7.0,
    "small_sigma": sim_area_width / 30.0
}

# 2. Define the Fixed Target Point for GDOP Calculation and Existing Fixed Sensors
# This is the single, fixed target point for which we want to optimize GDOP by adding one more sensor.
fixed_target_for_gdop_calc = np.array([sim_area_width / 2, sim_area_height / 2 + 150])  # e.g., (500, 650)
print(f"用于GDOP计算的固定目标点: {fixed_target_for_gdop_calc}")

# MODIFIED: Generate existing_fixed_sensors in the third quadrant relative to fixed_target_for_gdop_calc
num_existing_sensors = 4
sensor_spread_from_target = sim_area_width / 5.0  # How far sensors can be from target (e.g., 200m)
np.random.seed(44)  # Use a new seed for different sensor positions
existing_fixed_sensors = []
print(
    f"正在生成 {num_existing_sensors} 个已存在的固定传感器，位于目标点 ({fixed_target_for_gdop_calc[0]:.1f}, {fixed_target_for_gdop_calc[1]:.1f}) 的第三象限...")
for i in range(num_existing_sensors):
    # Generate random negative offsets
    offset_x = np.random.uniform(-sensor_spread_from_target, -1.0)  # Ensure it's strictly negative offset
    offset_y = np.random.uniform(-sensor_spread_from_target, -1.0)  # Ensure it's strictly negative offset

    sensor_x = fixed_target_for_gdop_calc[0] + offset_x
    sensor_y = fixed_target_for_gdop_calc[1] + offset_y

    # Clip to simulation boundaries
    sensor_x = np.clip(sensor_x, 0, sim_area_width)
    sensor_y = np.clip(sensor_y, 0, sim_area_height)

    existing_fixed_sensors.append(np.array([sensor_x, sensor_y]))
    print(f"  已存在的传感器 {i + 1}: ({sensor_x:.2f}, {sensor_y:.2f})")

print(f"已定义 {num_existing_sensors} 个已存在的固定传感器: {existing_fixed_sensors}")

# 3. Sensor Weight Calculation Parameters
small_error_coeff = 0.01 ** 2  # 测距误差的标准差是距离的1% (Std dev of range error is 1% of distance)
epsilon_variance = (0.1) ** 2  # 最小方差，避免权重无穷大 (Minimum variance to avoid infinite weight)

# Instantiate GDOP Calculator
gdop_calc_instance = GDOPCalculator()

# 4. Calculate "Sensor Placement GDOP Heatmap" (Situation A)
# Heatmap value at (gx,gy) = GDOP for fixed_target_for_gdop_calc if a new sensor is placed at (gx,gy)
print("正在计算传感器放置GDOP效益图 (这可能需要一些时间)...")
sensor_placement_gdop_heatmap = np.zeros_like(X_plot_mesh)

for i in range(plot_grid_points):
    for j in range(plot_grid_points):
        potential_new_sensor_loc = np.array([X_plot_mesh[i, j], Y_plot_mesh[i, j]])

        # Combine existing sensors with the potential new one
        current_sensor_network = existing_fixed_sensors + [potential_new_sensor_loc]
        num_total_sensors_for_calc = len(current_sensor_network)

        # Calculate weights for ALL sensors in this temporary network, relative to the fixed target
        current_sensor_weights = np.zeros(num_total_sensors_for_calc)
        for k, sensor_loc in enumerate(current_sensor_network):
            dist_to_fixed_target = np.linalg.norm(sensor_loc - fixed_target_for_gdop_calc)
            variance = small_error_coeff * (dist_to_fixed_target ** 2) + epsilon_variance
            current_sensor_weights[k] = 1.0 / variance

        # Calculate GDOP for the fixed target using this combined network
        gdop_val = gdop_calc_instance.calculate_weighted_gdop_at_target(
            fixed_target_for_gdop_calc,
            current_sensor_network,
            current_sensor_weights
        )
        sensor_placement_gdop_heatmap[i, j] = gdop_val
print("传感器放置GDOP效益图计算完成。")

# --- Visualize 2D Sensor Placement GDOP Heatmap ---
plt.figure(figsize=(9, 7.5))
gdop_display_clip = 10.0
im_gdop_placement = plt.imshow(np.clip(sensor_placement_gdop_heatmap, 0, gdop_display_clip),
                               extent=[0, sim_area_width, 0, sim_area_height],
                               origin='lower', cmap='viridis_r', aspect='auto')

existing_sensors_array = np.array(existing_fixed_sensors)
plt.scatter(existing_sensors_array[:, 0], existing_sensors_array[:, 1],
            c='red', s=100, marker='^', label='已存在的固定传感器 (第三象限)', edgecolors='black')
plt.scatter(fixed_target_for_gdop_calc[0], fixed_target_for_gdop_calc[1],
            c='magenta', s=150, marker='X', label='固定的GDOP目标点', edgecolors='black')

plt.colorbar(im_gdop_placement, label=f'固定目标的GDOP值 (Clipped at {gdop_display_clip})')
plt.xlabel('新传感器的潜在X坐标 (m)')
plt.ylabel('新传感器的潜在Y坐标 (m)')
plt.title(f'传感器放置GDOP效益图 (目标点: {fixed_target_for_gdop_calc.round(1)}, 固定传感器在目标第三象限)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# 5. Normalize this new GDOP map to Create a Mask (0-1, where 1 means placing sensor there gives good GDOP for target)
gdop_cap_for_mask = 10.0
gdop_finite_for_mask = np.nan_to_num(sensor_placement_gdop_heatmap, nan=float('inf'), posinf=float('inf'), neginf=0.0)
gdop_capped_for_mask = np.clip(gdop_finite_for_mask, 0, gdop_cap_for_mask)

min_g_mask = np.min(gdop_capped_for_mask)
max_g_mask = np.max(gdop_capped_for_mask)
print(f"传感器放置GDOP图的值 (裁剪到{gdop_cap_for_mask}后) 用于遮罩: min={min_g_mask:.3f}, max={max_g_mask:.3f}")

if (max_g_mask - min_g_mask) < 1e-6:
    normalized_gdop_for_mask = np.ones_like(gdop_capped_for_mask) if min_g_mask >= (
                gdop_cap_for_mask - 1e-6) else np.zeros_like(gdop_capped_for_mask)
else:
    normalized_gdop_for_mask = (gdop_capped_for_mask - min_g_mask) / (max_g_mask - min_g_mask)

final_placement_gdop_mask = 1.0 - normalized_gdop_for_mask
print(
    f"最终的传感器放置效益遮罩 min: {np.min(final_placement_gdop_mask):.3f}, max: {np.max(final_placement_gdop_mask):.3f}, mean: {np.mean(final_placement_gdop_mask):.3f}")

# --- Visualize 2D GDOP Placement Mask ---
plt.figure(figsize=(9, 7.5))
im_mask_vis = plt.imshow(final_placement_gdop_mask,
                         extent=[0, sim_area_width, 0, sim_area_height],
                         origin='lower', cmap='viridis', aspect='auto', vmin=0, vmax=1)

plt.scatter(existing_sensors_array[:, 0], existing_sensors_array[:, 1],
            c='red', s=100, marker='^', label='已存在的固定传感器 (第三象限)', edgecolors='black')
plt.scatter(fixed_target_for_gdop_calc[0], fixed_target_for_gdop_calc[1],
            c='black', s=150, marker='X', label='固定的GDOP目标点', edgecolors='white')

plt.colorbar(im_mask_vis, label='传感器放置效益遮罩值 (1 = 对目标GDOP最佳)')
plt.xlabel('新传感器的潜在X坐标 (m)')
plt.ylabel('新传感器的潜在Y坐标 (m)')
plt.title(f'2D传感器放置效益遮罩图 (目标点: {fixed_target_for_gdop_calc.round(1)}, 固定传感器在目标第三象限)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# --- Loop through different sigmas for the reward function and plot 3D MASKED REWARD ---
for sigma_label, current_reward_sigma_val in sigmas_for_reward_func.items():
    Z_original_reward_dist = reward_function_2d(X_plot_mesh, Y_plot_mesh,
                                                K_total_reward_val, current_reward_sigma_val,
                                                reward_gaussian_center_x, reward_gaussian_center_y)

    Z_final_masked_reward = Z_original_reward_dist * final_placement_gdop_mask
    max_masked_reward_val = np.max(Z_final_masked_reward)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X_plot_mesh, Y_plot_mesh, Z_final_masked_reward,
                           cmap='viridis', edgecolor='none', rstride=1, cstride=1)

    plot_title_str = (
        f"传感器放置效益遮罩后的奖励 (Sigma: {sigma_label}, $\sigma_R = {current_reward_sigma_val:.1f}$)\n"
        f"最大遮罩后奖励: {max_masked_reward_val:.4f}\n"
        f"(高斯奖励中心: ({reward_gaussian_center_x},{reward_gaussian_center_y}), GDOP目标点: {fixed_target_for_gdop_calc.round(1)}, 固定传感器在目标第三象限)")
    ax.set_title(plot_title_str, fontsize=9)
    ax.set_xlabel('X 坐标 (m)', fontsize=9)
    ax.set_ylabel('Y 坐标 (m)', fontsize=9)
    ax.set_zlabel('遮罩后奖励值', fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=8)

    current_max_z_val = Z_final_masked_reward.max()
    ax.set_zlim(0, current_max_z_val * 1.1 if current_max_z_val > 0 else 1)

    fig.colorbar(surf, shrink=0.6, aspect=15, ax=ax, label='遮罩后奖励值')

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.show()

print("所有可视化已完成。")
