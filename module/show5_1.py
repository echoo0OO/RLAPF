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

    def calculate_gdop_for_fixed_target(self,
                                        fixed_target_point: np.ndarray,
                                        current_sensor_network: List[np.ndarray],
                                        sensor_weights_for_fixed_target: np.ndarray
                                        ) -> float:
        """
        情况A的GDOP计算: 计算固定目标点在使用给定传感器网络和预计算权重时的加权GDOP值。

        Args:
            fixed_target_point: 计算GDOP的固定目标点坐标 [x, y]
            current_sensor_network: 当前使用的传感器位置列表 (包括已存在的和假设新增的)
            sensor_weights_for_fixed_target: current_sensor_network中各传感器相对于fixed_target_point的权重

        Returns:
            加权GDOP值
        """
        if not isinstance(fixed_target_point, np.ndarray):
            fixed_target_point = np.array(fixed_target_point)

        if not current_sensor_network or len(current_sensor_network) < 2 or \
                len(current_sensor_network) != len(sensor_weights_for_fixed_target):
            return float('inf')

        G_list = []
        valid_weights_for_G = []  # Weights corresponding to sensors that form G

        for i, sensor_loc in enumerate(current_sensor_network):
            if not isinstance(sensor_loc, np.ndarray):
                sensor_loc = np.array(sensor_loc)

            dx = sensor_loc[0] - fixed_target_point[0]
            dy = sensor_loc[1] - fixed_target_point[1]
            distance_to_target = np.sqrt(dx ** 2 + dy ** 2)

            if distance_to_target < 1e-6:
                continue

            G_list.append([dx / distance_to_target, dy / distance_to_target])
            valid_weights_for_G.append(sensor_weights_for_fixed_target[i])

        if len(G_list) < 2:
            return float('inf')

        G_matrix = np.array(G_list)
        W_matrix = np.diag(valid_weights_for_G)

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


# --- Helper function to calculate sensor placement GDOP heatmap for a given target ---
def calculate_sensor_placement_gdop(
        X_mesh, Y_mesh, grid_pts, existing_sensors,
        target_to_eval_gdop_for, err_coeff, eps_var, gdop_calculator_instance
):
    heatmap = np.zeros_like(X_mesh)
    print(f"正在为目标点 {target_to_eval_gdop_for.round(1)} 计算传感器放置GDOP效益图...")
    for r_idx in range(grid_pts):
        for c_idx in range(grid_pts):
            potential_new_sensor = np.array([X_mesh[r_idx, c_idx], Y_mesh[r_idx, c_idx]])
            current_network = existing_sensors + [potential_new_sensor]
            current_weights = np.zeros(len(current_network))
            for k, s_loc in enumerate(current_network):
                dist = np.linalg.norm(s_loc - target_to_eval_gdop_for)
                variance = err_coeff * (dist ** 2) + eps_var
                current_weights[k] = 1.0 / variance

            heatmap[r_idx, c_idx] = gdop_calculator_instance.calculate_gdop_for_fixed_target(
                target_to_eval_gdop_for, current_network, current_weights
            )
    print(f"目标点 {target_to_eval_gdop_for.round(1)} 的GDOP效益图计算完成。")
    return heatmap


# --- Helper function to normalize GDOP map and create mask ---
def create_gdop_mask_from_heatmap(gdop_heatmap, cap_value):
    finite_gdop = np.nan_to_num(gdop_heatmap, nan=float('inf'), posinf=float('inf'), neginf=0.0)
    capped_gdop = np.clip(finite_gdop, 0, cap_value)
    min_g = np.min(capped_gdop)
    max_g = np.max(capped_gdop)
    print(f"  用于遮罩的GDOP值 (裁剪到{cap_value}后): min={min_g:.3f}, max={max_g:.3f}")
    if (max_g - min_g) < 1e-6:
        norm_gdop = np.ones_like(capped_gdop) if min_g >= (cap_value - 1e-6) else np.zeros_like(capped_gdop)
    else:
        norm_gdop = (capped_gdop - min_g) / (max_g - min_g)
    mask = 1.0 - norm_gdop
    print(f"  生成的遮罩 min: {np.min(mask):.3f}, max: {np.max(mask):.3f}, mean: {np.mean(mask):.3f}")
    return mask


# --- Main Simulation and Plotting ---

# 1. Simulation Area and Parameters
sim_area_width = 1000.0
sim_area_height = 1000.0
plot_grid_points = 200  # Kept at 50 due to doubled GDOP calculation
x_plot_coords = np.linspace(0, sim_area_width, plot_grid_points)
y_plot_coords = np.linspace(0, sim_area_height, plot_grid_points)
X_plot_mesh, Y_plot_mesh = np.meshgrid(x_plot_coords, y_plot_coords)

# Reward Targets and their Sigmas
target1_coords_reward_center = np.array([400.0, 600.0])
sigma_target1 = sim_area_width / 10.0
target2_coords_reward_center = np.array([600.0, 400.0])
sigma_target2 = sim_area_width / 15.0
K_total_reward_val_single_target = 50000.0

# 2. Define Existing Fixed Sensors
area_center_x = sim_area_width / 2
area_center_y = sim_area_height / 2
num_existing_sensors = 4
sensor_spread_from_area_center = sim_area_width / 4.0
np.random.seed(45)
existing_fixed_sensors = []
print(
    f"正在生成 {num_existing_sensors} 个已存在的固定传感器，位于区域中心 ({area_center_x:.1f}, {area_center_y:.1f}) 的第三象限...")
for i in range(num_existing_sensors):
    offset_x = np.random.uniform(-sensor_spread_from_area_center, -10.0)
    offset_y = np.random.uniform(-sensor_spread_from_area_center, -10.0)
    sensor_x = area_center_x + offset_x;
    sensor_y = area_center_y + offset_y
    sensor_x = np.clip(sensor_x, 0, sim_area_width);
    sensor_y = np.clip(sensor_y, 0, sim_area_height)
    existing_fixed_sensors.append(np.array([sensor_x, sensor_y]))
    print(f"  已存在的传感器 {i + 1}: ({sensor_x:.2f}, {sensor_y:.2f})")

# 3. Sensor Error Parameters for Weight Calculation
small_error_coeff = 0.01 ** 2
epsilon_variance = (0.1) ** 2
gdop_calc_instance = GDOPCalculator()

# 4. Calculate SEPARATE "Sensor Placement GDOP Heatmaps" for T1 and T2
gdop_heatmap_T1 = calculate_sensor_placement_gdop(
    X_plot_mesh, Y_plot_mesh, plot_grid_points, existing_fixed_sensors,
    target1_coords_reward_center, small_error_coeff, epsilon_variance, gdop_calc_instance
)
gdop_heatmap_T2 = calculate_sensor_placement_gdop(
    X_plot_mesh, Y_plot_mesh, plot_grid_points, existing_fixed_sensors,
    target2_coords_reward_center, small_error_coeff, epsilon_variance, gdop_calc_instance
)

# --- Visualize 2D Sensor Placement GDOP Heatmaps (one for each target) ---
gdop_display_clip_val = 10.0
existing_sensors_array_vis = np.array(existing_fixed_sensors)

# For T1
plt.figure(figsize=(9, 7.5))
im_gdop_T1 = plt.imshow(np.clip(gdop_heatmap_T1, 0, gdop_display_clip_val),
                        extent=[0, sim_area_width, 0, sim_area_height],
                        origin='lower', cmap='viridis_r', aspect='auto')
plt.scatter(existing_sensors_array_vis[:, 0], existing_sensors_array_vis[:, 1],
            c='red', s=100, marker='^', label='已存在的固定传感器', edgecolors='black')
plt.scatter(target1_coords_reward_center[0], target1_coords_reward_center[1],
            c='magenta', s=150, marker='X', label=f'GDOP固定目标点 T1', edgecolors='black')
plt.colorbar(im_gdop_T1, label=f'目标T1的GDOP值 (Clipped at {gdop_display_clip_val})')
plt.xlabel('新传感器的潜在X坐标 (m)');
plt.ylabel('新传感器的潜在Y坐标 (m)')
plt.title(f'传感器放置GDOP效益图 (针对目标T1)')
plt.legend(fontsize='small');
plt.grid(True, linestyle='--', alpha=0.5);
plt.show()

# For T2
plt.figure(figsize=(9, 7.5))
im_gdop_T2 = plt.imshow(np.clip(gdop_heatmap_T2, 0, gdop_display_clip_val),
                        extent=[0, sim_area_width, 0, sim_area_height],
                        origin='lower', cmap='viridis_r', aspect='auto')
plt.scatter(existing_sensors_array_vis[:, 0], existing_sensors_array_vis[:, 1],
            c='red', s=100, marker='^', label='已存在的固定传感器', edgecolors='black')
plt.scatter(target2_coords_reward_center[0], target2_coords_reward_center[1],
            c='cyan', s=150, marker='P', label=f'GDOP固定目标点 T2', edgecolors='black')  # Different color for T2
plt.colorbar(im_gdop_T2, label=f'目标T2的GDOP值 (Clipped at {gdop_display_clip_val})')
plt.xlabel('新传感器的潜在X坐标 (m)');
plt.ylabel('新传感器的潜在Y坐标 (m)')
plt.title(f'传感器放置GDOP效益图 (针对目标T2)')
plt.legend(fontsize='small');
plt.grid(True, linestyle='--', alpha=0.5);
plt.show()

# 5. Normalize EACH GDOP heatmap to create respective Masks
gdop_mask_cap_val = 10.0
print("\n为目标T1创建遮罩:")
gdop_mask_T1 = create_gdop_mask_from_heatmap(gdop_heatmap_T1, gdop_mask_cap_val)
print("\n为目标T2创建遮罩:")
gdop_mask_T2 = create_gdop_mask_from_heatmap(gdop_heatmap_T2, gdop_mask_cap_val)

# --- Visualize 2D GDOP Masks ---
# For T1 Mask
plt.figure(figsize=(9, 7.5))
im_mask_T1 = plt.imshow(gdop_mask_T1, extent=[0, sim_area_width, 0, sim_area_height],
                        origin='lower', cmap='viridis', aspect='auto', vmin=0, vmax=1)
plt.scatter(existing_sensors_array_vis[:, 0], existing_sensors_array_vis[:, 1], c='red', s=100, marker='^',
            label='已存在的固定传感器', edgecolors='black')
plt.scatter(target1_coords_reward_center[0], target1_coords_reward_center[1], c='black', s=150, marker='X',
            label=f'GDOP固定目标点 T1', edgecolors='white')
plt.colorbar(im_mask_T1, label='传感器放置效益遮罩值 (对T1)');
plt.xlabel('新传感器的潜在X坐标 (m)');
plt.ylabel('新传感器的潜在Y坐标 (m)')
plt.title(f'2D传感器放置效益遮罩图 (针对目标T1)');
plt.legend(fontsize='small');
plt.grid(True, linestyle='--', alpha=0.5);
plt.show()

# For T2 Mask
plt.figure(figsize=(9, 7.5))
im_mask_T2 = plt.imshow(gdop_mask_T2, extent=[0, sim_area_width, 0, sim_area_height],
                        origin='lower', cmap='viridis', aspect='auto', vmin=0, vmax=1)
plt.scatter(existing_sensors_array_vis[:, 0], existing_sensors_array_vis[:, 1], c='red', s=100, marker='^',
            label='已存在的固定传感器', edgecolors='black')
plt.scatter(target2_coords_reward_center[0], target2_coords_reward_center[1], c='grey', s=150, marker='P',
            label=f'GDOP固定目标点 T2', edgecolors='white')
plt.colorbar(im_mask_T2, label='传感器放置效益遮罩值 (对T2)');
plt.xlabel('新传感器的潜在X坐标 (m)');
plt.ylabel('新传感器的潜在Y坐标 (m)')
plt.title(f'2D传感器放置效益遮罩图 (针对目标T2)');
plt.legend(fontsize='small');
plt.grid(True, linestyle='--', alpha=0.5);
plt.show()

# 6. Calculate Individual Rewards, Apply Respective Masks, then Combine with MAX
reward_T1_original = reward_function_2d(X_plot_mesh, Y_plot_mesh, K_total_reward_val_single_target, sigma_target1,
                                        target1_coords_reward_center[0], target1_coords_reward_center[1])
reward_T2_original = reward_function_2d(X_plot_mesh, Y_plot_mesh, K_total_reward_val_single_target, sigma_target2,
                                        target2_coords_reward_center[0], target2_coords_reward_center[1])

Z_original_combined_reward_sum = reward_T1_original + reward_T2_original  # For visualizing original potential

# Apply respective masks
final_reward_T1_masked = reward_T1_original * gdop_mask_T1
final_reward_T2_masked = reward_T2_original * gdop_mask_T2

# Combine using MAX
Z_final_combined_reward_max_of_masked = np.maximum(final_reward_T1_masked, final_reward_T2_masked)
max_overall_masked_reward_val = np.max(Z_final_combined_reward_max_of_masked)

# --- Visualize 3D Original Combined Reward (Sum) ---
fig_orig_sum_reward = plt.figure(figsize=(10, 8))
ax_orig_sum_reward = fig_orig_sum_reward.add_subplot(111, projection='3d')
surf_orig_sum_reward = ax_orig_sum_reward.plot_surface(X_plot_mesh, Y_plot_mesh, Z_original_combined_reward_sum,
                                                       cmap='coolwarm', edgecolor='none', rstride=1, cstride=1)
ax_orig_sum_reward.set_title("原始整体奖励分布 (双目标点之和)", fontsize=12)
ax_orig_sum_reward.set_xlabel('X 坐标 (m)', fontsize=9);
ax_orig_sum_reward.set_ylabel('Y 坐标 (m)', fontsize=9);
ax_orig_sum_reward.set_zlabel('奖励值', fontsize=9)
ax_orig_sum_reward.tick_params(axis='both', which='major', labelsize=8)
fig_orig_sum_reward.colorbar(surf_orig_sum_reward, shrink=0.6, aspect=15, ax=ax_orig_sum_reward,
                             label='原始奖励值 (T1+T2)')
plt.tight_layout();
plt.show()

# --- Visualize 3D Final Combined Reward (Max of individually masked rewards) ---
fig_final_masked_reward = plt.figure(figsize=(10, 8))
ax_final_masked_reward = fig_final_masked_reward.add_subplot(111, projection='3d')
surf_final_masked_reward = ax_final_masked_reward.plot_surface(X_plot_mesh, Y_plot_mesh,
                                                               Z_final_combined_reward_max_of_masked,
                                                               cmap='coolwarm', edgecolor='none', rstride=1, cstride=1)
ax_final_masked_reward.set_title(
    f"最终组合奖励 (Max(T1_masked, T2_masked), 最大值: {max_overall_masked_reward_val:.2f})", fontsize=10)
ax_final_masked_reward.set_xlabel('X 坐标 (m)', fontsize=9);
ax_final_masked_reward.set_ylabel('Y 坐标 (m)', fontsize=9);
ax_final_masked_reward.set_zlabel('遮罩后奖励值', fontsize=9)
ax_final_masked_reward.tick_params(axis='both', which='major', labelsize=8)
current_max_z_final_masked = Z_final_combined_reward_max_of_masked.max()
ax_final_masked_reward.set_zlim(0, current_max_z_final_masked * 1.1 if current_max_z_final_masked > 0 else 1)
fig_final_masked_reward.colorbar(surf_final_masked_reward, shrink=0.6, aspect=15, ax=ax_final_masked_reward,
                                 label='最终组合奖励值')
plt.tight_layout();
plt.show()

# 7. Local Reward Comparison Plots (3D Surface Plots)
local_window_radius_cells = int(plot_grid_points / 5)


def plot_local_3d_comparison(target_name, target_coords_val, original_reward_component,
                             individually_masked_reward_component, X_mesh, Y_mesh, window_radius_cells_val,
                             z_limit_orig=None, z_limit_masked_individual=None):
    center_x_idx = np.argmin(np.abs(x_plot_coords - target_coords_val[0]))
    center_y_idx = np.argmin(np.abs(y_plot_coords - target_coords_val[1]))
    y_start = max(0, center_y_idx - window_radius_cells_val);
    y_end = min(plot_grid_points, center_y_idx + window_radius_cells_val + 1)
    x_start = max(0, center_x_idx - window_radius_cells_val);
    x_end = min(plot_grid_points, center_x_idx + window_radius_cells_val + 1)

    local_X = X_mesh[y_start:y_end, x_start:x_end];
    local_Y = Y_mesh[y_start:y_end, x_start:x_end]
    local_original_Z = original_reward_component[y_start:y_end, x_start:x_end]
    local_individually_masked_Z = individually_masked_reward_component[y_start:y_end, x_start:x_end]

    fig_local = plt.figure(figsize=(14, 6.5))
    fig_local.suptitle(
        f"目标点 {target_name} ({target_coords_val[0]},{target_coords_val[1]}) 附近局部3D奖励对比 (针对其自身的GDOP效益遮罩)",
        fontsize=12)

    # Plot Original Local Reward (3D)
    ax_orig_local = fig_local.add_subplot(1, 2, 1, projection='3d')
    surf_orig_local = ax_orig_local.plot_surface(local_X, local_Y, local_original_Z, cmap='coolwarm', edgecolor='none',
                                                 rstride=1, cstride=1)
    ax_orig_local.set_title(f"原始奖励 ({target_name}贡献)")
    ax_orig_local.set_xlabel("X (m)");
    ax_orig_local.set_ylabel("Y (m)");
    ax_orig_local.set_zlabel("奖励值")
    max_val_orig = np.max(local_original_Z)
    ax_orig_local.set_zlim(0, max_val_orig * 1.1 if max_val_orig > 0 else 1)
    if z_limit_orig: ax_orig_local.set_zlim(0, z_limit_orig)  # Override if provided
    fig_local.colorbar(surf_orig_local, ax=ax_orig_local, shrink=0.7, aspect=10, label="原始奖励值")

    # Plot Individually Masked Local Reward (3D)
    ax_masked_local = fig_local.add_subplot(1, 2, 2, projection='3d')
    surf_masked_local = ax_masked_local.plot_surface(local_X, local_Y, local_individually_masked_Z, cmap='coolwarm',
                                                     edgecolor='none', rstride=1, cstride=1)
    ax_masked_local.set_title(f"单独遮罩后的奖励 ({target_name}_masked)")
    ax_masked_local.set_xlabel("X (m)");
    ax_masked_local.set_ylabel("Y (m)");
    ax_masked_local.set_zlabel("遮罩后奖励值")
    max_val_masked_ind = np.max(local_individually_masked_Z)
    ax_masked_local.set_zlim(0, max_val_masked_ind * 1.1 if max_val_masked_ind > 0 else 1)
    if z_limit_masked_individual: ax_masked_local.set_zlim(0, z_limit_masked_individual)  # Override if provided
    fig_local.colorbar(surf_masked_local, ax=ax_masked_local, shrink=0.7, aspect=10, label="单独遮罩后奖励值")

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    plt.show()


# Determine common Z limits for local plots for better visual comparison
max_original_reward_for_T1_local = np.max(reward_T1_original)
max_original_reward_for_T2_local = np.max(reward_T2_original)

max_individually_masked_reward_T1_local = np.max(final_reward_T1_masked)
max_individually_masked_reward_T2_local = np.max(final_reward_T2_masked)

plot_local_3d_comparison("T1 (奖励中心)", target1_coords_reward_center, reward_T1_original, final_reward_T1_masked,
                         X_plot_mesh, Y_plot_mesh, local_window_radius_cells,
                         z_limit_orig=max_original_reward_for_T1_local * 1.1,
                         z_limit_masked_individual=max_individually_masked_reward_T1_local * 1.1)
plot_local_3d_comparison("T2 (奖励中心)", target2_coords_reward_center, reward_T2_original, final_reward_T2_masked,
                         X_plot_mesh, Y_plot_mesh, local_window_radius_cells,
                         z_limit_orig=max_original_reward_for_T2_local * 1.1,
                         z_limit_masked_individual=max_individually_masked_reward_T2_local * 1.1)

print("所有可视化已完成。")
