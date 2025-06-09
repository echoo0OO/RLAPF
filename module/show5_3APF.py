import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Dict, Optional
from scipy.spatial.distance import cdist
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['STZhongsong']  # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


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
                                        fixed_target_point_3d: np.ndarray,
                                        current_sensor_network_3d: List[np.ndarray],
                                        sensor_weights_for_fixed_target: np.ndarray
                                        ) -> float:
        """
        情况A的GDOP计算 (3D Version): 计算固定目标点在使用给定传感器网络和预计算权重时的加权GDOP值。
        """
        if not isinstance(fixed_target_point_3d, np.ndarray) or fixed_target_point_3d.shape != (3,):
            if isinstance(fixed_target_point_3d, (list, tuple)) and len(fixed_target_point_3d) == 3:
                fixed_target_point_3d = np.array(fixed_target_point_3d)
            else:
                if isinstance(fixed_target_point_3d, np.ndarray) and fixed_target_point_3d.shape == (2,):
                    fixed_target_point_3d = np.append(fixed_target_point_3d, 0.0)
                else:
                    fixed_target_point_3d = np.array([0, 0, 0])

        if not current_sensor_network_3d or len(current_sensor_network_3d) < 3 or \
                len(current_sensor_network_3d) != len(sensor_weights_for_fixed_target):
            return float('inf')

        G_list = []
        valid_weights_for_G = []

        for i, sensor_loc_3d in enumerate(current_sensor_network_3d):
            if not isinstance(sensor_loc_3d, np.ndarray) or sensor_loc_3d.shape != (3,):
                if isinstance(sensor_loc_3d, (list, tuple)) and len(sensor_loc_3d) == 3:
                    sensor_loc_3d = np.array(sensor_loc_3d)
                else:
                    if isinstance(sensor_loc_3d, np.ndarray) and sensor_loc_3d.shape == (2,):
                        sensor_loc_3d = np.append(sensor_loc_3d, 0.0)
                    else:
                        sensor_loc_3d = np.array([0, 0, 0])

            diff_vector = sensor_loc_3d - fixed_target_point_3d
            distance_to_target = np.linalg.norm(diff_vector)

            if distance_to_target < 1e-6:
                continue

            unit_vector = diff_vector / distance_to_target
            G_list.append(unit_vector)
            valid_weights_for_G.append(sensor_weights_for_fixed_target[i])

        if len(G_list) < 3:
            return float('inf')

        G_matrix = np.array(G_list)
        W_matrix = np.diag(valid_weights_for_G)

        try:
            GtWG = G_matrix.T @ W_matrix @ G_matrix
            if GtWG.shape != (3, 3) or np.linalg.det(GtWG) < 1e-10:
                return float('inf')

            GtWG_inv = np.linalg.inv(GtWG)
            pdop = np.sqrt(np.trace(GtWG_inv))
            return pdop
        except np.linalg.LinAlgError:
            return float('inf')


# --- APF-like Reward Function ---
def reward_function_apf(x_coords_mesh, y_coords_mesh, R_max, r_threshold, xc, yc):
    """
    实现了图片中展示的APF形式的混合奖励函数。
    为了保证平滑性 (C1连续)，内部已设置 c1 = R_max / 3。
    """
    # Enforce a minimum r_threshold of 10
    effective_r_threshold = max(r_threshold, 10.0)

    c1 = R_max / 3.0
    d = np.sqrt((x_coords_mesh - xc) ** 2 + (y_coords_mesh - yc) ** 2)
    reward = np.zeros_like(d)

    # Use effective_r_threshold for calculations
    parabolic_mask = (d <= effective_r_threshold)
    reward[parabolic_mask] = R_max - (c1 * d[parabolic_mask] ** 2) / (effective_r_threshold ** 2)

    hyperbolic_mask = (d > effective_r_threshold)
    d_hyperbolic = d[hyperbolic_mask]
    d_hyperbolic[d_hyperbolic == 0] = 1e-6
    reward[hyperbolic_mask] = (effective_r_threshold * (R_max - c1)) / d_hyperbolic

    return reward


# --- Helper function to calculate sensor placement GDOP heatmap ---
def calculate_sensor_placement_gdop_3d(
        X_mesh, Y_mesh, grid_pts, existing_sensors_3d,
        target_to_eval_gdop_for_3d, err_coeff, eps_var, gdop_calculator_instance,
        z_coord_new_sensor
):
    heatmap = np.zeros_like(X_mesh)
    print(
        f"正在为目标点 {target_to_eval_gdop_for_3d[:2].round(1)} (z={target_to_eval_gdop_for_3d[2]}) 计算3D传感器放置GDOP效益图...")
    for r_idx in range(grid_pts):
        for c_idx in range(grid_pts):
            potential_new_sensor_3d = np.array([X_mesh[r_idx, c_idx], Y_mesh[r_idx, c_idx], z_coord_new_sensor])
            current_network_3d = existing_sensors_3d + [potential_new_sensor_3d]
            current_weights_3d = np.zeros(len(current_network_3d))
            for k, s_loc_3d in enumerate(current_network_3d):
                dist_3d = np.linalg.norm(s_loc_3d - target_to_eval_gdop_for_3d)
                variance = err_coeff * (dist_3d ** 2) + eps_var
                current_weights_3d[k] = 1.0 / variance

            heatmap[r_idx, c_idx] = gdop_calculator_instance.calculate_gdop_for_fixed_target(
                target_to_eval_gdop_for_3d, current_network_3d, current_weights_3d
            )
    print(
        f"目标点 {target_to_eval_gdop_for_3d[:2].round(1)} (z={target_to_eval_gdop_for_3d[2]}) 的3D GDOP效益图计算完成。")
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


# --- Helper function for overall 3D comparison plot ---
def plot_overall_3d_comparison(X_mesh, Y_mesh, original_Z, masked_Z, title_suffix=""):
    fig = plt.figure(figsize=(14, 6.5))
    fig.suptitle(f'整体奖励3D对比图 {title_suffix}', fontsize=14)

    # Z limit for both plots based on the original reward's max value for fair comparison
    z_max = np.max(original_Z)
    z_limit = z_max * 1.1 if z_max > 0 else 1

    # Plot Original Reward
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X_mesh, Y_mesh, original_Z, cmap='coolwarm', edgecolor='none', rstride=1, cstride=1)
    ax1.set_title("遮罩前 (原始奖励之和)")
    ax1.set_xlabel("X (m)");
    ax1.set_ylabel("Y (m)");
    ax1.set_zlabel("奖励值")
    ax1.set_zlim(0, z_limit)
    fig.colorbar(surf1, ax=ax1, shrink=0.7, aspect=10, label="原始奖励值")

    # Plot Masked Reward
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(X_mesh, Y_mesh, masked_Z, cmap='viridis', edgecolor='none', rstride=1, cstride=1)
    ax2.set_title("遮罩后 (Max组合)")
    ax2.set_xlabel("X (m)");
    ax2.set_ylabel("Y (m)");
    ax2.set_zlabel("遮罩后奖励值")
    ax2.set_zlim(0, z_limit)  # Use same z-limit
    fig.colorbar(surf2, ax=ax2, shrink=0.7, aspect=10, label="遮罩后奖励值")

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    plt.show()


# --- Helper function for local 3D comparison plot ---
def plot_local_3d_comparison(target_name, target_xy_coords_val, original_reward_component,
                             individually_masked_reward_component, X_mesh, Y_mesh, window_radius_cells_val,
                             z_limit_orig=None, z_limit_masked_individual=None):
    center_x_idx = np.argmin(np.abs(x_plot_coords - target_xy_coords_val[0]))
    center_y_idx = np.argmin(np.abs(y_plot_coords - target_xy_coords_val[1]))
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
        f"目标点 {target_name} ({target_xy_coords_val[0]},{target_xy_coords_val[1]}) 附近局部3D奖励对比 (APF奖励)",
        fontsize=12)

    ax_orig_local = fig_local.add_subplot(1, 2, 1, projection='3d')
    surf_orig_local = ax_orig_local.plot_surface(local_X, local_Y, local_original_Z, cmap='coolwarm', edgecolor='none',
                                                 rstride=1, cstride=1)
    ax_orig_local.set_title(f"原始奖励 ({target_name}贡献)")
    ax_orig_local.set_xlabel("X (m)");
    ax_orig_local.set_ylabel("Y (m)");
    ax_orig_local.set_zlabel("奖励值")
    max_val_orig = np.max(local_original_Z)
    z_lim_o = max_val_orig * 1.1 if max_val_orig > 0 else 1
    if z_limit_orig: z_lim_o = z_limit_orig
    ax_orig_local.set_zlim(0, z_lim_o)
    fig_local.colorbar(surf_orig_local, ax=ax_orig_local, shrink=0.7, aspect=10, label="原始奖励值")

    ax_masked_local = fig_local.add_subplot(1, 2, 2, projection='3d')
    surf_masked_local = ax_masked_local.plot_surface(local_X, local_Y, local_individually_masked_Z, cmap='viridis',
                                                     edgecolor='none', rstride=1, cstride=1)
    ax_masked_local.set_title(f"单独遮罩后的奖励 ({target_name}_masked)")
    ax_masked_local.set_xlabel("X (m)");
    ax_masked_local.set_ylabel("Y (m)");
    ax_masked_local.set_zlabel("遮罩后奖励值")
    max_val_masked_ind = np.max(local_individually_masked_Z)
    z_lim_m = max_val_masked_ind * 1.1 if max_val_masked_ind > 0 else 1
    if z_limit_masked_individual: z_lim_m = z_limit_masked_individual
    ax_masked_local.set_zlim(0, z_lim_m)
    fig_local.colorbar(surf_masked_local, ax=ax_masked_local, shrink=0.7, aspect=10, label="单独遮罩后奖励值")

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    plt.show()


# --- Main Simulation and Plotting ---

# 1. Simulation Area and Parameters
sim_area_width = 1000.0
sim_area_height = 1000.0
plot_grid_points = 200
x_plot_coords = np.linspace(0, sim_area_width, plot_grid_points)
y_plot_coords = np.linspace(0, sim_area_height, plot_grid_points)
X_plot_mesh, Y_plot_mesh = np.meshgrid(x_plot_coords, y_plot_coords)

Z_TARGET_HEIGHT = 0.0
Z_SENSOR_HEIGHT = 60.0

target1_coords_xy_reward_center = np.array([400.0, 600.0])
r_threshold_target1 = 5
target2_coords_xy_reward_center = np.array([600.0, 400.0])
r_threshold_target2 = 10
R_max_value = 100.0

# 2. Define Existing Fixed Sensors (3D) AND the Fixed Target for GDOP evaluation (3D)
area_center_x = sim_area_width / 2
area_center_y = sim_area_height / 2
num_existing_sensors = 4
sensor_spread_from_area_center = sim_area_width / 4.0
np.random.seed(45)
existing_fixed_sensors_3d = []
print(
    f"正在生成 {num_existing_sensors} 个已存在的固定传感器 (z={Z_SENSOR_HEIGHT}m)，位于区域中心 ({area_center_x:.1f}, {area_center_y:.1f}) 的第三象限...")
for i in range(num_existing_sensors):
    offset_x = np.random.uniform(-sensor_spread_from_area_center, -10.0)
    offset_y = np.random.uniform(-sensor_spread_from_area_center, -10.0)
    sensor_x = area_center_x + offset_x;
    sensor_y = area_center_y + offset_y
    sensor_x = np.clip(sensor_x, 0, sim_area_width);
    sensor_y = np.clip(sensor_y, 0, sim_area_height)
    existing_fixed_sensors_3d.append(np.array([sensor_x, sensor_y, Z_SENSOR_HEIGHT]))
    print(f"  已存在的传感器 {i + 1}: ({sensor_x:.2f}, {sensor_y:.2f}, {Z_SENSOR_HEIGHT:.1f})")

fixed_target_T1_for_gdop_3d = np.array(
    [target1_coords_xy_reward_center[0], target1_coords_xy_reward_center[1], Z_TARGET_HEIGHT])
fixed_target_T2_for_gdop_3d = np.array(
    [target2_coords_xy_reward_center[0], target2_coords_xy_reward_center[1], Z_TARGET_HEIGHT])
print(f"用于“情况A”GDOP评估的特定固定目标点T1 (3D): {fixed_target_T1_for_gdop_3d}")
print(f"用于“情况A”GDOP评估的特定固定目标点T2 (3D): {fixed_target_T2_for_gdop_3d}")

# 3. Sensor Error Parameters and GDOP Calculator
small_error_coeff = 0.01 ** 2
epsilon_variance = (0.1) ** 2
gdop_calc_instance = GDOPCalculator()

# 4. Calculate SEPARATE GDOP Heatmaps for T1 and T2
gdop_heatmap_T1 = calculate_sensor_placement_gdop_3d(
    X_plot_mesh, Y_plot_mesh, plot_grid_points, existing_fixed_sensors_3d,
    fixed_target_T1_for_gdop_3d, small_error_coeff, epsilon_variance, gdop_calc_instance,
    Z_SENSOR_HEIGHT
)
gdop_heatmap_T2 = calculate_sensor_placement_gdop_3d(
    X_plot_mesh, Y_plot_mesh, plot_grid_points, existing_fixed_sensors_3d,
    fixed_target_T2_for_gdop_3d, small_error_coeff, epsilon_variance, gdop_calc_instance,
    Z_SENSOR_HEIGHT
)

# 5. Normalize EACH GDOP heatmap to create respective Masks
gdop_mask_cap_val = 10.0
print("\n为目标T1创建遮罩 (基于3D PDOP):")
gdop_mask_T1 = create_gdop_mask_from_heatmap(gdop_heatmap_T1, gdop_mask_cap_val)
print("\n为目标T2创建遮罩 (基于3D PDOP):")
gdop_mask_T2 = create_gdop_mask_from_heatmap(gdop_heatmap_T2, gdop_mask_cap_val)

# --- Visualize 2D GDOP Masks ---
existing_sensors_array_vis_xy = np.array([s[:2] for s in existing_fixed_sensors_3d])

plt.figure(figsize=(9, 7.5))
im_mask_T1 = plt.imshow(gdop_mask_T1, extent=[0, sim_area_width, 0, sim_area_height],
                        origin='lower', cmap='viridis', aspect='auto', vmin=0, vmax=1)
plt.scatter(existing_sensors_array_vis_xy[:, 0], existing_sensors_array_vis_xy[:, 1], c='red', s=100, marker='^',
            label='已存在的固定传感器 (XY投影)', edgecolors='black')
plt.scatter(fixed_target_T1_for_gdop_3d[0], fixed_target_T1_for_gdop_3d[1], c='black', s=150, marker='X',
            label=f'GDOP固定目标点 T1 (XY投影)', edgecolors='white')
plt.colorbar(im_mask_T1, label='传感器放置效益遮罩值 (对T1)');
plt.xlabel('新传感器的潜在X坐标 (m)');
plt.ylabel('新传感器的潜在Y坐标 (m)')
plt.title(f'2D传感器放置效益遮罩图 (针对目标T1, 3D PDOP)');
plt.legend(fontsize='small');
plt.grid(True, linestyle='--', alpha=0.5);
plt.show()

plt.figure(figsize=(9, 7.5))
im_mask_T2 = plt.imshow(gdop_mask_T2, extent=[0, sim_area_width, 0, sim_area_height],
                        origin='lower', cmap='viridis', aspect='auto', vmin=0, vmax=1)
plt.scatter(existing_sensors_array_vis_xy[:, 0], existing_sensors_array_vis_xy[:, 1], c='red', s=100, marker='^',
            label='已存在的固定传感器 (XY投影)', edgecolors='black')
plt.scatter(fixed_target_T2_for_gdop_3d[0], fixed_target_T2_for_gdop_3d[1], c='grey', s=150, marker='P',
            label=f'GDOP固定目标点 T2 (XY投影)', edgecolors='white')
plt.colorbar(im_mask_T2, label='传感器放置效益遮罩值 (对T2)');
plt.xlabel('新传感器的潜在X坐标 (m)');
plt.ylabel('新传感器的潜在Y坐标 (m)')
plt.title(f'2D传感器放置效益遮罩图 (针对目标T2, 3D PDOP)');
plt.legend(fontsize='small');
plt.grid(True, linestyle='--', alpha=0.5);
plt.show()

# 6. Calculate Rewards, Apply Masks, and Combine
reward_T1_original = reward_function_apf(X_plot_mesh, Y_plot_mesh, R_max_value, r_threshold_target1,
                                         target1_coords_xy_reward_center[0], target1_coords_xy_reward_center[1])
reward_T2_original = reward_function_apf(X_plot_mesh, Y_plot_mesh, R_max_value, r_threshold_target2,
                                         target2_coords_xy_reward_center[0], target2_coords_xy_reward_center[1])
Z_original_combined_reward_sum = reward_T1_original + reward_T2_original
final_reward_T1_masked = reward_T1_original * gdop_mask_T1
final_reward_T2_masked = reward_T2_original * gdop_mask_T2
Z_final_combined_reward_max_of_masked = np.maximum(final_reward_T1_masked, final_reward_T2_masked)

# --- Visualize Overall 3D Comparison ---
plot_overall_3d_comparison(X_plot_mesh, Y_plot_mesh, Z_original_combined_reward_sum,
                           Z_final_combined_reward_max_of_masked, "(APF奖励)")

# 7. Local Reward Comparison Plots (3D Surface Plots)
local_window_radius_cells = int(plot_grid_points / 5)
max_orig_T1 = np.max(reward_T1_original);
max_orig_T2 = np.max(reward_T2_original)
max_masked_T1 = np.max(final_reward_T1_masked);
max_masked_T2 = np.max(final_reward_T2_masked)

plot_local_3d_comparison("T1", target1_coords_xy_reward_center, reward_T1_original, final_reward_T1_masked,
                         X_plot_mesh, Y_plot_mesh, local_window_radius_cells, z_limit_orig=max_orig_T1 * 1.1,
                         z_limit_masked_individual=max_masked_T1 * 1.1)
plot_local_3d_comparison("T2", target2_coords_xy_reward_center, reward_T2_original, final_reward_T2_masked,
                         X_plot_mesh, Y_plot_mesh, local_window_radius_cells, z_limit_orig=max_orig_T2 * 1.1,
                         z_limit_masked_individual=max_masked_T2 * 1.1)

print("所有可视化已完成。")
