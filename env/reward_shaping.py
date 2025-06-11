
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from scipy.spatial.distance import cdist
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题

def reward_function_apf(x_coords_mesh, y_coords_mesh, R_max, r_threshold, xc, yc):
    """
    APF形式的混合奖励函数，来自show5_3APF.py
    """
    effective_r_threshold = max(r_threshold, 20.0)*3 #最小值10→20
    c1 = R_max / 3.0
    d = np.sqrt((x_coords_mesh - xc) ** 2 + (y_coords_mesh - yc) ** 2)
    reward = np.zeros_like(d)

    parabolic_mask = (d <= effective_r_threshold)
    reward[parabolic_mask] = R_max - (c1 * d[parabolic_mask] ** 2) / (effective_r_threshold ** 2)

    hyperbolic_mask = (d > effective_r_threshold)
    d_hyperbolic = d[hyperbolic_mask]
    d_hyperbolic[d_hyperbolic == 0] = 1e-6  # 避免除以零
    reward[hyperbolic_mask] = (effective_r_threshold * (R_max - c1)) / d_hyperbolic

    return reward


class GDOPCalculator:
    """GDOP（几何精度因子）计算器"""

    def calculate_gdop_for_fixed_target(self,
                                        fixed_target_point_3d: np.ndarray,
                                        current_sensor_network_3d: List[np.ndarray],
                                        sensor_weights: np.ndarray
                                        ) -> float:
        """计算加权GDOP值。"""
        # ... (此函数的代码直接从 show5_3APF.py 复制过来) ...
        # (为简洁省略，逻辑与您提供的文件一致)
        if not current_sensor_network_3d or len(current_sensor_network_3d) < 3:
            return float('inf')

        G_list = []
        for i, sensor_loc_3d in enumerate(current_sensor_network_3d):
            diff_vector = sensor_loc_3d - fixed_target_point_3d
            distance_to_target = np.linalg.norm(diff_vector)
            if distance_to_target < 1e-6: continue
            unit_vector = diff_vector / distance_to_target
            G_list.append(unit_vector)

        if len(G_list) < 3: return float('inf')

        G_matrix = np.array(G_list)
        W_matrix = np.diag(sensor_weights[:len(G_matrix)])

        try:
            GtWG = G_matrix.T @ W_matrix @ G_matrix
            if np.linalg.det(GtWG) < 1e-10: return float('inf')
            GtWG_inv = np.linalg.inv(GtWG)
            pdop = np.sqrt(np.trace(GtWG_inv))
            return pdop
        except np.linalg.LinAlgError:
            return float('inf')


def calculate_gdop_heatmap_for_sensor(
        local_X_mesh, local_Y_mesh,
        measurement_points_3d: List[np.ndarray],
        sensor_to_locate_3d: np.ndarray,
        gdop_calculator: GDOPCalculator,
        g0: float
) -> np.ndarray:
    """
    为单个传感器在局部视图上计算GDOP热力图。
    热力图的每个点的值代表“如果无人机飞到该点并进行一次测量，对定位该传感器的GDOP改善程度”。
    """
    heatmap = np.full_like(local_X_mesh, float('inf'))

    for r_idx in range(local_X_mesh.shape[0]):
        for c_idx in range(local_X_mesh.shape[1]):
            # 潜在的无人机新位置（即热力图上的一个点）
            potential_uav_pos_3d = np.array([local_X_mesh[r_idx, c_idx], local_Y_mesh[r_idx, c_idx], 60.0])  # 假设无人机高度60

            # 包含新测量点在内的网络
            current_network_3d = measurement_points_3d + [potential_uav_pos_3d]

            # 计算权重 (1/方差)
            weights = np.zeros(len(current_network_3d))
            for k, p_loc_3d in enumerate(current_network_3d):
                dist_3d = np.linalg.norm(p_loc_3d - sensor_to_locate_3d)
                variance = g0 * (dist_3d ** 2)
                weights[k] = 1.0 / variance if variance > 1e-9 else 1e9

            heatmap[r_idx, c_idx] = gdop_calculator.calculate_gdop_for_fixed_target(
                sensor_to_locate_3d, current_network_3d, weights
            )

    return heatmap


def create_gdop_mask_from_heatmap(gdop_heatmap, cap_value=20.0):
    """将GDOP热力图转换为[0, 1]的遮罩。GDOP值越小，遮罩值越大（越好）。"""
    # ... (此函数的代码直接从 show5_3APF.py 复制过来) ...
    finite_gdop = np.nan_to_num(gdop_heatmap, nan=float('inf'), posinf=float('inf'))
    capped_gdop = np.clip(finite_gdop, 0, cap_value)
    min_g = np.min(capped_gdop)
    max_g = np.max(capped_gdop)
    if (max_g - min_g) < 1e-6:
        return np.ones_like(capped_gdop) if min_g < cap_value else np.zeros_like(capped_gdop)
    norm_gdop = (capped_gdop - min_g) / (max_g - min_g)
    mask = 1.0 - norm_gdop
    return mask