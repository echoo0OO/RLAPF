import numpy as np

def reward_function_apf(x_coords_mesh, y_coords_mesh, R_max, r_threshold, xc, yc):
    """
    APF形式的混合奖励函数，来自show5_3APF.py
    """
    effective_r_threshold = max(r_threshold, 10.0)
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