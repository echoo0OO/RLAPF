import gymnasium as gym
from gymnasium import spaces
import numpy as np


# --- 辅助函数 ---
# 建议将这些函数放在一个独立的 'env_utils.py' 文件中
def poisson_disk_sampling(region_size, num_points, min_dist, np_random):
    """
    使用泊松盘采样生成均匀分布且相互分离的点。
    :param region_size: 区域大小 (正方形边长)
    :param num_points: 要生成的点的数量
    :param min_dist: 点之间的最小距离
    :param np_random: gymnasium环境的随机数生成器
    :return: 点的数组 (x, y)
    """
    width, height = region_size
    points = []
    active_list = []

    # 创建一个加速网格
    cell_size = min_dist / np.sqrt(2)
    grid_width = int(np.ceil(width / cell_size))
    grid_height = int(np.ceil(height / cell_size))
    grid = [None] * (grid_width * grid_height)

    def get_grid_coords(p):
        return int(p[0] / cell_size), int(p[1] / cell_size)

    def add_point(p):
        points.append(p)
        gx, gy = get_grid_coords(p)
        grid[gy * grid_width + gx] = p
        active_list.append(p)

    # 初始点
    initial_point = np_random.uniform([0, 0], [width, height])
    add_point(initial_point)

    while active_list and len(points) < num_points:
        idx = np_random.integers(len(active_list))
        p = active_list[idx]
        found = False
        for _ in range(30):  # k = 30
            angle = 2 * np.pi * np_random.random()
            radius = np_random.uniform(min_dist, 2 * min_dist)
            new_point = p + np.array([radius * np.cos(angle), radius * np.sin(angle)])

            if 0 <= new_point[0] < width and 0 <= new_point[1] < height:
                gx, gy = get_grid_coords(new_point)
                valid = True
                # 检查周围9个格子
                for i in range(max(0, gx - 2), min(grid_width, gx + 3)):
                    for j in range(max(0, gy - 2), min(grid_height, gy + 3)):
                        neighbor = grid[j * grid_width + i]
                        if neighbor is not None and np.linalg.norm(new_point - neighbor) < min_dist:
                            valid = False
                            break
                    if not valid:
                        break

                if valid:
                    add_point(new_point)
                    found = True
                    if len(points) >= num_points:
                        return np.array(points)
        if not found:
            active_list.pop(idx)

    return np.array(points)


