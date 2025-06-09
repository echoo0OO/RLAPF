import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

def poisson_disk_sampling(width: float, height: float, min_distance: float, 
                         boundary_margin: float, num_sensors: int, 
                         max_attempts: int = 1000) -> np.ndarray:
    """
    泊松圆盘采样算法生成传感器位置
    
    Args:
        width: 区域宽度
        height: 区域高度  
        min_distance: 传感器间最小距离
        boundary_margin: 距离边界的最小距离
        num_sensors: 需要生成的传感器数量
        max_attempts: 最大尝试次数
        
    Returns:
        传感器位置数组 shape: (num_sensors, 2)
    """
    sensors = []
    attempts = 0
    
    # 有效区域边界
    x_min, x_max = boundary_margin, width - boundary_margin
    y_min, y_max = boundary_margin, height - boundary_margin
    
    while len(sensors) < num_sensors and attempts < max_attempts:
        # 随机生成候选位置
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        candidate = np.array([x, y])
        
        # 检查与已有传感器的距离
        valid = True
        for existing_sensor in sensors:
            if np.linalg.norm(candidate - existing_sensor) < min_distance:
                valid = False
                break
                
        if valid:
            sensors.append(candidate)
            
        attempts += 1
    
    if len(sensors) < num_sensors:
        print(f"警告：只生成了{len(sensors)}个传感器，少于要求的{num_sensors}个")
    
    return np.array(sensors)

def generate_sensor_network(area_size: Tuple[float, float] = (1000.0, 1000.0),
                          num_sensors: int = 5,
                          min_distance: float = 250.0,
                          boundary_margin: float = 100.0) -> dict:
    """
    生成传感器网络配置
    
    Returns:
        包含传感器真实位置、估计位置、初始不确定性等信息的字典
    """
    width, height = area_size
    
    # 生成传感器真实位置
    true_positions = poisson_disk_sampling(
        width, height, min_distance, boundary_margin, num_sensors
    )
    
    # 生成估计位置（在真实位置周围50m范围内随机）
    estimation_noise = 50.0
    estimated_positions = []
    
    for true_pos in true_positions:
        # 在真实位置周围生成估计位置
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(0, estimation_noise)
        noise = np.array([distance * np.cos(angle), distance * np.sin(angle)])
        estimated_pos = true_pos + noise
        estimated_positions.append(estimated_pos)
    
    estimated_positions = np.array(estimated_positions)
    
    # 初始不确定性半径
    initial_uncertainty_radius = np.full(num_sensors, 100.0)
    
    # 初始数据量（随机分配）
    initial_data_amounts = np.random.uniform(50, 200, num_sensors)
    
    return {
        'true_positions': true_positions,
        'estimated_positions': estimated_positions,
        'initial_uncertainty_radius': initial_uncertainty_radius,
        'initial_data_amounts': initial_data_amounts,
        'area_size': area_size,
        'num_sensors': num_sensors
    }

def visualize_sensor_network(sensor_config: dict, drone_position: Optional[np.ndarray] = None):
    """可视化传感器网络"""
    plt.figure(figsize=(10, 10))
    
    true_pos = sensor_config['true_positions']
    est_pos = sensor_config['estimated_positions']
    uncertainty = sensor_config['initial_uncertainty_radius']
    
    # 绘制真实位置
    plt.scatter(true_pos[:, 0], true_pos[:, 1], c='red', s=100, 
                marker='o', label='真实位置', alpha=0.8)
    
    # 绘制估计位置和不确定性圆
    for i, (est, radius) in enumerate(zip(est_pos, uncertainty)):
        circle = plt.Circle(est, radius, fill=False, color='blue', alpha=0.5)
        plt.gca().add_patch(circle)
        plt.scatter(est[0], est[1], c='blue', s=80, marker='x', alpha=0.8)
        plt.text(est[0]+20, est[1]+20, f'S{i}', fontsize=10)
    
    # 绘制无人机位置
    if drone_position is not None:
        plt.scatter(drone_position[0], drone_position[1], c='green', s=150, 
                   marker='^', label='无人机', alpha=0.9)
    
    plt.xlim(0, sensor_config['area_size'][0])
    plt.ylim(0, sensor_config['area_size'][1])
    plt.xlabel('X坐标 (m)')
    plt.ylabel('Y坐标 (m)')
    plt.title('传感器网络布局')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    # 测试代码
    config = generate_sensor_network()
    print("传感器网络配置：")
    print(f"传感器数量: {config['num_sensors']}")
    print(f"真实位置:\n{config['true_positions']}")
    print(f"估计位置:\n{config['estimated_positions']}")
    
    # 可视化
    visualize_sensor_network(config, np.array([0, 0])) 