import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from scipy.spatial.distance import cdist
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题

class GDOPCalculator:
    """GDOP（几何精度因子）计算器"""
    
    def __init__(self, area_size: Tuple[float, float] = (1000.0, 1000.0), 
                 grid_resolution: float = 10.0): # 分辨率之前为50.0
        """
        初始化GDOP计算器
        
        Args:
            area_size: 区域大小 (width, height)
            grid_resolution: 网格分辨率（米）
        """
        self.area_size = area_size
        self.grid_resolution = grid_resolution
        
        # 创建网格
        self.x_grid = np.arange(0, area_size[0] + grid_resolution, grid_resolution)
        self.y_grid = np.arange(0, area_size[1] + grid_resolution, grid_resolution)
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)
        
        # GDOP热力图
        self.gdop_heatmap = np.zeros_like(self.X)
        
        # 传感器相关数据
        self.sensor_positions = None
        self.sensor_ranging_points = None
        
    def calculate_gdop_at_point(self, point: np.ndarray, ranging_points: List[np.ndarray]) -> float:
        """
        计算指定点的GDOP值
        
        Args:
            point: 计算点坐标 [x, y]
            ranging_points: 测距点列表
            
        Returns:
            GDOP值
        """
        if len(ranging_points) < 3:
            return float('inf')  # 测距点不足
        
        # 构建几何矩阵G
        G = []
        for ranging_point in ranging_points:
            dx = ranging_point[0] - point[0]
            dy = ranging_point[1] - point[1]
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance < 1e-6:  # 避免除零
                continue
                
            # 单位方向向量
            unit_x = dx / distance
            unit_y = dy / distance
            G.append([unit_x, unit_y])
        
        if len(G) < 3:
            return float('inf')
        
        G = np.array(G)
        
        try:
            # 计算GDOP = sqrt(trace((G^T * G)^-1))
            GtG = G.T @ G
            if np.linalg.det(GtG) < 1e-10:
                return float('inf')
            
            GtG_inv = np.linalg.inv(GtG)
            gdop = np.sqrt(np.trace(GtG_inv))
            
            return gdop
            
        except np.linalg.LinAlgError:
            return float('inf')
    
    def update_sensor_gdop_heatmap(self, sensor_id: int, sensor_position: np.ndarray, 
                                  ranging_points: List[np.ndarray]) -> np.ndarray:
        """
        更新单个传感器的GDOP热力图
        
        Args:
            sensor_id: 传感器ID
            sensor_position: 传感器位置
            ranging_points: 该传感器的测距点列表
            
        Returns:
            该传感器的GDOP热力图
        """
        sensor_gdop_map = np.zeros_like(self.X)
        
        if len(ranging_points) < 3:
            # 测距点不足，返回高GDOP值
            sensor_gdop_map.fill(100.0)
            return sensor_gdop_map
        
        # 为每个传感器选择最近的9个测距点
        if len(ranging_points) > 9:
            # 计算距离并选择最近的9个点
            distances = cdist([sensor_position], ranging_points)[0]
            closest_indices = np.argsort(distances)[:9]
            selected_points = [ranging_points[i] for i in closest_indices]
        else:
            selected_points = ranging_points
        
        # 计算网格上每个点的GDOP
        for i in range(len(self.y_grid)):
            for j in range(len(self.x_grid)):
                point = np.array([self.X[i, j], self.Y[i, j]])
                gdop = self.calculate_gdop_at_point(point, selected_points)
                sensor_gdop_map[i, j] = min(gdop, 100.0)  # 限制最大GDOP值
        
        return sensor_gdop_map
    
    def update_global_gdop_heatmap(self, sensors_data: Dict[int, Dict]) -> np.ndarray:
        """
        更新全局GDOP热力图
        
        Args:
            sensors_data: 传感器数据字典
                格式: {sensor_id: {'position': np.ndarray, 'ranging_points': List[np.ndarray], 'data_amount': float}}
                
        Returns:
            全局GDOP热力图
        """
        if not sensors_data:
            self.gdop_heatmap.fill(20.0)
            return self.gdop_heatmap
        
        # 计算每个传感器的GDOP热力图
        sensor_gdop_maps = {}
        total_data_amount = 0
        
        for sensor_id, data in sensors_data.items():
            sensor_position = data['position']
            ranging_points = data['ranging_points']
            data_amount = data.get('data_amount', 1.0)
            
            sensor_gdop_map = self.update_sensor_gdop_heatmap(
                sensor_id, sensor_position, ranging_points
            )
            sensor_gdop_maps[sensor_id] = sensor_gdop_map
            total_data_amount += data_amount
        
        # 根据剩余数据量加权平均
        self.gdop_heatmap.fill(0.0)
        
        for sensor_id, data in sensors_data.items():
            data_amount = data.get('data_amount', 1.0)
            weight = data_amount / total_data_amount if total_data_amount > 0 else 1.0 / len(sensors_data)
            self.gdop_heatmap += weight * sensor_gdop_maps[sensor_id]
        
        return self.gdop_heatmap
    
    def get_gdop_features(self, drone_position: np.ndarray, feature_radius: float = 100.0) -> np.ndarray:
        """
        提取无人机周围的GDOP特征
        
        Args:
            drone_position: 无人机位置
            feature_radius: 特征提取半径
            
        Returns:
            GDOP特征向量
        """
        # 找到无人机位置在网格中的索引
        x_idx = int(drone_position[0] / self.grid_resolution)
        y_idx = int(drone_position[1] / self.grid_resolution)
        
        # 计算特征提取范围
        radius_cells = int(feature_radius / self.grid_resolution)
        
        # 提取局部GDOP值
        y_min = max(0, y_idx - radius_cells)
        y_max = min(self.gdop_heatmap.shape[0], y_idx + radius_cells + 1)
        x_min = max(0, x_idx - radius_cells)
        x_max = min(self.gdop_heatmap.shape[1], x_idx + radius_cells + 1)
        
        local_gdop = self.gdop_heatmap[y_min:y_max, x_min:x_max]
        
        # 计算统计特征
        features = [
            np.mean(local_gdop),           # 平均GDOP
            np.std(local_gdop),            # GDOP标准差
            np.min(local_gdop),            # 最小GDOP
            np.max(local_gdop),            # 最大GDOP
            np.median(local_gdop),         # 中位数GDOP
        ]
        
        # 添加方向性特征（8个方向的平均GDOP）
        center_y, center_x = local_gdop.shape[0] // 2, local_gdop.shape[1] // 2
        directions = []
        
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            dx = int(radius_cells * np.cos(angle))
            dy = int(radius_cells * np.sin(angle))
            
            target_y = center_y + dy
            target_x = center_x + dx
            
            if 0 <= target_y < local_gdop.shape[0] and 0 <= target_x < local_gdop.shape[1]:
                directions.append(local_gdop[target_y, target_x])
            else:
                directions.append(20.0)  # 边界外设为高GDOP
        
        features.extend(directions)
        
        return np.array(features)
    
    def get_best_positioning_location(self, current_position: np.ndarray, 
                                    search_radius: float = 200.0) -> np.ndarray:
        """
        在指定半径内找到GDOP最小的位置
        
        Args:
            current_position: 当前位置
            search_radius: 搜索半径
            
        Returns:
            最佳定位位置
        """
        # 在搜索半径内找到GDOP最小的网格点
        distances = np.sqrt((self.X - current_position[0])**2 + (self.Y - current_position[1])**2)
        valid_mask = distances <= search_radius
        
        if not np.any(valid_mask):
            return current_position
        
        # 在有效区域内找到GDOP最小值
        valid_gdop = np.where(valid_mask, self.gdop_heatmap, np.inf)
        min_idx = np.unravel_index(np.argmin(valid_gdop), valid_gdop.shape)
        
        best_position = np.array([self.X[min_idx], self.Y[min_idx]])
        return best_position
    
    def visualize_gdop_heatmap(self, drone_position: Optional[np.ndarray] = None,
                              sensor_positions: Optional[List[np.ndarray]] = None,
                              title: str = "GDOP热力图"):
        """可视化GDOP热力图"""
        plt.figure(figsize=(12, 10))
        
        # 绘制GDOP热力图
        im = plt.imshow(self.gdop_heatmap, extent=[0, self.area_size[0], 0, self.area_size[1]], 
                       origin='lower', cmap='viridis_r', alpha=0.8)
        plt.colorbar(im, label='GDOP值')
        
        # 绘制传感器位置
        if sensor_positions is not None:
            sensor_pos = np.array(sensor_positions)
            plt.scatter(sensor_pos[:, 0], sensor_pos[:, 1], c='red', s=100, 
                       marker='s', label='传感器', alpha=0.9)
        
        # 绘制无人机位置
        if drone_position is not None:
            plt.scatter(drone_position[0], drone_position[1], c='white', s=150, 
                       marker='^', label='无人机', alpha=0.9, edgecolors='black')
        
        plt.xlabel('X坐标 (m)')
        plt.ylabel('Y坐标 (m)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def get_gdop_penalty(self, drone_position: np.ndarray) -> float:
        """
        计算当前位置的GDOP惩罚值
        
        Args:
            drone_position: 无人机位置
            
        Returns:
            GDOP惩罚值（0-1之间，值越大惩罚越大）
        """
        # 获取当前位置的GDOP值
        x_idx = int(np.clip(drone_position[0] / self.grid_resolution, 0, len(self.x_grid) - 1))
        y_idx = int(np.clip(drone_position[1] / self.grid_resolution, 0, len(self.y_grid) - 1))
        
        current_gdop = self.gdop_heatmap[y_idx, x_idx]
        
        # 将GDOP值映射到0-1的惩罚值
        # GDOP值越大，惩罚越大
        max_gdop = 20.0  # 设定最大GDOP阈值
        penalty = min(current_gdop / max_gdop, 1.0)
        
        return penalty
    
    def get_gdop_value(self, position: np.ndarray) -> float:
        """
        获取指定位置的GDOP值
        
        Args:
            position: 位置坐标 [x, y]
            
        Returns:
            该位置的GDOP值
        """
        # 将位置转换为网格索引
        x_idx = int(np.clip(position[0] / self.grid_resolution, 0, len(self.x_grid) - 1))
        y_idx = int(np.clip(position[1] / self.grid_resolution, 0, len(self.y_grid) - 1))
        
        return float(self.gdop_heatmap[y_idx, x_idx])

if __name__ == "__main__":
    # 测试代码
    gdop_calc = GDOPCalculator()
    
    # 模拟传感器数据
    sensors_data = {
        0: {
            'position': np.array([200, 200]),
            'ranging_points': [
                np.array([100, 100]), np.array([150, 150]), np.array([200, 100]),
                np.array([250, 150]), np.array([180, 180])
            ],
            'data_amount': 100.0
        },
        1: {
            'position': np.array([800, 800]),
            'ranging_points': [
                np.array([700, 700]), np.array([750, 750]), np.array([800, 700]),
                np.array([850, 750]), np.array([780, 780])
            ],
            'data_amount': 150.0
        }
    }
    
    # 更新GDOP热力图
    gdop_calc.update_global_gdop_heatmap(sensors_data)
    
    # 可视化
    drone_pos = np.array([400, 400])
    sensor_positions = [data['position'] for data in sensors_data.values()]
    gdop_calc.visualize_gdop_heatmap(drone_pos, sensor_positions)
    
    # 提取特征
    features = gdop_calc.get_gdop_features(drone_pos)
    print(f"GDOP特征: {features}")
    
    # 计算惩罚
    penalty = gdop_calc.get_gdop_penalty(drone_pos)
    print(f"GDOP惩罚: {penalty:.3f}") 