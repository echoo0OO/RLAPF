import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from env.env_utils import  poisson_disk_sampling
from env.reward_shaping import (
    reward_function_apf,
    GDOPCalculator,
    calculate_gdop_heatmap_for_sensor,
    create_gdop_mask_from_heatmap
)
from env.uncertain_model import UncertaintyModel

# 这是一个完整的、可运行的环境模板
# 您需要将'...'部分替换为您自己的环境模拟逻辑

class DroneNavigationEnv(gym.Env):
    """
    一个适用于混合动作PPO和注意力架构的自定义无人机导航环境模板。

    观测空间 (Observation Space):
        - 'map': 一个多通道图像 (奖励图, 无人机位置图等)
        - 'sensors': 一个N*M的矩阵，代表N个传感器的M个状态
        - 'action_mask': 一个用于离散动作的掩码

    动作空间 (Action Space):
        - 'discrete': 多维离散动作 (例如，选择目标传感器)
        - 'continuous': 连续动作 (例如，速度、角速度)
    """

    def __init__(self, config):
        super().__init__()

        # --- 环境和模型参数 ---
        self.map_channels = config.get("map_channels", 2)
        self.img_h = config.get("img_h", 84)
        self.img_w = config.get("img_w", 84)
        self.num_sensors = config.get("num_sensors", 5)
        self.sensor_dim = config.get("sensor_dim", 4)
        self.action_dis_dim = config.get("action_dis_dim", 1)
        self.action_dis_len = config.get("action_dis_len", 2)
        self.action_con_dim = config.get("action_con_dim", 2)
        self.area_size = config.get("area_size", (1000.0, 1000.0))
        self.solo_SN_data = config.get("solo_SN_data", 2e7)
        self.max_steps_per_episode = config.get("max_steps", 500)
        self.local_view_size = config.get("local_view_size", 150.0)  # 局部视图为 150m x 150m

        # --- 1. 定义复杂的观测空间 (使用 spaces.Dict) ---
        self.observation_space = spaces.Dict({
            "map": spaces.Box(low=0, high=255, shape=(self.map_channels, self.img_h, self.img_w), dtype=np.float32), #局部地图150*150
            "sensors": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_sensors, self.sensor_dim),
                                  dtype=np.float32),
            # "action_mask": spaces.Box(low=0, high=1, shape=(self.action_dis_dim, self.action_dis_len), dtype=np.int8)
        })

        # --- 2. 定义混合动作空间 (使用 spaces.Dict) ---
        self.action_space = spaces.Dict({
            "discrete": spaces.Discrete(self.action_dis_len),
            "continuous": spaces.Box(low=-1.0, high=1.0, shape=(self.action_con_dim,), dtype=np.float32)
        })
        # --- 保存真实的动作边界 ---
        self.real_action_bounds = {
            'direction': {'low': -np.pi, 'high': np.pi},
            'speed': {'low': 0.0, 'high': 30.0}
        }
        self.last_comm_success = True  # 初始假设通信是“成功”的，避免一开始就限速
        # --- 初始化环境状态 (示例) ---
        #环境参数
        self.area_size = (1000.0, 1000.0)
        self.num_sensors = 5
        self.max_speed = 30.0  # m/s
        self.time_slot = 1.0  # s
        self.solo_SN_data = 2e7 #20Mbits
        # 通信参数
        self.bandwidth = 1e6 # 1MHz
        self.path_loss_exponent = 2
        self.reference_distance = 1.0 # m
        self.reference_loss = -60  # dB
        self.transmit_power = 0.1  # W
        self.noise_power = 1e-11  # W (-110dBm)
        self.snr_threshold = 10.0  # dB
        self.delay_penalty_coefficient = 2e7 #时延系数=单个SN数据量
        # 定位参数
        # 在 __init__ 中实例化 GDOPCalculator
        self.gdop_calc = GDOPCalculator()
        self.g0 = config.get("g0", 1.125e-5) # 测量噪声方差系数
        self.uncertainty_model = UncertaintyModel(num_sensors=self.num_sensors)
        self.R_max_base = 100.0
        # 用于触发定位模型更新的计数器
        self.ranging_update_interval = 10
        self.ranging_point_counter = 0
        # 状态变量
        # 这些变量会在reset时被正确初始化
        self.drone_position = None
        self.sensor_true_positions = None
        self.sensor_estimated_positions = None
        self.sensor_estimated_radii = None
        self.sensor_data_amounts = None
        self.sensor_states = None
        self.reward_map = None
        self.current_step = 0
        self.trajectory_save_freq = 5
        # --- 新增: 追踪定位稳定性的状态变量 ---
        self.radius_stable_steps = 0  # 连续多少步半径变化很小
        self.stable_threshold = 3  # 需要连续多少步才算稳定
        # 存储上一步的半径，用于比较变化
        self.previous_radii = None

        # 记录和可视化
        self.trajectory = []
        self.communication_log = []
        self.localization_log = []
        self.reward_components = []

        print("环境已初始化！")
        print(f"观测空间: {self.observation_space}")
        print(f"动作空间: {self.action_space}")

    def _unnormalize_action(self, norm_action):
        """将[-1, 1]的归一化动作映射回真实物理值"""
        # norm_action 是一个 shape=(2,) 的numpy数组

        # 映射方向: 从[-1, 1]映射到[-pi, pi]
        low_dir = self.real_action_bounds['direction']['low']
        high_dir = self.real_action_bounds['direction']['high']
        real_direction = low_dir + (norm_action[0] + 1.0) * 0.5 * (high_dir - low_dir)

        # 映射速度: 从[-1, 1]映射到[0, 30]
        low_speed = self.real_action_bounds['speed']['low']
        high_speed = self.real_action_bounds['speed']['high']
        real_speed = low_speed + (norm_action[1] + 1.0) * 0.5 * (high_speed - low_speed)

        return real_direction, real_speed

    def _generate_local_view_grid(self):
        """
        辅助函数：为当前无人机位置生成局部视图的坐标网格。
        """
        half_view = self.local_view_size / 2.0
        center_x, center_y = self.drone_position

        min_x = center_x - half_view
        max_x = center_x + half_view
        min_y = center_y - half_view
        max_y = center_y + half_view

        local_x_coords = np.linspace(min_x, max_x, self.img_w)
        local_y_coords = np.linspace(min_y, max_y, self.img_h)

        return np.meshgrid(local_x_coords, local_y_coords)

    def _calculate_gdop_mask_for_sensor(self, sensor_id: int, local_X_mesh, local_Y_mesh) -> np.ndarray:
        """
        为单个传感器计算其在当前局部视图上的GDOP遮罩。
        """
        # a. 从不确定性模型中获取所有历史测距点
        all_points_history = self.uncertainty_model.ranging_points
        all_points_flat = [pos for sensor_history in all_points_history for pos in sensor_history]

        # b. 挑选距离当前传感器估计位置最近的10个点
        sensor_est_pos_2d = self.sensor_estimated_positions[sensor_id]

        if not all_points_flat:
            # 如果没有任何历史测量点，返回一个表示“任何地方都好”的中性遮罩
            return np.ones((self.img_h, self.img_w), dtype=np.float32)

        all_points_array_2d = np.array(all_points_flat)
        distances = np.linalg.norm(all_points_array_2d - sensor_est_pos_2d, axis=1)

        num_to_select = min(10, len(all_points_array_2d))
        closest_points_indices = np.argsort(distances)[:num_to_select]

        # 将选中的2D测距点转换为3D（加上无人机高度）
        measurement_points_3d = [
            np.append(all_points_array_2d[i], self.drone_height) for i in closest_points_indices
        ]

        # 要定位的目标传感器也转换为3D（高度为0）
        sensor_to_locate_3d = np.append(sensor_est_pos_2d, 0.0)

        # c. 调用工具函数计算GDOP热力图
        # 注意：这一步计算量非常大！
        gdop_heatmap = calculate_gdop_heatmap_for_sensor(
            local_X_mesh, local_Y_mesh,
            measurement_points_3d,
            sensor_to_locate_3d,
            self.gdop_calc,
            self.g0
        )

        # d. 调用工具函数将热力图转换为遮罩
        gdop_mask = create_gdop_mask_from_heatmap(gdop_heatmap, cap_value=20.0)

        return gdop_mask

    def _compute_current_local_reward_map(self):
        """
        【新增】
        根据当前状态（无人机位置、传感器状态），计算并返回当前的局部奖励地图。
        这个函数是当前步骤中所有奖励地图计算的唯一来源。
        """
        # a. 为当前局部视图生成坐标网格
        local_X_mesh, local_Y_mesh = self._generate_local_view_grid()

        # b. 为每个传感器计算其贡献的奖励层
        map_reward_layers = []
        for i in range(self.num_sensors):
            # 如果传感器数据已采完，则其吸引力（R_max）为0
            current_R_max = self.R_max_base if self.sensor_data_amounts[i] > 0 else 0.0

            # 计算APF奖励层
            apf_layer = reward_function_apf(
                local_X_mesh, local_Y_mesh,
                R_max=current_R_max,
                r_threshold=self.sensor_estimated_radii[i],
                xc=self.sensor_estimated_positions[i][0],
                yc=self.sensor_estimated_positions[i][1]
            )

            # 计算GDOP遮罩层
            gdop_mask_layer = self._calculate_gdop_mask_for_sensor(i, local_X_mesh, local_Y_mesh)

            # 应用遮罩并添加到列表中
            map_reward_layers.append(apf_layer * gdop_mask_layer)

        # c. 融合所有层，得到最终的局部奖励地图
        if not map_reward_layers:  # 如果列表为空
            return np.zeros((self.img_h, self.img_w), dtype=np.float32)

        final_local_reward_map = np.maximum.reduce(map_reward_layers)
        return final_local_reward_map

    def _calculate_reward(self, current_reward_map):
        """
        【修改】
        计算当前时间步的总奖励。它现在接收一个预先计算好的奖励地图作为参数。
        """
        # --- 1. 获取基于地图的奖励 ---
        # 无人机在局部视图中永远在中心
        center_h, center_w = self.img_h // 2, self.img_w // 2
        map_based_reward = current_reward_map[center_h, center_w]

        # --- 2. 计算时延惩罚 ---
        data_collected = self.solo_SN_data - self.sensor_data_amounts
        total_data_collected = np.sum(data_collected)
        delay_penalty = - (total_data_collected / self.delay_penalty_coefficient)

        # --- 3. 计算总奖励 ---
        total_reward = map_based_reward + delay_penalty

        return total_reward

    def _get_obs(self):
        """辅助函数：根据当前环境状态生成一个符合观测空间的字典。"""
        # 1. 动态计算局部奖励地图
        local_reward_map = self._compute_current_local_reward_map()

        # 2. 创建无人机位置图
        # 在局部视图中，无人机永远在中心
        local_drone_pos_map = np.zeros((self.img_h, self.img_w), dtype=np.float32)
        center_h, center_w = self.img_h // 2, self.img_w // 2
        local_drone_pos_map[center_h, center_w] = 1.0

        # 3. !! 恢复：将两个通道堆叠起来 !!
        map_obs = np.stack([local_reward_map, local_drone_pos_map], axis=0)

        # --- 动态生成离散动作掩码 ---
        action_mask_dis = np.ones((self.action_dis_dim, self.action_dis_len), dtype=np.int8)
        is_stable = self.radius_stable_steps >= self.stable_threshold

        if is_stable:
            # 定位稳定阶段：优先进行通信, 强制选择动作0 (通信)
            action_mask_dis[0, 0] = 1  # 通信可用
            action_mask_dis[0, 1] = 0  # 定位不可用 (强制通信)

        # --- 动态生成连续动作掩码 ---
        # 默认情况下，所有连续动作都可用，范围是 [-1, 1]
        action_mask_con = np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=np.float32)

        # 仅当定位稳定且需要优先通信时，才施加方向约束

        # if is_stable: 试试不管定位稳不稳定都使用连续动作遮罩

        # a. 确定通信目标 (与 _execute_communication 逻辑一致)
        eligible_sensors_mask = self.sensor_data_amounts > 0
        horizontal_dist_to_est_center = np.linalg.norm(
            self.drone_position - self.sensor_estimated_positions, axis=1
        )
        max_horizontal_dist = horizontal_dist_to_est_center + self.sensor_estimated_radii
        max_dist_3d_all = np.sqrt(max_horizontal_dist ** 2 + self.drone_height ** 2)
        # 将不合格的传感器的距离设置为无穷大，使其永远不会被选中
        distances_to_consider = np.where(eligible_sensors_mask, max_dist_3d_all, np.inf)
        # 选择合格者中距离最近的传感器
        target_sensor_idx = np.argmin(distances_to_consider)

        # b. 计算指向目标传感器的方向 (角度)
        target_pos = self.sensor_estimated_positions[target_sensor_idx]
        direction_vector = target_pos - self.drone_position

        # 使用 arctan2 计算从-pi到pi的角度
        target_direction_rad = np.arctan2(direction_vector[1], direction_vector[0])

        # c. 定义一个可接受的角度范围 (例如，目标方向 ± 15度)
        angle_tolerance_rad = np.deg2rad(15)
        min_angle_rad = target_direction_rad - angle_tolerance_rad
        max_angle_rad = target_direction_rad + angle_tolerance_rad

        # d. 将真实角度范围 [-pi, pi] 映射到归一化范围 [-1, 1]
        # 归一化函数: norm_val = (real_val - low) / (high - low) * 2 - 1
        low_dir_real = self.real_action_bounds['direction']['low']  # -pi
        high_dir_real = self.real_action_bounds['direction']['high']  # +pi

        norm_min_angle = (min_angle_rad - low_dir_real) / (high_dir_real - low_dir_real) * 2 - 1
        norm_max_angle = (max_angle_rad - low_dir_real) / (high_dir_real - low_dir_real) * 2 - 1

        # 处理角度环绕问题 (例如，目标是-175度，范围可能跨越-180/180度)
        # 简单处理：如果最小归一化值大于最大值，说明跨越了边界，暂时不加约束
        if norm_min_angle < norm_max_angle:
            action_mask_con[0, :] = [norm_min_angle, norm_max_angle]

        # e. (可选) 也可以对速度施加约束，例如，强制高速飞行
        # --- 新增规则2：如果上一步通信失败，约束速度 ---
        if not self.last_comm_success:
            # a. 定义真实速度的高速范围 [20, 30] m/s
            high_speed_min_real = 20.0
            high_speed_max_real = 30.0
            # b. 获取速度的真实边界 [0, 30] m/s
            low_speed_real = self.real_action_bounds['speed']['low']
            high_speed_real = self.real_action_bounds['speed']['high']
            # c. 将真实的高速范围映射到归一化范围 [-1, 1]
            # 归一化函数: norm_val = (real_val - low) / (high - low) * 2 - 1
            norm_min_speed = (high_speed_min_real - low_speed_real) / (high_speed_real - low_speed_real) * 2 - 1
            norm_max_speed = (high_speed_max_real - low_speed_real) / (high_speed_real - low_speed_real) * 2 - 1
            # d. 应用速度约束
            action_mask_con[1, :] = [norm_min_speed, norm_max_speed]
            print(f"上一步通信失败，应用速度约束: [{norm_min_speed:.2f}, {norm_max_speed:.2f}]")

        return {
            "map": map_obs,
            "sensors": self.sensor_states,
            "action_mask_discrete": action_mask_dis,
            "action_mask_continuous": action_mask_con
        }


    def _get_info(self):
        """辅助函数：返回一些用于调试的额外信息。"""
        return {
            "drone_position": self.drone_position.copy(),
            "steps": self.current_step,
            "stable_steps": self.radius_stable_steps,
            "remaining_data_total": np.sum(self.sensor_data_amounts)
        }

    def reset(self, seed=None, options=None):
        """
        重置环境到初始状态。
        """
        super().reset(seed=seed)
        # --- 清空日志列表 ---
        self.trajectory = []
        self.communication_log = []
        self.localization_log = []
        self.reward_components = []  # 如果也想记录奖励的分解项
        # ... 在这里实现重置环境的逻辑 ...
        self.radius_stable_steps = 0
        self.previous_radii = np.full(self.num_sensors, 100.0)  # 初始半径都是100m
        # --- 重置通信状态 ---
        self.last_comm_success = True
        # 1. 重置内部计数器
        self.current_step = 0

        # 2. 生成传感器的真实位置 (源自 sensornet.py)
        # 使用泊松盘采样保证传感器之间有最小距离
        self.sensor_true_positions = poisson_disk_sampling(
            self.area_size, self.num_sensors, min_dist=250.0, np_random=self.np_random
        )

        # 3. 初始化传感器的估计状态 (源自 sensornet.py)
        # a. 初始估计半径都为100m
        initial_radius = 100.0
        # 让不确定性模型根据真实位置和初始半径生成自己的初始状态
        self.uncertainty_model.initialize_states(
            true_positions=self.sensor_true_positions,
            initial_radius=initial_radius,
            np_random=self.np_random
        )
        # 重置测量历史
        self.ranging_point_counter = 0
        # b. 从不确定性模型中获取初始状态，用于环境自身的状态变量
        self.sensor_estimated_positions = self.uncertainty_model.estimated_positions.copy()
        self.sensor_estimated_radii = self.uncertainty_model.uncertainty_radii.copy()
        self.previous_radii = self.sensor_estimated_radii.copy()  # 初始化 previous_radii

        # c. 初始化传感器数据量
        self.sensor_data_amounts = np.full(self.num_sensors, self.solo_SN_data)

        # 4. 初始化无人机状态
        self.drone_position = np.array([0.0, 0.0])
        self.trajectory.append(self.drone_position.copy())
        self.drone_height = 60.0 # 60m

        # 5. 构建符合观测空间的'sensors'状态矩阵
        # 格式: [est_x, est_y, uncertainty_radius, data_volume]
        self.sensor_states = np.concatenate([
            self.sensor_estimated_positions,
            self.sensor_estimated_radii.reshape(-1, 1),  # reshape为列向量
            self.sensor_data_amounts.reshape(-1, 1)  # reshape为列向量
        ], axis=1).astype(np.float32)

        # 6. reset不需要计算全局奖励图。
        #    它只需要调用一次_get_obs()来生成初始观测即可。
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _execute_communication(self):
        """
        执行与单个传感器的通信逻辑。
        """
        # 1. 找到所有数据量大于零的合格传感器
        eligible_sensors_mask = self.sensor_data_amounts > 0
        # 如果没有合格的传感器（所有数据都采集完了），则直接返回
        if not np.any(eligible_sensors_mask):
            print("所有传感器数据均已采集完毕，跳过通信。")
            # 保持上一步通信状态为成功，避免不必要的限速
            self.last_comm_success = True
            return
        # 1. 计算到每个传感器的“最大可能3D距离”用于目标选择
        # a. 无人机到每个传感器估计中心的水平距离
        horizontal_dist_to_est_center = np.linalg.norm(
            self.drone_position - self.sensor_estimated_positions, axis=1
        )
        # b. 最大可能水平距离 = 中心水平距离 + 不确定性半径
        max_horizontal_dist = horizontal_dist_to_est_center + self.sensor_estimated_radii

        # c. 使用勾股定理计算最大可能3D距离
        max_dist_3d = np.sqrt(max_horizontal_dist ** 2 + self.drone_height ** 2)

        # 3. 在选择目标时，只考虑合格的传感器
        # 将不合格的传感器的距离设置为无穷大，使其永远不会被选中
        distances_to_consider = np.where(eligible_sensors_mask, max_dist_3d, np.inf)
        # 选择合格者中距离最近的传感器
        target_sensor_idx = np.argmin(distances_to_consider)

        # 3. 计算与该目标传感器的“真实3D距离”
        true_sensor_pos = self.sensor_true_positions[target_sensor_idx]
        true_horizontal_dist = np.linalg.norm(self.drone_position - true_sensor_pos)
        true_dist_3d = np.sqrt(true_horizontal_dist ** 2 + self.drone_height ** 2)

        # 4. 根据最大距离预测信道增益和信噪比 (SNR)

        # 将参考损耗从dB转换为线性尺度
        ref_gain_linear = 10 ** (self.reference_loss / 10)

        # 计算路径损耗因子
        path_loss_factor_max = ref_gain_linear * (self.reference_distance / max_dist_3d[target_sensor_idx]) ** self.path_loss_exponent

        # 计算接收功率
        received_power_est = self.transmit_power * path_loss_factor_max

        # 计算线性信噪比
        snr_linear_est = received_power_est / self.noise_power

        # 转换为dB以便比较
        # snr_db = 10 * np.log10(snr_linear)

        # 5. 判断通信是否成功并计算传输的数据量
        transmitted_data = 0.0
        snr_linear = 0.0
        if snr_linear_est >= self.snr_threshold:
            self.last_comm_success = True
            # 通信成功，使用香农公式计算吞吐量 (bps)
            # 计算路径损耗因子
            path_loss_factor = ref_gain_linear * (self.reference_distance / true_dist_3d) ** self.path_loss_exponent
            # 计算接收功率
            received_power = self.transmit_power * path_loss_factor
            # 计算线性信噪比
            snr_linear = received_power / self.noise_power
            throughput_bps = self.bandwidth * np.log2(1 + snr_linear)
            # 在一个时隙内传输的数据量
            transmitted_data = throughput_bps * self.time_slot
            print(
                f"与传感器 {target_sensor_idx} 通信成功。SNR: {snr_linear:.2f} , 传输数据: {transmitted_data / 1e6:.2f} Mbits")
        else:
            self.last_comm_success = False
            print(f"与传感器 {target_sensor_idx} 通信失败。SNR: {snr_linear_est:.2f}  (低于阈值 {self.snr_threshold} )")
        # 6. 更新传感器的剩余数据量
        current_data = self.sensor_data_amounts[target_sensor_idx]
        self.sensor_data_amounts[target_sensor_idx] = max(0, current_data - transmitted_data)
        # --- 新增：记录通信日志 ---
        log_entry = {
            'step': self.current_step,
            'action': 'communication',
            'target_sensor': target_sensor_idx,
            'snr_linear': snr_linear,
            'transmitted_data_Mbits': transmitted_data / 1e6,
            'remaining_data': self.sensor_data_amounts.copy()  # 记录所有传感器当前的数据量
        }
        self.communication_log.append(log_entry)

    def _execute_localization(self):
        """
        【修正版】
        执行对所有传感器的单次测距，并根据条件更新不确定性模型。
        该版本明确使用2D水平距离进行所有计算。
        """
        # 执行定位时，认为通信状态是“好的” !!
        self.last_comm_success = True
        # 1. 计算到所有传感器的真实水平距离 (2D)
        true_horizontal_dist = np.linalg.norm(self.drone_position - self.sensor_true_positions, axis=1)

        # 2. 根据2D距离模型生成带噪声的测量值
        # 方差 variance = g0 * (distance^2)
        variances_2d = self.g0 * (true_horizontal_dist ** 2)
        std_devs_2d = np.sqrt(variances_2d)
        measured_distances_2d = self.np_random.normal(loc=true_horizontal_dist, scale=std_devs_2d)

        # 3. 将2D测量数据添加到不确定性模型中
        for i in range(self.num_sensors):
            # 调用 uncertainty_model 的接口，传入2D测量值
            self.uncertainty_model.add_ranging_measurement(
                sensor_id=i,
                drone_position=self.drone_position,
                measured_distance=measured_distances_2d[i]
            )

        # 4. 检查是否需要触发模型更新
        self.ranging_point_counter += 1
        if self.ranging_point_counter >= self.ranging_update_interval:
            print(f"已收集 {self.ranging_point_counter} 个新测距点，触发不确定性模型更新...")
            self.ranging_point_counter = 0  # 重置计数器

            # --- 关键修改：在这里更新稳定计数器 ---
            # a. 先保存更新前的半径，用于比较
            radii_before_update = self.uncertainty_model.uncertainty_radii.copy()

            # b. 对每个传感器调用更新方法，这会改变模型内部的估计值
            for i in range(self.num_sensors):
                self.uncertainty_model.update_sensor_estimate(sensor_id=i)
            # c. 从模型中获取更新后的状态，同步到环境自身的状态变量中
            self.sensor_estimated_positions = self.uncertainty_model.estimated_positions.copy()
            self.sensor_estimated_radii = self.uncertainty_model.uncertainty_radii.copy()
            # --- 新增：在模型更新后记录定位日志 ---
            log_entry = {
                'step': self.current_step,
                'action': 'localization_update',  # 标记这是一个更新事件
                'est_positions': self.sensor_estimated_positions.copy(),
                'est_radii': self.sensor_estimated_radii.copy()
            }
            self.localization_log.append(log_entry)
            # d. 现在，在半径真正更新后，再检查稳定性
            radius_change = np.abs(self.sensor_estimated_radii - radii_before_update)
            if np.all(radius_change < 0.5):  # 假设变化小于0.5米算稳定
                self.radius_stable_steps += 1
                print(f"半径变化小，稳定计数器增加到: {self.radius_stable_steps}")
            else:
                self.radius_stable_steps = 0  # 如果有任何一个半径变化大，则重置计数器
                print(f"半径变化大，稳定计数器重置为: 0")
            print(f"更新后半径: {np.round(self.sensor_estimated_radii, 2)}")



    def step(self, action):
        """
        在环境中执行一步。
        """
        self.current_step  += 1
        # --- 1. 解包混合动作 ---
        discrete_action = action["discrete"]
        normalized_continuous_action = action["continuous"]

        # 将归一化动作转换为真实动作
        real_direction, real_speed = self._unnormalize_action(normalized_continuous_action)
        print(f"无人机移动角度: [{real_direction:.2f},无人机移动速度 {real_speed:.2f}]")
        # --- 2. 在这里实现您的核心环境动力学 ---
        # ... 根据 discrete_action 和 continuous_action 更新环境状态 ...
        # --- 更新无人机位置 (基于连续动作) ---
        # a. 将速度和方向（极坐标）转换为速度向量 (vx, vy)（笛卡尔坐标）
        move_vector = np.array([
            real_speed * np.cos(real_direction),
            real_speed * np.sin(real_direction)
        ])

        # b. 根据速度和时间步长更新位置: new_pos = old_pos + velocity * time
        self.drone_position += move_vector * self.time_slot
        # c. 边界检查，确保无人机不会飞出定义的区域
        self.drone_position = np.clip(
            self.drone_position,
            [0.0, 0.0],  # 区域的左下角边界 (x_min, y_min)
            [self.area_size[0], self.area_size[1]]  # 区域的右上角边界 (x_max, y_max)
        )
        # d. 记录轨迹点，用于后续的可视化
        if self.current_step % self.trajectory_save_freq == 0:
            self.trajectory.append(self.drone_position.copy())

        if discrete_action == 0:
            # 执行通信逻辑...
            self._execute_communication()
        elif discrete_action == 1:
            # 执行定位逻辑...
            self._execute_localization()

        # 5. 更新用于观测的'sensors'矩阵
        self.sensor_states = np.concatenate([
            self.sensor_estimated_positions,
            self.sensor_estimated_radii.reshape(-1, 1),
            self.sensor_data_amounts.reshape(-1, 1)
        ], axis=1).astype(np.float32)

        # 6. 计算奖励
        # 【注意】奖励函数现在需要在这里调用
        current_reward_map = self._compute_current_local_reward_map()
        reward = self._calculate_reward(current_reward_map)
        self.current_step += 1

        # --- 7. 判断 episode 是否结束 ---
        # !! 关键修改：当所有传感器数据量为0时，任务完成 !!
        all_data_collected = np.all(self.sensor_data_amounts == 0)
        max_steps_reached = self.current_step >= self.max_steps_per_episode

        terminated = bool(all_data_collected)
        truncated = bool(max_steps_reached)

        # --- 8. 获取下一步的观测和信息 ---
        observation = self._get_obs()
        info = self._get_info()

        # --- 9. 任务成功或超时后的额外处理 ---
        if terminated:
            # 任务成功完成，给予一个大的正奖励
            reward += 200  # 您可以调整这个数值
            print(f"所有数据采集完毕！任务成功。Episode 结束。")

        if truncated:
            print(f"达到最大步数 {self.max_steps_per_episode}，任务超时。Episode 结束。")

        return observation, reward, terminated, truncated, info