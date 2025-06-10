import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from env.env_utils import  poisson_disk_sampling
from env.reward_shaping import reward_function_apf
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
            "action_mask": spaces.Box(low=0, high=1, shape=(self.action_dis_dim, self.action_dis_len), dtype=np.int8)
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
        # 定位参数
        self.g0 = config.get("g0", 1.125e-5) # 测量噪声方差系数
        self.uncertainty_model = UncertaintyModel(num_sensors=self.num_sensors)
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

    def _get_obs(self):
        """辅助函数：根据当前环境状态生成一个符合观测空间的字典。"""
        # ... 在这里实现生成观测字典的逻辑 ...
        # 例如，根据self.drone_position更新位置地图
        drone_pos_map = np.zeros((self.img_h, self.img_w), dtype=np.float32)
        if self.drone_position is not None:
            # 将无人机位置(归一化到图像坐标)在地图上标记为1
            pos_x = int(self.drone_position[0] * self.img_w)
            pos_y = int(self.drone_position[1] * self.img_h)
            drone_pos_map[pos_y, pos_x] = 1

        # 组合成多通道地图
        map_obs = np.stack([self.reward_map, drone_pos_map], axis=0)

        # --- 动态生成动作掩码 ---
        # 动作0: 通信, 动作1: 定位
        action_mask = np.zeros((self.action_dis_dim, self.action_dis_len), dtype=np.int8)

        # 判断定位是否稳定
        if self.radius_stable_steps < self.stable_threshold:
            # 定位不稳定阶段：强制或优先进行定位
            # 掩码为 [0, 1]，意味着只有动作1（定位）是可用的
            action_mask[0, 1] = 1
        else:
            # 定位稳定阶段：优先进行通信
            # 掩码为 [1, 1]，意味着两个动作都可用，让智能体自己决定
            # 也可以设为 [1, 0] 来强制通信，但允许两个动作通常更灵活
            action_mask[0, 0] = 1
            action_mask[0, 1] = 1
        return {
            "map": map_obs,
            "sensors": self.sensor_states,
            "action_mask": action_mask
        }


    def _get_info(self):
        """辅助函数：返回一些用于调试的额外信息。"""
        return {
            "drone_position": self.drone_position,
            "steps": self.current_step
        }

    def reset(self, seed=None, options=None):
        """
        重置环境到初始状态。
        """
        super().reset(seed=seed)

        # ... 在这里实现重置环境的逻辑 ...
        self.radius_stable_steps = 0
        self.previous_radii = np.full(self.num_sensors, 100.0)  # 初始半径都是100m
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
        # 1. 计算到每个传感器的“最大可能3D距离”用于目标选择
        # a. 无人机到每个传感器估计中心的水平距离
        horizontal_dist_to_est_center = np.linalg.norm(
            self.drone_position - self.sensor_estimated_positions, axis=1
        )
        # b. 最大可能水平距离 = 中心水平距离 + 不确定性半径
        max_horizontal_dist = horizontal_dist_to_est_center + self.sensor_estimated_radii

        # c. 使用勾股定理计算最大可能3D距离
        max_dist_3d = np.sqrt(max_horizontal_dist ** 2 + self.drone_height ** 2)

        # 2. 选择最大可能距离最近的传感器作为通信目标
        target_sensor_idx = np.argmin(max_dist_3d)

        # 3. 计算与该目标传感器的“真实3D距离”
        true_sensor_pos = self.sensor_true_positions[target_sensor_idx]
        true_horizontal_dist = np.linalg.norm(self.drone_position - true_sensor_pos)
        true_dist_3d = np.sqrt(true_horizontal_dist ** 2 + self.drone_height ** 2)

        # 4. 根据最大距离预测信道增益和信噪比 (SNR)
        # 将参考损耗从dB转换为线性尺度
        ref_gain_linear = 10 ** (self.reference_loss / 10)

        # 计算路径损耗因子
        path_loss_factor_max = ref_gain_linear * (self.reference_distance / max_dist_3d) ** self.path_loss_exponent

        # 计算接收功率
        received_power_est = self.transmit_power * path_loss_factor_max

        # 计算线性信噪比
        snr_linear_est = received_power_est / self.noise_power

        # 转换为dB以便比较
        # snr_db = 10 * np.log10(snr_linear)

        # 5. 判断通信是否成功并计算传输的数据量
        transmitted_data = 0.0
        if snr_linear_est >= self.snr_threshold:
            # 通信成功，使用香农公式计算吞吐量 (bps)
            # 计算路径损耗因子
            path_loss_factor = ref_gain_linear * (self.reference_distance / max_dist_3d) ** self.path_loss_exponent
            # 计算接收功率
            received_power = self.transmit_power * path_loss_factor
            # 计算线性信噪比
            snr_linear = received_power / self.noise_power
            throughput_bps = self.bandwidth * np.log2(1 + snr_linear)
            # 在一个时隙内传输的数据量
            transmitted_data = throughput_bps * self.time_slot
            print(
                f"与传感器 {target_sensor_idx} 通信成功。SNR: {snr_linear:.2f} dB, 传输数据: {transmitted_data / 1e6:.2f} Mbits")
        else:
            print(f"与传感器 {target_sensor_idx} 通信失败。SNR: {snr_linear_est:.2f} dB (低于阈值 {self.snr_threshold_db} dB)")

        # 6. 更新传感器的剩余数据量
        current_data = self.sensor_data_amounts[target_sensor_idx]
        self.sensor_data_amounts[target_sensor_idx] = max(0, current_data - transmitted_data)

    def _execute_localization(self):
        """
        【修正版】
        执行对所有传感器的单次测距，并根据条件更新不确定性模型。
        该版本明确使用2D水平距离进行所有计算。
        """
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
            self.trajectory_points.append(self.drone_position.copy())

        if discrete_action == 0:
            # 执行通信逻辑...
            self._execute_communication()
        elif discrete_action == 1:
            # 执行定位逻辑...
            self._execute_localization()

        # 示例：简单的奖励 (离目标越近奖励越高)
        target_sensor_idx = discrete_action[0]
        target_pos = self.sensor_states[target_sensor_idx, :2]  # 假设前两位是位置
        distance = np.linalg.norm(self.drone_position - target_pos)
        reward = -distance  # 距离越近，负奖励越小

        self.current_step += 1

        # --- 3. 判断 episode 是否结束 ---
        # 如果达到目标，或者发生碰撞等
        terminated = bool(distance < 0.05)
        # 如果达到最大步数
        truncated = bool(self.current_step >= self.max_steps_per_episode)

        # --- 4. 获取下一步的观测和信息 ---
        observation = self._get_obs()
        info = self._get_info()

        if terminated:
            reward += 10  # 到达目标的额外奖励
            print(f"目标达成！奖励: {reward}")

        return observation, reward, terminated, truncated, info