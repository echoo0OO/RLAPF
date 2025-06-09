import gymnasium as gym
from gymnasium import spaces
import numpy as np

from env.env_utils import  poisson_disk_sampling
from env.reward_shaping import reward_function_apf


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
        self.max_steps_per_episode = config.get("max_steps", 1000)

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
        self.path_loss_exponent = 2
        self.reference_distance = 1.0 # m
        self.reference_loss = -60  # dB
        self.transmit_power = 0.1  # W
        self.noise_power = 1e-11  # W (-110dBm)
        self.snr_threshold = 10.0  # dB
        # 状态变量
        # --- 状态变量 ---
        # 这些变量会在reset时被正确初始化
        self.drone_position = None
        self.sensor_true_positions = None
        self.sensor_estimated_positions = None
        self.sensor_estimated_radii = None
        self.sensor_data_amounts = None
        self.sensor_states = None
        self.reward_map = None
        self.current_step = 0

        # 为地图生成预先计算网格坐标
        self.x_map_coords = np.linspace(0, self.area_size[0], self.img_w)
        self.y_map_coords = np.linspace(0, self.area_size[1], self.img_h)
        self.X_mesh, self.Y_mesh = np.meshgrid(self.x_map_coords, self.y_map_coords)

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

        # 获取动作掩码，例如，如果某个传感器数据已满，则不可选
        action_mask = np.ones((self.action_dis_dim, self.action_dis_len), dtype=np.int8)
        # ... 更新掩码的逻辑 ...

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
        self.sensor_estimated_radii = np.full(self.num_sensors, initial_radius)

        # b. 根据真实位置和半径生成初始估计位置
        self.sensor_estimated_positions = np.array([
            true_pos + self.np_random.uniform(-initial_radius / 2, initial_radius / 2, 2)
            for true_pos in self.sensor_true_positions
        ])

        # c. 初始化传感器数据量
        self.sensor_data_amounts = np.full(self.num_sensors, self.solo_SN_data)

        # 4. 初始化无人机状态
        self.drone_position = np.array([0.0, 0.0])

        # 5. 构建符合观测空间的'sensors'状态矩阵
        # 格式: [est_x, est_y, uncertainty_radius, data_volume]
        self.sensor_states = np.concatenate([
            self.sensor_estimated_positions,
            self.sensor_estimated_radii.reshape(-1, 1),  # reshape为列向量
            self.sensor_data_amounts.reshape(-1, 1)  # reshape为列向量
        ], axis=1).astype(np.float32)

        # 6. 初始化奖励地图 (源自 show5_3APF.py)
        # a. 为每个传感器生成一个APF奖励层和一个全1的GDOP遮罩层
        all_sensor_rewards = []
        for i in range(self.num_sensors):
            xc, yc = self.sensor_estimated_positions[i]

            # 计算APF奖励表面
            # R_max和r_threshold可以作为配置传入或在这里定义
            reward_layer = reward_function_apf(
                self.X_mesh, self.Y_mesh, R_max=100.0, r_threshold=100.0, xc=xc, yc=yc
            )

            # 初始GDOP遮罩为全1
            gdop_mask_layer = np.ones((self.img_h, self.img_w), dtype=np.float32)

            # 应用遮罩
            masked_reward = reward_layer * gdop_mask_layer
            all_sensor_rewards.append(masked_reward)

        # b. 将所有传感器的奖励层融合成一个单一的奖励地图
        # 使用np.maximum，地图上每个点的值等于它能从所有传感器中获得的最大奖励
        if all_sensor_rewards:
            self.reward_map = np.maximum.reduce(all_sensor_rewards).astype(np.float32)
        else:  # 以防万一没有传感器
            self.reward_map = np.zeros((self.img_h, self.img_w), dtype=np.float32)

        # 7. 获取最终的观测和信息
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """
        在环境中执行一步。
        """
        # --- 1. 解包混合动作 ---
        discrete_action = action["discrete"]
        normalized_continuous_action = action["continuous"]

        # 将归一化动作转换为真实动作
        real_direction, real_speed = self._unnormalize_action(normalized_continuous_action)

        # --- 2. 在这里实现您的核心环境动力学 ---
        # ... 根据 discrete_action 和 continuous_action 更新环境状态 ...
        # (例如，更新无人机位置、更新传感器数据量、计算奖励)
        if discrete_action == 0:
            # 执行通信逻辑...
            pass
        elif discrete_action == 1:
            # 执行定位逻辑...
            pass

        # 示例：简单的移动
        move_vec = (continuous_action / 10.0)  # 缩放动作效果
        self.drone_position += move_vec
        self.drone_position = np.clip(self.drone_position, 0, 1)  # 限制在边界内

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