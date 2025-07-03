# 文件名: env/DroneNavigationEnv.py
# 版本: AttributeError 最终修复版

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from env.env_utils import poisson_disk_sampling
from env.uncertain_model import UncertaintyModel
from env.maneuver_controllers import SpiralManeuverController

class DroneNavigationEnv(gym.Env):
    """
    【AttributeError 最终修复版】
    - 修复了所有未初始化属性的错误，包括 snr_threshold_linear 和 drone_height。
    - 统一了状态获取方式，始终从 self.uncertainty_model 获取最新状态。
    - 保留了HRL状态机。
    """
    metadata = {'render_modes': []}

    def __init__(self, config):
        super().__init__()
        # --- 环境和模型参数 ---

        self.maneuver_reward_bonus = 30.0  # 可以调整这个值

        self.num_sensors = config.get("num_sensors", 5)
        self.area_size = config.get("area_size", (1000.0, 1000.0))
        self.solo_SN_data = config.get("solo_SN_data", 6e7)
        self.max_steps_per_episode = config.get("max_steps_per_episode", 500)
        self.MANEUVER_TOTAL_STEPS = 5

        # --- 观测与动作空间定义 ---
        self.global_state_dim = 4
        self.per_sensor_dim = 6
        self.observation_dim = self.global_state_dim + self.num_sensors * self.per_sensor_dim
        self.observation_space = spaces.Dict({
            "vector": spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32)
        })
        self.action_space = spaces.Dict({
            "discrete": spaces.Discrete(2),
            "continuous": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        })
        self.real_action_bounds = {'direction': {'low': -np.pi, 'high': np.pi}, 'speed': {'low': 0.0, 'high': 30.0}}

        # --- 环境物理参数 ---
        self.drone_height = 60.0  # 确保无人机高度被定义
        self.max_speed = 30.0
        self.time_slot = 1.0

        # --- 通信参数 ---
        self.bandwidth = 1e6
        self.path_loss_exponent = 2.0
        self.reference_distance = 1.0
        self.reference_loss = -60
        self.transmit_power = 0.1
        self.noise_power = 1e-12
        self.snr_threshold = 2.0  # dB

        # 【BUG修复】: 根据dB值计算并存储线性阈值
        self.snr_threshold_linear = 10 ** (self.snr_threshold / 10.0)

        # --- 定位模型 ---
        self.uncertainty_model = UncertaintyModel(num_sensors=self.num_sensors)

        # --- 状态变量 ---
        self.drone_position = None
        self.drone_velocity = None
        self.sensor_true_positions = None
        self.sensor_data_amounts = None
        self.current_step = 0

        # --- HRL状态机变量 ---
        self.current_mode = 'APPROACHING'
        self.maneuver_target_sensor_id = -1
        self.maneuver_steps_left = 0

        # 【新增】用于在 _calculate_reward 内部比较的状态属性
        self.last_dist_to_closest_target = None
        self.radii_before_action = None

        # 【新增】传感器级别的稳定状态标志
        self.is_sensor_stable = np.full(self.num_sensors, False, dtype=bool)

        self.spiral_controller = SpiralManeuverController(
            total_speed=20.0,
            start_radius=150.0,  # 可调参数
            min_radius=20.0,  # 可调参数
            time_slot=self.time_slot
        )

        # --- 新的双重稳定状态追踪 ---
        # 机制1: 软稳定 (用于引导通信)
        self.soft_stable_threshold = 3
        self.soft_radius_change_threshold = 1.0
        self.is_sensor_soft_stable = np.full(self.num_sensors, False, dtype=bool)
        self.radius_stable_counters_soft = np.zeros(self.num_sensors, dtype=int)
        # 机制2: 硬稳定 (用于全局屏蔽)
        self.hard_stable_threshold = 5
        self.hard_radius_change_threshold = 0.5
        self.is_sensor_hard_stable = np.full(self.num_sensors, False, dtype=bool)
        self.radius_stable_counters_hard = np.zeros(self.num_sensors, dtype=int)
        # ----------------------------
        # --- 日志 ---
        self.trajectory, self.communication_log, self.localization_log = [], [], []
        self.trajectory_save_freq = 5

        print("环境已初始化 (AttributeError 最终修复版)！")

    def _unnormalize_action(self, norm_action):
        low_dir, high_dir = self.real_action_bounds['direction']['low'], self.real_action_bounds['direction']['high']
        real_direction = low_dir + (norm_action[0] + 1.0) * 0.5 * (high_dir - low_dir)
        low_speed, high_speed = self.real_action_bounds['speed']['low'], self.real_action_bounds['speed']['high']
        real_speed = low_speed + (norm_action[1] + 1.0) * 0.5 * (high_speed - low_speed)
        return real_direction, real_speed

    def _calculate_reward(self):
        """
                计算当前时间步的总奖励，包含所有惩罚项和正向塑形奖励。
                """
        est_pos = self.uncertainty_model.estimated_positions
        est_radii = self.uncertainty_model.uncertainty_radii

        # --- 1. 基础惩罚项 (距离惩罚和延迟惩罚) ---
        eligible_mask = self.sensor_data_amounts > 0
        current_min_dist = np.linalg.norm(self.area_size)
        if np.any(eligible_mask):
            distances = np.linalg.norm(self.drone_position - est_pos[eligible_mask], axis=1)
            current_min_dist = np.min(distances)

        dist_penalty = current_min_dist / np.linalg.norm(self.area_size)
        delay_penalty = np.sum(self.sensor_data_amounts) / (self.solo_SN_data * self.num_sensors)
        base_reward = -1.0 * dist_penalty - 0.5 * delay_penalty

        # --- 2. 接近奖励 ---
        approach_reward = 0
        if self.last_dist_to_closest_target is not None and np.any(eligible_mask):
            distance_diff = self.last_dist_to_closest_target - current_min_dist
            approach_reward_coeff = 0.01
            approach_reward = distance_diff * approach_reward_coeff

        # --- 3. 不确定性降低奖励 ---
        localization_reward = 0
        if self.radii_before_action is not None:
            radii_after_action = est_radii.copy()
            radius_reduction = np.sum(self.radii_before_action - radii_after_action)
            localization_reward_coeff = 0.1
            localization_reward = max(0, radius_reduction * localization_reward_coeff)

        # --- 4. 整合所有奖励 ---
        total_reward = base_reward + approach_reward + localization_reward

        # --- 5. 更新用于下一次比较的状态 ---
        self.last_dist_to_closest_target = current_min_dist

        return total_reward

    def _get_obs(self):
        est_pos = self.uncertainty_model.estimated_positions
        est_radii = self.uncertainty_model.uncertainty_radii
        global_part = np.array([*self.drone_velocity, *self.drone_position], dtype=np.float32)
        sensor_parts = []
        for i in range(self.num_sensors):
            rel_vec = est_pos[i] - self.drone_position
            sensor_part = np.array([rel_vec[0], rel_vec[1], self.sensor_data_amounts[i],
                                    est_pos[i, 0], est_pos[i, 1], est_radii[i]], dtype=np.float32)
            sensor_parts.append(sensor_part)
        final_vector = np.concatenate([global_part] + sensor_parts, axis=0)
        return {"vector": final_vector}

    def _get_info(self):
        return {"steps": self.current_step, "current_mode": self.current_mode}

    def _get_action_mask(self):
        """
        【全新完整实现】计算当前步的高级动态动作掩码。
        """
        # --- 1. 离散动作掩码 ---
        mask_discrete = np.ones((1, 2), dtype=np.int8)  # 默认都允许 [通信, 定位]

        # 检查所有需要收集数据的传感器
        unfinished_mask = self.sensor_data_amounts > 0

        # 如果没有任何需要收集数据的传感器，则禁止定位
        if not np.any(unfinished_mask):
            mask_discrete[0, 1] = 0
        else:
            # 获取所有未完成任务的传感器的硬稳定状态
            hard_stable_status_of_unfinished = self.is_sensor_hard_stable[unfinished_mask]

            # **全局硬屏蔽逻辑**: 如果所有未完成的传感器都已达到硬稳定，则禁止“定位”
            if np.all(hard_stable_status_of_unfinished):
                mask_discrete[0, 1] = 0

        # --- 2. 连续动作掩码 ---
        mask_continuous = np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=np.float32)  # 默认全范围

        # **通信时方向限制逻辑**:
        # 这个逻辑应该只在“智能体将要执行通信”这个前提下应用。
        # 我们可以在 _get_obs 时就计算好，智能体在选择动作时会看到这个限制。
        if self._is_in_comm_range():
            est_pos = self.uncertainty_model.estimated_positions
            distances = np.linalg.norm(self.drone_position - est_pos, axis=1)
            # 只考虑需要通信的目标
            distances[~unfinished_mask] = np.inf

            if not np.all(np.isinf(distances)):
                target_idx = np.argmin(distances)
                target_vec = est_pos[target_idx] - self.drone_position
                target_angle = np.arctan2(target_vec[1], target_vec[0])

                angle_allowance = np.pi / 6.0  # 30度
                min_angle = target_angle - angle_allowance
                max_angle = target_angle + angle_allowance

                # 归一化到 [-1, 1]
                low_b, high_b = self.real_action_bounds['direction']['low'], self.real_action_bounds['direction'][
                    'high']
                norm_min = (min_angle - low_b) / (high_b - low_b) * 2 - 1
                norm_max = (max_angle - low_b) / (high_b - low_b) * 2 - 1

                mask_continuous[0, 0] = np.clip(norm_min, -1.0, 1.0)
                mask_continuous[0, 1] = np.clip(norm_max, -1.0, 1.0)

        return mask_discrete, mask_continuous

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 重置所有稳定性相关的状态
        self.is_sensor_soft_stable.fill(False)
        self.radius_stable_counters_soft.fill(0)
        self.is_sensor_hard_stable.fill(False)
        self.radius_stable_counters_hard.fill(0)

        self.trajectory, self.communication_log, self.localization_log = [], [], []
        self.current_step = 0

        self.current_mode = 'APPROACHING'
        self.maneuver_target_sensor_id = -1
        self.maneuver_steps_left = 0

        margin = 100.0
        w, h = self.area_size[0] - 2 * margin, self.area_size[1] - 2 * margin
        points = poisson_disk_sampling((w, h), self.num_sensors, 250.0, self.np_random)
        self.sensor_true_positions = points + np.array([margin, margin])

        self.uncertainty_model.initialize_states(self.sensor_true_positions, 100.0, self.np_random)
        self.sensor_data_amounts = np.full(self.num_sensors, self.solo_SN_data)

        self.drone_position = np.array([200.0, 200.0])
        self.drone_velocity = np.zeros(2)
        self.trajectory.append(self.drone_position.copy())

        # 计算初始距离，但不计算奖励
        eligible_mask = self.sensor_data_amounts > 0
        if np.any(eligible_mask):
            distances = np.linalg.norm(self.drone_position - self.uncertainty_model.estimated_positions[eligible_mask],
                                       axis=1)
            self.last_dist_to_closest_target = np.min(distances)
        else:
            self.last_dist_to_closest_target = np.linalg.norm(self.area_size)



        return self._get_obs(), self._get_info()

    def _is_in_comm_range(self):
        est_pos = self.uncertainty_model.estimated_positions
        est_radii = self.uncertainty_model.uncertainty_radii
        eligible_mask = self.sensor_data_amounts > 0
        if not np.any(eligible_mask): return False

        ref_gain_linear = 10 ** (self.reference_loss / 10.0)
        gain_const = self.transmit_power * ref_gain_linear * (self.reference_distance ** self.path_loss_exponent)

        for idx in np.where(eligible_mask)[0]:
            dist_center = np.linalg.norm(self.drone_position - est_pos[idx])
            radii = min(est_radii[idx], 100)
            worst_h_dist = dist_center + radii
            worst_3d_dist = np.sqrt(worst_h_dist ** 2 + self.drone_height ** 2) + 1e-6
            path_loss = (1 / worst_3d_dist) ** self.path_loss_exponent
            rx_power_est = gain_const * path_loss
            snr_est = rx_power_est / self.noise_power
            # 现在 self.snr_threshold_linear 存在了
            if snr_est >= self.snr_threshold_linear:
                return True
        return False

    def _execute_approach_policy(self):
        est_pos = self.uncertainty_model.estimated_positions
        eligible_mask = self.sensor_data_amounts > 0
        if not np.any(eligible_mask):
            self.drone_velocity = np.zeros(2)
        else:
            eligible_pos = est_pos[eligible_mask]
            distances = np.linalg.norm(self.drone_position - eligible_pos, axis=1)
            target_pos = eligible_pos[np.argmin(distances)]
            direction_vec = target_pos - self.drone_position
            real_direction = np.arctan2(direction_vec[1], direction_vec[0])
            self.drone_velocity = np.array(
                [self.max_speed * np.cos(real_direction), self.max_speed * np.sin(real_direction)])

        self.drone_position += self.drone_velocity * self.time_slot
        self.drone_position = np.clip(self.drone_position, [0, 0], self.area_size)
        if self._is_in_comm_range():
            self.current_mode = 'DECIDING'

    def _execute_maneuver_policy(self):
        target_id = self.maneuver_target_sensor_id
        # =================== 补丁 2 开始 ===================
        # 检查目标ID是否有效，以及机动步数是否用尽
        if target_id == -1 or self.maneuver_steps_left < 0:
            self.current_mode = 'DECIDING'
            return False

        # 新的“软稳定”判断：即使目标稳定，也允许至少执行一步机动。
        # 只有在第二次及以后进入时，才检查软稳定状态。
        is_stable = self.is_sensor_soft_stable[target_id]
        if is_stable and self.maneuver_steps_left < self.MANEUVER_TOTAL_STEPS - 1:
            self.current_mode = 'DECIDING'
            return False
        # =================== 补丁 2 结束 ===================
        # --- 1. 递减步数 ---
        self.maneuver_steps_left -= 1
        target_pos = self.uncertainty_model.estimated_positions[target_id]

        # 调用螺旋控制器计算移动向量
        move_vector = self.spiral_controller.calculate_move_vector(
            self.drone_position, target_pos
        )

        # 如果控制器返回零向量（表示已完成），则切换模式
        if np.linalg.norm(move_vector) < 1e-6:
            self.current_mode = 'DECIDING'
            return

        # 更新无人机位置和速度
        self.drone_position += move_vector
        self.drone_velocity = move_vector / self.time_slot
        self.drone_position = np.clip(self.drone_position, [0, 0], self.area_size)

        self._execute_localization()
        return True

    def _execute_communication(self):
        """
            【优化版】
            - 移除了多余的SNR判断。
            - 明确了“只与最近的可通信目标”通信的逻辑。
            """
        est_pos = self.uncertainty_model.estimated_positions
        est_radii = self.uncertainty_model.uncertainty_radii
        eligible_mask = self.sensor_data_amounts > 0

        if not np.any(eligible_mask):
            return  # 如果没有需要通信的目标，直接返回

        # --- 核心修改部分 ---

        # 1. 计算到所有有效目标的最坏情况3D距离
        dist_to_center = np.linalg.norm(self.drone_position - est_pos, axis=1)
        max_h_dist = dist_to_center + est_radii
        max_3d_dist = np.sqrt(max_h_dist ** 2 + self.drone_height ** 2)

        # 2. 将无效目标（数据已采完）的距离设为无穷大
        dist_to_consider = np.where(eligible_mask, max_3d_dist, np.inf)

        # 3. 如果不存在任何有效目标，则返回
        if np.all(np.isinf(dist_to_consider)):
            return

        # 4. 找到距离最近的有效目标
        target_idx = np.argmin(dist_to_consider)

        # 5. 【移除冗余判断】
        # 我们在这里假设，既然被允许执行通信，那么这个最近的目标一定是可通信的。
        # 因此，直接计算实际的吞吐量并传输数据。
        # 之前的 if snr_est_lin >= self.snr_threshold_linear: 判断被移除。

        # 计算真实SNR和吞吐量
        ref_gain_lin = 10 ** (self.reference_loss / 10)
        true_h_dist = np.linalg.norm(self.drone_position - self.sensor_true_positions[target_idx])
        true_3d_dist = np.sqrt(true_h_dist ** 2 + self.drone_height ** 2)

        # 防止除以零
        if true_3d_dist < 1e-6: true_3d_dist = 1e-6

        path_loss_true = ref_gain_lin * (self.reference_distance / true_3d_dist) ** self.path_loss_exponent
        snr_lin = (self.transmit_power * path_loss_true) / self.noise_power

        # 注意：如果SNR低于某个极小值，log2(1+snr)可能为负或0，导致无数据传输，这是正常的。
        throughput = self.bandwidth * np.log2(1 + snr_lin)
        transmitted_data = throughput * self.time_slot

        # 确保传输数据不为负
        if transmitted_data < 0:
            transmitted_data = 0.0

        # 更新数据量
        self.sensor_data_amounts[target_idx] = max(0, self.sensor_data_amounts[target_idx] - transmitted_data)

        # 记录日志
        self.communication_log.append({
            'step': self.current_step, 'target_sensor': target_idx,
            'transmitted_data_Mbits': transmitted_data / 1e6,
            'remaining_data': self.sensor_data_amounts.copy()
        })

    def _execute_localization(self):
        true_h_dists = np.linalg.norm(self.drone_position - self.sensor_true_positions, axis=1)
        variances = self.uncertainty_model.g0 * (true_h_dists ** 2)
        std_devs = np.sqrt(variances)
        measured_dists = self.np_random.normal(loc=true_h_dists, scale=std_devs)

        for i in range(self.num_sensors):
            self.uncertainty_model.g_mgf_update(
                sensor_id=i,
                drone_position=self.drone_position.copy(),
                measured_distance=measured_dists[i],
                measurement_variance=variances[i]
            )

        changes = np.abs(self.radii_before_action - self.uncertainty_model.uncertainty_radii.copy())
        for i in range(self.num_sensors):
            # 更新软稳定
            if changes[i] < self.soft_radius_change_threshold:
                self.radius_stable_counters_soft[i] += 1
            else:
                self.radius_stable_counters_soft[i] = 0
                self.is_sensor_soft_stable[i] = False
            if self.radius_stable_counters_soft[i] >= self.soft_stable_threshold:
                self.is_sensor_soft_stable[i] = True

            # 更新硬稳定
            if changes[i] < self.hard_radius_change_threshold:
                self.radius_stable_counters_hard[i] += 1
            else:
                self.radius_stable_counters_hard[i] = 0
                self.is_sensor_hard_stable[i] = False
            if self.radius_stable_counters_hard[i] >= self.hard_stable_threshold:
                self.is_sensor_hard_stable[i] = True

        self.localization_log.append({
            'step': self.current_step,
            'est_positions': self.uncertainty_model.estimated_positions.copy(),
            'est_radii': self.uncertainty_model.uncertainty_radii.copy()
        })


    def step(self, action):
        self.current_step += 1

        # 定义一个变量来捕获本次step是否执行了有效机动
        maneuver_executed_successfully = False

        # 核心逻辑：如果一个传感器的数据被采集完了，那么它就不再是“软稳定”状态，
        # 因为我们不再关心它的临时稳定性了。这为下一轮可能的任务做准备。
        # 硬稳定状态一旦达成，则保持不变。
        data_finished_mask = self.sensor_data_amounts <= 1e-3
        self.is_sensor_soft_stable[data_finished_mask] = False
        self.radius_stable_counters_soft[data_finished_mask] = 0

        # 【核心修改】在执行任何动作之前，记录状态快照
        self.radii_before_action = self.uncertainty_model.uncertainty_radii.copy()

        if self.current_mode == 'DECIDING' and not self._is_in_comm_range():
            # 如果飞出了决策范围，强制切换回接近模式
            self.current_mode = 'APPROACHING'

        if self.current_mode == 'APPROACHING':
            self._execute_approach_policy()

        elif self.current_mode == 'EXECUTING_MANEUVER':
            self._execute_maneuver_policy()

        elif self.current_mode == 'DECIDING':
            discrete_action = action["discrete"]
            real_direction, real_speed = self._unnormalize_action(action["continuous"])

            self.drone_velocity = np.array([real_speed * np.cos(real_direction), real_speed * np.sin(real_direction)])
            self.drone_position += self.drone_velocity * self.time_slot
            self.drone_position = np.clip(self.drone_position, [0, 0], self.area_size)

            if discrete_action == 0:
                self._execute_communication()
            elif discrete_action == 1:

                eligible_mask = self.sensor_data_amounts > 0
                if np.any(eligible_mask):
                    # 计算到所有未完成任务的传感器的距离
                    distances = np.linalg.norm(self.drone_position - self.uncertainty_model.estimated_positions, axis=1)
                    distances[~eligible_mask] = np.inf  # 屏蔽已完成的
                    self.maneuver_target_sensor_id = np.argmin(distances)
                else:
                    # 如果没有目标了，则设为-1，机动会立即退出
                    self.maneuver_target_sensor_id = -1

                # 2. 然后再切换模式并设置步数
                self.current_mode = 'EXECUTING_MANEUVER'
                self.maneuver_steps_left = self.MANEUVER_TOTAL_STEPS

                maneuver_executed_successfully = self._execute_maneuver_policy()

                eligible_mask = self.sensor_data_amounts > 0
                if np.any(eligible_mask):
                    distances = np.linalg.norm(self.drone_position - self.uncertainty_model.estimated_positions, axis=1)
                    distances[~eligible_mask] = np.inf
                    self.maneuver_target_sensor_id = np.argmin(distances)
                else:
                    self.maneuver_target_sensor_id = -1



        # --- 奖励计算 ---
        # 现在，所有计算都在这个函数内部完成
        reward = self._calculate_reward()
        if maneuver_executed_successfully:
            reward += self.maneuver_reward_bonus
        # --- 公共逻辑 (轨迹记录, 终止判断) ---
        if self.current_step % self.trajectory_save_freq == 0:
            self.trajectory.append(self.drone_position.copy())
        all_data_collected = np.all(self.sensor_data_amounts <= 1e-3)
        max_steps_reached = self.current_step >= self.max_steps_per_episode
        terminated = bool(all_data_collected)
        truncated = bool(max_steps_reached)

        # 最终任务奖励/惩罚
        if terminated: reward += 200.0
        if truncated: reward -= 500.0

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info