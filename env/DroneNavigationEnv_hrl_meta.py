# 文件名: env/DroneNavigationEnv.py
# 版本: HRL 增强终止条件最终版 (APPROACH任务同时考虑通信范围和定位几何)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from env.ekf_model import EKF_Model
from env.env_utils import poisson_disk_sampling
from env.maneuver_controllers import InformationDrivenManeuverController


class DroneNavigationEnv(gym.Env):
    """
    【HRL 增强终止条件最终版】
    - APPROACH 任务的完成条件被增强为：必须同时满足“在通信范围内”和“对目标的定位几何构型良好”。
    - 这使得HRL的决策更具前瞻性，为后续的COMMUNICATE和LOCALIZE任务创造更有利的条件。
    """
    metadata = {'render_modes': []}
    # 定义高层任务的ID
    TASK_APPROACH = 0
    TASK_COMMUNICATE = 1
    TASK_LOCALIZE = 2
    NUM_HIGH_LEVEL_TASKS = 3
    def __init__(self, config, low_level_agent):
        super().__init__()
        # 存储传入的低层智能体，用于在step中生成低层动作
        self.low_level_agent_controller = low_level_agent
        self.sector_coverage_distance_thresh = config.get("sector_coverage_distance_thresh", 400.0)
        # --- 环境和模型参数 ---
        self.num_sensors = config.get("num_sensors", 5)
        self.area_size = config.get("area_size", (1000.0, 1000.0))
        self.solo_SN_data = config.get("solo_SN_data", 2e7)
        self.max_steps_per_episode = config.get("max_steps_per_episode", 1000)

        # 现在这是一个最大承诺步数
        self.max_task_commitment_steps = config.get("max_task_commitment_steps", 5)

        # 为每个任务定义灵活终止的阈值
        self.comm_termination_data_thresh = 1e-3  # 数据剩余量低于此值算完成

        # 为APPROACH任务的几何构型部分定义终止条件
        self.approach_termination_sector_coverage = config.get("approach_termination_sector_coverage", 11)
        self.num_sectors_for_check = 1210

        # --- 观测空间定义 ---
        self.global_state_dim = 4
        self.per_sensor_dim = 6
        self.Condition = 3
        self.observation_dim = self.global_state_dim + self.num_sensors * self.per_sensor_dim + self.Condition
        self.observation_space = spaces.Dict({
            "vector": spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32)
        })

        # 动作空间现在是高层的离散动作空间
        self.action_space = spaces.Discrete(self.NUM_HIGH_LEVEL_TASKS)

        # 定义低层连续动作的真实物理边界
        self.real_action_bounds = {'direction': {'low': -np.pi, 'high': np.pi}, 'speed': {'low': 0.0, 'high': 30.0}}

        # --- 环境物理和通信参数 ---
        self.drone_height = 60.0
        self.max_speed = 30.0
        self.time_slot = 1.0
        self.bandwidth = 1e6
        self.path_loss_exponent = 2.0
        self.reference_distance = 1.0
        self.reference_loss = -60
        self.transmit_power = 0.1
        self.noise_power = 1e-12
        self.snr_threshold = 2.0
        self.snr_threshold_linear = 10 ** (self.snr_threshold / 10.0)

        # --- 定位模型 ---
        self.uncertainty_model = EKF_Model(num_sensors=self.num_sensors)

        # --- 状态变量 ---
        self.drone_position = None
        self.drone_velocity = None
        self.sensor_true_positions = None
        self.sensor_data_amounts = None
        self.current_step = 0
        self.last_dist_to_closest_target = None
        self.radii_before_action = None
        self.is_sensor_hard_stable = np.full(self.num_sensors, False, dtype=bool)
        self.maneuver_controller = InformationDrivenManeuverController(
            total_speed=self.max_speed,  # 使用环境的最大速度
            optimal_radius=config.get("maneuver_optimal_radius", 60.0),  # 期望的绕飞半径
            time_slot=self.time_slot
        )
        # --- 日志 ---
        self.trajectory, self.communication_log, self.localization_log = [], [], []

    def _unnormalize_action(self, norm_action):
        low_dir, high_dir = self.real_action_bounds['direction']['low'], self.real_action_bounds['direction']['high']
        real_direction = low_dir + (norm_action[0] + 1.0) * 0.5 * (high_dir - low_dir)
        low_speed, high_speed = self.real_action_bounds['speed']['low'], self.real_action_bounds['speed']['high']
        real_speed = low_speed + (norm_action[1] + 1.0) * 0.5 * (high_speed - low_speed)
        return real_direction, real_speed

    def _calculate_reward(self,transmitted_data_in_bits=0.0,high_level_action=None, condition_obs=None):
        """
               【最终优化版 - 简洁高效】
               - 移除状态导向的定位惩罚，因为它在当前环境下很快失效。
               - 强化对“行为”的直接奖励：成功通信 和 成功定位（降低不确定性）。
               - 保持对时间和任务延迟的基础惩罚。
               """
        # --- 1. 基础成本惩罚 ---
        # a. 时间成本：激励智能体用更少的步数完成任务。
        time_step_penalty = -0.4
        # b. 任务延迟成本：剩余数据总量越大，惩罚越大。
        delay_penalty = -0.5 * (np.sum(self.sensor_data_amounts) / (self.solo_SN_data * self.num_sensors))
        data_transmission_reward = 1.2 * (transmitted_data_in_bits / 1e6)  # 每传输1Mbit，奖励 1.0
        total_true_error = 0.0
        for i in range(self.num_sensors):
            # 获取传感器的真实位置
            true_pos = self.sensor_true_positions[i]
            # 获取传感器的估计位置
            est_pos = self.uncertainty_model.estimated_positions[i]
            # 计算欧氏距离作为真实误差
            error = np.linalg.norm(true_pos - est_pos)
            total_true_error += error
        localization_penalty = -0.5*total_true_error/250.0
        action_match_reward = 0.0
        if high_level_action is not None and condition_obs is not None and len(condition_obs) == 3:
            cond_not_in_comm_range = condition_obs[0]
            cond_comm_task_pending_and_in_range = condition_obs[1]
            cond_localize_task_pending_and_in_range = condition_obs[2]

            # 规则1: 如果不在通信范围内 (cond_not_in_comm_range为1) 且选择了 APPROACH
            if cond_not_in_comm_range == 1.0 and high_level_action == self.TASK_APPROACH:
                action_match_reward += 1.0  # 奖励1

            # 规则2: 如果在通信范围内且通信任务未完成 (cond_comm_task_pending_and_in_range为1) 且选择了 COMMUNICATE
            if cond_comm_task_pending_and_in_range == 1.0 and high_level_action == self.TASK_COMMUNICATE:
                action_match_reward += 1.0  # 奖励1

            # 规则3: 如果在通信范围内且定位任务未完成 (cond_localize_task_pending_and_in_range为1) 且选择了 LOCALIZE
            if cond_localize_task_pending_and_in_range == 1.0 and high_level_action == self.TASK_LOCALIZE:
                action_match_reward += 1.0  # 奖励1
        # --- 3. 整合总奖励 ---
        total_reward = (time_step_penalty +
                        delay_penalty +
                        data_transmission_reward +
                        localization_penalty+
                        action_match_reward)
        # 更新状态以备下次计算（这部分不是奖励，但在这里计算很方便）
        est_pos = self.uncertainty_model.estimated_positions
        eligible_mask_for_dist = self.sensor_data_amounts > self.comm_termination_data_thresh
        if np.any(eligible_mask_for_dist):
            current_min_dist = np.min(np.linalg.norm(self.drone_position - est_pos[eligible_mask_for_dist], axis=1))
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
        target_id, _, _ = self._get_current_target_info()

        # 初始化布尔条件
        # 第一个布尔值：当不在通信范围内时为1，否则为0
        # 第二个布尔值：当处于通信范围内且通信任务未完成时为1，否则为0
        # 第三个布尔值：当处于通信范围内且定位任务未完成时为1，否则为0

        cond_not_in_comm_range = 0.0
        cond_comm_task_pending_and_in_range = 0.0
        cond_localize_task_pending_and_in_range = 0.0

        if target_id != -1:  # 确保存在一个有效目标
            is_in_comm_range = self._is_in_comm_range()  # 调用现有函数获取通信范围判断

            if not is_in_comm_range:
                cond_not_in_comm_range = 1.0  # 状态1：远离，需要APPROACH
            else:  # 处于通信范围内
                # 检查当前主要目标的子任务是否都已完成
                is_comm_task_done = self.sensor_data_amounts[target_id] <= self.comm_termination_data_thresh
                num_covered_sectors = self.uncertainty_model.get_sector_coverage(
                    sensor_id=target_id,
                    num_sectors=self.num_sectors_for_check
                )
                is_localize_task_done = (num_covered_sectors >= self.approach_termination_sector_coverage)
                # 状态2：在可操作区域内，且当前目标有事可做
                if not is_comm_task_done:  # 如果通信任务未完成
                    cond_comm_task_pending_and_in_range = 1.0
                if not is_localize_task_done:  # 如果定位任务未完成
                    cond_localize_task_pending_and_in_range = 1.0
        # 将这些布尔条件（转换为浮点数）打包成一个 NumPy 数组
        condition_part = np.array([
            cond_not_in_comm_range,
            cond_comm_task_pending_and_in_range,
            cond_localize_task_pending_and_in_range
        ], dtype=np.float32)
        # 最终的观测向量
        final_vector = np.concatenate([global_part] + sensor_parts + [condition_part], axis=0)
        return {"vector": final_vector}


    def _get_info(self):
        return {"steps": self.current_step}

    def _get_current_target_info(self):
        """
        【修正版】辅助函数，获取当前最近的、有任何未完成子任务的传感器ID和位置。
        一个传感器被视为“未完成”，只要它的通信任务 或 定位任务中至少有一个没有完成。
        """
        # 1. 检查每个传感器的通信任务是否完成
        comm_done_mask = self.sensor_data_amounts <= self.comm_termination_data_thresh
        # 2. 检查每个传感器的定位任务是否完成
        localize_done_mask = np.full(self.num_sensors, False, dtype=bool)
        for i in range(self.num_sensors):
            num_covered_sectors = self.uncertainty_model.get_sector_coverage(
                sensor_id=i,
                num_sectors=self.num_sectors_for_check
            )
            if num_covered_sectors >= self.approach_termination_sector_coverage:
                localize_done_mask[i] = True
        # 3. 确定“完全完成”的传感器：通信和定位都已完成
        fully_done_mask = comm_done_mask & localize_done_mask
        # 4. “有效”的传感器是那些“不完全完成”的
        eligible_mask = ~fully_done_mask
        # 如果所有传感器都已完全完成，则没有有效目标
        if not np.any(eligible_mask):
            return -1, None, np.inf
        # 5. 在所有有效目标中，找到距离最近的那个
        est_pos = self.uncertainty_model.estimated_positions
        distances = np.linalg.norm(self.drone_position - est_pos, axis=1)
        # 将无效目标（已完全完成的）的距离设为无穷大
        distances[~eligible_mask] = np.inf
        # 如果由于某种原因（例如所有有效目标都在无穷远处），则返回无目标
        if np.all(np.isinf(distances)):
            return -1, None, np.inf
        target_idx = np.argmin(distances)
        return target_idx, est_pos[target_idx], distances[target_idx]

    def _check_task_completion(self, task_id, target_sensor_id):
        """检查当前高层任务是否已经可以提前终止，增强了APPROACH的条件"""
        if target_sensor_id == -1:
            return True

        if task_id == self.TASK_APPROACH:
            # 复合条件: 1. 在通信范围内 AND 2. 几何构型良好
            is_in_range = self._is_in_comm_range()
            return is_in_range

        elif task_id == self.TASK_COMMUNICATE:
            return self.sensor_data_amounts[target_sensor_id] <= self.comm_termination_data_thresh

        elif task_id == self.TASK_LOCALIZE:
            # 【核心修复】定位任务的完成，由扇区覆盖决定
            num_covered_sectors = self.uncertainty_model.get_sector_coverage(
                sensor_id=target_sensor_id,
                num_sectors=self.num_sectors_for_check
            )
            return num_covered_sectors >= self.approach_termination_sector_coverage

        return False

    def get_high_level_action_mask(self):
        """
        【审视与微调】
        基于新的 _get_current_target_info，重新检查此函数的逻辑。
        """
        mask = np.ones(self.NUM_HIGH_LEVEL_TASKS, dtype=np.int8)  # [APPROACH, COMMUNICATE, LOCALIZE]
        # 1. 确定当前是否存在一个有效的主要目标
        #    【新行为】target_id 现在是最近的、有任何未完成子任务的传感器。
        target_id, _, _ = self._get_current_target_info()
        # 如果世界上已经没有任何需要处理的目标了，屏蔽所有动作
        #    【新行为】现在这表示所有传感器的通信和定位任务都已完成。
        if target_id == -1:
            mask[:] = 0
            return mask
        # 2. 检查无人机对于这个主要目标，是否处于“可操作区域”
        #    “可操作区域”的定义 = 在通信范围内
        is_in_comm_range = self._is_in_comm_range()
        # 3. 检查当前主要目标的子任务是否都已完成
        is_comm_task_done = self.sensor_data_amounts[target_id] <= self.comm_termination_data_thresh
        num_covered_sectors = self.uncertainty_model.get_sector_coverage(
            sensor_id=target_id,
            num_sectors=self.num_sectors_for_check
        )
        is_localize_task_done = (num_covered_sectors >= self.approach_termination_sector_coverage)
        # 4. 根据状态机的三个状态，生成掩码
        if not is_in_comm_range:
            # 状态1：远离。必须 APPROACH。
            mask[self.TASK_COMMUNICATE] = 0
            mask[self.TASK_LOCALIZE] = 0
        elif is_comm_task_done and is_localize_task_done:
            # 状态3：当前目标已处理完毕。必须强制再次 APPROACH 去寻找下一个目标。
            mask[self.TASK_COMMUNICATE] = 0
            mask[self.TASK_LOCALIZE] = 0
        else:
            # 状态2：在可操作区域内，且当前目标有事可做。
            # 禁止 APPROACH，允许操作。
            mask[self.TASK_APPROACH] = 0
            if is_comm_task_done:
                mask[self.TASK_COMMUNICATE] = 0
            if is_localize_task_done:
                mask[self.TASK_LOCALIZE] = 0

        return mask

    def _get_low_level_action_mask(self, task_id):
        """根据当前高层任务，为低层智能体生成连续动作的掩码"""
        mask_continuous = np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=np.float32)
        target_idx, target_pos, _ = self._get_current_target_info()

        if target_idx == -1:
            # 如果没有目标，就地悬停 (速度为0)
            mask_continuous[1, :] = [-1.0, -1.0]  # 归一化速度为-1.0，对应真实速度0
            return {'continuous_mask': mask_continuous}

        if task_id == self.TASK_APPROACH:
            # 强制：以最大速度飞向最近的目标
            direction_vec = target_pos - self.drone_position
            target_angle = np.arctan2(direction_vec[1], direction_vec[0])
            low_b, high_b = self.real_action_bounds['direction']['low'], self.real_action_bounds['direction']['high']
            norm_angle = (target_angle - low_b) / (high_b - low_b) * 2 - 1
            mask_continuous[0, :] = np.clip([norm_angle - 1e-4, norm_angle + 1e-4], -1.0, 1.0)
            mask_continuous[1, :] = [1.0, 1.0]
        elif task_id == self.TASK_COMMUNICATE:
            # 限制：方向大致朝向目标，速度可以慢一些以保持连接
            target_vec = target_pos - self.drone_position
            target_angle = np.arctan2(target_vec[1], target_vec[0])
            angle_allowance = np.pi / 4.0
            min_angle, max_angle = target_angle - angle_allowance, target_angle + angle_allowance
            low_b, high_b = self.real_action_bounds['direction']['low'], self.real_action_bounds['direction']['high']
            norm_min = (min_angle - low_b) / (high_b - low_b) * 2 - 1
            norm_max = (max_angle - low_b) / (high_b - low_b) * 2 - 1
            mask_continuous[0, :] = np.clip([norm_min, norm_max], -1.0, 1.0)
            mask_continuous[1, :] = [-0.6, 0.2]  # 归一化速度，对应真实速度的大约 6m/s 到 18m/s
        elif task_id == self.TASK_LOCALIZE:
            # 1. 获取该传感器的历史测距点
            history_points = self.uncertainty_model.ranging_points[target_idx]

            # 2. 使用控制器计算最优的移动向量
            move_vector = self.maneuver_controller.calculate_move_vector(
                drone_pos=self.drone_position,
                sensor_est_pos=target_pos,
                history_points=history_points
            )
            # --- 【BUG修复】补全后续逻辑 ---
            move_norm = np.linalg.norm(move_vector)
            if move_norm > 1e-6:
                # 3. 从移动向量中提取期望的方向和速度
                target_direction_angle = np.arctan2(move_vector[1], move_vector[0])
                target_speed = move_norm / self.time_slot
                target_speed = min(target_speed, self.max_speed)

                # 4. 将期望值转换为归一化的、狭窄的掩码
                angle_low_b, angle_high_b = self.real_action_bounds['direction']['low'], \
                self.real_action_bounds['direction']['high']
                norm_angle = (target_direction_angle - angle_low_b) / (angle_high_b - angle_low_b) * 2 - 1
                angle_tolerance = 0.1
                mask_continuous[0, :] = np.clip([norm_angle - angle_tolerance, norm_angle + angle_tolerance], -1.0, 1.0)

                speed_low_b, speed_high_b = self.real_action_bounds['speed']['low'], self.real_action_bounds['speed'][
                    'high']
                norm_speed = (target_speed - speed_low_b) / (speed_high_b - speed_low_b) * 2 - 1
                speed_tolerance = 0.1
                mask_continuous[1, :] = np.clip([norm_speed - speed_tolerance, norm_speed + speed_tolerance], -1.0, 1.0)
            else:
                # 回退机制：悬停
                mask_continuous[1, :] = [-1.0, -1.0]
        return {'continuous_mask': mask_continuous}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.trajectory, self.communication_log, self.localization_log = [], [], []
        self.current_step = 0
        margin = 100.0
        w, h = self.area_size[0] - 2 * margin, self.area_size[1] - 2 * margin
        points = poisson_disk_sampling((w, h), self.num_sensors, 250.0, self.np_random)
        self.sensor_true_positions = points + np.array([margin, margin])
        self.uncertainty_model.initialize_states(self.sensor_true_positions, 100.0, self.np_random)
        self.sensor_data_amounts = np.full(self.num_sensors, self.solo_SN_data)
        self.drone_position = np.array([0.0, 0.0])
        self.drone_velocity = np.zeros(2)
        self.trajectory.append(self.drone_position.copy())
        eligible_mask = self.sensor_data_amounts > self.comm_termination_data_thresh
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
        # 1. 定义通信任务未完成的传感器
        comm_not_done_mask = self.sensor_data_amounts > self.comm_termination_data_thresh
        # 2. 定义定位任务未完成的传感器
        localize_not_done_mask = np.full(self.num_sensors, False, dtype=bool)
        for i in range(self.num_sensors):
            num_covered_sectors = self.uncertainty_model.get_sector_coverage(
                sensor_id=i,
                num_sectors=self.num_sectors_for_check
            )
            if num_covered_sectors < self.approach_termination_sector_coverage:
                localize_not_done_mask[i] = True
        # 3. 任何一个任务未完成，传感器就是“合格”的
        eligible_mask = comm_not_done_mask | localize_not_done_mask
        if not np.any(eligible_mask):
            return False
        ref_gain_linear = 10 ** (self.reference_loss / 10.0)
        gain_const = self.transmit_power * ref_gain_linear * (self.reference_distance ** self.path_loss_exponent)
        for idx in np.where(eligible_mask)[0]:
            dist_center = np.linalg.norm(self.drone_position - est_pos[idx])
            radii = min(est_radii[idx], 100)  # Cap radius for stability
            worst_h_dist = dist_center + radii
            worst_3d_dist = np.sqrt(worst_h_dist ** 2 + self.drone_height ** 2) + 1e-6
            path_loss = (1 / worst_3d_dist) ** self.path_loss_exponent
            rx_power_est = gain_const * path_loss
            snr_est = rx_power_est / self.noise_power
            if snr_est >= self.snr_threshold_linear: return True
        return False

    def _execute_communication(self):
        est_pos = self.uncertainty_model.estimated_positions
        est_radii = self.uncertainty_model.uncertainty_radii

        # 1. 定义通信任务未完成的传感器（即有剩余数据的传感器）
        comm_not_done_mask = self.sensor_data_amounts > self.comm_termination_data_thresh
        # 如果所有传感器都没有数据需要传输了，直接返回 -1
        if not np.any(comm_not_done_mask):
            return 0.0
        ref_gain_linear = 10 ** (self.reference_loss / 10.0)
        gain_const = self.transmit_power * ref_gain_linear * (self.reference_distance ** self.path_loss_exponent)
        min_dist_to_comm_sensor = np.inf
        target_idx = -1
        # 遍历所有“有剩余数据”的传感器
        for idx in np.where(comm_not_done_mask)[0]:
            dist_center = np.linalg.norm(self.drone_position - est_pos[idx])
            radii = min(est_radii[idx], 100)  # 限制半径以保持稳定性
            worst_h_dist = dist_center + radii  # 考虑估计不确定性，计算最坏情况下的水平距离
            worst_3d_dist = np.sqrt(worst_h_dist ** 2 + self.drone_height ** 2) + 1e-6  # 考虑高度，计算3D距离

            path_loss = (1 / worst_3d_dist) ** self.path_loss_exponent
            rx_power_est = gain_const * path_loss
            snr_est = rx_power_est / self.noise_power

            # 如果该传感器满足通信条件
            if snr_est >= self.snr_threshold_linear:
                # 检查这是否是目前找到的距离最小的满足通信条件的传感器
                if dist_center < min_dist_to_comm_sensor:
                    min_dist_to_comm_sensor = dist_center
                    target_idx = idx
        if target_idx == -1:
            return 0.0
        ref_gain_lin = 10 ** (self.reference_loss / 10);
        true_h_dist = np.linalg.norm(self.drone_position - self.sensor_true_positions[target_idx]);
        true_3d_dist = np.sqrt(true_h_dist ** 2 + self.drone_height ** 2)
        if true_3d_dist < 1e-6: true_3d_dist = 1e-6
        path_loss_true = ref_gain_lin * (self.reference_distance / true_3d_dist) ** self.path_loss_exponent;
        snr_lin = (self.transmit_power * path_loss_true) / self.noise_power
        throughput = self.bandwidth * np.log2(1 + snr_lin);
        transmitted_data = throughput * self.time_slot
        if transmitted_data < 0: transmitted_data = 0.0
        self.sensor_data_amounts[target_idx] = max(0, self.sensor_data_amounts[target_idx] - transmitted_data)
        self.communication_log.append(
            {'step': self.current_step, 'target_sensor': target_idx, 'transmitted_data_Mbits': transmitted_data / 1e6,
             'remaining_data': self.sensor_data_amounts.copy()})
        return transmitted_data

    def _execute_localization(self):
        """
        【修改版】进行完整的定位流程：
        对所有传感器进行测距、记录测距点，并对满足条件的传感器进行EKF更新。
        """
        current_drone_pos = self.drone_position
        true_h_dists = np.linalg.norm(current_drone_pos - self.sensor_true_positions, axis=1)
        variances = self.uncertainty_model.g0 * (true_h_dists ** 2)
        std_devs = np.sqrt(variances)
        measured_dists = self.np_random.normal(loc=true_h_dists, scale=std_devs)
        measured_dists_2 = self.np_random.normal(loc=true_h_dists, scale=std_devs)
        measured_dists_3 = self.np_random.normal(loc=true_h_dists, scale=std_devs)

        # 将原本只针对 target_id 的逻辑，扩展到所有传感器
        for i in range(self.num_sensors):
            dist = np.linalg.norm(current_drone_pos - self.uncertainty_model.estimated_positions[i])
            if dist <= self.sector_coverage_distance_thresh:
                self.uncertainty_model.add_ranging_point(i, current_drone_pos.copy())

            # 2. 检查该传感器是否满足EKF更新的几何前提条件
            num_covered_sectors = self.uncertainty_model.get_sector_coverage(
                sensor_id=i,
                num_sectors=self.num_sectors_for_check
            )
            # 使用4作为阈值，如你所要求
            if num_covered_sectors >= 4 and dist <= self.sector_coverage_distance_thresh:
                # 3. 如果满足条件，则对该传感器执行EKF更新
                self.uncertainty_model.update(
                    sensor_id=i,
                    drone_position=current_drone_pos.copy(),
                    measured_distance=measured_dists[i],
                    measurement_variance=variances[i]
                )
                self.uncertainty_model.update(
                    sensor_id=i,
                    drone_position=current_drone_pos.copy(),
                    measured_distance=measured_dists_2[i],
                    measurement_variance=variances[i]
                )
                self.uncertainty_model.update(
                    sensor_id=i,
                    drone_position=current_drone_pos.copy(),
                    measured_distance=measured_dists_3[i],
                    measurement_variance=variances[i]
                )

        # (这部分用于追踪“硬稳定”状态的逻辑保持不变)

        # (日志记录逻辑保持不变)
        self.localization_log.append(
            {'step': self.current_step,
             'est_positions': self.uncertainty_model.estimated_positions.copy(),
             'est_radii': self.uncertainty_model.uncertainty_radii.copy()}
        )

    def _apply_movement(self, continuous_action):
        real_direction, real_speed = self._unnormalize_action(continuous_action)
        self.drone_velocity = np.array([real_speed * np.cos(real_direction), real_speed * np.sin(real_direction)])
        self.drone_position += self.drone_velocity * self.time_slot
        self.drone_position = np.clip(self.drone_position, [0, 0], self.area_size)

    def step(self, high_level_action):
        accumulated_reward = 0.0
        task_completion_reward = 0.0
        terminated, truncated = False, False
        initial_target_id, _, _ = self._get_current_target_info()
        for i in range(self.max_task_commitment_steps):
            # 1. 检查整个Episode是否已完成
            final_target_id, _, _ = self._get_current_target_info()
            if final_target_id == -1:
                terminated = True
                break
            # 2. 检查是否超时
            if self.current_step >= self.max_steps_per_episode:
                truncated = True
                break
                # 3. 检查当前高层动作是否还合法
            # current_high_level_mask = self.get_high_level_action_mask()
            # if current_high_level_mask[high_level_action] == 0:
            #     break

            self.current_step += 1

            flat_obs = gym.spaces.utils.flatten(self.observation_space, self._get_obs())
            low_level_mask = self._get_low_level_action_mask(high_level_action)
            low_level_action, low_val, low_logp = self.low_level_agent_controller.select_action(
                flat_obs, high_level_action, low_level_mask
            )
            self._apply_movement(low_level_action)
            transmitted_data_this_step = 0.0
            # 核心逻辑：APPROACH和LOCALIZE任务都需要执行测距和可能的EKF更新
            if high_level_action == self.TASK_COMMUNICATE:
                transmitted_data_this_step = self._execute_communication()
            elif high_level_action == self.TASK_LOCALIZE:  # For both APPROACH and LOCALIZE tasks
                self._execute_localization()
            else:
                self._execute_localization()

            reward = self._calculate_reward(transmitted_data_this_step)
            accumulated_reward += reward
            # self.low_level_agent_controller.buffer.store(
            #     obs=flat_obs, act=low_level_action, rew=reward, val=low_val,
            #     logp=low_logp, task_id=high_level_action
            # )

            self.trajectory.append(self.drone_position.copy())
        if not (terminated or truncated):
            final_target_id, _, _ = self._get_current_target_info()
            if final_target_id == -1:
                terminated = True
            if self.current_step >= self.max_steps_per_episode:
                truncated = True
        accumulated_reward += task_completion_reward
        if terminated: accumulated_reward += 100.0
        if truncated: accumulated_reward -= 500.0

        return self._get_obs(), accumulated_reward, terminated, truncated, self._get_info()