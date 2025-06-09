import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt

from envs.sensornet import generate_sensor_network, visualize_sensor_network
from envs.uncertain_model import UncertaintyModel
from envs.GDOP import GDOPCalculator
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题


class DroneEnv(gym.Env):
    """无人机传感器网络数据采集环境"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化无人机环境

        Args:
            config: 环境配置参数
        """
        super().__init__()

        # 环境参数
        self.config = config or {}
        self.area_size = self.config.get('area_size', (1000.0, 1000.0))
        self.num_sensors = self.config.get('num_sensors', 5)
        self.max_speed = self.config.get('max_speed', 30.0)  # m/s
        self.time_slot = self.config.get('time_slot', 1.0)  # s
        self.max_episode_steps = self.config.get('max_episode_steps', 500)

        # GPU优化选项
        self.enable_fast_mode = self.config.get('enable_fast_mode', False)
        self.gdop_resolution = self.config.get('gdop_resolution', 100.0)
        self.disable_realtime_visualization = self.config.get('disable_realtime_visualization', False)  # 只禁用实时可视化
        self.reduce_computation = self.config.get('reduce_computation', False)

        # 通信参数
        self.path_loss_exponent = self.config.get('path_loss_exponent', 2)
        self.reference_distance = self.config.get('reference_distance', 1.0)  # m
        self.reference_loss = self.config.get('reference_loss', -60)  # dB
        self.transmit_power = self.config.get('transmit_power', 0.1)  # W
        self.noise_power = self.config.get('noise_power', 1e-11)  # W (-110dBm)
        self.snr_threshold = self.config.get('snr_threshold', 10.0)  # dB

        # 初始化组件
        self.uncertainty_model = UncertaintyModel(self.num_sensors)

        # 根据快速模式设置GDOP计算器
        if self.enable_fast_mode:
            # 快速模式：降低分辨率，减少计算量
            self.gdop_calculator = GDOPCalculator(self.area_size, grid_resolution=self.gdop_resolution)
        else:
            # 标准模式
            self.gdop_calculator = GDOPCalculator(self.area_size)

        # 快速模式的计算缓存
        if self.enable_fast_mode:
            self.gdop_update_counter = 0
            self.gdop_update_interval = 5  # 每5步更新一次GDOP

        # 状态变量
        self.drone_position = np.array([0.0, 0.0])
        self.drone_velocity = np.array([0.0, 0.0])
        self.current_step = 0
        self.sensor_data_amounts = None
        self.sensor_config = None

        # 渐进式奖励系统
        self.cumulative_data_collected = 0.0
        self.cumulative_uncertainty_reduction = 0.0
        self.initial_total_data = 0.0
        self.initial_total_uncertainty = 0.0
        self.reward_history = []  # 用于奖励平滑
        self.learning_stage = 'exploration'  # exploration, exploitation, optimization

        # 动作技能系统
        self.communication_skill = 0.0  # 通信技能等级 [0, 1]
        self.localization_skill = 0.0  # 定位技能等级 [0, 1]
        self.communication_attempts = 0  # 通信尝试次数
        self.localization_attempts = 0  # 定位尝试次数
        self.successful_communications = 0  # 成功通信次数
        self.successful_localizations = 0  # 成功定位次数

        # 动作空间：[通信/定位(离散), 角度(连续), 速度(连续)]
        # 离散动作：0=通信, 1=定位
        # 连续动作：[角度(-π到π), 速度(0到max_speed)]
        self.action_space = spaces.Dict({
            'discrete': spaces.Discrete(2),
            'continuous': spaces.Box(
                low=np.array([-np.pi, 0.0]),
                high=np.array([np.pi, self.max_speed]),
                dtype=np.float32
            )
        })

        # 状态空间维度计算
        state_dim = self._calculate_state_dimension()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        # 记录和可视化
        self.trajectory = []
        self.communication_log = []
        self.localization_log = []
        self.reward_components = []

    def _calculate_state_dimension(self) -> int:
        """计算状态空间维度"""
        # 无人机状态：位置(2) + 速度(2) = 4
        drone_state_dim = 4

        # 传感器状态：每个传感器的估计位置(2) + 不确定性半径(1) + 剩余数据量(1) = 4 * num_sensors
        sensor_state_dim = 4 * self.num_sensors

        # GDOP特征：13维（5个统计特征 + 8个方向特征）
        gdop_feature_dim = 13

        # 时间特征：当前步数归一化(1)
        time_feature_dim = 1

        return drone_state_dim + sensor_state_dim + gdop_feature_dim + time_feature_dim

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        super().reset(seed=seed)

        # 生成传感器网络
        self.sensor_config = generate_sensor_network(
            area_size=self.area_size,
            num_sensors=self.num_sensors
        )

        # 初始化无人机状态
        self.drone_position = np.array([0.0, 0.0])
        self.drone_velocity = np.array([0.0, 0.0])
        self.current_step = 0

        # 初始化传感器数据量
        self.sensor_data_amounts = self.sensor_config['initial_data_amounts'].copy()

        # 初始化不确定性模型
        self.uncertainty_model = UncertaintyModel(self.num_sensors)
        self.uncertainty_model.estimated_positions = self.sensor_config['estimated_positions'].copy()
        self.uncertainty_model.uncertainty_radii = self.sensor_config['initial_uncertainty_radius'].copy()

        # 初始化GDOP计算器
        self.gdop_calculator = GDOPCalculator(self.area_size)

        # 重置渐进式奖励系统
        self.cumulative_data_collected = 0.0
        self.cumulative_uncertainty_reduction = 0.0
        self.initial_total_data = np.sum(self.sensor_data_amounts)
        self.initial_total_uncertainty = np.sum(self.uncertainty_model.uncertainty_radii)
        self.reward_history = []
        self.learning_stage = 'exploration'

        # 重置动作技能系统
        self.communication_skill = 0.0
        self.localization_skill = 0.0
        self.communication_attempts = 0
        self.localization_attempts = 0
        self.successful_communications = 0
        self.successful_localizations = 0

        # 清空记录
        self.trajectory = [self.drone_position.copy()]
        self.communication_log = []
        self.localization_log = []
        self.reward_components = []

        # 生成动作掩码
        self.action_mask = self._generate_action_mask()

        # 返回初始状态
        state = self._get_state()
        info = self._get_info()

        return state, info

    def step(self, action: Dict) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步动作"""
        discrete_action = action['discrete']
        continuous_action = action['continuous']

        angle, speed = continuous_action[0], continuous_action[1]

        # 更新无人机位置
        self._update_drone_position(angle, speed)

        # 计算新的渐进式奖励
        reward_components = self._calculate_progressive_reward(discrete_action)

        # 执行动作
        if discrete_action == 0:  # 通信
            action_result = self._perform_communication_v3()
        else:  # 定位
            action_result = self._perform_localization_v3()

        # 更新奖励组件
        reward_components.update(action_result)

        # 计算最终奖励
        total_reward = self._finalize_reward(reward_components)

        # 更新学习阶段
        self._update_learning_stage()

        # 检查终止条件
        done = self._check_done()
        truncated = self.current_step >= self.max_episode_steps

        self.current_step += 1

        # 更新动作掩码
        self.action_mask = self._generate_action_mask()

        # 记录轨迹
        self.trajectory.append(self.drone_position.copy())

        # 记录奖励组件（用于调试）
        reward_components['step'] = self.current_step
        reward_components['action_type'] = 'communication' if discrete_action == 0 else 'localization'
        reward_components['drone_position'] = self.drone_position.copy()
        reward_components['learning_stage'] = self.learning_stage
        self.reward_components.append(reward_components)

        state = self._get_state()
        info = self._get_info()

        # 在info中添加奖励组件信息
        info['reward_components'] = reward_components

        return state, total_reward, done, truncated, info

    def _update_drone_position(self, angle: float, speed: float):
        """更新无人机位置"""
        # 计算速度向量
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)
        self.drone_velocity = np.array([vx, vy])

        # 更新位置
        new_position = self.drone_position + self.drone_velocity * self.time_slot

        # 边界约束
        new_position[0] = np.clip(new_position[0], 0, self.area_size[0])
        new_position[1] = np.clip(new_position[1], 0, self.area_size[1])

        self.drone_position = new_position

    def _calculate_progressive_reward(self, discrete_action: int) -> Dict:
        """计算渐进式奖励"""
        reward_components = {
            'baseline_penalty': 0.0,
            'progress_reward': 0.0,
            'efficiency_bonus': 0.0,
            'stage_bonus': 0.0,
            'action_reward': 0.0,
            'total_reward': 0.0
        }

        # 1. 基线惩罚：从负值开始，随着进度逐渐减少
        current_progress = self._get_overall_progress()
        baseline_penalty = -0.5 * (1.0 - current_progress)  # 从-0.5逐渐减少到0
        reward_components['baseline_penalty'] = baseline_penalty

        # 2. 进度奖励：基于累积成就，严格递增（添加安全上限）
        data_progress = 0.0
        uncertainty_progress = 0.0

        if self.initial_total_data > 1e-6:
            data_progress = min(1.0, max(0.0, self.cumulative_data_collected / self.initial_total_data))

        if self.initial_total_uncertainty > 1e-6:
            uncertainty_progress = min(1.0,
                                       max(0.0, self.cumulative_uncertainty_reduction / self.initial_total_uncertainty))

        overall_progress = (data_progress + uncertainty_progress) / 2.0
        overall_progress = np.clip(overall_progress, 0.0, 1.0)

        # 非线性进度奖励，早期增长缓慢，后期增长加速（添加安全上限）
        progress_reward = min(2.0, 2.0 * (overall_progress ** 1.5))  # 范围[0, 2.0]
        reward_components['progress_reward'] = progress_reward

        # 3. 效率奖励：基于时间效率（添加安全检查）
        time_efficiency = max(0, min(1.0, 1.0 - (self.current_step / max(1, self.max_episode_steps))))
        efficiency_bonus = min(0.3, 0.3 * time_efficiency * overall_progress)  # 只有在有进度时才给效率奖励
        reward_components['efficiency_bonus'] = efficiency_bonus

        # 4. 学习阶段奖励：根据当前学习阶段给予不同的奖励偏向
        stage_bonus = self._calculate_stage_bonus(discrete_action)
        stage_bonus = np.clip(stage_bonus, 0.0, 0.2)  # 限制阶段奖励范围
        reward_components['stage_bonus'] = stage_bonus

        return reward_components

    def _perform_communication_v3(self) -> Dict:
        """执行通信动作 - 第四版（基于技能熟练度）"""
        result = {'action_reward': 0.0, 'data_collected': 0.0}

        # 增加通信尝试次数
        self.communication_attempts += 1

        # 基础动作奖励：从负值开始，随技能提升
        base_action_reward = -0.1 + 0.3 * self.communication_skill  # 范围[-0.1, 0.2]

        # 找到可通信的传感器
        communicable_sensors = []
        for i in range(self.num_sensors):
            if self.sensor_data_amounts[i] > 0:
                worst_distance = self.uncertainty_model.get_worst_case_distance(i, self.drone_position)
                snr = self._calculate_snr(worst_distance)

                if snr >= self.snr_threshold:
                    value = self.sensor_data_amounts[i] / 100.0
                    communicable_sensors.append((i, snr, value))

        if communicable_sensors:
            # 选择最有价值的传感器
            best_sensor = max(communicable_sensors, key=lambda x: x[1])  #之前按value排序
            sensor_id, snr, value = best_sensor

            # 计算数据采集
            true_distance = np.linalg.norm(
                np.array([
                    self.drone_position[0] - self.sensor_config['true_positions'][sensor_id][0],
                    self.drone_position[1] - self.sensor_config['true_positions'][sensor_id][1],
                    60 - 0  # 高度差
                ])
            )
            actual_snr = self._calculate_snr(true_distance)
            data_collected = self._calculate_data_collection_v2(actual_snr)

            # 更新累积数据
            collected = min(data_collected, self.sensor_data_amounts[sensor_id])
            self.sensor_data_amounts[sensor_id] -= collected
            self.cumulative_data_collected += collected

            # 成功通信计数
            if collected > 0:
                self.successful_communications += 1

            # 效果奖励：基于实际采集量和技能等级
            if collected > 0:
                effect_reward = 0.1 * (collected / 20.0) * (1.0 + self.communication_skill)  # 技能越高效果越好
            else:
                effect_reward = -0.05  # 连接成功但没有数据的小惩罚

            # 总动作奖励
            action_reward = base_action_reward + effect_reward
            result['action_reward'] = np.clip(action_reward, -0.2, 0.4)  # 限制范围
            result['data_collected'] = collected

            # 记录通信
            self.communication_log.append({
                'step': self.current_step,
                'sensor_id': sensor_id,
                'position': self.drone_position.copy(),
                'data_collected': collected,
                'snr': actual_snr,
                'reward': result['action_reward']
            })
        else:
            # 无法通信：技能影响惩罚程度
            penalty = -0.15 + 0.1 * self.communication_skill  # 技能越高惩罚越小
            result['action_reward'] = max(penalty, -0.15)

        # 更新通信技能
        self._update_communication_skill()

        return result

    def _perform_localization_v3(self) -> Dict:
        """执行定位动作 - 第四版（基于技能熟练度）"""
        result = {'action_reward': 0.0, 'uncertainty_reduction': 0.0}

        # 增加定位尝试次数
        self.localization_attempts += 1

        # 基础动作奖励：从负值开始，随技能提升
        base_action_reward = -0.08 + 0.25 * self.localization_skill  # 范围[-0.08, 0.17]

        total_uncertainty_reduction = 0.0
        measurement_quality = 0.0  # 测量质量评估

        # 为所有传感器添加测距测量(水平测距)
        for i in range(self.num_sensors):
            true_distance = np.linalg.norm(
                self.drone_position - self.sensor_config['true_positions'][i]
            )

            # 技能影响测量精度
            #noise_factor = 1.0 - 0.5 * self.localization_skill  # 技能越高噪声越小
            #g0 = 1.125e-5
            variances = 1.125e-5 * true_distance ** 2
            measured_distance = true_distance + np.random.normal(0, variances)
            old_radius = self.uncertainty_model.uncertainty_radii[i]
            self.uncertainty_model.add_ranging_measurement(i, self.drone_position, measured_distance)

            # 评估测量质量
            distance_error = abs(measured_distance - true_distance)
            measurement_quality += max(0, 1.0 - distance_error / 50.0)  # 50m为基准误差

            # 每5个测距点更新一次
            if len(self.uncertainty_model.ranging_points[i]) % 5 == 0:
                self.uncertainty_model.update_sensor_estimate(i)
                new_radius = self.uncertainty_model.uncertainty_radii[i]

                if new_radius < old_radius:
                    uncertainty_reduction = old_radius - new_radius
                    total_uncertainty_reduction += uncertainty_reduction
                    self.cumulative_uncertainty_reduction += uncertainty_reduction

        # 计算平均测量质量
        avg_measurement_quality = measurement_quality / self.num_sensors

        # 效果奖励：基于不确定性减少和测量质量
        if total_uncertainty_reduction > 0:
            effect_reward = 0.08 * min(total_uncertainty_reduction / 10.0, 1.0) * (1.0 + self.localization_skill)
            self.successful_localizations += 1
        else:
            # 基于测量质量给予小奖励，避免完全惩罚
            effect_reward = 0.02 * avg_measurement_quality - 0.01

        # 总动作奖励
        action_reward = base_action_reward + effect_reward
        result['action_reward'] = np.clip(action_reward, -0.15, 0.3)  # 限制范围
        result['uncertainty_reduction'] = total_uncertainty_reduction

        # 更新GDOP热力图
        self._update_gdop_heatmap()

        # 记录定位
        self.localization_log.append({
            'step': self.current_step,
            'position': self.drone_position.copy(),
            'uncertainty_radii': self.uncertainty_model.uncertainty_radii.copy(),
            'uncertainty_reduction': total_uncertainty_reduction,
            'measurement_quality': avg_measurement_quality,
            'reward': result['action_reward']
        })

        # 更新定位技能
        self._update_localization_skill()

        return result

    def _calculate_snr(self, distance: float) -> float:
        """计算信噪比（dB）"""
        # 避免距离过小导致的计算问题
        distance = max(distance, 1.0)

        # 路径损耗模型
        path_loss_db = self.reference_loss + 10 * self.path_loss_exponent * np.log10(
            distance / self.reference_distance
        )

        # 接收功率
        received_power_db = 10 * np.log10(self.transmit_power) + path_loss_db
        received_power_w = 10 ** (received_power_db / 10)

        # 信噪比
        snr = received_power_w / self.noise_power
        snr_db = 10 * np.log10(max(snr, 1e-10))  # 避免log(0)

        # 限制SNR范围，避免极值
        snr_db = np.clip(snr_db, -20.0, 60.0)

        return snr_db

    def _calculate_data_collection_v2(self, snr_db: float) -> float:
        """计算数据采集量 - 改进版本"""
        # 确保SNR在合理范围内
        snr_db = np.clip(snr_db, -20.0, 60.0)

        # 更稳定的数据采集模型
        if snr_db >= 20:
            data_rate = 20.0  # 高质量连接
        elif snr_db >= 10:
            data_rate = 15.0  # 中等质量连接
        elif snr_db >= 0:
            data_rate = 10.0  # 低质量连接
        else:
            data_rate = 5.0  # 最低质量连接

        return data_rate

    def _get_overall_progress(self) -> float:
        """获取整体进度"""
        data_progress = 0.0
        uncertainty_progress = 0.0

        # 安全的进度计算，避免除零错误
        if self.initial_total_data > 1e-6:  # 使用小的阈值而不是0
            data_progress = min(1.0, max(0.0, self.cumulative_data_collected / self.initial_total_data))

        if self.initial_total_uncertainty > 1e-6:  # 使用小的阈值而不是0
            uncertainty_progress = min(1.0,
                                       max(0.0, self.cumulative_uncertainty_reduction / self.initial_total_uncertainty))

        overall_progress = (data_progress + uncertainty_progress) / 2.0
        overall_progress = np.clip(overall_progress, 0.0, 1.0)  # 强制限制范围

        # 调试信息（仅在异常值时打印）
        if overall_progress > 1.0 or overall_progress < 0:
            print(f"警告: 异常的进度值 - 数据进度:{data_progress:.3f}, 不确定性进度:{uncertainty_progress:.3f}")
            print(f"  累积数据:{self.cumulative_data_collected:.1f}/{self.initial_total_data:.1f}")
            print(
                f"  累积不确定性减少:{self.cumulative_uncertainty_reduction:.1f}/{self.initial_total_uncertainty:.1f}")

        return overall_progress

    def _calculate_stage_bonus(self, discrete_action: int) -> float:
        """计算学习阶段奖励"""
        overall_progress = self._get_overall_progress()

        if self.learning_stage == 'exploration':
            # 探索阶段：鼓励尝试不同动作
            return 0.05
        elif self.learning_stage == 'exploitation':
            # 开发阶段：鼓励高效动作
            if discrete_action == 0:  # 通信
                return 0.08 if overall_progress < 0.7 else 0.04
            else:  # 定位
                return 0.06
        else:  # optimization
            # 优化阶段：鼓励精确执行
            return 0.1 * overall_progress

    def _update_learning_stage(self):
        """更新学习阶段"""
        overall_progress = self._get_overall_progress()

        if overall_progress < 0.3:
            self.learning_stage = 'exploration'
        elif overall_progress < 0.7:
            self.learning_stage = 'exploitation'
        else:
            self.learning_stage = 'optimization'

    def _finalize_reward(self, reward_components: Dict) -> float:
        """计算最终奖励并进行自适应平滑"""
        # 计算原始总奖励
        raw_total = (reward_components['baseline_penalty'] +
                     reward_components['progress_reward'] +
                     reward_components['efficiency_bonus'] +
                     reward_components['stage_bonus'] +
                     reward_components['action_reward'])

        # 检查原始奖励是否异常
        if abs(raw_total) > 10.0:
            print(f"警告: 异常的原始奖励值 {raw_total:.3f}")
            print(f"  组件: 基线:{reward_components['baseline_penalty']:.3f}, "
                  f"进度:{reward_components['progress_reward']:.3f}, "
                  f"效率:{reward_components['efficiency_bonus']:.3f}, "
                  f"阶段:{reward_components['stage_bonus']:.3f}, "
                  f"动作:{reward_components['action_reward']:.3f}")
            # 异常奖励直接裁剪
            raw_total = np.clip(raw_total, -1.0, 5.0)

        # 自适应奖励平滑
        if len(self.reward_history) > 0:
            # 计算历史奖励的标准差，用于自适应平滑
            recent_history = self.reward_history[-10:] if len(self.reward_history) >= 10 else self.reward_history
            hist_std = np.std(recent_history)

            # 根据历史波动调整平滑强度
            if hist_std > 1.0:  # 高波动时使用强平滑
                alpha = 0.5
            elif hist_std > 0.5:  # 中等波动时使用中等平滑
                alpha = 0.7
            else:  # 低波动时使用轻微平滑
                alpha = 0.85

            # 使用指数移动平均进行平滑
            smoothed_reward = alpha * raw_total + (1 - alpha) * np.mean(recent_history)
        else:
            smoothed_reward = raw_total

        # 更新奖励历史
        self.reward_history.append(raw_total)
        if len(self.reward_history) > 30:  # 增加历史窗口大小
            self.reward_history.pop(0)

        # 确保奖励在合理范围内
        final_reward = np.clip(smoothed_reward, -0.8, 3.5)

        # 如果最终奖励被裁剪，发出警告
        if abs(final_reward - smoothed_reward) > 0.1:
            print(f"警告: 奖励被显著裁剪 {smoothed_reward:.3f} -> {final_reward:.3f}")

        reward_components['raw_total'] = raw_total
        reward_components['smoothed_total'] = smoothed_reward
        reward_components['total_reward'] = final_reward

        return final_reward

    def _update_gdop_heatmap(self):
        """更新GDOP热力图"""
        # 快速模式：减少GDOP更新频率
        if self.enable_fast_mode:
            self.gdop_update_counter += 1
            if self.gdop_update_counter % self.gdop_update_interval != 0:
                return  # 跳过本次更新

        sensors_data = {}
        for i in range(self.num_sensors):
            sensors_data[i] = {
                'position': self.uncertainty_model.estimated_positions[i],
                'ranging_points': self.uncertainty_model.ranging_points[i],
                'data_amount': self.sensor_data_amounts[i]
            }

        self.gdop_calculator.update_global_gdop_heatmap(sensors_data)

    def _check_done(self) -> bool:
        """检查是否完成任务"""
        # 数据采集完成度检查（更宽松的条件）
        total_initial_data = np.sum(self.sensor_config['initial_data_amounts'])
        remaining_data = np.sum(self.sensor_data_amounts)
        data_completion_rate = 1.0 - (remaining_data / total_initial_data)
        data_complete = data_completion_rate >= 0.95  # 从完全采集改为95%

        # 定位收敛检查（更宽松的条件）
        avg_uncertainty = np.mean(self.uncertainty_model.uncertainty_radii)
        localization_converged = avg_uncertainty <= 5.0  # 从1.0m改为5.0m

        # 早期成功终止：数据收集达到90%且平均不确定性低于10m
        early_success = (data_completion_rate >= 0.9 and avg_uncertainty <= 10.0)

        return data_complete or early_success

    def _generate_action_mask(self) -> Dict:
        """生成动作掩码"""
        # 返回PPO_Hybrid期望的字典格式
        return {
            'discrete_mask': np.ones(2, dtype=np.float32),  # 离散动作掩码：[通信, 定位]
            'continuous_mask': np.ones(2, dtype=np.float32)  # 连续动作掩码：[角度, 速度]
        }

    def _get_state(self) -> np.ndarray:
        """获取当前状态"""
        state_components = []

        # 无人机状态
        state_components.extend(self.drone_position)
        state_components.extend(self.drone_velocity)

        # 传感器状态
        for i in range(self.num_sensors):
            state_components.extend(self.uncertainty_model.estimated_positions[i])
            state_components.append(self.uncertainty_model.uncertainty_radii[i])
            state_components.append(self.sensor_data_amounts[i])

        # GDOP特征
        gdop_features = self.gdop_calculator.get_gdop_features(self.drone_position)
        state_components.extend(gdop_features)

        # 时间特征
        time_feature = self.current_step / self.max_episode_steps
        state_components.append(time_feature)

        return np.array(state_components, dtype=np.float32)

    def _get_info(self) -> Dict:
        """获取环境信息"""
        return {
            'drone_position': self.drone_position.copy(),
            'sensor_data_amounts': self.sensor_data_amounts.copy(),
            'uncertainty_radii': self.uncertainty_model.uncertainty_radii.copy(),
            'step': self.current_step,
            'data_completion_rate': 1.0 - np.sum(self.sensor_data_amounts) / np.sum(
                self.sensor_config['initial_data_amounts'])
        }

    def render(self, mode: str = 'human'):
        """渲染环境"""
        # 快速模式下禁用可视化
        if self.disable_realtime_visualization:
            return

        if mode == 'human':
            self.visualize_environment()

    def visualize_environment(self, save_path: Optional[str] = None):
        """
        可视化环境状态（4个子图）
        """
        # 更严格的GDOP热力图尺寸检查和降采样
        heatmap_shape = self.gdop_calculator.gdop_heatmap.shape
        max_pixels_per_dim = 200  # 每个维度最大200像素
        total_max_pixels = 20000  # 总像素数最大20000

        skip_gdop_visualization = (
                heatmap_shape[0] > max_pixels_per_dim or
                heatmap_shape[1] > max_pixels_per_dim or
                heatmap_shape[0] * heatmap_shape[1] > total_max_pixels
        )

        # 如果热力图太大，尝试降采样
        gdop_heatmap_to_display = None
        if not skip_gdop_visualization:
            gdop_heatmap_to_display = self.gdop_calculator.gdop_heatmap
        elif heatmap_shape[0] * heatmap_shape[1] <= 200000:  # 允许更大的降采样空间
            # 计算降采样因子
            downsample_factor_x = max(1, int(np.ceil(heatmap_shape[0] / max_pixels_per_dim)))
            downsample_factor_y = max(1, int(np.ceil(heatmap_shape[1] / max_pixels_per_dim)))

            try:
                # 执行降采样
                from scipy import ndimage
                gdop_heatmap_to_display = ndimage.zoom(
                    self.gdop_calculator.gdop_heatmap,
                    (1 / downsample_factor_x, 1 / downsample_factor_y),
                    order=1  # 双线性插值
                )
                skip_gdop_visualization = False
                print(f"GDOP热力图已降采样：{heatmap_shape} -> {gdop_heatmap_to_display.shape}")
            except Exception as e:
                print(f"GDOP热力图降采样失败: {e}")
                skip_gdop_visualization = True

        if skip_gdop_visualization:
            print(f"警告: GDOP热力图尺寸过大 {heatmap_shape}，将跳过GDOP可视化")

        # 使用固定且安全的图像尺寸
        plt.rcParams['figure.max_open_warning'] = 0  # 禁用图像数量警告
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9), dpi=72)  # 降低DPI

        # 1. 传感器网络和无人机轨迹
        ax1.set_title(f'Sensor Network & Drone Trajectory (Step {self.current_step})')
        ax1.set_xlim(0, self.area_size[0])
        ax1.set_ylim(0, self.area_size[1])

        # 获取真实和估计位置
        true_pos = np.array([self.sensor_config['true_positions'][i] for i in range(self.num_sensors)])
        est_pos = np.array([self.uncertainty_model.estimated_positions[i] for i in range(self.num_sensors)])
        uncertainty = self.uncertainty_model.uncertainty_radii

        # 绘制传感器真实位置（红色圆点）
        ax1.scatter(true_pos[:, 0], true_pos[:, 1], c='red', s=100, marker='o',
                    label='True Sensor Positions', alpha=0.8, edgecolors='darkred')

        # 绘制传感器估计位置和不确定性圆圈（蓝色×和圆圈）
        for i, (est, radius) in enumerate(zip(est_pos, uncertainty)):
            circle = plt.Circle(est, radius, fill=False, color='blue', alpha=0.5, linestyle='--')
            ax1.add_patch(circle)
            ax1.scatter(est[0], est[1], c='blue', s=80, marker='x', alpha=0.8, linewidths=2)
            ax1.text(est[0] + 20, est[1] + 20, f'S{i}\n±{radius:.1f}m\n{self.sensor_data_amounts[i]:.1f}MB',
                     fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

        # 改进的轨迹绘制逻辑
        trajectory_drawn = False
        if len(self.trajectory) >= 2:  # 至少需要2个点才能画线
            trajectory = np.array(self.trajectory)

            print(f"调试: 轨迹点数量={len(trajectory)}, 起点={trajectory[0]}, 终点={trajectory[-1]}")

            # 检查轨迹是否有实际移动（不是所有点都一样）
            has_movement = False
            if len(trajectory) >= 2:
                for i in range(1, len(trajectory)):
                    distance = np.linalg.norm(trajectory[i] - trajectory[i - 1])
                    if distance > 1.0:  # 移动距离大于1米才算有效移动
                        has_movement = True
                        break

            if has_movement:
                # 智能采样：保留关键点和最新点
                max_trajectory_points = 100  # 增加轨迹点数量限制
                if len(trajectory) > max_trajectory_points:
                    # 保留起点、终点和均匀分布的中间点
                    indices = [0]  # 起点
                    middle_indices = np.linspace(1, len(trajectory) - 2, max_trajectory_points - 2, dtype=int)
                    indices.extend(middle_indices)
                    indices.append(len(trajectory) - 1)  # 终点
                    trajectory = trajectory[indices]

                # 绘制轨迹线
                ax1.plot(trajectory[:, 0], trajectory[:, 1], 'g-', alpha=0.8, linewidth=2.5,
                         label='Drone Trajectory', zorder=5)

                # 绘制轨迹点
                ax1.scatter(trajectory[:, 0], trajectory[:, 1], c='green', s=10, alpha=0.6, zorder=6)

                trajectory_drawn = True
                print(f"✅ 轨迹已绘制: {len(trajectory)} 个点")
            else:
                print(f"⚠️ 轨迹无实际移动，跳过绘制")
        else:
            print(f"⚠️ 轨迹点不足: {len(self.trajectory)} 个点 (需要至少2个点)")

        # 绘制无人机当前位置
        ax1.scatter(self.drone_position[0], self.drone_position[1], c='green', s=200,
                    marker='^', label='Current Drone', alpha=0.9, edgecolors='darkgreen', linewidth=2, zorder=10)

        # 绘制起始位置
        ax1.scatter(0, 0, c='orange', s=150, marker='*', label='Start Position', alpha=0.9, zorder=8)

        # 如果没有绘制轨迹，添加说明文本
        if not trajectory_drawn:
            ax1.text(0.02, 0.98, f'轨迹点数: {len(self.trajectory)}\n(需要至少2个有效移动点才显示轨迹)',
                     transform=ax1.transAxes, va='top', ha='left', fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')

        # 2. GDOP热力图（条件渲染）
        ax2.set_title(f'GDOP Information (Step {self.current_step})')

        if not skip_gdop_visualization and gdop_heatmap_to_display is not None:
            try:
                im = ax2.imshow(gdop_heatmap_to_display,
                                extent=[0, self.area_size[0], 0, self.area_size[1]],
                                origin='lower', cmap='viridis_r', alpha=0.8)

                # 简化colorbar，避免额外的复杂度
                try:
                    cbar = plt.colorbar(im, ax=ax2, label='GDOP Value', shrink=0.8)
                except:
                    pass  # 如果colorbar失败，跳过

                # 标记无人机当前位置
                ax2.scatter(self.drone_position[0], self.drone_position[1], c='white', s=150,
                            marker='^', alpha=0.9, edgecolors='black', linewidth=2)

            except Exception as e:
                print(f"GDOP热力图渲染失败: {e}")
                skip_gdop_visualization = True

        if skip_gdop_visualization:
            # 显示GDOP统计信息而不是热力图
            current_gdop = self.gdop_calculator.get_gdop_value(self.drone_position)
            gdop_stats = f'''GDOP热力图尺寸过大，显示统计信息:

当前位置GDOP: {current_gdop:.2f}
热力图尺寸: {heatmap_shape}
最小GDOP: {np.min(self.gdop_calculator.gdop_heatmap):.2f}
最大GDOP: {np.max(self.gdop_calculator.gdop_heatmap):.2f}
平均GDOP: {np.mean(self.gdop_calculator.gdop_heatmap):.2f}'''

            ax2.text(0.5, 0.5, gdop_stats, transform=ax2.transAxes,
                     ha='center', va='center', fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Y Position (m)')

        # 3. 不确定性收敛曲线（简化数据点）
        ax3.set_title('Uncertainty Convergence')

        if self.localization_log:
            steps = [entry['step'] for entry in self.localization_log]
            # 限制数据点数量
            max_data_points = 100
            if len(steps) > max_data_points:
                indices = np.linspace(0, len(steps) - 1, max_data_points, dtype=int)
                steps = [steps[i] for i in indices]

                for i in range(self.num_sensors):
                    uncertainties = [self.localization_log[j]['uncertainty_radii'][i] for j in indices]
                    ax3.plot(steps, uncertainties, label=f'Sensor {i}', linewidth=1.5)
            else:
                for i in range(self.num_sensors):
                    uncertainties = [entry['uncertainty_radii'][i] for entry in self.localization_log]
                    ax3.plot(steps, uncertainties, label=f'Sensor {i}', linewidth=1.5)

        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Uncertainty Radius (m)')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 110)

        # 添加收敛目标线
        ax3.axhline(y=5.0, color='red', linestyle='--', alpha=0.7, label='Target (5m)')
        ax3.legend(fontsize=8)

        # 4. 数据采集进度（简化数据点）
        ax4.set_title('Data Collection Progress')

        if self.communication_log:
            # 重建数据采集历史
            initial_data = self.sensor_config['initial_data_amounts'].copy()
            current_data = initial_data.copy()

            steps = [0]
            completion_rates = [0.0]  # 初始完成率为0

            # 限制通信日志数据点
            comm_log = self.communication_log
            max_comm_points = 100
            if len(comm_log) > max_comm_points:
                indices = np.linspace(0, len(comm_log) - 1, max_comm_points, dtype=int)
                comm_log = [comm_log[i] for i in indices]

            for log_entry in comm_log:
                step = log_entry['step']
                sensor_id = log_entry['sensor_id']
                collected = log_entry['data_collected']

                current_data[sensor_id] -= collected
                total_collected = np.sum(initial_data) - np.sum(current_data)
                completion_rate = (total_collected / np.sum(initial_data)) * 100

                steps.append(step)
                completion_rates.append(completion_rate)

            ax4.plot(steps, completion_rates, 'b-', linewidth=2, label='Overall Progress')

        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('Completion Rate (%)')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 105)

        # 添加完成目标线
        ax4.axhline(y=100, color='green', linestyle='--', alpha=0.7, label='Target (100%)')
        ax4.legend(fontsize=8)

        # 使用更安全的布局调整
        try:
            plt.tight_layout(pad=1.5)
        except:
            try:
                plt.tight_layout(pad=0.5)  # 尝试更小的padding
            except:
                plt.subplots_adjust(hspace=0.4, wspace=0.4)  # 备用方案

        if save_path:
            try:
                # 预检查图像尺寸
                fig_width_inch, fig_height_inch = fig.get_size_inches()
                dpi = 50  # 使用非常低的DPI
                width_pixels = fig_width_inch * dpi
                height_pixels = fig_height_inch * dpi

                # 验证像素尺寸不超过限制
                max_dimension = 32000  # 安全边界，远小于65536
                if width_pixels > max_dimension or height_pixels > max_dimension:
                    print(f"警告: 图像尺寸过大 ({width_pixels}x{height_pixels})，降低DPI重试")
                    dpi = min(dpi, int(max_dimension / max(fig_width_inch, fig_height_inch)))

                # 使用最安全的保存参数
                plt.savefig(save_path,
                            dpi=dpi,
                            bbox_inches=None,  # 不使用tight
                            facecolor='white',
                            format='png')  # 移除不支持的optimize参数
                print(f"图像已保存: {save_path} (尺寸: {int(fig_width_inch * dpi)}x{int(fig_height_inch * dpi)})")
                plt.close()
            except Exception as e:
                print(f"图像保存失败，使用默认显示: {e}")
                try:
                    plt.show()
                except:
                    pass
                plt.close()
        else:
            try:
                plt.show()
            except Exception as e:
                print(f"图像显示失败: {e}")
                plt.close()

    def get_performance_metrics(self) -> Dict:
        """获取性能指标"""
        total_initial_data = np.sum(self.sensor_config['initial_data_amounts'])
        remaining_data = np.sum(self.sensor_data_amounts)

        metrics = {
            'data_completion_rate': 1.0 - (remaining_data / total_initial_data),
            'average_uncertainty': np.mean(self.uncertainty_model.uncertainty_radii),
            'max_uncertainty': np.max(self.uncertainty_model.uncertainty_radii),
            'total_steps': self.current_step,
            'communication_count': len(self.communication_log),
            'localization_count': len(self.localization_log),
            'trajectory_length': len(self.trajectory),
            'final_position': self.drone_position.copy()
        }

        return metrics

    def _calculate_data_collection(self, snr_db: float) -> float:
        """计算数据采集量 - 兼容性方法"""
        return self._calculate_data_collection_v2(snr_db)

    def _perform_communication(self) -> Dict:
        """执行通信动作 - 兼容性方法 (已弃用，请使用_perform_communication_v3)"""
        print("警告: 使用了已弃用的_perform_communication方法")
        return self._perform_communication_v3()

    def _perform_localization(self) -> Dict:
        """执行定位动作 - 兼容性方法 (已弃用，请使用_perform_localization_v3)"""
        print("警告: 使用了已弃用的_perform_localization方法")
        return self._perform_localization_v3()

    def _calculate_completion_reward(self) -> float:
        """计算完成任务奖励 - 兼容性方法 (已弃用)"""
        print("警告: 使用了已弃用的_calculate_completion_reward方法")
        return 0.0  # 新版本不再使用单独的完成奖励

    def get_reward_statistics(self) -> Dict:
        """获取奖励统计信息，用于分析训练进展"""
        if not self.reward_components:
            return {}

        # 计算最近10步的奖励统计
        recent_rewards = self.reward_components[-10:] if len(self.reward_components) >= 10 else self.reward_components

        stats = {
            'total_episodes': len(self.reward_components),
            'recent_average_total': np.mean([r['total_reward'] for r in recent_rewards]),
            'recent_average_baseline': np.mean([r['baseline_penalty'] for r in recent_rewards]),
            'recent_average_action': np.mean([r['action_reward'] for r in recent_rewards]),
            'recent_average_progress': np.mean([r['progress_reward'] for r in recent_rewards]),
            'recent_average_efficiency': np.mean([r['efficiency_bonus'] for r in recent_rewards]),
            'recent_average_stage': np.mean([r['stage_bonus'] for r in recent_rewards]),
            'current_learning_stage': self.learning_stage,
            'overall_progress': self._get_overall_progress(),
            'reward_trend': 'increasing' if len(recent_rewards) > 1 and
                                            recent_rewards[-1]['total_reward'] > recent_rewards[0][
                                                'total_reward'] else 'stable',

            # 技能系统统计
            'communication_skill': self.communication_skill,
            'localization_skill': self.localization_skill,
            'communication_success_rate': self.successful_communications / max(1, self.communication_attempts),
            'localization_success_rate': self.successful_localizations / max(1, self.localization_attempts),
            'total_communication_attempts': self.communication_attempts,
            'total_localization_attempts': self.localization_attempts
        }

        return stats

    def _update_communication_skill(self):
        """更新通信技能等级"""
        if self.communication_attempts > 0:
            # 基于成功率和经验累积更新技能
            success_rate = self.successful_communications / self.communication_attempts
            experience_factor = min(1.0, self.communication_attempts / 50.0)  # 50次尝试达到满经验

            # 技能等级 = 成功率 * 经验因子
            self.communication_skill = min(1.0, success_rate * experience_factor)

    def _update_localization_skill(self):
        """更新定位技能等级"""
        if self.localization_attempts > 0:
            # 基于成功率和经验累积更新技能
            success_rate = self.successful_localizations / self.localization_attempts
            experience_factor = min(1.0, self.localization_attempts / 50.0)  # 50次尝试达到满经验

            # 技能等级 = 成功率 * 经验因子
            self.localization_skill = min(1.0, success_rate * experience_factor)


# 兼容性适配器
class WetEnv(DroneEnv):
    """为了兼容现有代码的适配器类"""

    def __init__(self, data=None):
        # 忽略data参数，使用默认配置
        super().__init__()

        # 兼容原有接口
        self.machine_qty = self.num_sensors
        self.action_len = 2  # 离散动作数量

        # 模拟原有记录
        self.record_eqp_plan = []
        self.record_wip_move = []
        self.record_acid_density = []
        self.record_acid_lifetime = []

    @property
    def observation_space(self):
        """兼容原有观察空间接口"""
        return spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._calculate_state_dimension(),),
            dtype=np.float32
        )

    @property
    def action_space(self):
        """兼容原有动作空间接口"""
        # 返回离散动作空间维度
        return spaces.Box(
            low=0, high=1,
            shape=(self.num_sensors,),
            dtype=np.int32
        )