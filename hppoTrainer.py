import argparse
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams

from hppo.hppo_actionmask import *
from hppo.hppo_utils import *
from util.configuration import *
from util.util import *
from env.drone_env import DroneEnv
from config import TrainingConfig
from hppo.hppo_actionmask import PPO_Hybrid


def setup_chinese_fonts():
    """设置matplotlib中文字体支持"""
    try:
        # 尝试常见的中文字体
        chinese_fonts = [
            'SimHei',  # 黑体 (Windows)
            'Microsoft YaHei',  # 微软雅黑 (Windows)
            'DejaVu Sans',  # DejaVu (跨平台)
            'WenQuanYi Micro Hei',  # 文泉驿微米黑 (Linux)
            'Noto Sans CJK SC',  # 思源黑体 (跨平台)
            'PingFang SC',  # 苹方 (macOS)
            'Heiti SC',  # 黑体 (macOS)
            'STHeiti',  # 华文黑体 (macOS)
        ]

        # 尝试设置中文字体
        font_set = False
        for font_name in chinese_fonts:
            try:
                rcParams['font.sans-serif'] = [font_name] + rcParams['font.sans-serif']
                rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

                # 测试字体是否可用
                fig, ax = plt.subplots(figsize=(1, 1))
                ax.text(0.5, 0.5, '测试', fontsize=10)
                plt.close(fig)

                print(f"✓ 中文字体设置成功: {font_name}")
                font_set = True
                break
            except Exception:
                continue

        if not font_set:
            print("⚠ 警告: 未找到合适的中文字体，将使用英文标签")
            return False

        return True

    except Exception as e:
        print(f"⚠ 字体设置失败: {e}，将使用英文标签")
        return False


@timer
class Trainer(object):
    """
    优化的强化学习训练器
    支持配置文件驱动的训练和改进的监控系统
    """

    def __init__(self, config=None):
        """
        初始化训练器

        Args:
            config: TrainingConfig实例或训练模式字符串
        """
        if isinstance(config, str):
            self.config = TrainingConfig(config)
        elif isinstance(config, TrainingConfig):
            self.config = config
        elif config is None:
            self.config = TrainingConfig('standard')
        else:
            # 兼容旧版参数格式
            self.config = self._convert_old_args(config)

        # 设置中文字体支持
        self.chinese_font_available = setup_chinese_fonts()

        # 从配置中加载参数
        self._load_config_params()

        # 初始化环境
        self.env = DroneEnv(self.config.env_config)
        self.machine_qty = self.env.num_sensors
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dis_dim = 1
        self.action_con_dim = 2
        self.action_len = 2

        # 训练历史记录
        self.history = {}
        self.total_rewards = []
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'data_completion_rates': [],
            'avg_uncertainties': [],
            'communication_counts': [],
            'localization_counts': []
        }

        # 创建保存目录
        self._setup_save_directory()

        # 打印配置信息
        self.config.print_config()

    def _load_config_params(self):
        """从配置中加载训练参数"""
        self.device = self.config.device
        self.max_episodes = self.config.max_episodes
        self.buffer_size = self.config.buffer_size
        self.batch_size = self.config.batch_size
        self.agent_save_freq = self.config.agent_save_freq
        self.agent_update_freq = self.config.agent_update_freq
        self.experiment_name = self.config.experiment_name

        # 学习器超参数
        self.mid_dim = self.config.mid_dim
        self.lr_actor = self.config.lr_actor
        self.lr_critic = self.config.lr_critic
        self.lr_std = self.config.lr_std
        self.lr_decay_rate = self.config.lr_decay_rate
        self.target_kl_dis = self.config.target_kl_dis
        self.target_kl_con = self.config.target_kl_con
        self.gamma = self.config.gamma
        self.lam = self.config.lam
        self.epochs_update = self.config.epochs_update
        self.v_iters = self.config.v_iters
        self.eps_clip = self.config.eps_clip
        self.max_norm_grad = self.config.max_norm_grad
        self.init_log_std = self.config.init_log_std
        self.coeff_dist_entropy = self.config.coeff_dist_entropy
        self.random_seed = self.config.random_seed
        self.if_use_active_selection = self.config.if_use_active_selection

    def _setup_save_directory(self):
        """设置保存目录"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.save_path = f'log/{self.config.mode}_{timestamp}/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # 保存配置信息
        config_info = {
            'mode': self.config.mode,
            'timestamp': timestamp,
            'device': str(self.device),
            'max_episodes': self.max_episodes,
            'env_config': self.config.env_config
        }
        np.save(self.save_path + 'training_config.npy', config_info)

    def _convert_old_args(self, args):
        """兼容旧版参数格式的转换器"""
        config = TrainingConfig('standard')

        # 从args中提取参数并设置到config
        if hasattr(args, 'max_episodes'):
            config.max_episodes = args.max_episodes
        if hasattr(args, 'lr_actor'):
            config.lr_actor = args.lr_actor
        if hasattr(args, 'experiment_name'):
            config.experiment_name = args.experiment_name

        return config

    def push_history_dis(self, obs, action_mask, act_dis, logp_act_dis, val):
        self.history = {
            'obs': obs,
            'action_mask': action_mask,
            'act_dis': act_dis,
            'logp_act_dis': logp_act_dis,
            'val': val
        }

    def push_history_hybrid(self, obs, action_mask, act_dis, act_con, logp_act_dis, logp_act_con, val):
        self.history = {
            'obs': obs,
            'action_mask': action_mask,
            'act_dis': act_dis,
            'act_con': act_con,
            'logp_act_dis': logp_act_dis,
            'logp_act_con': logp_act_con,
            'val': val
        }

    def unbatchify(self, value_action_logp):
        """
        解析智能体选择动作的返回值

        Args:
            value_action_logp: PPO_Hybrid.select_action的返回值 (state_value, actions, logp_actions)

        Returns:
            state_value, actions, logp_actions
        """
        if isinstance(value_action_logp, (tuple, list)) and len(value_action_logp) == 3:
            # 处理PPO_Hybrid返回的三元组格式
            state_value = value_action_logp[0]
            actions = value_action_logp[1]  # (action_dis, action_con)
            logp_actions = value_action_logp[2]  # (logp_dis, logp_con)
            return state_value, actions, logp_actions
        elif isinstance(value_action_logp, dict):
            # 处理字典格式（向后兼容）
            state_value = value_action_logp[0]
            actions = value_action_logp[1]
            logp_actions = value_action_logp[2]
            return state_value, actions, logp_actions
        else:
            raise ValueError(f"无法解析value_action_logp的格式: {type(value_action_logp)}")

    def initialize_agents(self, random_seed):
        """初始化智能体"""
        return PPO_Hybrid(
            self.obs_dim, self.action_dis_dim, self.action_len, self.action_con_dim,
            self.mid_dim, self.lr_actor, self.lr_critic, self.lr_decay_rate,
            self.buffer_size, self.target_kl_dis, self.target_kl_con,
            self.gamma, self.lam, self.epochs_update, self.v_iters,
            self.eps_clip, self.max_norm_grad, self.coeff_dist_entropy,
            random_seed, self.device, self.lr_std, self.init_log_std,
            self.if_use_active_selection
        )

    def train(self, worker_idx=1):
        """
        执行训练循环

        Args:
            worker_idx: 工作器索引
        """
        print(f"\n开始训练 - 工作器 {worker_idx}")

        agent = self.initialize_agents(worker_idx)
        norm_mean = np.zeros(self.obs_dim)
        norm_std = np.ones(self.obs_dim)

        i_episode = 0
        start_time = time.time()

        try:
            while i_episode < self.max_episodes:
                episode_start = time.time()

                # 收集一个episode
                with torch.no_grad():
                    episode_data = self._collect_episode(agent, norm_mean, norm_std, i_episode)

                # 记录训练指标
                self._record_episode_metrics(episode_data, i_episode)

                # 定期更新智能体
                if i_episode % self.agent_update_freq == 0 and i_episode > 0:
                    norm_mean = agent.buffer.filter()[0]
                    norm_std = agent.buffer.filter()[1]
                    if i_episode > self.agent_save_freq:
                        agent.update(self.batch_size)
                    agent.buffer.clear()

                # 定期保存和可视化
                if i_episode % self.agent_save_freq == 0 and i_episode > 0:
                    self._save_checkpoint(i_episode)
                    self._visualize_progress(i_episode)

                i_episode += 1

                # 估算剩余时间
                if i_episode % 50 == 0:
                    elapsed = time.time() - start_time
                    avg_time_per_episode = elapsed / i_episode
                    remaining_episodes = self.max_episodes - i_episode
                    eta = remaining_episodes * avg_time_per_episode

                    # GPU优化模式下减少详细输出
                    if self.config.mode == 'gpu_optimized':
                        # GPU模式下更简洁的进度报告
                        progress_percent = i_episode / self.max_episodes * 100
                        recent_avg = np.mean(self.total_rewards[-10:]) if len(self.total_rewards) >= 10 else 0
                        print(
                            f"进度: {i_episode}/{self.max_episodes} ({progress_percent:.1f}%) | 近期平均奖励: {recent_avg:.2f} | ETA: {eta / 60:.1f}分钟")
                    else:
                        print(
                            f"训练进度: {i_episode}/{self.max_episodes} ({i_episode / self.max_episodes * 100:.1f}%), 预计剩余时间: {eta / 60:.1f}分钟")

        except KeyboardInterrupt:
            print(f"\n训练被中断，已完成 {i_episode} 个episode")
        except Exception as e:
            print(f"\n训练过程中发生错误: {e}")
            raise
        finally:
            # 保存最终结果
            self._save_final_results()
            print(f"\n训练完成! 总用时: {(time.time() - start_time) / 60:.1f}分钟")

    def _collect_episode(self, agent, norm_mean, norm_std, episode_num):
        """收集一个episode的数据"""
        state, info = self.env.reset()
        action_mask = self.env.action_mask

        total_reward = 0
        step_count = 0
        episode_data = {
            'reward': 0,
            'steps': 0,
            'metrics': None,
            'success': False
        }

        while True:
            # 状态归一化
            observations_norm = (state - norm_mean) / np.maximum(norm_std, 1e-6)

            # 选择动作
            value_action_logp = agent.select_action(observations_norm, action_mask)
            values, actions, logp_actions = self.unbatchify(value_action_logp)

            # 构建环境动作
            action_dict = {
                'discrete': actions[0],
                'continuous': actions[1]
            }

            next_state, reward, done, truncated, info = self.env.step(action_dict)

            # 存储经验
            action_mask_flat = action_mask['discrete_mask']
            self.push_history_hybrid(state, action_mask_flat, actions[0], actions[1],
                                     logp_actions[0], logp_actions[1], values)
            agent.buffer.store_hybrid(
                self.history['obs'], self.history['action_mask'],
                self.history['act_dis'], self.history['act_con'],
                reward, self.history['val'],
                self.history['logp_act_dis'], self.history['logp_act_con']
            )

            total_reward += reward
            step_count += 1
            state = next_state
            action_mask = self.env.action_mask

            if done or truncated:
                episode_data['reward'] = total_reward
                episode_data['steps'] = step_count
                episode_data['metrics'] = self.env.get_performance_metrics()
                episode_data['success'] = done and not truncated

                agent.buffer.finish_path(0)
                break

        return episode_data

    def _record_episode_metrics(self, episode_data, episode_num):
        """记录episode指标"""
        self.total_rewards.append(episode_data['reward'])

        # 详细指标记录
        metrics = episode_data['metrics']
        if metrics:
            self.training_metrics['episode_rewards'].append(episode_data['reward'])
            self.training_metrics['episode_lengths'].append(episode_data['steps'])
            self.training_metrics['data_completion_rates'].append(metrics['data_completion_rate'])
            self.training_metrics['avg_uncertainties'].append(metrics['average_uncertainty'])
            self.training_metrics['communication_counts'].append(metrics['communication_count'])
            self.training_metrics['localization_counts'].append(metrics['localization_count'])

        # 获取奖励组件统计
        reward_stats = self.env.get_reward_statistics()

        # GPU优化模式下减少打印频率
        print_interval = 100 if self.config.mode == 'gpu_optimized' else 10  # GPU模式下改为100轮打印一次

        # 定期打印详细进度
        if episode_num % print_interval == 0:
            recent_rewards = self.total_rewards[-10:] if len(self.total_rewards) >= 10 else self.total_rewards
            avg_reward = np.mean(recent_rewards)

            # 奖励分析
            reward_trend = self._analyze_reward_trend()

            print(
                f"Episode {episode_num}: 奖励={episode_data['reward']:.2f}, 步数={episode_data['steps']}, 10轮平均={avg_reward:.2f}")

            if metrics:
                print(
                    f"  数据完成率: {metrics['data_completion_rate']:.2%}, 平均不确定性: {metrics['average_uncertainty']:.1f}m")

            # GPU优化模式下减少详细输出
            if self.config.mode != 'gpu_optimized':
                # 打印奖励组件分析
                if reward_stats:
                    print(f"  奖励组件 - 基线:{reward_stats['recent_average_baseline']:.3f}, "
                          f"动作:{reward_stats['recent_average_action']:.3f}, "
                          f"进度:{reward_stats['recent_average_progress']:.3f}")
                    print(f"  学习阶段: {reward_stats['current_learning_stage']}, "
                          f"整体进度: {reward_stats['overall_progress']:.1%}")
                    print(f"  技能等级 - 通信:{reward_stats['communication_skill']:.3f}, "
                          f"定位:{reward_stats['localization_skill']:.3f}")
                    print(f"  成功率 - 通信:{reward_stats['communication_success_rate']:.1%}, "
                          f"定位:{reward_stats['localization_success_rate']:.1%}")
                    print(f"  收敛趋势: {reward_trend}")
            else:
                # GPU优化模式下的简化输出
                if reward_stats:
                    print(
                        f"  学习阶段: {reward_stats['current_learning_stage']}, 进度: {reward_stats['overall_progress']:.1%}, 趋势: {reward_trend}")

        # 检查奖励收敛性（GPU优化模式下减少频率）
        convergence_check_interval = 200 if self.config.mode == 'gpu_optimized' else 50  # GPU模式下改为200轮检查一次
        if episode_num % convergence_check_interval == 0 and episode_num > 100:
            self._check_convergence_status(episode_num)

    def _analyze_reward_trend(self, window_size=20):
        """分析奖励趋势"""
        if len(self.total_rewards) < window_size:
            return "数据不足"

        recent_rewards = self.total_rewards[-window_size:]

        # 计算线性趋势
        x = np.arange(len(recent_rewards))
        slope, _ = np.polyfit(x, recent_rewards, 1)

        if slope > 0.01:
            return "📈 上升趋势"
        elif slope < -0.01:
            return "📉 下降趋势"
        else:
            return "➡️ 稳定"

    def _check_convergence_status(self, episode_num):
        """检查奖励收敛状态并提供稳定性分析"""
        if len(self.total_rewards) < 100:
            return

        # 分析最近50轮和之前50轮的奖励
        recent_50 = self.total_rewards[-50:]
        previous_50 = self.total_rewards[-100:-50]

        recent_mean = np.mean(recent_50)
        previous_mean = np.mean(previous_50)
        recent_std = np.std(recent_50)
        previous_std = np.std(previous_50)

        improvement = recent_mean - previous_mean
        stability_change = recent_std - previous_std

        print(f"\n🔍 收敛性分析 (Episode {episode_num}):")
        print(f"  最近50轮平均奖励: {recent_mean:.3f} (标准差: {recent_std:.3f})")
        print(f"  相比前50轮改善: {improvement:.3f}")
        print(f"  稳定性变化: {stability_change:.3f} ({'恶化' if stability_change > 0 else '改善'})")

        # 稳定性评估
        if recent_std > 1.5:
            print("  ⚠️ 奖励波动很大，建议降低学习率或增加平滑")
            self._provide_stability_suggestions(recent_std, recent_mean)
        elif recent_std > 0.8:
            print("  📊 奖励波动适中，可能需要微调参数")
        elif recent_std < 0.3:
            print("  ✅ 奖励非常稳定")
        else:
            print("  🔄 奖励稳定性良好")

        # 收敛判断
        if improvement > 0.1 and recent_std < 0.5:
            print("  ✅ 奖励正在收敛且稳定")
        elif improvement > 0.05:
            print("  📈 奖励持续改善")
        elif abs(improvement) < 0.02 and recent_std < 0.4:
            print("  🎯 可能已达到收敛条件")
        elif recent_std > 1.0:
            print("  ⚠️ 奖励波动过大，建议调整训练参数")
        else:
            print("  🔄 训练进行中，需要更多数据")

        # 检查奖励趋势
        if len(self.total_rewards) >= 200:
            self._analyze_long_term_trend(episode_num)

    def _provide_stability_suggestions(self, std_dev, mean_reward):
        """提供稳定性改善建议"""
        print(f"\n💡 稳定性改善建议:")

        if std_dev > 2.0:
            print("  🔧 建议大幅降低学习率 (减少到当前的 0.5倍)")
            print("  📊 考虑增加批处理大小")
            print("  🎯 使用更严格的梯度裁剪")
        elif std_dev > 1.5:
            print("  🔧 建议适度降低学习率 (减少到当前的 0.7倍)")
            print("  📈 考虑减少epochs_update")
            print("  🎛️ 调整eps_clip参数 (建议0.05-0.08)")

        if mean_reward < 0:
            print("  📉 平均奖励为负，可能需要:")
            print("     - 检查奖励函数设计")
            print("     - 降低探索强度")
            print("     - 增加基线奖励")
        elif mean_reward > 3.0:
            print("  📈 奖励较高但不稳定，建议:")
            print("     - 使用更保守的更新策略")
            print("     - 增加奖励平滑")

        print("  🔄 也可以尝试使用 'stable_gpu' 配置模式")

    def _analyze_long_term_trend(self, episode_num):
        """分析长期训练趋势"""
        if len(self.total_rewards) < 200:
            return

        # 分成4个阶段分析
        quarter_size = len(self.total_rewards) // 4
        quarters = [
            self.total_rewards[i * quarter_size:(i + 1) * quarter_size]
            for i in range(4)
        ]

        quarter_means = [np.mean(q) for q in quarters]
        quarter_stds = [np.std(q) for q in quarters]

        print(f"\n📊 长期趋势分析:")
        print(f"  各阶段平均奖励: {[f'{m:.2f}' for m in quarter_means]}")
        print(f"  各阶段稳定性: {[f'{s:.2f}' for s in quarter_stds]}")

        # 趋势判断
        if quarter_means[-1] > quarter_means[0] * 1.1:
            print("  📈 长期上升趋势明显")
        elif quarter_means[-1] < quarter_means[0] * 0.9:
            print("  📉 长期下降趋势，需要检查")
        else:
            print("  ➡️ 长期趋势平稳")

        # 稳定性趋势
        if quarter_stds[-1] < quarter_stds[0] * 0.8:
            print("  ✅ 稳定性显著改善")
        elif quarter_stds[-1] > quarter_stds[0] * 1.2:
            print("  ⚠️ 稳定性有所恶化")
        else:
            print("  🔄 稳定性基本保持")

    def _visualize_progress(self, episode_num):
        """可视化训练进度"""
        # GPU优化模式下完全跳过训练过程中的可视化，只在最终生成
        if self.config.mode == 'gpu_optimized':
            return  # 直接返回，不保存任何中间图像

        # 其他模式正常保存环境可视化
        save_path = self.save_path + f'environment_ep{episode_num}.png'
        self.env.visualize_environment(save_path)

        # 生成奖励分析图表
        if episode_num % 100 == 0 and len(self.total_rewards) > 20:
            self._create_reward_analysis_plot(episode_num)

    def _create_reward_analysis_plot(self, episode_num):
        """创建奖励分析图表"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        # 根据字体支持情况选择标签语言
        if self.chinese_font_available:
            labels = {
                'title1': '奖励收敛曲线',
                'title2': '奖励分布',
                'title3': '奖励组件变化',
                'title4': '学习曲线 (分段平均)',
                'xlabel1': '训练轮次',
                'ylabel1': '总奖励',
                'xlabel2': '奖励值',
                'ylabel2': '频次',
                'xlabel3': '最近步数',
                'ylabel3': '奖励值',
                'xlabel4': '训练阶段',
                'ylabel4': '平均奖励',
                'original_reward': '原始奖励',
                'moving_avg': '移动平均',
                'mean': '均值',
                'baseline_penalty': '基线惩罚',
                'action_reward': '动作奖励',
                'progress_reward': '进度奖励',
                'trend': '趋势',
                'main_title': f'训练分析 - Episode {episode_num}'
            }
        else:
            labels = {
                'title1': 'Reward Convergence',
                'title2': 'Reward Distribution',
                'title3': 'Reward Components',
                'title4': 'Learning Curve',
                'xlabel1': 'Episodes',
                'ylabel1': 'Total Reward',
                'xlabel2': 'Reward Value',
                'ylabel2': 'Frequency',
                'xlabel3': 'Recent Steps',
                'ylabel3': 'Reward Value',
                'xlabel4': 'Training Stage',
                'ylabel4': 'Average Reward',
                'original_reward': 'Original Reward',
                'moving_avg': 'Moving Average',
                'mean': 'Mean',
                'baseline_penalty': 'Baseline Penalty',
                'action_reward': 'Action Reward',
                'progress_reward': 'Progress Reward',
                'trend': 'Trend',
                'main_title': f'Training Analysis - Episode {episode_num}'
            }

        # 1. 奖励历史和移动平均
        episodes = range(len(self.total_rewards))
        ax1.plot(episodes, self.total_rewards, alpha=0.3, color='lightblue', label=labels['original_reward'])

        if len(self.total_rewards) > 10:
            window = min(20, len(self.total_rewards) // 5)
            moving_avg = np.convolve(self.total_rewards, np.ones(window) / window, mode='valid')
            ax1.plot(range(window - 1, len(self.total_rewards)), moving_avg,
                     color='red', linewidth=2, label=f"{labels['moving_avg']}({window})")

        ax1.set_title(labels['title1'])
        ax1.set_xlabel(labels['xlabel1'])
        ax1.set_ylabel(labels['ylabel1'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 奖励分布直方图
        ax2.hist(self.total_rewards, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(np.mean(self.total_rewards), color='red', linestyle='--',
                    label=f"{labels['mean']}: {np.mean(self.total_rewards):.2f}")
        ax2.set_title(labels['title2'])
        ax2.set_xlabel(labels['xlabel2'])
        ax2.set_ylabel(labels['ylabel2'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 奖励组件分析（如果有数据）
        if hasattr(self.env, 'reward_components') and self.env.reward_components:
            components = self.env.reward_components[-min(50, len(self.env.reward_components)):]

            # 检查组件是否包含新的字段名
            if len(components) > 0 and 'baseline_penalty' in components[0]:
                baseline_penalties = [c['baseline_penalty'] for c in components]
                action_rewards = [c['action_reward'] for c in components]
                progress_rewards = [c['progress_reward'] for c in components]

                x = range(len(components))
                ax3.plot(x, baseline_penalties, label=labels['baseline_penalty'], alpha=0.8, color='red')
                ax3.plot(x, action_rewards, label=labels['action_reward'], alpha=0.8, color='blue')
                ax3.plot(x, progress_rewards, label=labels['progress_reward'], alpha=0.8, color='green')

                ax3.set_title(labels['title3'])
                ax3.set_xlabel(labels['xlabel3'])
                ax3.set_ylabel(labels['ylabel3'])
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            else:
                # 如果没有新字段，显示提示信息
                ax3.text(0.5, 0.5, '奖励组件数据不可用\n(可能使用了旧的奖励系统)',
                         transform=ax3.transAxes, ha='center', va='center', fontsize=12)
                ax3.set_title(labels['title3'])
        else:
            ax3.text(0.5, 0.5, '无奖励组件数据', transform=ax3.transAxes, ha='center', va='center')
            ax3.set_title(labels['title3'])

        # 4. 学习曲线（分段分析）
        if len(self.total_rewards) > 50:
            # 将训练过程分为10段
            segment_size = max(5, len(self.total_rewards) // 10)
            segments = []
            segment_means = []

            for i in range(0, len(self.total_rewards), segment_size):
                segment = self.total_rewards[i:i + segment_size]
                if len(segment) >= 3:  # 至少3个数据点
                    segments.append(i // segment_size)
                    segment_means.append(np.mean(segment))

            ax4.plot(segments, segment_means, marker='o', linewidth=2, markersize=6, color='purple')
            ax4.set_title(labels['title4'])
            ax4.set_xlabel(labels['xlabel4'])
            ax4.set_ylabel(labels['ylabel4'])
            ax4.grid(True, alpha=0.3)

            # 添加趋势线
            if len(segments) > 2:
                z = np.polyfit(segments, segment_means, 1)
                p = np.poly1d(z)
                ax4.plot(segments, p(segments), "--", alpha=0.8, color='red',
                         label=f"{labels['trend']} ({z[0]:.3f})")
                ax4.legend()

        plt.suptitle(labels['main_title'], fontsize=16)
        plt.tight_layout()
        plt.savefig(self.save_path + f'reward_analysis_ep{episode_num}.png',
                    dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"  💾 奖励分析图表已保存: reward_analysis_ep{episode_num}.png")

    def _save_checkpoint(self, episode_num):
        """保存训练检查点"""
        checkpoint_data = {
            'episode': episode_num,
            'total_rewards': self.total_rewards,
            'training_metrics': self.training_metrics,
            'config': self.config.mode
        }
        np.save(self.save_path + f'checkpoint_ep{episode_num}.npy', checkpoint_data)

    def _save_final_results(self):
        """保存最终训练结果"""
        # 保存详细的训练历史
        final_data = {
            'total_rewards': self.total_rewards,
            'training_metrics': self.training_metrics,
            'config_mode': self.config.mode,
            'final_performance': self.env.get_performance_metrics()
        }
        np.save(self.save_path + 'final_training_results.npy', final_data)

        # 生成训练总结
        self._generate_training_summary()

    def _generate_training_summary(self):
        """生成训练总结报告"""
        print(f"\n{'=' * 50}")
        print("训练总结报告")
        print(f"{'=' * 50}")

        if self.total_rewards:
            print(f"总训练轮次: {len(self.total_rewards)}")
            print(f"平均奖励: {np.mean(self.total_rewards):.3f}")
            print(f"最高奖励: {np.max(self.total_rewards):.3f}")
            print(f"最低奖励: {np.min(self.total_rewards):.3f}")
            print(f"奖励标准差: {np.std(self.total_rewards):.3f}")

            if len(self.total_rewards) >= 100:
                last_100 = self.total_rewards[-100:]
                print(f"最后100轮平均奖励: {np.mean(last_100):.3f}")

            if len(self.total_rewards) >= 50:
                first_half = self.total_rewards[:len(self.total_rewards) // 2]
                second_half = self.total_rewards[len(self.total_rewards) // 2:]
                improvement = np.mean(second_half) - np.mean(first_half)
                print(f"训练改善程度: {improvement:.3f}")

        # 环境性能指标
        if hasattr(self, 'env'):
            final_metrics = self.env.get_performance_metrics()
            print(f"\n最终环境性能:")
            print(f"数据收集完成率: {final_metrics['data_completion_rate']:.2%}")
            print(f"平均定位不确定性: {final_metrics['average_uncertainty']:.2f}m")
            print(f"通信次数: {final_metrics['communication_count']}")
            print(f"定位次数: {final_metrics['localization_count']}")

    def save_data(self):
        """保存训练数据并生成可视化"""
        if not self.total_rewards:
            print("警告: 没有训练数据可保存")
            return

        # 保存基础数据
        np.save(self.save_path + 'total_reward_history.npy', self.total_rewards)

        # 生成综合分析图表
        self._create_comprehensive_plots()

    def _create_comprehensive_plots(self):
        """创建综合分析图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 根据字体支持情况选择标签语言
        if self.chinese_font_available:
            labels = {
                'title1': '总奖励历史',
                'title2': '奖励分布',
                'title3': 'Episode长度变化',
                'title4': '学习曲线(分段平均)',
                'xlabel1': '训练轮次',
                'ylabel1': '总奖励',
                'xlabel2': '奖励值',
                'ylabel2': '频次',
                'xlabel3': '训练轮次',
                'ylabel3': '步数',
                'xlabel4': '训练阶段',
                'ylabel4': '平均奖励',
                'moving_avg': '移动平均',
                'mean': '平均值'
            }
        else:
            labels = {
                'title1': 'Total Reward History',
                'title2': 'Reward Distribution',
                'title3': 'Episode Length Changes',
                'title4': 'Learning Curve (Segmented)',
                'xlabel1': 'Episodes',
                'ylabel1': 'Total Reward',
                'xlabel2': 'Reward Value',
                'ylabel2': 'Frequency',
                'xlabel3': 'Episodes',
                'ylabel3': 'Steps',
                'xlabel4': 'Training Stage',
                'ylabel4': 'Average Reward',
                'moving_avg': 'Moving Average',
                'mean': 'Mean'
            }

        # 1. 总奖励历史
        axes[0, 0].plot(self.total_rewards, alpha=0.7, color='blue', linewidth=1)
        if len(self.total_rewards) > 20:
            window_size = min(20, len(self.total_rewards) // 5)
            moving_avg = np.convolve(self.total_rewards, np.ones(window_size) / window_size, mode='valid')
            axes[0, 0].plot(range(window_size - 1, len(self.total_rewards)), moving_avg,
                            color='red', linewidth=2, label=f"{labels['moving_avg']} ({window_size})")
            axes[0, 0].legend()
        axes[0, 0].set_title(labels['title1'])
        axes[0, 0].set_xlabel(labels['xlabel1'])
        axes[0, 0].set_ylabel(labels['ylabel1'])
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 奖励分布
        axes[0, 1].hist(self.total_rewards, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].axvline(np.mean(self.total_rewards), color='red', linestyle='--',
                           label=f"{labels['mean']}: {np.mean(self.total_rewards):.2f}")
        axes[0, 1].set_title(labels['title2'])
        axes[0, 1].set_xlabel(labels['xlabel2'])
        axes[0, 1].set_ylabel(labels['ylabel2'])
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 训练指标（如果有）
        if self.training_metrics['episode_lengths']:
            axes[1, 0].plot(self.training_metrics['episode_lengths'], color='orange', alpha=0.7)
            axes[1, 0].set_title(labels['title3'])
            axes[1, 0].set_xlabel(labels['xlabel3'])
            axes[1, 0].set_ylabel(labels['ylabel3'])
            axes[1, 0].grid(True, alpha=0.3)

        # 4. 学习曲线对比
        if len(self.total_rewards) > 50:
            segment_size = len(self.total_rewards) // 10
            segments = [self.total_rewards[i:i + segment_size] for i in range(0, len(self.total_rewards), segment_size)]
            segment_means = [np.mean(seg) for seg in segments if seg]

            axes[1, 1].plot(range(len(segment_means)), segment_means, marker='o', linewidth=2, color='purple')
            axes[1, 1].set_title(labels['title4'])
            axes[1, 1].set_xlabel(labels['xlabel4'])
            axes[1, 1].set_ylabel(labels['ylabel4'])
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_path + 'training_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='Device.')
    parser.add_argument('--max_episodes', type=int, default=1001, help='The max episodes per agent per run.')
    parser.add_argument('--buffer_size', type=int, default=6000, help='The maximum size of the PPOBuffer.')
    parser.add_argument('--batch_size', type=int, default=64, help='The sample batch size.')
    parser.add_argument('--agent_save_freq', type=int, default=10, help='The frequency of the agent saving.')
    parser.add_argument('--agent_update_freq', type=int, default=10, help='The frequency of the agent updating.')
    parser.add_argument('--lr_actor', type=float, default=0.0001,
                        help='The learning rate of actor_con.')  # 从0.0003降为0.0001
    parser.add_argument('--lr_actor_param', type=float, default=0.0001,
                        help='The learning rate of critic.')  # 从0.001降为0.0001
    parser.add_argument('--lr_std', type=float, default=0.001, help='The learning rate of log_std.')  # 从0.004降为0.001
    parser.add_argument('--lr_decay_rate', type=float, default=0.998,
                        help='Factor of learning rate decay.')  # 从0.995改为0.998，衰减更慢
    parser.add_argument('--mid_dim', type=list, default=[256, 128, 64], help='The middle dimensions of both nets.')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted of future rewards.')
    parser.add_argument('--lam', type=float, default=0.95,  # 从0.8改为0.95，提高GAE估计稳定性
                        help='Lambda for GAE-Lambda. (Always between 0 and 1, close to 1.)')
    parser.add_argument('--epochs_update', type=int, default=10,  # 从20降为10，减少每次更新的epochs
                        help='Maximum number of gradient descent steps to take on policy loss per epoch. (Early stopping may cause optimizer to take fewer than this.)')
    parser.add_argument('--v_iters', type=int, default=1,
                        help='Number of gradient descent steps to take on value function per epoch.')
    parser.add_argument('--target_kl_dis', type=float, default=0.01,  # 从0.025降为0.01，更严格的KL限制
                        help='Roughly what KL divergence we think is appropriate between new and old policies after an update. This will get used for early stopping. (Usually small, 0.01 or 0.05.)')
    parser.add_argument('--target_kl_con', type=float, default=0.02,  # 从0.05降为0.02，更严格的KL限制
                        help='Roughly what KL divergence we think is appropriate between new and old policies after an update. This will get used for early stopping. (Usually small, 0.01 or 0.05.)')
    parser.add_argument('--eps_clip', type=float, default=0.1,
                        help='The clip ratio when calculate surr.')  # 从0.2降为0.1，更保守的裁剪
    parser.add_argument('--max_norm_grad', type=float, default=2.0,
                        help='max norm of the gradients.')  # 从5.0降为2.0，更严格的梯度裁剪
    parser.add_argument('--init_log_std', type=float, default=-1.5,  # 从-1.0改为-1.5，初始探索更保守
                        help='The initial log_std of Normal in continuous pattern.')
    parser.add_argument('--coeff_dist_entropy', type=float, default=0.01,  # 从0.005改为0.01，增加探索
                        help='The coefficient of distribution entropy.')
    parser.add_argument('--random_seed', type=int, default=1, help='The random seed.')
    parser.add_argument('--record_mark', type=str, default='renaissance',
                        help='The mark that differentiates different experiments.')
    parser.add_argument('--if_use_active_selection', type=bool, default=False,
                        help='Whether use active selection in the exploration.')
    parser.add_argument('--experiment_name', type=str, default='optimized_drone',
                        help='The name of the experiment.')  # 改名以区分优化版本

    version_no = "RTS-T2-20240507164500"
    mode = "main_train"
    data_source = "pk"
    wetConfig = {"version_no": version_no, "mode": mode, "data_source": data_source}
    parser.add_argument('--wetConfig', type=dict, default=wetConfig, help='wet config')

    args = parser.parse_args()

    # training through multiprocess
    trainer = Trainer(args)
    trainer.train(1)
    trainer.save_data()
