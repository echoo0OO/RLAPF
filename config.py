"""
无人机传感器网络强化学习训练配置文件
支持不同的训练模式和环境设置
"""

import torch

class TrainingConfig:
    """训练配置类"""
    
    def __init__(self, mode='standard'):
        """
        初始化训练配置
        
        Args:
            mode: 训练模式 ['standard', 'fast', 'stable', 'debug']
        """
        self.mode = mode
        self._load_config()
    
    def _load_config(self):
        """根据模式加载相应配置"""
        
        # 基础配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.random_seed = 1
        self.experiment_name = f'drone_training_{self.mode}'
        
        # 环境配置
        self.env_config = {
            'area_size': (1000.0, 1000.0),
            'num_sensors': 5,
            'max_speed': 30.0,
            'time_slot': 1.0,
            'max_episode_steps': 500
        }
        
        if self.mode == 'standard':
            self._load_standard_config()
        elif self.mode == 'fast':
            self._load_fast_config()
        elif self.mode == 'stable':
            self._load_stable_config()
        elif self.mode == 'debug':
            self._load_debug_config()
        elif self.mode == 'gpu_optimized':
            self._load_gpu_optimized_config()
        elif self.mode == 'stable_gpu':
            self._load_stable_gpu_config()
        else:
            raise ValueError(f"未知的训练模式: {self.mode}")

    def _load_standard_config(self):
        """标准训练配置 - 平衡性能和稳定性"""
        self.max_episodes = 1000
        self.buffer_size = 6000
        self.batch_size = 64
        self.agent_save_freq = 100
        self.agent_update_freq = 10

        # 更保守的超参数设置
        self.lr_actor = 0.00005  # 从0.0001进一步降低到0.00005
        self.lr_critic = 0.00005  # 从0.0001进一步降低到0.00005
        self.lr_std = 0.0005  # 从0.001降低到0.0005
        self.lr_decay_rate = 0.9995  # 从0.998改为0.9995，更慢的衰减
        self.mid_dim = [256, 128, 64]
        self.gamma = 0.99
        self.lam = 0.95
        self.epochs_update = 8  # 从10降为8
        self.v_iters = 1
        self.target_kl_dis = 0.005  # 从0.01进一步降为0.005
        self.target_kl_con = 0.01   # 从0.02进一步降为0.01
        self.eps_clip = 0.08  # 从0.1进一步降为0.08
        self.max_norm_grad = 1.5  # 从2.0降为1.5
        self.init_log_std = -2.0  # 从-1.5改为-2.0，更保守的初始探索
        self.coeff_dist_entropy = 0.005  # 从0.01降为0.005
        self.if_use_active_selection = False

    def _load_fast_config(self):
        """快速训练配置 - 适用于测试和调试"""
        self.max_episodes = 200
        self.buffer_size = 2000
        self.batch_size = 32
        self.agent_save_freq = 50
        self.agent_update_freq = 5

        self.lr_actor = 0.0001  # 从0.0003降为0.0001
        self.lr_critic = 0.0001  # 从0.0003降为0.0001
        self.lr_std = 0.001  # 从0.002降为0.001
        self.lr_decay_rate = 0.998  # 从0.995改为0.998
        self.mid_dim = [128, 64]
        self.gamma = 0.99
        self.lam = 0.9
        self.epochs_update = 5
        self.v_iters = 1
        self.target_kl_dis = 0.01  # 从0.02降为0.01
        self.target_kl_con = 0.02  # 从0.04降为0.02
        self.eps_clip = 0.12  # 从0.15降为0.12
        self.max_norm_grad = 2.5  # 从3.0降为2.5
        self.init_log_std = -1.5  # 从-1.0改为-1.5
        self.coeff_dist_entropy = 0.01  # 从0.02降为0.01
        self.if_use_active_selection = False

    def _load_stable_config(self):
        """稳定训练配置 - 最大化稳定性"""
        self.max_episodes = 2000
        self.buffer_size = 8000
        self.batch_size = 128
        self.agent_save_freq = 200
        self.agent_update_freq = 20

        self.lr_actor = 0.00005
        self.lr_critic = 0.00005
        self.lr_std = 0.0005
        self.lr_decay_rate = 0.999
        self.mid_dim = [512, 256, 128]
        self.gamma = 0.995
        self.lam = 0.97
        self.epochs_update = 15
        self.v_iters = 2
        self.target_kl_dis = 0.005
        self.target_kl_con = 0.01
        self.eps_clip = 0.05
        self.max_norm_grad = 1.0
        self.init_log_std = -2.0
        self.coeff_dist_entropy = 0.005
        self.if_use_active_selection = False

    def _load_debug_config(self):
        """调试配置 - 小规模快速测试"""
        self.max_episodes = 50
        self.buffer_size = 1000
        self.batch_size = 16
        self.agent_save_freq = 10
        self.agent_update_freq = 5

        self.lr_actor = 0.001
        self.lr_critic = 0.001
        self.lr_std = 0.01
        self.lr_decay_rate = 0.99
        self.mid_dim = [64, 32, 16]
        self.gamma = 0.9
        self.lam = 0.8
        self.epochs_update = 3
        self.v_iters = 1
        self.target_kl_dis = 0.05
        self.target_kl_con = 0.1
        self.eps_clip = 0.3
        self.max_norm_grad = 5.0
        self.init_log_std = 0.0
        self.coeff_dist_entropy = 0.05
        self.if_use_active_selection = False

    def _load_gpu_optimized_config(self):
        """GPU优化配置 - 最大化GPU利用率和训练速度"""
        self.max_episodes = 1000
        self.buffer_size = 12000  # 增大缓冲区，更好利用GPU批处理
        self.batch_size = 256     # 大幅增加批处理大小，充分利用GPU
        self.agent_save_freq = 200  # 减少保存频率
        self.agent_update_freq = 20  # 减少更新频率，累积更多经验

        # 适中的学习率，快速收敛
        self.lr_actor = 0.0001
        self.lr_critic = 0.0001
        self.lr_std = 0.0008
        self.lr_decay_rate = 0.998
        self.mid_dim = [256, 128]  # 简化网络结构，减少计算量
        self.gamma = 0.99
        self.lam = 0.95
        self.epochs_update = 6    # 减少训练epochs，加快每次更新
        self.v_iters = 1
        self.target_kl_dis = 0.01
        self.target_kl_con = 0.02
        self.eps_clip = 0.1
        self.max_norm_grad = 2.0
        self.init_log_std = -1.5
        self.coeff_dist_entropy = 0.01
        self.if_use_active_selection = False

        # GPU优化的环境配置
        self.env_config = {
            'area_size': (1000.0, 1000.0),
            'num_sensors': 5,
            'max_speed': 30.0,
            'time_slot': 1.0,
            'max_episode_steps': 300,  # 减少episode长度
            'enable_fast_mode': True,  # 启用快速模式
            'gdop_resolution': 200.0,  # 降低GDOP分辨率
            'disable_realtime_visualization': True,  # 禁用实时可视化（而不是完全禁用）
            'reduce_computation': True  # 减少计算密集度
        }

    def _load_stable_gpu_config(self):
        """稳定GPU配置 - 平衡速度和稳定性"""
        self.max_episodes = 1000
        self.buffer_size = 8000   # 适中的缓冲区大小
        self.batch_size = 128     # 适中的批处理大小，提高稳定性
        self.agent_save_freq = 100
        self.agent_update_freq = 15  # 稍微频繁的更新

        # 更保守的学习率设置
        self.lr_actor = 0.00005   # 降低学习率提高稳定性
        self.lr_critic = 0.00005  # 降低学习率提高稳定性
        self.lr_std = 0.0005      # 降低标准差学习率
        self.lr_decay_rate = 0.9995  # 更慢的学习率衰减
        self.mid_dim = [256, 128, 64]  # 稍微深一点的网络，提高表达能力
        self.gamma = 0.995        # 更高的折扣因子
        self.lam = 0.96           # 提高GAE-Lambda参数
        self.epochs_update = 8    # 适中的更新epochs
        self.v_iters = 1
        self.target_kl_dis = 0.008  # 更严格的KL散度限制
        self.target_kl_con = 0.015  # 更严格的KL散度限制
        self.eps_clip = 0.08      # 更保守的裁剪范围
        self.max_norm_grad = 1.5  # 更严格的梯度裁剪
        self.init_log_std = -2.0  # 更保守的初始探索
        self.coeff_dist_entropy = 0.008  # 适中的熵系数
        self.if_use_active_selection = False

        # 稳定性优化的环境配置
        self.env_config = {
            'area_size': (1000.0, 1000.0),
            'num_sensors': 5,
            'max_speed': 30.0,
            'time_slot': 1.0,
            'max_episode_steps': 400,  # 稍长的episode
            'enable_fast_mode': True,
            'gdop_resolution': 150.0,  # 适中的GDOP分辨率
            'disable_realtime_visualization': True,
            'reduce_computation': True
        }

    def get_wetconfig(self):
        """获取wetConfig格式的配置"""
        version_no = f"DRONE-{self.mode.upper()}-20241201"
        return {
            "version_no": version_no,
            "mode": f"{self.mode}_train",
            "data_source": self.mode
        }

    def print_config(self):
        """打印当前配置信息"""
        print(f"\n=== 训练配置 ({self.mode.upper()}) ===")
        print(f"设备: {self.device}")
        print(f"最大训练轮次: {self.max_episodes}")
        print(f"缓冲区大小: {self.buffer_size}")
        print(f"批处理大小: {self.batch_size}")
        print(f"学习率 - Actor: {self.lr_actor}, Critic: {self.lr_critic}")
        print(f"网络结构: {self.mid_dim}")
        print(f"环境设置: {self.env_config}")
        print("=" * 40)

    @property
    def available_modes(self):
        """获取所有可用的配置模式"""
        return ['standard', 'fast', 'stable', 'debug', 'gpu_optimized', 'stable_gpu']

    def _select_config_by_mode(self):
        """根据模式选择对应的配置"""
        if self.mode == 'standard':
            self._load_standard_config()
        elif self.mode == 'fast':
            self._load_fast_config()
        elif self.mode == 'stable':
            self._load_stable_config()
        elif self.mode == 'debug':
            self._load_debug_config()
        elif self.mode == 'gpu_optimized':
            self._load_gpu_optimized_config()
        elif self.mode == 'stable_gpu':
            self._load_stable_gpu_config()
        else:
            print(f"警告: 未知配置模式 '{self.mode}'，使用标准配置")
            self._load_standard_config()