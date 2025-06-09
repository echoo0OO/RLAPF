#!/usr/bin/env python3
"""
无人机传感器网络强化学习训练主程序
支持多种训练模式和配置
"""

import argparse
import sys
from hppoTrainer import Trainer
from config import TrainingConfig

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='无人机传感器网络强化学习训练')
    
    parser.add_argument('--mode', type=str, default='standard', 
                       choices=['standard', 'fast', 'stable', 'debug'],
                       help='训练模式选择')
    
    parser.add_argument('--episodes', type=int, default=None,
                       help='自定义训练轮次数（覆盖配置文件设置）')
    
    parser.add_argument('--test-only', action='store_true',
                       help='仅运行快速测试，不进行完整训练')
    
    parser.add_argument('--config-info', action='store_true',
                       help='显示配置信息并退出')
    
    args = parser.parse_args()
    
    # 显示配置信息
    if args.config_info:
        print("可用的训练模式配置:")
        for mode in ['standard', 'fast', 'stable', 'debug']:
            config = TrainingConfig(mode)
            print(f"\n{mode.upper()}:")
            print(f"  训练轮次: {config.max_episodes}")
            print(f"  学习率: Actor={config.lr_actor}, Critic={config.lr_critic}")
            print(f"  网络结构: {config.mid_dim}")
            print(f"  适用场景: {_get_mode_description(mode)}")
        return
    
    # 快速测试模式
    if args.test_only:
        print("运行快速测试...")
        try:
            from quick_test import main as test_main
            success = test_main()
            if success:
                print("✓ 快速测试通过")
                sys.exit(0)
            else:
                print("✗ 快速测试失败")
                sys.exit(1)
        except ImportError:
            print("错误: 无法导入测试模块")
            sys.exit(1)
    
    # 正常训练模式
    print(f"启动训练 - 模式: {args.mode}")
    
    # 创建配置
    config = TrainingConfig(args.mode)
    
    # 自定义训练轮次
    if args.episodes is not None:
        config.max_episodes = args.episodes
        print(f"自定义训练轮次: {args.episodes}")
    
    # 创建训练器并开始训练
    try:
        trainer = Trainer(config)
        trainer.train()
        trainer.save_data()
        
        print("\n训练完成! 查看 log/ 目录获取详细结果")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n训练过程中发生错误: {e}")
        sys.exit(1)

def _get_mode_description(mode):
    """获取模式描述"""
    descriptions = {
        'standard': '平衡性能和稳定性，适合正常训练',
        'fast': '快速训练，适合测试和调试',
        'stable': '最大化稳定性，适合生产环境',
        'debug': '小规模测试，快速验证功能'
    }
    return descriptions.get(mode, '未知模式')

if __name__ == '__main__':
    main() 