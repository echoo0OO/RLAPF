#!/usr/bin/env python3
"""
GPU优化训练脚本
针对CUDA设备优化的高速训练配置
"""

import torch
import time
import os
from hppoTrainer import Trainer
from config import TrainingConfig


def check_gpu_status():
    """检查GPU状态"""
    print("=" * 50)
    print("GPU状态检查")
    print("=" * 50)

    if torch.cuda.is_available():
        print(f"✅ CUDA可用")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.current_device()}")
        print(f"GPU名称: {torch.cuda.get_device_name()}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

        # 检查GPU内存使用
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"可用内存: {torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()} bytes")

        return True
    else:
        print("❌ CUDA不可用，将使用CPU训练（会很慢）")
        return False


def optimize_pytorch_settings():
    """优化PyTorch设置以提高GPU性能"""
    print("\n" + "=" * 50)
    print("PyTorch性能优化设置")
    print("=" * 50)

    # 启用cuDNN的自动优化
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("✅ 启用cuDNN基准模式")

        # 设置GPU内存分配策略
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        print("✅ 优化CUDA内存分配")

    # 设置线程数
    torch.set_num_threads(4)  # 不要设置太高，避免与GPU竞争
    print(f"✅ 设置CPU线程数: {torch.get_num_threads()}")

    # 禁用梯度检查以提高速度（在生产环境中）
    torch.autograd.set_detect_anomaly(False)
    print("✅ 禁用梯度异常检测（提高速度）")


def main():
    """主训练函数"""
    print("🚀 启动GPU优化训练")
    print("=" * 50)

    # 检查GPU状态
    gpu_available = check_gpu_status()

    # 优化PyTorch设置
    optimize_pytorch_settings()

    # 检查奖励波动问题，提供配置建议
    print(f"\n📋 配置模式选择:")
    print(f"  🚄 gpu_optimized: 最大化训练速度 (可能有奖励波动)")
    print(f"  🎯 stable_gpu: 平衡速度和稳定性 (推荐)")
    print(f"  🔧 stable: 最大化稳定性 (较慢)")

    # 默认使用稳定GPU配置，除非用户有特殊需求
    config_mode = 'stable_gpu'  # 从gpu_optimized改为stable_gpu

    # 创建配置
    config = TrainingConfig(config_mode)

    # 如果GPU不可用，降级到CPU优化配置
    if not gpu_available:
        print("\n⚠️ GPU不可用，切换到CPU优化模式")
        config.batch_size = min(config.batch_size, 64)  # 限制批处理大小
        config.buffer_size = min(config.buffer_size, 4000)  # 限制缓冲区
        config.device = torch.device('cpu')
        # CPU模式下进一步降低学习率提高稳定性
        config.lr_actor *= 0.7
        config.lr_critic *= 0.7

    print(f"\n📊 训练配置概览:")
    print(f"配置模式: {config_mode.upper()}")
    print(f"设备: {config.device}")
    print(f"批处理大小: {config.batch_size}")
    print(f"缓冲区大小: {config.buffer_size}")
    print(f"更新频率: {config.agent_update_freq}")
    print(f"网络结构: {config.mid_dim}")
    print(f"学习率: Actor={config.lr_actor}, Critic={config.lr_critic}")
    print(f"最大轮次: {config.max_episodes}")

    # 创建训练器
    trainer = Trainer(config)

    print(f"\n🎯 开始稳定性优化训练...")
    print(f"💡 提示: 如果仍有奖励波动，训练器会自动提供调优建议")
    start_time = time.time()

    # 初始化训练时间变量
    training_time = 0.0

    try:
        # 执行训练
        trainer.train(worker_idx=1)

        # 计算训练时间
        training_time = time.time() - start_time

        # 保存结果并生成最终分析
        print(f"\n🎨 正在生成训练结果分析...")

        # 保存训练数据
        trainer._save_final_results()

        # 生成最终的完整分析图表
        if trainer.total_rewards:
            print("📊 生成训练分析图表...")
            trainer._create_comprehensive_plots()

            print("📈 生成奖励分析图表...")
            trainer._create_reward_analysis_plot(len(trainer.total_rewards))

            print("📊 完整训练分析图表已生成")

        # 生成最终环境状态可视化
        print("🗺️ 生成最终环境状态图...")
        final_env_path = trainer.save_path + 'final_environment_state.png'
        trainer.env.visualize_environment(final_env_path)
        print(f"🖼️ 最终环境状态已保存: {final_env_path}")

        # 生成最终性能总结
        print("📋 生成性能总结报告...")
        performance_metrics = trainer.env.get_performance_metrics()
        reward_stats = trainer.env.get_reward_statistics()

        summary_text = f"""
训练总结报告
{'=' * 50}
训练配置: {config_mode.upper()} 模式
设备: {config.device}
训练轮次: {len(trainer.total_rewards)}
训练用时: {training_time / 60:.1f}分钟

奖励表现:
- 平均奖励: {sum(trainer.total_rewards) / len(trainer.total_rewards):.3f}
- 最高奖励: {max(trainer.total_rewards):.3f}
- 最低奖励: {min(trainer.total_rewards):.3f}
- 最终50轮平均: {sum(trainer.total_rewards[-50:]) / min(50, len(trainer.total_rewards)):.3f}

环境表现:
- 数据收集完成率: {performance_metrics['data_completion_rate']:.1%}
- 平均定位不确定性: {performance_metrics['average_uncertainty']:.1f}m
- 通信次数: {performance_metrics['communication_count']}
- 定位次数: {performance_metrics['localization_count']}

技能发展:
- 通信技能: {reward_stats.get('communication_skill', 0):.3f}
- 定位技能: {reward_stats.get('localization_skill', 0):.3f}
- 通信成功率: {reward_stats.get('communication_success_rate', 0):.1%}
- 定位成功率: {reward_stats.get('localization_success_rate', 0):.1%}

训练效率:
- 平均每轮用时: {training_time / len(trainer.total_rewards):.2f}秒
- 训练速度: {len(trainer.total_rewards) / (training_time / 60):.1f} episodes/分钟
"""

        # 保存总结报告
        with open(trainer.save_path + 'training_summary.txt', 'w', encoding='utf-8') as f:
            f.write(summary_text)

        print("📄 训练总结报告已保存")
        print("💾 所有结果已保存完成")

        print(f"\n✅ 训练完成！")
        print(f"总用时: {training_time / 60:.1f}分钟")
        print(f"平均每轮用时: {training_time / config.max_episodes:.2f}秒")

        # 显示最终性能指标
        if trainer.total_rewards:
            final_avg_reward = sum(trainer.total_rewards[-50:]) / min(50, len(trainer.total_rewards))
            print(f"最终平均奖励(最后50轮): {final_avg_reward:.3f}")

            # 计算训练速度提升
            episodes_per_minute = len(trainer.total_rewards) / (training_time / 60)
            print(f"训练速度: {episodes_per_minute:.1f} episodes/分钟")

    except KeyboardInterrupt:
        training_time = time.time() - start_time
        print(f"\n⏹️ 训练被用户中断")
        print(f"已训练时间: {training_time / 60:.1f}分钟")

    except Exception as e:
        training_time = time.time() - start_time
        print(f"\n❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 GPU内存已清理")


if __name__ == "__main__":
    main()