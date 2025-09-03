"""
================================================================================
第1章：多臂赌博机 - 完整演示
Chapter 1: Multi-Armed Bandits - Complete Demonstration
================================================================================

运行第1章的所有内容，包括：
1. 基础概念和环境
2. ε-贪婪算法
3. UCB算法
4. 梯度赌博机算法
5. 综合比较和分析

Run all content from Chapter 1, including:
1. Basic concepts and environment
2. ε-greedy algorithm
3. UCB algorithm
4. Gradient bandit algorithm
5. Comprehensive comparison and analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import sys
from typing import Dict, List, Any
from tqdm import tqdm

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

# 导入所有模块
from ch01_bandits.bandit_introduction import (
    BanditProblemDefinition,
    MultiArmedBandit,
    ActionValueEstimation,
    demonstrate_chapter1_basics
)
from ch01_bandits.epsilon_greedy import (
    EpsilonGreedyAgent,
    EpsilonGreedyAnalysis,
    compare_epsilon_greedy_variants
)
from ch01_bandits.ucb_algorithm import (
    UCBAgent,
    UCBPrinciple,
    UCBTheoreticalAnalysis,
    compare_ucb_variants
)
from ch01_bandits.gradient_bandit import (
    GradientBanditAgent,
    GradientBanditPrinciple,
    GradientBanditAnalysis,
    compare_all_algorithms
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置绘图风格
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


# ================================================================================
# 第1章总结
# Chapter 1 Summary
# ================================================================================

def chapter1_summary():
    """
    第1章知识总结
    Chapter 1 Knowledge Summary
    """
    print("\n" + "="*80)
    print("第1章：多臂赌博机 - 知识总结")
    print("Chapter 1: Multi-Armed Bandits - Knowledge Summary")
    print("="*80)
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                         第1章 核心知识点总结                              ║
    ║                    Chapter 1 Core Knowledge Summary                      ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    
    1. 多臂赌博机问题 Multi-Armed Bandit Problem
    ═══════════════════════════════════════════════
    
    定义 Definition:
    • k个动作，每个有未知期望奖励 q*(a)
      k actions, each with unknown expected reward q*(a)
    • 目标：最大化累积奖励 Σ R_t
      Goal: Maximize cumulative reward
    • 核心挑战：探索vs利用权衡
      Core challenge: Exploration vs exploitation trade-off
    
    应用 Applications:
    • A/B测试 A/B testing
    • 推荐系统 Recommendation systems
    • 临床试验 Clinical trials
    • 资源分配 Resource allocation
    
    2. 动作价值估计 Action-Value Estimation
    ═══════════════════════════════════════════════
    
    样本平均 Sample Average:
    Q_t(a) = Σ R_i / N_t(a)
    
    增量更新 Incremental Update:
    Q_{n+1} = Q_n + α[R_n - Q_n]
    
    关键模式 Key Pattern:
    NewEstimate = OldEstimate + StepSize × [Target - OldEstimate]
    
    这个模式贯穿整个强化学习！
    This pattern runs through all of RL!
    
    3. 算法对比 Algorithm Comparison
    ═══════════════════════════════════════════════
    
    ┌─────────────┬──────────────┬─────────────┬──────────────┬───────────────┐
    │  Algorithm  │   Selection  │  Exploration│  Regret      │  Best For     │
    ├─────────────┼──────────────┼─────────────┼──────────────┼───────────────┤
    │ ε-greedy    │  Random      │  ε param    │  O(T)        │  Simple       │
    │             │  exploration │             │  (fixed ε)   │  problems     │
    ├─────────────┼──────────────┼─────────────┼──────────────┼───────────────┤
    │ UCB         │  Optimistic  │  Automatic  │  O(ln T)     │  Stationary   │
    │             │  (upper CI)  │  √(ln t/n)  │              │  problems     │
    ├─────────────┼──────────────┼─────────────┼──────────────┼───────────────┤
    │ Gradient    │  Softmax     │  Temperature│  O(√T)       │  Need soft    │
    │ Bandit      │  policy      │  or entropy │              │  policies     │
    └─────────────┴──────────────┴─────────────┴──────────────┴───────────────┘
    
    4. 关键见解 Key Insights
    ═══════════════════════════════════════════════
    
    探索的必要性 Necessity of Exploration:
    • 没有探索 → 可能永远错过最优动作
      No exploration → May miss optimal action forever
    • 过度探索 → 浪费在次优动作上
      Too much exploration → Waste on suboptimal actions
    
    不同探索策略 Different Exploration Strategies:
    • 随机探索（ε-greedy）：简单但不智能
      Random (ε-greedy): Simple but not intelligent
    • 乐观探索（UCB）：优先探索不确定的
      Optimistic (UCB): Prioritize uncertain actions
    • 概率探索（Gradient）：自然的随机性
      Probabilistic (Gradient): Natural stochasticity
    
    平稳vs非平稳 Stationary vs Non-stationary:
    • 平稳：可以逐渐减少探索
      Stationary: Can gradually reduce exploration
    • 非平稳：需要持续探索
      Non-stationary: Need continuous exploration
    
    5. 与后续章节的联系 Connection to Later Chapters
    ═══════════════════════════════════════════════════
    
    • 赌博机 → 单状态MDP
      Bandits → Single-state MDP
    • 价值估计 → 价值函数
      Value estimation → Value functions
    • 梯度赌博机 → 策略梯度
      Gradient bandit → Policy gradient
    • 探索策略 → 通用探索方法
      Exploration strategies → General exploration methods
    
    6. 实践建议 Practical Recommendations
    ═══════════════════════════════════════════════
    
    选择算法 Choosing Algorithm:
    • 快速原型：ε-greedy with ε=0.1
      Quick prototype: ε-greedy
    • 理论保证：UCB with c=2
      Theoretical guarantee: UCB
    • 需要随机策略：Gradient bandit
      Need stochastic policy: Gradient
    
    参数调优 Parameter Tuning:
    • ε: 从0.1开始，根据性能调整
      Start with 0.1, adjust based on performance
    • c: 理论值√2，实践常用2
      Theoretical √2, practical 2
    • α: 0.1是好的起点
      0.1 is a good starting point
    
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║  记住：多臂赌博机是强化学习的"Hello World"，掌握它是理解RL的第一步！        ║
    ║  Remember: Bandits are RL's "Hello World", mastering them is step 1!     ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)


def run_complete_experiments(cfg: DictConfig):
    """
    运行完整的第1章实验
    Run complete Chapter 1 experiments
    
    Args:
        cfg: Hydra配置
    """
    print("\n" + "="*80)
    print("运行第1章完整实验")
    print("Running Complete Chapter 1 Experiments")
    print("="*80)
    
    # 实验参数
    k = cfg.get('k', 10)  # 臂数
    n_runs = cfg.get('n_runs', 100)  # 运行次数
    n_steps = cfg.get('n_steps', 1000)  # 步数
    
    print(f"\n实验配置 Experiment Configuration:")
    print(f"  Arms k = {k}")
    print(f"  Runs = {n_runs}")
    print(f"  Steps = {n_steps}")
    
    # 所有算法
    algorithms = {
        'ε-greedy (ε=0.1)': EpsilonGreedyAgent(k=k, epsilon=0.1),
        'ε-greedy (decay)': EpsilonGreedyAgent(k=k, epsilon=0.5, epsilon_decay=0.995),
        'UCB (c=2)': UCBAgent(k=k, c=2.0),
        'Gradient (baseline)': GradientBanditAgent(k=k, alpha=0.1, use_baseline=True),
    }
    
    # 运行实验
    all_results = {}
    
    for name, agent in algorithms.items():
        print(f"\n测试 Testing: {name}")
        rewards = []
        optimal = []
        regrets = []
        
        for run in tqdm(range(n_runs), desc=f"{name}", leave=False):
            # 创建环境
            env = MultiArmedBandit(k=k, seed=run)
            
            # 重置智能体
            agent.reset()
            
            # 运行
            episode_data = agent.run_episode(env, n_steps)
            
            rewards.append(episode_data['rewards'])
            optimal.append(episode_data['optimal_actions'])
            regrets.append(episode_data['regrets'])
        
        all_results[name] = {
            'rewards': np.array(rewards),
            'optimal': np.array(optimal),
            'regrets': np.array(regrets)
        }
    
    # 绘制综合结果
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # 1. 平均奖励
    ax1 = axes[0, 0]
    for name, data in all_results.items():
        mean_rewards = np.mean(data['rewards'], axis=0)
        ax1.plot(mean_rewards, label=name, alpha=0.8)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Average Reward Over Time')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. 最优动作比例
    ax2 = axes[0, 1]
    for name, data in all_results.items():
        optimal_rate = np.mean(data['optimal'], axis=0) * 100
        ax2.plot(optimal_rate, label=name, alpha=0.8)
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Optimal Action %')
    ax2.set_title('Optimal Action Selection Rate')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])
    
    # 3. 累积遗憾
    ax3 = axes[0, 2]
    for name, data in all_results.items():
        mean_regrets = np.mean(data['regrets'], axis=0)
        ax3.plot(mean_regrets, label=name, alpha=0.8)
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Cumulative Regret')
    ax3.set_title('Cumulative Regret')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. 学习曲线（移动平均）
    ax4 = axes[1, 0]
    window = 50
    for name, data in all_results.items():
        mean_rewards = np.mean(data['rewards'], axis=0)
        smoothed = np.convolve(mean_rewards, np.ones(window)/window, mode='valid')
        ax4.plot(smoothed, label=name, alpha=0.8)
    ax4.set_xlabel('Steps')
    ax4.set_ylabel('Smoothed Reward')
    ax4.set_title(f'Learning Curves (window={window})')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. 最终性能箱线图
    ax5 = axes[1, 1]
    final_rewards = []
    labels = []
    for name, data in all_results.items():
        # 最后100步的平均
        final = np.mean(data['rewards'][:, -100:], axis=1)
        final_rewards.append(final)
        labels.append(name.split('(')[0].strip())
    
    bp = ax5.boxplot(final_rewards, labels=labels, patch_artist=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax5.set_ylabel('Final Average Reward')
    ax5.set_title('Final Performance Distribution')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. 性能指标表
    ax6 = axes[1, 2]
    ax6.axis('tight')
    ax6.axis('off')
    
    # 计算性能指标
    metrics_data = []
    for name in all_results:
        final_reward = np.mean(all_results[name]['rewards'][:, -100:])
        final_optimal = np.mean(all_results[name]['optimal'][:, -100:]) * 100
        total_regret = np.mean(all_results[name]['regrets'][:, -1])
        metrics_data.append([
            name.split('(')[0].strip(),
            f"{final_reward:.3f}",
            f"{final_optimal:.1f}%",
            f"{total_regret:.0f}"
        ])
    
    table = ax6.table(cellText=metrics_data,
                     colLabels=['Algorithm', 'Final Reward', 'Optimal %', 'Total Regret'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # 设置表头样式
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置行颜色
    for i in range(1, len(metrics_data) + 1):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax6.set_title('Performance Metrics Summary', pad=20)
    
    plt.suptitle('Chapter 1: Multi-Armed Bandits - Complete Results', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # 保存图表
    if cfg.get('save_plots', False):
        plot_dir = Path(cfg.get('plot_dir', 'outputs/plots'))
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_dir / 'chapter1_complete_results.png', dpi=150, bbox_inches='tight')
        print(f"\n图表已保存到 Plot saved to: {plot_dir / 'chapter1_complete_results.png'}")
    
    return fig, all_results


def interactive_demo():
    """
    交互式演示
    Interactive demonstration
    """
    print("\n" + "="*80)
    print("交互式多臂赌博机演示")
    print("Interactive Multi-Armed Bandit Demo")
    print("="*80)
    
    print("\n欢迎来到多臂赌博机游戏！")
    print("Welcome to the Multi-Armed Bandit Game!")
    print("\n你面前有10台老虎机，每台的赢钱概率不同。")
    print("You face 10 slot machines, each with different winning probabilities.")
    print("你有100次机会，目标是赚最多的钱！")
    print("You have 100 chances, goal is to make the most money!")
    
    # 创建环境
    env = MultiArmedBandit(k=10, seed=42)
    total_reward = 0
    n_plays = 100
    history = []
    
    # 显示初始状态（作弊模式：显示真实价值）
    show_truth = input("\n是否显示真实价值（作弊模式）？ Show true values (cheat mode)? (y/n): ")
    if show_truth.lower() == 'y':
        true_values = env.get_true_values()
        optimal = env.get_optimal_action()
        print(f"\n真实价值 True values:")
        for i, val in enumerate(true_values):
            marker = " ← 最优 BEST!" if i == optimal else ""
            print(f"  机器 Machine {i}: {val:.3f}{marker}")
    
    print("\n" + "-"*60)
    print("开始游戏！Start playing!")
    print("-"*60)
    
    for play in range(n_plays):
        print(f"\n第 {play+1}/{n_plays} 次")
        
        # 显示当前统计
        if play > 0:
            print(f"当前总奖励 Current total: {total_reward:.2f}")
            print(f"平均奖励 Average reward: {total_reward/play:.3f}")
        
        # 选择动作
        while True:
            try:
                action = int(input(f"选择机器 (0-9) Choose machine (0-9): "))
                if 0 <= action <= 9:
                    break
                else:
                    print("请输入0-9之间的数字 Please enter 0-9")
            except ValueError:
                print("请输入数字 Please enter a number")
            except KeyboardInterrupt:
                print("\n游戏结束！Game over!")
                return
        
        # 执行动作
        reward = env.step(action)
        total_reward += reward
        history.append((action, reward))
        
        print(f"→ 奖励 Reward: {reward:.3f}")
        
        # 每20次显示统计
        if (play + 1) % 20 == 0:
            print("\n" + "="*40)
            print(f"统计 Statistics (前{play+1}次):")
            action_counts = np.zeros(10)
            action_rewards = np.zeros(10)
            for a, r in history:
                action_counts[a] += 1
                action_rewards[a] += r
            
            print("机器 | 次数 | 平均奖励")
            print("-"*30)
            for i in range(10):
                if action_counts[i] > 0:
                    avg = action_rewards[i] / action_counts[i]
                    print(f"  {i}  |  {int(action_counts[i]):2d}  | {avg:.3f}")
            print("="*40)
    
    # 游戏结束
    print("\n" + "="*60)
    print("游戏结束！Game Over!")
    print("="*60)
    
    print(f"\n最终得分 Final Score: {total_reward:.2f}")
    print(f"平均每次 Average per play: {total_reward/n_plays:.3f}")
    
    # 分析表现
    optimal_action = env.get_optimal_action()
    optimal_count = sum(1 for a, _ in history if a == optimal_action)
    print(f"\n你选择最优机器的比例 Optimal action rate: {optimal_count/n_plays*100:.1f}%")
    
    # 理论最优
    true_values = env.get_true_values()
    theoretical_best = true_values[optimal_action] * n_plays
    print(f"理论最优得分 Theoretical best: {theoretical_best:.2f}")
    print(f"你的效率 Your efficiency: {total_reward/theoretical_best*100:.1f}%")
    
    # 评价
    efficiency = total_reward / theoretical_best
    if efficiency > 0.9:
        print("\n太棒了！你几乎达到了理论最优！")
        print("Excellent! You almost reached theoretical optimum!")
    elif efficiency > 0.7:
        print("\n不错！你找到了好机器！")
        print("Good! You found good machines!")
    elif efficiency > 0.5:
        print("\n还可以，但还有提升空间。")
        print("OK, but there's room for improvement.")
    else:
        print("\n需要更多探索来找到好机器！")
        print("Need more exploration to find good machines!")


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    主函数：运行第1章完整内容
    Main function: Run complete Chapter 1 content
    
    Args:
        cfg: Hydra配置
    """
    print("\n" + "="*80)
    print("Sutton & Barto《强化学习导论》")
    print("第1章：多臂赌博机")
    print("\nReinforcement Learning: An Introduction")
    print("Chapter 1: Multi-Armed Bandits")
    print("="*80)
    
    # 选择运行模式
    print("\n选择运行模式 Select mode:")
    print("1. 完整演示 (Full demonstration)")
    print("2. 快速实验 (Quick experiment)")
    print("3. 交互游戏 (Interactive game)")
    print("4. 仅显示总结 (Summary only)")
    
    try:
        mode = input("\n请选择 (1-4) Please select (1-4): ").strip()
    except KeyboardInterrupt:
        print("\n退出 Exiting...")
        return
    
    if mode == '1':
        # 完整演示
        print("\n运行完整演示...")
        print("Running full demonstration...")
        
        # 1. 基础概念
        print("\n" + "="*60)
        print("Part 1: 基础概念 Basic Concepts")
        print("="*60)
        demonstrate_chapter1_basics()
        
        # 2. ε-贪婪
        print("\n" + "="*60)
        print("Part 2: ε-贪婪算法 ε-Greedy Algorithm")
        print("="*60)
        EpsilonGreedyAnalysis.parameter_sensitivity_study()
        compare_epsilon_greedy_variants()
        
        # 3. UCB
        print("\n" + "="*60)
        print("Part 3: UCB算法 UCB Algorithm")
        print("="*60)
        UCBPrinciple.explain_ucb_principle()
        UCBTheoreticalAnalysis.demonstrate_regret_growth()
        compare_ucb_variants()
        
        # 4. 梯度赌博机
        print("\n" + "="*60)
        print("Part 4: 梯度赌博机 Gradient Bandit")
        print("="*60)
        GradientBanditPrinciple.explain_principle()
        GradientBanditAnalysis.demonstrate_convergence()
        
        # 5. 综合比较
        print("\n" + "="*60)
        print("Part 5: 综合比较 Comprehensive Comparison")
        print("="*60)
        compare_all_algorithms()
        
    elif mode == '2':
        # 快速实验
        print("\n运行快速实验...")
        print("Running quick experiment...")
        
        # 修改配置为快速版本
        cfg.n_runs = 20
        cfg.n_steps = 500
        
        run_complete_experiments(cfg)
        
    elif mode == '3':
        # 交互游戏
        interactive_demo()
        
    elif mode == '4':
        # 仅显示总结
        pass
    else:
        print("无效选择 Invalid selection")
        return
    
    # 显示总结
    chapter1_summary()
    
    # 显示所有图表
    print("\n显示所有图表...")
    print("Showing all plots...")
    plt.show()
    
    print("\n" + "="*80)
    print("第1章演示完成！")
    print("Chapter 1 Demo Complete!")
    print("\n下一章：第2章 - 有限马尔可夫决策过程")
    print("Next: Chapter 2 - Finite Markov Decision Processes")
    print("="*80)


if __name__ == "__main__":
    # 如果直接运行，使用默认配置
    import sys
    from omegaconf import OmegaConf
    
    # 创建默认配置
    default_cfg = OmegaConf.create({
        'k': 10,
        'n_runs': 100,
        'n_steps': 1000,
        'save_plots': True,
        'plot_dir': 'outputs/plots'
    })
    
    # 运行
    print("使用默认配置运行 Running with default config")
    main(default_cfg)