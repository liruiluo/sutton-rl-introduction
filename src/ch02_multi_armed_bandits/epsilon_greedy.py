"""
================================================================================
第2.3节 & 第2.4节：增量实现与ε-贪婪方法
Section 2.3 & 2.4: Incremental Implementation and ε-Greedy Methods

根据 Sutton & Barto《强化学习：导论》第二版
Based on Sutton & Barto "Reinforcement Learning: An Introduction" 2nd Edition
================================================================================

在第2章中，Sutton & Barto 介绍了多臂赌博机问题作为强化学习的简化版本。
这里我们实现书中的核心算法：ε-贪婪方法。

书中原话（第2.2节）：
"The simplest action selection rule is to select one of the actions with the 
highest estimated value... We call this the greedy action selection method."

"ε-greedy methods: Most of the time they choose the action with the highest 
estimated action value (exploiting), but with small probability ε they instead 
select randomly from among all the actions (exploring)."

让我们通过代码深入理解这个最基础但极其重要的算法。

================================================================================
书中的10臂赌博机测试平台 (The 10-armed Testbed)
================================================================================

Sutton & Barto 在第2.3节描述了一个标准测试环境：
"Consider the following learning problem. You are faced repeatedly with a choice
among k different options, or actions. After each choice you receive a numerical
reward chosen from a stationary probability distribution that depends on the 
action you selected."

在10臂测试平台中：
1. 每个动作的真实价值 q*(a) 从标准正态分布 N(0,1) 中选取
2. 选择动作a后，奖励从 N(q*(a), 1) 中采样
3. 这创建了一个有噪声但稳定的环境

================================================================================
"""

import numpy as np
from typing import List, Optional, Dict, Any
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

from .bandit_introduction import BaseBanditAgent, MultiArmedBandit

# 设置日志
logger = logging.getLogger(__name__)


# ================================================================================
# 第2.3节：增量实现的动作价值估计
# Section 2.3: Incremental Implementation of Action-Value Estimates
# ================================================================================

class IncrementalEstimation:
    """
    书中第2.4节的增量更新公式 (Incremental Update Formula)
    
    Sutton & Barto 推导了一个优雅的增量更新公式：
    
    Q_{n+1} = Q_n + (1/n)[R_n - Q_n]
    
    更一般的形式（第2.5节）：
    NewEstimate ← OldEstimate + StepSize [Target - OldEstimate]
    
    这个公式的美妙之处：
    1. 不需要存储所有历史奖励
    2. 计算复杂度是常数 O(1)
    3. [Target - OldEstimate] 是"误差"项
    4. StepSize 控制学习速度
    
    书中原话：
    "The update rule is of a form that occurs frequently throughout this book.
    The general form is:
    NewEstimate ← OldEstimate + StepSize [Target - OldEstimate]"
    
    物理意义：
    - Target: 我们想要逼近的值（这里是新奖励R_n）
    - OldEstimate: 当前的估计
    - 误差项: 告诉我们估计偏离了多少
    - StepSize: 我们多大程度上相信这个新信息
    """
    
    @staticmethod
    def update_rule_derivation():
        """
        展示书中第2.4节的推导过程
        
        从定义开始：
        Q_n = (R_1 + R_2 + ... + R_{n-1}) / (n-1)
        
        推导增量形式：
        Q_{n+1} = (R_1 + R_2 + ... + R_n) / n
                = (R_1 + R_2 + ... + R_{n-1} + R_n) / n
                = [(n-1)Q_n + R_n] / n
                = Q_n + (1/n)[R_n - Q_n]
        
        这就是书中的公式(2.3)！
        """
        print("="*70)
        print("增量更新公式推导（书中公式2.3）")
        print("Derivation of Incremental Update (Equation 2.3)")
        print("="*70)
        print()
        print("定义 Definition:")
        print("  Q_n = (R_1 + R_2 + ... + R_{n-1}) / (n-1)")
        print()
        print("推导 Derivation:")
        print("  Q_{n+1} = (R_1 + R_2 + ... + R_n) / n")
        print("          = [(n-1)Q_n + R_n] / n")
        print("          = Q_n + (1/n)[R_n - Q_n]")
        print()
        print("这就是增量更新公式！")
        print("This is the incremental update formula!")
        print()
        print("优势 Advantages:")
        print("  1. 内存需求：O(1) 而非 O(n)")
        print("  2. 计算复杂度：O(1) 每步")
        print("  3. 在线学习：可以实时更新")
        print("="*70)


# ================================================================================
# 第2.2节 & 2.4节：ε-贪婪动作选择
# Section 2.2 & 2.4: ε-Greedy Action Selection
# ================================================================================

class EpsilonGreedyAgent(BaseBanditAgent):
    """
    ε-贪婪算法实现（对应书中第2.2节）
    ε-Greedy Algorithm Implementation (Section 2.2)
    
    书中的定义（第29页）：
    "ε-greedy action selection: With probability 1-ε select greedily, 
     and with probability ε select randomly."
    
    数学表示：
    A_t = {
        argmax_a Q_t(a),     以概率 1-ε （贪婪）
        random action,       以概率 ε   （探索）
    }
    
    书中讨论的关键点：
    1. ε = 0: 纯贪婪（pure greedy），可能卡在次优
    2. ε = 0.1: 书中实验的典型值
    3. ε = 1: 纯随机（pure random），不利用学习结果
    
    书中Figure 2.2显示：
    - ε = 0.1 在长期表现最好
    - ε = 0.01 收敛慢但最终性能好
    - ε = 0 容易卡在次优动作
    """
    
    def __init__(self, 
                 k: int = None,                      # 书中的k（动作数）
                 n_arms: int = None,                 # 向后兼容
                 epsilon: float = 0.1,               # 书中典型值0.1
                 epsilon_decay: Optional[float] = None,  # 第2.8节提到的衰减
                 epsilon_min: float = 0.01,          # 保持最小探索
                 initial_value: float = 0.0,         # Q_1(a)，书中第2.6节
                 step_size_mode: str = 'sample_average',  # 第2.5节的步长
                 **kwargs):
        """
        初始化ε-贪婪智能体
        
        参数对应书中概念：
        -------------------
        k: 动作数量（书中记号k）
        epsilon: 探索概率（书中记号ε）
        initial_value: 初始动作价值Q_1(a)（第2.6节：乐观初始值）
        step_size_mode: 步长选择
            - 'sample_average': α = 1/n（第2.3节）
            - 'constant': α = 常数（第2.5节）
        
        书中第2.6节关于初始值的讨论：
        "All the methods we have discussed so far are dependent to some extent on
        the initial action-value estimates, Q_1(a). In the language of statistics,
        these methods are biased by their initial estimates."
        
        乐观初始值（Optimistic Initial Values）：
        - 设置Q_1(a) = 5（当真实值在[-1,1]范围）
        - 鼓励早期探索
        - 是一种简单但有效的探索技术
        """
        # 处理参数
        if n_arms is not None:
            k = n_arms
        if k is None:
            raise ValueError("必须提供k（动作数量）Must provide k (number of actions)")
        
        # 保存ε相关参数
        self.epsilon_initial = epsilon
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # 记录统计信息（用于复现Figure 2.2）
        self.epsilon_history = [epsilon]
        self.exploration_count = 0  # 探索次数
        self.exploitation_count = 0  # 利用次数
        
        # 调用父类初始化
        super().__init__(
            k=k, 
            initial_value=initial_value,
            step_size_mode=step_size_mode,
            **kwargs
        )
        
        logger.info(f"初始化ε-贪婪智能体: ε={epsilon}, decay={epsilon_decay}, min={epsilon_min}")
    
    def select_action(self) -> int:
        """
        ε-贪婪动作选择（书中算法2.1的核心）
        
        书中伪代码（第33页）：
        if random() < ε:
            A ← random action
        else:
            A ← argmax_a Q(a)
        
        实现细节：
        1. 生成随机数判断探索还是利用
        2. 探索时均匀随机选择
        3. 利用时选择最高估值动作
        4. 如有多个最大值，随机打破平局（书中第2.2节提到）
        """
        # ε-贪婪决策
        if np.random.random() < self.epsilon:
            # 探索：随机选择（书中："select randomly")
            action = np.random.choice(self.k)
            self.exploration_count += 1
        else:
            # 利用：选择最优（书中："select greedily")
            # 注意：书中提到要随机打破平局
            max_value = np.max(self.Q)
            max_actions = np.where(self.Q == max_value)[0]
            action = np.random.choice(max_actions)
            self.exploitation_count += 1
        
        # ε衰减（如果设置了）
        if self.epsilon_decay is not None:
            self.epsilon = max(
                self.epsilon * self.epsilon_decay, 
                self.epsilon_min
            )
            if len(self.epsilon_history) < 10000:  # 限制历史长度
                self.epsilon_history.append(self.epsilon)
        
        return action
    
    def reset(self) -> None:
        """重置智能体到初始状态"""
        self.epsilon = self.epsilon_initial
        self.epsilon_history = [self.epsilon]
        self.exploration_count = 0
        self.exploitation_count = 0
        super().reset()
        logger.info(f"重置ε-贪婪智能体，ε恢复到{self.epsilon_initial}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取学习统计信息"""
        stats = super().get_statistics()
        stats.update({
            'current_epsilon': self.epsilon,
            'exploration_count': self.exploration_count,
            'exploitation_count': self.exploitation_count,
            'exploration_ratio': self.exploration_count / max(1, self.exploration_count + self.exploitation_count),
            'epsilon_history': self.epsilon_history,
        })
        return stats


# ================================================================================
# 第2.3节：复现书中Figure 2.2
# Section 2.3: Reproducing Figure 2.2 from the Book
# ================================================================================

class Figure2_2Reproduction:
    """
    复现Sutton & Barto书中的Figure 2.2
    
    书中描述（第29页）：
    "Figure 2.2: Average performance of ε-greedy action-value methods on the 
    10-armed testbed. These data are averages over 2000 runs with different 
    bandit problems. All methods used sample averages as their action-value 
    estimates."
    
    实验设置：
    - 2000个不同的10臂赌博机问题
    - 每个问题运行1000步
    - 比较ε = 0, 0.01, 0.1的性能
    - 两个指标：平均奖励和最优动作百分比
    """
    
    @staticmethod
    def run_experiment(epsilon_values: List[float] = [0.0, 0.01, 0.1],
                       n_bandits: int = 2000,
                       n_steps: int = 1000,
                       show_plot: bool = True) -> Dict:
        """
        运行书中的实验
        
        参数：
            epsilon_values: 要测试的ε值（书中用[0, 0.01, 0.1]）
            n_bandits: 赌博机问题数量（书中用2000）
            n_steps: 每个问题的步数（书中用1000）
            show_plot: 是否显示图表
        
        返回：
            实验结果字典
        """
        print("="*70)
        print("复现 Sutton & Barto Figure 2.2")
        print("Reproducing Sutton & Barto Figure 2.2")
        print("="*70)
        print(f"\n实验设置 Experimental Setup:")
        print(f"  - {n_bandits} 个10臂赌博机问题 (10-armed bandit problems)")
        print(f"  - 每个问题运行 {n_steps} 步 (steps per problem)")
        print(f"  - 测试ε值: {epsilon_values}")
        print()
        
        results = {}
        
        for epsilon in epsilon_values:
            print(f"运行 ε = {epsilon}...")
            
            # 存储所有运行的结果
            all_rewards = np.zeros((n_bandits, n_steps))
            all_optimal_actions = np.zeros((n_bandits, n_steps))
            
            for i in tqdm(range(n_bandits), desc=f"ε={epsilon}"):
                # 创建新的赌博机问题（书中：different bandit problems）
                bandit = MultiArmedBandit(k=10)
                
                # 创建ε-贪婪智能体
                agent = EpsilonGreedyAgent(
                    k=10, 
                    epsilon=epsilon,
                    initial_value=0.0,  # 书中默认用0
                    step_size_mode='sample_average'  # 书中用样本平均
                )
                
                # 运行实验
                for step in range(n_steps):
                    action = agent.select_action()
                    reward = bandit.step(action)
                    agent.update(action, reward)
                    
                    # 记录结果
                    all_rewards[i, step] = reward
                    all_optimal_actions[i, step] = (action == bandit.optimal_action)
            
            # 计算平均性能（书中：averages over 2000 runs）
            avg_rewards = np.mean(all_rewards, axis=0)
            optimal_action_percentage = np.mean(all_optimal_actions, axis=0) * 100
            
            results[epsilon] = {
                'average_rewards': avg_rewards,
                'optimal_action_percentage': optimal_action_percentage,
                'all_rewards': all_rewards,
                'all_optimal_actions': all_optimal_actions
            }
            
            print(f"  最终平均奖励: {avg_rewards[-1]:.3f}")
            print(f"  最终最优动作率: {optimal_action_percentage[-1]:.1f}%")
        
        if show_plot:
            Figure2_2Reproduction._plot_results(results, n_steps)
        
        return results
    
    @staticmethod
    def _plot_results(results: Dict, n_steps: int):
        """
        绘制类似书中Figure 2.2的图表
        
        书中有两个子图：
        1. 上图：平均奖励 vs 步数
        2. 下图：最优动作百分比 vs 步数
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        
        # 颜色方案（类似书中）
        colors = {0.0: 'green', 0.01: 'red', 0.1: 'blue'}
        
        # 上图：平均奖励
        ax = axes[0]
        for epsilon, data in results.items():
            color = colors.get(epsilon, 'gray')
            ax.plot(data['average_rewards'], 
                   color=color, 
                   label=f'ε = {epsilon}' if epsilon > 0 else 'ε = 0 (greedy)',
                   linewidth=2)
        
        ax.set_xlabel('Steps')
        ax.set_ylabel('Average\nreward', rotation=0, labelpad=40)
        ax.set_ylim([0, 1.6])
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_title('Average Reward on 10-Armed Testbed', fontsize=14)
        
        # 下图：最优动作百分比
        ax = axes[1]
        for epsilon, data in results.items():
            color = colors.get(epsilon, 'gray')
            ax.plot(data['optimal_action_percentage'], 
                   color=color, 
                   label=f'ε = {epsilon}' if epsilon > 0 else 'ε = 0 (greedy)',
                   linewidth=2)
        
        ax.set_xlabel('Steps')
        ax.set_ylabel('% Optimal\naction', rotation=0, labelpad=40)
        ax.set_ylim([0, 100])
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_title('% Optimal Action on 10-Armed Testbed', fontsize=14)
        
        plt.suptitle('Figure 2.2: ε-greedy Methods (Sutton & Barto)', 
                    fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()


# ================================================================================
# 第2.6节：乐观初始值
# Section 2.6: Optimistic Initial Values
# ================================================================================

class OptimisticGreedyAgent(EpsilonGreedyAgent):
    """
    使用乐观初始值的贪婪方法（书中第2.6节）
    
    书中原话（第34页）：
    "We call this technique for encouraging exploration optimistic initial values.
    We regard it as a simple trick that can be quite effective on stationary problems,
    but it is far from being a generally useful approach to encouraging exploration."
    
    关键思想：
    1. 设置Q_1(a) = 5（远高于真实值）
    2. 使用ε = 0（纯贪婪）
    3. 初始的高估值自然导致探索
    4. 随着更新，估值逐渐降到真实水平
    
    书中Figure 2.3显示：
    - 乐观初始值开始时探索更多
    - 最终性能与ε-贪婪相当
    - 对非稳态问题效果不好
    """
    
    def __init__(self, k: int, 
                 optimistic_value: float = 5.0,
                 **kwargs):
        """
        初始化乐观贪婪智能体
        
        参数：
            k: 动作数量
            optimistic_value: 乐观初始值（书中用5.0）
        """
        # 使用ε=0（纯贪婪）和乐观初始值
        super().__init__(
            k=k,
            epsilon=0.0,  # 纯贪婪！
            initial_value=optimistic_value,
            **kwargs
        )
        self.optimistic_value = optimistic_value
        
        print(f"初始化乐观贪婪智能体: Q_1(a) = {optimistic_value}")
        print("使用ε=0（纯贪婪），依靠乐观初始值探索")


# ================================================================================
# 衍生类和辅助功能（向后兼容）
# Derived Classes and Helper Functions (Backward Compatibility)
# ================================================================================

class DecayingEpsilonGreedy(EpsilonGreedyAgent):
    """带衰减的ε-贪婪（向后兼容）"""
    
    def __init__(self, k: int, 
                 initial_epsilon: float = 0.5,
                 decay_rate: float = 0.995,
                 min_epsilon: float = 0.01,
                 **kwargs):
        super().__init__(
            k=k,
            epsilon=initial_epsilon,
            epsilon_decay=decay_rate,
            epsilon_min=min_epsilon,
            **kwargs
        )


class AdaptiveEpsilonGreedy(EpsilonGreedyAgent):
    """
    自适应ε-贪婪（基于不确定性）
    
    不是书中的方法，但是一个有用的扩展。
    根据动作价值的分散程度调整ε。
    """
    
    def select_action(self) -> int:
        # 计算动作价值的标准差
        q_std = np.std(self.Q)
        q_range = np.max(self.Q) - np.min(self.Q) + 1e-10
        
        # 根据不确定性调整ε
        certainty = min(1.0, q_std / q_range)
        adaptive_epsilon = self.epsilon_min + \
                          (self.epsilon_initial - self.epsilon_min) * (1 - certainty)
        
        # 临时使用自适应ε
        original_epsilon = self.epsilon
        self.epsilon = adaptive_epsilon
        action = super().select_action()
        self.epsilon = original_epsilon
        
        return action


# 为向后兼容创建别名
EpsilonGreedyAnalyzer = Figure2_2Reproduction
EpsilonGreedyAnalysis = Figure2_2Reproduction


def compare_epsilon_greedy_variants(n_steps: int = 1000,
                                   n_runs: int = 20) -> Dict:
    """比较不同ε-贪婪变体（向后兼容）"""
    
    print("比较ε-贪婪变体（包括乐观初始值）")
    print("="*70)
    
    results = {}
    
    # 测试标准ε-贪婪
    epsilons = [0.0, 0.01, 0.1]
    fig_results = Figure2_2Reproduction.run_experiment(
        epsilon_values=epsilons,
        n_bandits=n_runs,
        n_steps=n_steps,
        show_plot=False
    )
    results.update(fig_results)
    
    # 测试乐观初始值
    print("\n测试乐观初始值方法...")
    all_rewards = []
    
    for i in tqdm(range(n_runs), desc="Optimistic"):
        bandit = MultiArmedBandit(k=10)
        agent = OptimisticGreedyAgent(k=10, optimistic_value=5.0)
        
        rewards = []
        for _ in range(n_steps):
            action = agent.select_action()
            reward = bandit.step(action)
            agent.update(action, reward)
            rewards.append(reward)
        
        all_rewards.append(rewards)
    
    results['optimistic'] = {
        'average_rewards': np.mean(all_rewards, axis=0),
        'final_performance': np.mean(all_rewards)
    }
    
    print(f"乐观初始值最终性能: {results['optimistic']['final_performance']:.3f}")
    
    return results


def demonstrate_epsilon_greedy():
    """演示ε-贪婪算法的关键概念"""
    
    print("\n" + "="*70)
    print("ε-贪婪算法演示")
    print("Demonstrating ε-Greedy Algorithm")
    print("="*70)
    
    # 1. 展示增量更新公式推导
    print("\n1. 增量更新公式")
    print("-" * 40)
    IncrementalEstimation.update_rule_derivation()
    
    # 2. 运行简单实验
    print("\n2. 简单10臂赌博机实验")
    print("-" * 40)
    
    bandit = MultiArmedBandit(k=10)
    agent = EpsilonGreedyAgent(k=10, epsilon=0.1)
    
    print(f"创建10臂赌博机")
    print(f"最优臂: {bandit.optimal_action}")
    print(f"使用ε=0.1的ε-贪婪算法")
    
    # 运行100步
    rewards = []
    for step in range(100):
        action = agent.select_action()
        reward = bandit.step(action)
        agent.update(action, reward)
        rewards.append(reward)
        
        if (step + 1) % 20 == 0:
            stats = agent.get_statistics()
            print(f"  步 {step+1}: 平均奖励={np.mean(rewards[-20:]):.3f}, "
                  f"探索率={stats['exploration_ratio']:.1%}")
    
    # 3. 显示最终结果
    print("\n3. 学习结果")
    print("-" * 40)
    print(f"动作选择次数: {agent.action_counts}")
    print(f"估计动作价值: {np.round(agent.Q, 2)}")
    print(f"真实动作价值: {np.round(bandit.q_true, 2)}")
    print(f"最常选择的动作: {np.argmax(agent.action_counts)}")
    print(f"真实最优动作: {bandit.optimal_action}")


# ================================================================================
# 主程序入口
# Main Program Entry
# ================================================================================

if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║" + " "*15 + "Sutton & Barto 第2章：多臂赌博机".center(38) + " "*15 + "║")
    print("║" + " "*10 + "Chapter 2: Multi-Armed Bandits - ε-Greedy".center(48) + " "*10 + "║")
    print("╚" + "═"*68 + "╝")
    
    print("\n欢迎来到强化学习第2章！")
    print("Welcome to Chapter 2 of Reinforcement Learning!")
    
    print("\n本程序实现了书中的ε-贪婪算法和Figure 2.2。")
    print("This implements the ε-greedy algorithm and Figure 2.2 from the book.")
    print("="*70)
    
    # 选择运行模式
    print("\n请选择运行模式 Select Mode:")
    print("1. 快速演示 (Quick Demo)")
    print("2. 复现Figure 2.2 (Reproduce Figure 2.2)")
    print("3. 完整实验 (Full Experiment)")
    
    choice = input("\n输入选择 (1/2/3，默认1): ").strip() or "1"
    
    if choice == "1":
        # 快速演示
        demonstrate_epsilon_greedy()
        
    elif choice == "2":
        # 复现Figure 2.2
        print("\n复现书中Figure 2.2...")
        print("注意：完整实验需要几分钟时间")
        print("Note: Full experiment takes several minutes")
        
        # 使用较小的样本数进行演示
        Figure2_2Reproduction.run_experiment(
            epsilon_values=[0.0, 0.01, 0.1],
            n_bandits=200,  # 书中用2000，这里用200加快速度
            n_steps=1000,
            show_plot=True
        )
        
    else:
        # 完整实验
        print("\n运行完整实验（包括乐观初始值）...")
        print("Running full experiment (including optimistic initial values)...")
        
        results = compare_epsilon_greedy_variants(
            n_steps=1000,
            n_runs=100
        )
        
        print("\n实验完成！")
        print("Experiment complete!")
    
    print("\n" + "="*70)
    print("总结 Summary")
    print("="*70)
    print("""
    书中关于ε-贪婪的关键要点 (Key Points from the Book):
    
    1. ε-贪婪平衡探索与利用 (Balances exploration and exploitation)
    2. ε=0.1通常是好的选择 (ε=0.1 is often a good choice)
    3. 增量更新节省内存和计算 (Incremental updates save memory and computation)
    4. 乐观初始值提供另一种探索方式 (Optimistic initial values provide alternative exploration)
    5. 没有普适最优的ε值 (No universally optimal ε value)
    
    下一节：UCB（上置信界）算法
    Next: UCB (Upper-Confidence-Bound) Algorithm
    """)