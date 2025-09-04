"""
================================================================================
第1.6节：ε-贪婪算法 - 最简单的探索策略
Section 1.6: ε-Greedy Algorithm - The Simplest Exploration Strategy
================================================================================

ε-贪婪是最经典的探索策略，简单但有效！
ε-greedy is the most classic exploration strategy, simple but effective!

核心思想 Core Idea:
- 以概率ε进行探索（随机选择）
- 以概率1-ε进行利用（选择最佳）
- With probability ε: explore (random selection)
- With probability 1-ε: exploit (select best)
"""

import numpy as np
from typing import List, Optional, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging

from .bandit_introduction import BaseBanditAgent, MultiArmedBandit

# 设置日志
logger = logging.getLogger(__name__)


# ================================================================================
# 第1.6.1节：ε-贪婪算法实现
# Section 1.6.1: ε-Greedy Algorithm Implementation
# ================================================================================

class EpsilonGreedyAgent(BaseBanditAgent):
    """
    ε-贪婪智能体
    ε-Greedy Agent
    
    这是强化学习中最基础的算法之一
    This is one of the most fundamental algorithms in RL
    
    数学原理 Mathematical Principle:
    
    动作选择策略 Action Selection Policy:
    π(a|Q) = {
        1 - ε + ε/|A|,  if a = argmax Q(a)  (贪婪动作)
        ε/|A|,          otherwise            (非贪婪动作)
    }
    
    其中 Where:
    - ε ∈ [0,1]: 探索率 Exploration rate
    - |A|: 动作空间大小 Size of action space
    - Q(a): 动作a的价值估计 Value estimate of action a
    """
    
    def __init__(self, k: int = None, 
                 n_arms: int = None,
                 epsilon: float = 0.1,
                 epsilon_decay: Optional[float] = None,
                 epsilon_min: float = 0.01,
                 **kwargs):
        """
        初始化ε-贪婪智能体
        Initialize ε-greedy agent
        
        Args:
            k: 动作数量 Number of actions
            epsilon: 初始探索率 Initial exploration rate
            epsilon_decay: ε衰减率（每步乘以此值）Decay rate for ε
            epsilon_min: 最小ε值 Minimum ε value
            **kwargs: 传递给父类的参数
        
        深入理解 Deep Understanding:
        
        1. ε的选择 Choice of ε:
           - ε=0: 纯利用（可能陷入次优）Pure exploitation
           - ε=1: 纯探索（不使用学到的知识）Pure exploration
           - ε=0.1: 常用的平衡值 Common balanced value
        
        2. ε衰减 ε Decay:
           - 固定ε: 简单但可能过度探索 Simple but may over-explore
           - 衰减ε: 开始多探索，后期多利用 More exploration early, more exploitation later
           - 自适应ε: 根据不确定性调整 Adapt based on uncertainty
        """
        # 处理n_arms参数（向后兼容）
        if n_arms is not None:
            k = n_arms
        if k is None:
            raise ValueError("必须提供k或n_arms参数")
            
        # 先设置epsilon相关属性，因为reset()会用到
        self.epsilon_initial = epsilon
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # 然后调用父类初始化（会调用reset）
        super().__init__(k, **kwargs)
        
        # 记录探索vs利用
        self.exploration_count = 0
        self.exploitation_count = 0
        
        logger.info(f"初始化ε-贪婪智能体: ε={epsilon}, "
                   f"decay={epsilon_decay}, min={epsilon_min}")
    
    def select_action(self) -> int:
        """
        使用ε-贪婪策略选择动作
        Select action using ε-greedy policy
        
        算法 Algorithm:
        1. 生成随机数r ~ U(0,1)
        2. 如果r < ε: 探索（随机选择）
        3. 否则: 利用（选择Q值最大的动作）
        
        Returns:
            选择的动作 Selected action
        """
        # ε-贪婪决策
        if np.random.random() < self.epsilon:
            # 探索：随机选择
            # Exploration: random selection
            action = np.random.randint(self.k)
            self.exploration_count += 1
            
            logger.debug(f"探索: 随机选择动作 {action}")
        else:
            # 利用：选择最佳动作
            # Exploitation: select best action
            
            # 处理并列最优的情况（随机打破平局）
            # Handle ties (random tie-breaking)
            best_actions = np.where(self.Q == np.max(self.Q))[0]
            action = np.random.choice(best_actions)
            self.exploitation_count += 1
            
            logger.debug(f"利用: 选择最佳动作 {action} (Q={self.Q[action]:.3f})")
        
        # ε衰减
        if self.epsilon_decay is not None:
            self.epsilon = max(self.epsilon_min, 
                             self.epsilon * self.epsilon_decay)
        
        return action
    
    def reset(self):
        """
        重置智能体
        Reset agent
        """
        super().reset()
        self.epsilon = self.epsilon_initial
        self.exploration_count = 0
        self.exploitation_count = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        Get statistics
        """
        stats = super().get_statistics()
        total_actions = self.exploration_count + self.exploitation_count
        
        if total_actions > 0:
            stats.update({
                'epsilon': self.epsilon,
                'exploration_rate': self.exploration_count / total_actions,
                'exploitation_rate': self.exploitation_count / total_actions,
                'exploration_count': self.exploration_count,
                'exploitation_count': self.exploitation_count
            })
        
        return stats


# ================================================================================
# 第1.6.2节：ε-贪婪算法分析
# Section 1.6.2: ε-Greedy Algorithm Analysis
# ================================================================================

class EpsilonGreedyAnalysis:
    """
    ε-贪婪算法的理论分析
    Theoretical Analysis of ε-Greedy Algorithm
    
    分析内容：
    1. 收敛性 Convergence
    2. 遗憾界 Regret Bound
    3. 参数敏感性 Parameter Sensitivity
    """
    
    @staticmethod
    def theoretical_analysis():
        """
        理论分析
        Theoretical Analysis
        """
        print("\n" + "="*80)
        print("ε-贪婪算法理论分析")
        print("ε-Greedy Algorithm Theoretical Analysis")
        print("="*80)
        
        print("""
        1. 收敛性分析 Convergence Analysis
        -----------------------------------
        
        定理：在平稳k臂赌博机问题中，如果：
        Theorem: In stationary k-armed bandit, if:
        
        (1) 所有动作被无限次尝试 All actions are tried infinitely
            lim_{t→∞} N_t(a) = ∞, ∀a
            
        (2) 探索率趋于零 Exploration rate goes to zero
            lim_{t→∞} ε_t = 0
            
        (3) 探索总量发散 Total exploration diverges
            Σ_{t=1}^∞ ε_t = ∞
        
        则 Then:
            Q_t(a) → q*(a), ∀a  (概率1 with probability 1)
        
        常见ε调度 Common ε Schedules:
        - ε_t = 1/t: 满足条件，保证收敛 Satisfies conditions
        - ε_t = 0.1: 不满足条件2，不保证收敛 Doesn't converge
        - ε_t = 0.99^t: 不满足条件3，可能过早停止探索 May stop exploring too early
        
        2. 遗憾界 Regret Bound
        ----------------------
        
        对于固定ε，期望遗憾 For fixed ε, expected regret:
        
        E[L_T] ≤ εT·Δ_max + (1-ε)·Σ_{a:Δ_a>0} Δ_a·P(选择a)
        
        其中 Where:
        - Δ_a = max_a q*(a) - q*(a): 次优性差距 Suboptimality gap
        - Δ_max: 最大差距 Maximum gap
        
        渐近遗憾 Asymptotic regret:
        lim_{T→∞} E[L_T]/T = ε·E[Δ]
        
        这意味着：
        - 固定ε导致线性遗憾 Fixed ε leads to linear regret
        - 需要衰减ε以获得亚线性遗憾 Need decaying ε for sublinear regret
        
        3. 最优ε选择 Optimal ε Selection
        ---------------------------------
        
        如果知道时间范围T和差距Δ：
        If horizon T and gaps Δ are known:
        
        ε*_T = min(1, C·√(k·ln(T)/T))
        
        其中C是依赖于问题的常数
        Where C is a problem-dependent constant
        
        实践建议 Practical Recommendations:
        - 短期任务：ε = 0.1 ~ 0.3
        - 长期任务：使用衰减，如 ε_t = min(0.1, 100/t)
        - 非平稳环境：固定小ε，如 ε = 0.01 ~ 0.1
        """)
    
    @staticmethod
    def parameter_sensitivity_study():
        """
        参数敏感性研究
        Parameter Sensitivity Study
        
        研究ε对性能的影响
        Study the effect of ε on performance
        """
        print("\n" + "="*80)
        print("参数敏感性研究：ε的影响")
        print("Parameter Sensitivity: Effect of ε")
        print("="*80)
        
        # 不同的ε值
        epsilons = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        n_bandits = 100  # 运行100个赌博机问题
        n_steps = 1000
        
        results = {eps: {'rewards': [], 'optimal_actions': []} 
                  for eps in epsilons}
        
        print("运行实验...")
        for eps in tqdm(epsilons, desc="Testing ε values"):
            for _ in range(n_bandits):
                # 创建环境和智能体
                env = MultiArmedBandit(k=10, seed=None)
                agent = EpsilonGreedyAgent(k=10, epsilon=eps)
                
                # 运行
                episode_data = agent.run_episode(env, n_steps)
                results[eps]['rewards'].append(episode_data['rewards'])
                results[eps]['optimal_actions'].append(episode_data['optimal_actions'])
        
        # 计算平均性能
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 左图：平均奖励
        ax1 = axes[0]
        for eps in epsilons:
            avg_rewards = np.mean(results[eps]['rewards'], axis=0)
            ax1.plot(avg_rewards, label=f'ε={eps}', alpha=0.7)
        
        ax1.set_xlabel('Steps / 步数')
        ax1.set_ylabel('Average Reward / 平均奖励')
        ax1.set_title('Average Reward vs ε / 平均奖励与ε的关系')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右图：最优动作比例
        ax2 = axes[1]
        for eps in epsilons:
            optimal_rate = np.mean(results[eps]['optimal_actions'], axis=0)
            ax2.plot(optimal_rate, label=f'ε={eps}', alpha=0.7)
        
        ax2.set_xlabel('Steps / 步数')
        ax2.set_ylabel('Optimal Action % / 最优动作比例')
        ax2.set_title('Optimal Action Selection vs ε / 最优动作选择与ε的关系')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        
        # 分析结果
        print("\n分析结果 Analysis Results:")
        print("-" * 60)
        print("ε=0 (纯利用): 快速收敛但可能陷入次优")
        print("ε=0.01-0.1: 良好的平衡，推荐值")
        print("ε=0.2-0.5: 持续探索，学习较慢")
        print("ε=1.0 (纯探索): 不利用学到的知识")
        
        return fig


# ================================================================================
# 第1.6.3节：改进的ε-贪婪变体
# Section 1.6.3: Improved ε-Greedy Variants
# ================================================================================

class AdaptiveEpsilonGreedy(EpsilonGreedyAgent):
    """
    自适应ε-贪婪算法
    Adaptive ε-Greedy Algorithm
    
    根据学习进度自动调整ε
    Automatically adjust ε based on learning progress
    """
    
    def __init__(self, k: int, 
                 adaptation_mode: str = 'ucb_based',
                 **kwargs):
        """
        初始化自适应ε-贪婪
        Initialize adaptive ε-greedy
        
        Args:
            adaptation_mode: 自适应模式
                - 'ucb_based': 基于不确定性
                - 'value_based': 基于价值差异
                - 'time_based': 基于时间
        """
        super().__init__(k, **kwargs)
        self.adaptation_mode = adaptation_mode
        self.uncertainty = np.ones(k)  # 不确定性估计
    
    def select_action(self) -> int:
        """
        自适应选择动作
        Adaptively select action
        """
        # 根据模式调整ε
        if self.adaptation_mode == 'ucb_based':
            # 基于不确定性：不确定性越大，探索越多
            avg_uncertainty = np.mean(self.uncertainty)
            self.epsilon = min(1.0, avg_uncertainty)
            
        elif self.adaptation_mode == 'value_based':
            # 基于价值差异：价值接近时多探索
            if np.max(self.Q) - np.min(self.Q) < 0.1:
                self.epsilon = 0.2
            else:
                self.epsilon = 0.05
                
        elif self.adaptation_mode == 'time_based':
            # 基于时间：随时间递减
            total_steps = np.sum(self.N)
            if total_steps > 0:
                self.epsilon = min(self.epsilon_initial, 
                                 10.0 / (10.0 + total_steps))
        
        return super().select_action()
    
    def update(self, action: int, reward: float):
        """
        更新并调整不确定性
        Update and adjust uncertainty
        """
        super().update(action, reward)
        
        # 更新不确定性估计
        if self.N[action] > 0:
            # 基于访问次数的不确定性
            self.uncertainty[action] = 1.0 / np.sqrt(self.N[action])


class DecayingEpsilonGreedy(EpsilonGreedyAgent):
    """
    衰减ε-贪婪算法
    Decaying ε-Greedy Algorithm
    
    使用不同的衰减策略
    Use different decay strategies
    """
    
    def __init__(self, k: int,
                 decay_mode: str = 'exponential',
                 **kwargs):
        """
        初始化衰减ε-贪婪
        
        Args:
            decay_mode: 衰减模式
                - 'exponential': 指数衰减 ε_t = ε_0 * decay^t
                - 'inverse': 反比衰减 ε_t = ε_0 / (1 + decay*t)
                - 'linear': 线性衰减 ε_t = max(ε_min, ε_0 - decay*t)
        """
        super().__init__(k, **kwargs)
        self.decay_mode = decay_mode
        self.time_step = 0
    
    def select_action(self) -> int:
        """
        使用衰减ε选择动作
        Select action with decaying ε
        """
        self.time_step += 1
        
        # 根据模式计算ε
        if self.decay_mode == 'exponential':
            if self.epsilon_decay:
                self.epsilon = self.epsilon_initial * (self.epsilon_decay ** self.time_step)
                
        elif self.decay_mode == 'inverse':
            decay_rate = 0.01 if self.epsilon_decay is None else self.epsilon_decay
            self.epsilon = self.epsilon_initial / (1 + decay_rate * self.time_step)
            
        elif self.decay_mode == 'linear':
            decay_rate = 0.001 if self.epsilon_decay is None else self.epsilon_decay
            self.epsilon = max(self.epsilon_min, 
                             self.epsilon_initial - decay_rate * self.time_step)
        
        # 确保ε在合理范围
        self.epsilon = max(self.epsilon_min, min(1.0, self.epsilon))
        
        return super().select_action()


# ================================================================================
# 第1.6.4节：ε-贪婪算法比较实验
# Section 1.6.4: ε-Greedy Algorithm Comparison Experiments
# ================================================================================

def compare_epsilon_greedy_variants():
    """
    比较不同ε-贪婪变体
    Compare different ε-greedy variants
    """
    print("\n" + "="*80)
    print("ε-贪婪算法变体比较")
    print("ε-Greedy Variants Comparison")
    print("="*80)
    
    # 算法配置
    agents_config = [
        ('Fixed ε=0.1', EpsilonGreedyAgent(k=10, epsilon=0.1)),
        ('Fixed ε=0.01', EpsilonGreedyAgent(k=10, epsilon=0.01)),
        ('Exponential Decay', DecayingEpsilonGreedy(k=10, epsilon=0.5, 
                                                    decay_mode='exponential',
                                                    epsilon_decay=0.995)),
        ('Inverse Decay', DecayingEpsilonGreedy(k=10, epsilon=0.5,
                                               decay_mode='inverse')),
        ('Adaptive (UCB)', AdaptiveEpsilonGreedy(k=10, epsilon=0.2,
                                                 adaptation_mode='ucb_based')),
    ]
    
    # 实验参数
    n_runs = 100
    n_steps = 1000
    
    # 运行实验
    results = {}
    
    print("运行对比实验...")
    for name, agent in tqdm(agents_config, desc="Testing agents"):
        rewards_all = []
        optimal_all = []
        regrets_all = []
        
        for run in range(n_runs):
            # 创建新环境
            env = MultiArmedBandit(k=10, seed=run)
            
            # 重置智能体
            agent.reset()
            
            # 运行回合
            episode_data = agent.run_episode(env, n_steps)
            
            rewards_all.append(episode_data['rewards'])
            optimal_all.append(episode_data['optimal_actions'])
            regrets_all.append(episode_data['regrets'])
        
        results[name] = {
            'rewards': np.mean(rewards_all, axis=0),
            'optimal': np.mean(optimal_all, axis=0),
            'regrets': np.mean(regrets_all, axis=0),
            'rewards_std': np.std(rewards_all, axis=0),
            'optimal_std': np.std(optimal_all, axis=0)
        }
    
    # 绘制结果
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 平均奖励
    ax1 = axes[0, 0]
    for name, data in results.items():
        ax1.plot(data['rewards'], label=name, alpha=0.8)
    ax1.set_xlabel('Steps / 步数')
    ax1.set_ylabel('Average Reward / 平均奖励')
    ax1.set_title('Average Reward Comparison / 平均奖励比较')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 最优动作比例
    ax2 = axes[0, 1]
    for name, data in results.items():
        ax2.plot(data['optimal'] * 100, label=name, alpha=0.8)
    ax2.set_xlabel('Steps / 步数')
    ax2.set_ylabel('Optimal Action % / 最优动作百分比')
    ax2.set_title('Optimal Action Selection / 最优动作选择率')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])
    
    # 3. 累积遗憾
    ax3 = axes[1, 0]
    for name, data in results.items():
        ax3.plot(data['regrets'], label=name, alpha=0.8)
    ax3.set_xlabel('Steps / 步数')
    ax3.set_ylabel('Cumulative Regret / 累积遗憾')
    ax3.set_title('Cumulative Regret / 累积遗憾')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 最终性能对比
    ax4 = axes[1, 1]
    final_rewards = [data['rewards'][-100:].mean() for name, data in results.items()]
    final_optimal = [data['optimal'][-100:].mean() * 100 for name, data in results.items()]
    
    x_pos = np.arange(len(agents_config))
    width = 0.35
    
    bars1 = ax4.bar(x_pos - width/2, final_rewards, width, label='Final Reward', alpha=0.8)
    ax4_2 = ax4.twinx()
    bars2 = ax4_2.bar(x_pos + width/2, final_optimal, width, label='Final Optimal %', 
                      color='orange', alpha=0.8)
    
    ax4.set_xlabel('Algorithm / 算法')
    ax4.set_ylabel('Final Average Reward / 最终平均奖励', color='blue')
    ax4_2.set_ylabel('Final Optimal % / 最终最优比例', color='orange')
    ax4.set_title('Final Performance Comparison / 最终性能比较')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([name.split()[0] for name, _ in agents_config], rotation=45)
    ax4.tick_params(axis='y', labelcolor='blue')
    ax4_2.tick_params(axis='y', labelcolor='orange')
    
    # 添加数值标签
    for bar, val in zip(bars1, final_rewards):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, final_optimal):
        ax4_2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                  f'{val:.0f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # 打印总结
    print("\n实验总结 Experiment Summary:")
    print("-" * 60)
    for name, _ in agents_config:
        final_reward = results[name]['rewards'][-100:].mean()
        final_optimal = results[name]['optimal'][-100:].mean() * 100
        final_regret = results[name]['regrets'][-1]
        
        print(f"\n{name}:")
        print(f"  最终平均奖励: {final_reward:.3f}")
        print(f"  最终最优动作比例: {final_optimal:.1f}%")
        print(f"  总遗憾: {final_regret:.1f}")
    
    print("\n关键发现 Key Findings:")
    print("1. 固定小ε(0.01)：收敛快但探索不足")
    print("2. 固定大ε(0.1)：持续探索但收敛慢")
    print("3. 衰减ε：平衡探索和利用，性能最佳")
    print("4. 自适应ε：根据不确定性调整，稳健性好")
    
    return fig


# ================================================================================
# 主函数：运行ε-贪婪算法演示
# Main Function: Run ε-Greedy Algorithm Demo
# ================================================================================

def main():
    """
    运行ε-贪婪算法完整演示
    Run complete ε-greedy algorithm demo
    """
    print("\n" + "="*80)
    print("第1.6节：ε-贪婪算法")
    print("Section 1.6: ε-Greedy Algorithm")
    print("="*80)
    
    # 1. 理论分析
    EpsilonGreedyAnalysis.theoretical_analysis()
    
    # 2. 参数敏感性研究
    fig1 = EpsilonGreedyAnalysis.parameter_sensitivity_study()
    
    # 3. 变体比较
    fig2 = compare_epsilon_greedy_variants()
    
    print("\n" + "="*80)
    print("ε-贪婪算法演示完成！")
    print("ε-Greedy Algorithm Demo Complete!")
    print("="*80)
    
    plt.show()
    
    return [fig1, fig2]


if __name__ == "__main__":
    main()