"""
================================================================================
第1.7节：UCB算法 - 基于置信度的乐观探索
Section 1.7: UCB Algorithm - Optimism-Based Exploration with Confidence
================================================================================

UCB (Upper Confidence Bound) 是一种更智能的探索策略
不是随机探索，而是优先探索不确定性高的动作

UCB is a smarter exploration strategy
Instead of random exploration, it prioritizes actions with high uncertainty

核心思想 Core Idea:
"面对不确定性时保持乐观" - 选择置信上界最高的动作
"Optimism in the face of uncertainty" - Select action with highest upper confidence bound
"""

import numpy as np
from typing import Optional, Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging

from .bandit_introduction import BaseBanditAgent, MultiArmedBandit

# 设置日志
logger = logging.getLogger(__name__)


# ================================================================================
# 第1.7.1节：UCB算法原理
# Section 1.7.1: UCB Algorithm Principle
# ================================================================================

class UCBPrinciple:
    """
    UCB算法的核心原理
    Core Principle of UCB Algorithm
    
    UCB来源于"乐观面对不确定性"原则
    UCB comes from the principle of "optimism in the face of uncertainty"
    """
    
    @staticmethod
    def explain_ucb_principle():
        """
        详解UCB原理
        Detailed Explanation of UCB Principle
        """
        print("\n" + "="*80)
        print("UCB算法原理")
        print("UCB Algorithm Principle")
        print("="*80)
        
        print("""
        1. 核心思想 Core Idea
        ---------------------
        
        UCB选择动作的准则：
        UCB action selection criterion:
        
        A_t = argmax_a [Q_t(a) + c·√(ln(t)/N_t(a))]
        
        其中 Where:
        - Q_t(a): 动作a的价值估计（利用项）
                 Value estimate of action a (exploitation term)
        - c·√(ln(t)/N_t(a)): 置信半径（探索项）
                            Confidence radius (exploration term)
        - c: 控制探索程度的参数
             Parameter controlling exploration degree
        - t: 当前时间步
             Current time step
        - N_t(a): 动作a被选择的次数
                 Number of times action a was selected
        
        2. 直观理解 Intuitive Understanding
        ------------------------------------
        
        想象每个动作的真实价值在一个置信区间内：
        Imagine each action's true value lies in a confidence interval:
        
        真实价值 q*(a) ∈ [Q_t(a) - U_t(a), Q_t(a) + U_t(a)]
        True value q*(a) ∈ [Q_t(a) - U_t(a), Q_t(a) + U_t(a)]
        
        UCB策略：选择上界最高的动作（乐观）
        UCB strategy: Select action with highest upper bound (optimistic)
        
        这确保了：
        This ensures:
        - 不确定的动作得到探索（U_t(a)大）
          Uncertain actions get explored (large U_t(a))
        - 好的动作得到利用（Q_t(a)大）
          Good actions get exploited (large Q_t(a))
        
        3. 数学基础 Mathematical Foundation
        ------------------------------------
        
        Hoeffding不等式 Hoeffding's Inequality:
        P[|Q_t(a) - q*(a)| ≥ u] ≤ 2·exp(-2·N_t(a)·u²)
        
        选择置信水平 1-δ，令：
        Choose confidence level 1-δ, let:
        
        u = √(ln(2/δ) / (2·N_t(a)))
        
        则以概率至少1-δ：
        Then with probability at least 1-δ:
        
        q*(a) ≤ Q_t(a) + √(ln(2/δ) / (2·N_t(a)))
        
        UCB使用 √(2·ln(t)/N_t(a)) 作为置信半径
        UCB uses √(2·ln(t)/N_t(a)) as confidence radius
        
        4. 关键性质 Key Properties
        --------------------------
        
        • 无参数调优：只需设置c（通常c=2）
          No parameter tuning: Only need to set c (usually c=2)
          
        • 自适应探索：随着N_t(a)增加，不确定性自动减少
          Adaptive exploration: Uncertainty automatically decreases as N_t(a) increases
          
        • 理论保证：对数遗憾界 O(ln T)
          Theoretical guarantee: Logarithmic regret bound O(ln T)
          
        • 确定性：相同情况下总是选择相同动作
          Deterministic: Always selects same action in same situation
        """)
        
        # 可视化UCB原理
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 左图：置信区间可视化
        ax1 = axes[0]
        k = 5
        Q_values = np.array([0.5, 0.3, 0.7, 0.4, 0.6])
        N_values = np.array([100, 20, 50, 10, 30])
        t = 200
        c = 2
        
        UCB_values = Q_values + c * np.sqrt(np.log(t) / N_values)
        confidence_radius = c * np.sqrt(np.log(t) / N_values)
        
        x_pos = np.arange(k)
        ax1.bar(x_pos, Q_values, label='Q(a) - 估计值', alpha=0.6, color='blue')
        ax1.errorbar(x_pos, Q_values, yerr=confidence_radius, 
                    fmt='none', color='red', capsize=5, capthick=2,
                    label='置信区间')
        ax1.scatter(x_pos, UCB_values, color='green', s=100, marker='*',
                   label='UCB值', zorder=5)
        
        # 标记最佳动作
        best_action = np.argmax(UCB_values)
        ax1.annotate(f'选择动作{best_action}', 
                    xy=(best_action, UCB_values[best_action]),
                    xytext=(best_action+0.5, UCB_values[best_action]+0.2),
                    arrowprops=dict(arrowstyle='->', color='green'),
                    fontsize=10)
        
        ax1.set_xlabel('Action / 动作')
        ax1.set_ylabel('Value / 价值')
        ax1.set_title('UCB Action Selection / UCB动作选择')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右图：探索奖励随时间变化
        ax2 = axes[1]
        time_steps = np.arange(1, 1001)
        N_counts = [10, 50, 100, 500]
        
        for N in N_counts:
            exploration_bonus = c * np.sqrt(np.log(time_steps) / N)
            ax2.plot(time_steps, exploration_bonus, label=f'N(a)={N}', alpha=0.7)
        
        ax2.set_xlabel('Time Step t / 时间步')
        ax2.set_ylabel('Exploration Bonus / 探索奖励')
        ax2.set_title('Exploration Bonus Over Time / 探索奖励随时间变化')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 1000])
        
        plt.tight_layout()
        return fig


# ================================================================================
# 第1.7.2节：UCB算法实现
# Section 1.7.2: UCB Algorithm Implementation
# ================================================================================

class UCBAgent(BaseBanditAgent):
    """
    UCB智能体
    UCB Agent
    
    实现Upper Confidence Bound算法
    Implements Upper Confidence Bound algorithm
    """
    
    def __init__(self, k: int = None,
                 n_arms: int = None,
                 c: float = 2.0,
                 **kwargs):
        """
        初始化UCB智能体
        Initialize UCB agent
        
        Args:
            k: 动作数量 Number of actions
            c: 探索参数 Exploration parameter
               - c=0: 纯利用 Pure exploitation
               - c=√2: 理论最优 Theoretically optimal
               - c=2: 实践常用值 Common practical value
            **kwargs: 传递给父类的参数
        
        深入理解 Deep Understanding:
        
        c参数的选择：
        Choice of c parameter:
        
        - 小c (< 1): 更多利用，快速收敛但可能次优
          Small c: More exploitation, fast convergence but may be suboptimal
          
        - 大c (> 2): 更多探索，收敛慢但更可能找到最优
          Large c: More exploration, slow convergence but more likely to find optimal
          
        - c = √(2ln(1/δ)): 理论值，其中δ是置信参数
          Theoretical value where δ is confidence parameter
        """
        # 处理n_arms参数（向后兼容）
        if n_arms is not None:
            k = n_arms
        if k is None:
            raise ValueError("必须提供k或n_arms参数")
            
        super().__init__(k, **kwargs)
        self.c = c
        self.t = 0  # 时间步计数器
        
        logger.info(f"初始化UCB智能体: k={k}, c={c}")
    
    def select_action(self, t: int = None) -> int:
        """
        使用UCB策略选择动作
        Select action using UCB policy
        
        UCB公式：
        UCB formula:
        UCB_t(a) = Q_t(a) + c·√(ln(t)/N_t(a))
        
        Args:
            t: 时间步（可选，如果不提供则使用内部计数器）
               Time step (optional, uses internal counter if not provided)
        
        Returns:
            选择的动作 Selected action
        """
        if t is not None:
            self.t = t
        else:
            self.t += 1
        
        # 如果有未尝试的动作，先尝试
        # If there are untried actions, try them first
        untried_actions = np.where(self.N == 0)[0]
        if len(untried_actions) > 0:
            action = np.random.choice(untried_actions)
            logger.debug(f"探索未尝试动作: {action}")
            return action
        
        # 计算所有动作的UCB值
        # Calculate UCB values for all actions
        ucb_values = self._calculate_ucb_values()
        
        # 选择UCB值最大的动作（处理并列）
        # Select action with maximum UCB value (handle ties)
        best_actions = np.where(ucb_values == np.max(ucb_values))[0]
        action = np.random.choice(best_actions)
        
        logger.debug(f"t={self.t}: 选择动作{action}, "
                    f"UCB={ucb_values[action]:.3f} "
                    f"(Q={self.Q[action]:.3f}, "
                    f"U={ucb_values[action]-self.Q[action]:.3f})")
        
        return action
    
    def _calculate_ucb_values(self) -> np.ndarray:
        """
        计算所有动作的UCB值
        Calculate UCB values for all actions
        
        Returns:
            UCB值数组 Array of UCB values
        """
        # 避免除零（虽然理论上不应该发生）
        # Avoid division by zero (though shouldn't happen theoretically)
        N_safe = np.maximum(self.N, 1e-10)
        
        # 计算探索奖励（置信半径）
        # Calculate exploration bonus (confidence radius)
        exploration_bonus = self.c * np.sqrt(np.log(self.t) / N_safe)
        
        # UCB = 估计值 + 探索奖励
        # UCB = estimate + exploration bonus
        ucb_values = self.Q + exploration_bonus
        
        # 对未尝试的动作设置为无穷大（优先探索）
        # Set infinity for untried actions (prioritize exploration)
        ucb_values[self.N == 0] = float('inf')
        
        return ucb_values
    
    def reset(self):
        """
        重置智能体
        Reset agent
        """
        super().reset()
        self.t = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        Get statistics
        """
        stats = super().get_statistics()
        
        if self.t > 0:
            ucb_values = self._calculate_ucb_values()
            stats.update({
                'time_step': self.t,
                'ucb_values': ucb_values,
                'exploration_bonus': ucb_values - self.Q,
                'c_parameter': self.c
            })
        
        return stats


# ================================================================================
# 第1.7.3节：UCB算法变体
# Section 1.7.3: UCB Algorithm Variants
# ================================================================================

class UCB2Agent(UCBAgent):
    """
    UCB2算法
    UCB2 Algorithm
    
    改进的UCB，使用分阶段采样减少计算
    Improved UCB using epochal sampling to reduce computation
    """
    
    def __init__(self, k: int, alpha: float = 0.5, **kwargs):
        """
        初始化UCB2
        
        Args:
            alpha: 控制epoch长度的参数
        """
        super().__init__(k, **kwargs)
        self.alpha = alpha
        self.epochs = np.zeros(k, dtype=int)  # 每个动作的epoch数
        self.epoch_counts = np.zeros(k, dtype=int)  # 当前epoch的计数
        
    def select_action(self) -> int:
        """
        UCB2动作选择
        使用epoch机制减少切换
        """
        self.t += 1
        
        # 初始化：每个动作至少尝试一次
        untried = np.where(self.N == 0)[0]
        if len(untried) > 0:
            return np.random.choice(untried)
        
        # 检查是否需要开始新的epoch
        for a in range(self.k):
            tau = self._compute_tau(self.epochs[a] + 1)
            if self.epoch_counts[a] >= tau - self._compute_tau(self.epochs[a]):
                self.epochs[a] += 1
                self.epoch_counts[a] = 0
        
        # 计算UCB2值
        ucb2_values = self.Q + self._compute_radius()
        return np.argmax(ucb2_values)
    
    def _compute_tau(self, r: int) -> int:
        """计算第r个epoch的长度"""
        return int(np.ceil((1 + self.alpha) ** r))
    
    def _compute_radius(self) -> np.ndarray:
        """计算置信半径"""
        radius = np.zeros(self.k)
        for a in range(self.k):
            if self.N[a] > 0:
                tau_r = self._compute_tau(self.epochs[a])
                radius[a] = np.sqrt((1 + self.alpha) * np.log(np.e * self.t / tau_r) / (2 * tau_r))
        return radius


class BayesianUCBAgent(UCBAgent):
    """
    贝叶斯UCB算法
    Bayesian UCB Algorithm
    
    使用贝叶斯方法估计置信区间
    Use Bayesian methods to estimate confidence intervals
    """
    
    def __init__(self, k: int, 
                 prior_mean: float = 0.0,
                 prior_std: float = 1.0,
                 **kwargs):
        """
        初始化贝叶斯UCB
        
        Args:
            prior_mean: 先验均值
            prior_std: 先验标准差
        """
        super().__init__(k, **kwargs)
        
        # 贝叶斯参数（假设高斯先验和似然）
        self.prior_mean = prior_mean
        self.prior_variance = prior_std ** 2
        
        # 后验参数
        self.posterior_mean = np.ones(k) * prior_mean
        self.posterior_variance = np.ones(k) * self.prior_variance
        
    def update(self, action: int, reward: float):
        """
        贝叶斯更新
        Bayesian update
        """
        super().update(action, reward)
        
        # 假设奖励方差为1（可以估计）
        reward_variance = 1.0
        
        # 贝叶斯更新公式（高斯-高斯共轭）
        # Bayesian update formula (Gaussian-Gaussian conjugate)
        
        # 精度（方差的倒数）
        prior_precision = 1.0 / self.posterior_variance[action]
        likelihood_precision = 1.0 / reward_variance
        
        # 更新后验方差
        posterior_precision = prior_precision + likelihood_precision
        self.posterior_variance[action] = 1.0 / posterior_precision
        
        # 更新后验均值
        self.posterior_mean[action] = (
            (prior_precision * self.posterior_mean[action] + 
             likelihood_precision * reward) / posterior_precision
        )
    
    def _calculate_ucb_values(self) -> np.ndarray:
        """
        计算贝叶斯UCB值
        Calculate Bayesian UCB values
        """
        # 使用后验分布的分位数作为UCB
        # Use quantile of posterior distribution as UCB
        
        # c参数控制置信水平（例如c=2对应95%置信区间）
        # c parameter controls confidence level (e.g., c=2 for 95% CI)
        ucb_values = self.posterior_mean + self.c * np.sqrt(self.posterior_variance)
        
        return ucb_values


class UCBTunedAgent(UCBAgent):
    """
    UCB-Tuned算法
    UCB-Tuned Algorithm
    
    自适应调整探索参数的UCB变体
    UCB variant with adaptive exploration parameter
    """
    
    def __init__(self, k: int, **kwargs):
        """初始化UCB-Tuned"""
        super().__init__(k, c=1.0, **kwargs)  # c会被动态调整
        
        # 记录奖励的平方（用于计算方差）
        self.reward_squares = np.zeros(k)
        
    def update(self, action: int, reward: float):
        """更新包括奖励平方"""
        super().update(action, reward)
        self.reward_squares[action] += reward ** 2
        
    def _calculate_ucb_values(self) -> np.ndarray:
        """
        计算UCB-Tuned值
        使用经验方差调整探索
        """
        ucb_values = np.zeros(self.k)
        
        for a in range(self.k):
            if self.N[a] == 0:
                ucb_values[a] = float('inf')
            else:
                # 计算经验方差
                mean_square = self.reward_squares[a] / self.N[a]
                variance = mean_square - self.Q[a] ** 2
                variance = max(0, variance)  # 数值稳定性
                
                # 调整的探索项
                V = variance + np.sqrt(2 * np.log(self.t) / self.N[a])
                tuned_radius = np.sqrt(np.log(self.t) / self.N[a] * min(0.25, V))
                
                ucb_values[a] = self.Q[a] + tuned_radius
        
        return ucb_values


# ================================================================================
# 第1.7.4节：UCB算法理论分析
# Section 1.7.4: UCB Algorithm Theoretical Analysis
# ================================================================================

class UCBTheoreticalAnalysis:
    """
    UCB算法的理论分析
    Theoretical Analysis of UCB Algorithm
    """
    
    @staticmethod
    def regret_analysis():
        """
        遗憾界分析
        Regret Bound Analysis
        """
        print("\n" + "="*80)
        print("UCB算法遗憾界分析")
        print("UCB Algorithm Regret Bound Analysis")
        print("="*80)
        
        print("""
        UCB算法的理论保证
        Theoretical Guarantees of UCB
        
        1. 遗憾上界 Regret Upper Bound
        --------------------------------
        
        定理（Auer et al., 2002）:
        Theorem (Auer et al., 2002):
        
        对于UCB算法with c = √2，期望遗憾满足：
        For UCB with c = √2, expected regret satisfies:
        
        E[L_T] ≤ 8·Σ_{a:Δ_a>0} (ln T / Δ_a) + (1 + π²/3)·Σ_{a:Δ_a>0} Δ_a
        
        其中 Where:
        - L_T: T步的累积遗憾 Cumulative regret after T steps
        - Δ_a = μ* - μ_a: 次优性差距 Suboptimality gap
        - μ*: 最优臂的期望奖励 Expected reward of optimal arm
        
        渐近行为 Asymptotic Behavior:
        E[L_T] = O(√(k·T·ln T))  (问题无关界 Problem-independent bound)
        E[L_T] = O((k/Δ)·ln T)   (问题相关界 Problem-dependent bound)
        
        2. 与其他算法比较 Comparison with Other Algorithms
        ---------------------------------------------------
        
        算法           遗憾界              优点              缺点
        Algorithm      Regret Bound        Pros              Cons
        
        ε-greedy      O(T^{2/3})          简单              线性遗憾(固定ε)
                                         Simple            Linear regret (fixed ε)
        
        UCB           O(ln T)             对数遗憾          需要知道时间范围
                                         Logarithmic       Needs time horizon
        
        Thompson      O(ln T)             贝叶斯最优        计算复杂
        Sampling                         Bayesian optimal  Computationally complex
        
        EXP3          O(√(T·ln k))        对抗性环境        保守
                                         Adversarial       Conservative
        
        3. 最优性 Optimality
        --------------------
        
        Lai-Robbins下界 Lower Bound:
        
        对于任何一致好的算法：
        For any uniformly good algorithm:
        
        lim inf_{T→∞} E[L_T] / ln T ≥ Σ_{a:Δ_a>0} Δ_a / KL(μ_a, μ*)
        
        UCB达到此下界的常数因子内！
        UCB achieves this bound within a constant factor!
        
        4. 实践考虑 Practical Considerations
        -------------------------------------
        
        • 有限时间性能：理论界在小T时较松
          Finite-time performance: Bounds are loose for small T
          
        • 参数调优：c=2通常比理论值c=√2更好
          Parameter tuning: c=2 often better than theoretical c=√2
          
        • 计算效率：O(1)选择时间，O(k)空间
          Computational efficiency: O(1) selection time, O(k) space
        """)
    
    @staticmethod
    def demonstrate_regret_growth():
        """
        演示遗憾增长
        Demonstrate Regret Growth
        """
        print("\n演示：不同算法的遗憾增长")
        print("Demo: Regret Growth of Different Algorithms")
        print("-" * 60)
        
        # 创建测试环境
        k = 10
        n_runs = 50
        n_steps = 2000
        
        # 不同算法
        from epsilon_greedy import EpsilonGreedyAgent
        
        algorithms = [
            ('UCB (c=2)', lambda: UCBAgent(k=k, c=2.0)),
            ('UCB (c=√2)', lambda: UCBAgent(k=k, c=np.sqrt(2))),
            ('ε-greedy (ε=0.1)', lambda: EpsilonGreedyAgent(k=k, epsilon=0.1)),
            ('ε-greedy (ε=0.01)', lambda: EpsilonGreedyAgent(k=k, epsilon=0.01)),
        ]
        
        results = {name: [] for name, _ in algorithms}
        
        print("运行实验...")
        for run in tqdm(range(n_runs), desc="Runs"):
            # 创建环境
            env = MultiArmedBandit(k=k, seed=run)
            
            for name, agent_factory in algorithms:
                agent = agent_factory()
                agent.reset()
                
                # 运行并记录遗憾
                episode_data = agent.run_episode(env, n_steps)
                results[name].append(episode_data['regrets'])
                
                env.reset()
        
        # 绘制结果
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 左图：累积遗憾
        ax1 = axes[0]
        for name in results:
            mean_regret = np.mean(results[name], axis=0)
            std_regret = np.std(results[name], axis=0)
            steps = np.arange(1, n_steps + 1)
            
            ax1.plot(steps, mean_regret, label=name, alpha=0.8)
            # 添加置信区间
            ax1.fill_between(steps, 
                            mean_regret - std_regret/np.sqrt(n_runs),
                            mean_regret + std_regret/np.sqrt(n_runs),
                            alpha=0.2)
        
        ax1.set_xlabel('Steps / 步数')
        ax1.set_ylabel('Cumulative Regret / 累积遗憾')
        ax1.set_title('Cumulative Regret Comparison / 累积遗憾比较')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右图：对数尺度（检验O(ln T)增长）
        ax2 = axes[1]
        for name in results:
            mean_regret = np.mean(results[name], axis=0)
            steps = np.arange(1, n_steps + 1)
            ax2.plot(steps, mean_regret, label=name, alpha=0.8)
        
        ax2.set_xlabel('Steps (log scale) / 步数（对数尺度）')
        ax2.set_ylabel('Cumulative Regret / 累积遗憾')
        ax2.set_title('Regret Growth Rate / 遗憾增长率')
        ax2.set_xscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 打印最终遗憾
        print("\n最终遗憾 Final Regret:")
        for name in results:
            final_regret = np.mean([r[-1] for r in results[name]])
            print(f"{name:20s}: {final_regret:.1f}")
        
        return fig


# ================================================================================
# 第1.7.5节：UCB算法比较实验
# Section 1.7.5: UCB Algorithm Comparison Experiments
# ================================================================================

def compare_ucb_variants():
    """
    比较不同UCB变体
    Compare different UCB variants
    """
    print("\n" + "="*80)
    print("UCB变体比较实验")
    print("UCB Variants Comparison Experiment")
    print("="*80)
    
    # 创建测试场景
    k = 10
    n_runs = 100
    n_steps = 1000
    
    # 不同UCB变体
    agents = [
        ('UCB (c=2)', UCBAgent(k=k, c=2.0)),
        ('UCB (c=1)', UCBAgent(k=k, c=1.0)),
        ('UCB-Tuned', UCBTunedAgent(k=k)),
        ('Bayesian UCB', BayesianUCBAgent(k=k)),
    ]
    
    results = {name: {'rewards': [], 'optimal': [], 'regrets': []} 
              for name, _ in agents}
    
    print("运行比较实验...")
    for run in tqdm(range(n_runs), desc="Running experiments"):
        env = MultiArmedBandit(k=k, seed=run)
        
        for name, agent in agents:
            agent.reset()
            episode_data = agent.run_episode(env, n_steps)
            
            results[name]['rewards'].append(episode_data['rewards'])
            results[name]['optimal'].append(episode_data['optimal_actions'])
            results[name]['regrets'].append(episode_data['regrets'])
            
            env.reset()
    
    # 绘制结果
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 平均奖励
    ax1 = axes[0, 0]
    for name in results:
        mean_rewards = np.mean(results[name]['rewards'], axis=0)
        ax1.plot(mean_rewards, label=name, alpha=0.8)
    ax1.set_xlabel('Steps / 步数')
    ax1.set_ylabel('Average Reward / 平均奖励')
    ax1.set_title('Average Reward Over Time / 平均奖励随时间变化')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 最优动作比例
    ax2 = axes[0, 1]
    for name in results:
        optimal_rate = np.mean(results[name]['optimal'], axis=0) * 100
        ax2.plot(optimal_rate, label=name, alpha=0.8)
    ax2.set_xlabel('Steps / 步数')
    ax2.set_ylabel('Optimal Action % / 最优动作百分比')
    ax2.set_title('Optimal Action Selection Rate / 最优动作选择率')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])
    
    # 3. 累积遗憾
    ax3 = axes[1, 0]
    for name in results:
        mean_regrets = np.mean(results[name]['regrets'], axis=0)
        ax3.plot(mean_regrets, label=name, alpha=0.8)
    ax3.set_xlabel('Steps / 步数')
    ax3.set_ylabel('Cumulative Regret / 累积遗憾')
    ax3.set_title('Cumulative Regret / 累积遗憾')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 箱线图：最终性能
    ax4 = axes[1, 1]
    final_rewards = {name: [np.mean(r[-100:]) for r in results[name]['rewards']] 
                    for name in results}
    
    box_data = [final_rewards[name] for name in results]
    box_labels = [name for name in results]
    
    bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], plt.cm.Set2.colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_ylabel('Final Average Reward / 最终平均奖励')
    ax4.set_title('Final Performance Distribution / 最终性能分布')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # 打印统计
    print("\n性能统计 Performance Statistics:")
    print("-" * 60)
    for name in results:
        final_reward = np.mean([np.mean(r[-100:]) for r in results[name]['rewards']])
        final_optimal = np.mean([np.mean(o[-100:]) for o in results[name]['optimal']]) * 100
        final_regret = np.mean([r[-1] for r in results[name]['regrets']])
        
        print(f"\n{name}:")
        print(f"  最终平均奖励: {final_reward:.3f}")
        print(f"  最终最优动作比例: {final_optimal:.1f}%")
        print(f"  总遗憾: {final_regret:.1f}")
    
    return fig


# ================================================================================
# 主函数：运行UCB算法演示
# Main Function: Run UCB Algorithm Demo
# ================================================================================

def main():
    """
    运行UCB算法完整演示
    Run complete UCB algorithm demo
    """
    print("\n" + "="*80)
    print("第1.7节：UCB（置信上界）算法")
    print("Section 1.7: UCB (Upper Confidence Bound) Algorithm")
    print("="*80)
    
    # 1. 原理解释
    fig1 = UCBPrinciple.explain_ucb_principle()
    
    # 2. 理论分析
    UCBTheoreticalAnalysis.regret_analysis()
    fig2 = UCBTheoreticalAnalysis.demonstrate_regret_growth()
    
    # 3. 变体比较
    fig3 = compare_ucb_variants()
    
    print("\n" + "="*80)
    print("UCB算法演示完成！")
    print("UCB Algorithm Demo Complete!")
    print("\n关键要点 Key Takeaways:")
    print("1. UCB平衡探索和利用，无需调参")
    print("2. 理论保证对数遗憾O(ln T)")
    print("3. 适合平稳环境，计算简单")
    print("4. UCB-Tuned等变体可能实践性能更好")
    print("="*80)
    
    plt.show()
    
    return [fig1, fig2, fig3]


if __name__ == "__main__":
    main()