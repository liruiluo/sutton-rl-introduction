"""
================================================================================
第4.1节：蒙特卡洛方法基础 - 从经验中学习
Section 4.1: Monte Carlo Foundations - Learning from Experience
================================================================================

蒙特卡洛方法是强化学习的重要转折点：从需要模型到不需要模型。
Monte Carlo methods are a turning point in RL: from model-based to model-free.

核心洞察：我们可以通过平均实际回报来估计期望回报
Core insight: We can estimate expected return by averaging actual returns

这基于大数定律：
This is based on the Law of Large Numbers:
随着样本增加，样本均值收敛到期望值
As samples increase, sample mean converges to expected value

比喻：就像通过多次投掷来估计硬币正面的概率
Analogy: Like estimating coin's probability by many tosses

MC的优势：
Advantages of MC:
1. 不需要环境模型（只需要能采样）
   No need for environment model (only need sampling)
2. 可以从实际或模拟经验中学习
   Can learn from actual or simulated experience
3. 不受马尔可夫性限制
   Not restricted by Markov property
4. 可以专注于感兴趣的状态子集
   Can focus on subset of states of interest

MC的劣势：
Disadvantages of MC:
1. 只适用于回合式任务
   Only works for episodic tasks
2. 需要等到回合结束才能更新
   Must wait until episode ends to update
3. 高方差（但无偏）
   High variance (but unbiased)
4. 收敛可能很慢
   Convergence can be slow
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time

# 导入基础组件
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.ch03_finite_mdp.mdp_framework import State, Action, MDPEnvironment
from src.ch03_finite_mdp.policies_and_values import (
    Policy, StateValueFunction, ActionValueFunction
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第4.1.1节：回合与回报
# Section 4.1.1: Episodes and Returns
# ================================================================================

@dataclass
class Experience:
    """
    单步经验
    Single-step experience
    
    这是MC方法的原子单位
    This is the atomic unit of MC methods
    """
    state: State
    action: Action
    reward: float
    next_state: State
    done: bool
    
    def __repr__(self):
        return f"({self.state.id}, {self.action.id}, {self.reward:.1f})"


@dataclass
class Episode:
    """
    完整的回合（轨迹）
    Complete episode (trajectory)
    
    MC的关键：需要完整的回合来计算回报
    Key to MC: Need complete episodes to compute returns
    
    一个回合是从初始状态到终止状态的完整序列：
    An episode is a complete sequence from initial to terminal state:
    S₀, A₀, R₁, S₁, A₁, R₂, ..., Sₜ (terminal)
    
    数学表示：
    Mathematical representation:
    τ = (S₀, A₀, R₁, S₁, ..., Sₜ)
    
    为什么需要完整回合？
    Why need complete episodes?
    因为需要知道未来所有奖励才能计算准确的回报
    Because we need all future rewards to compute accurate returns
    """
    experiences: List[Experience] = field(default_factory=list)
    
    def add_experience(self, exp: Experience):
        """
        添加一步经验
        Add one-step experience
        """
        self.experiences.append(exp)
    
    def length(self) -> int:
        """
        回合长度
        Episode length
        """
        return len(self.experiences)
    
    def is_complete(self) -> bool:
        """
        检查回合是否结束
        Check if episode is complete
        """
        if not self.experiences:
            return False
        return self.experiences[-1].done
    
    def get_states(self) -> List[State]:
        """
        获取所有访问的状态
        Get all visited states
        """
        states = [exp.state for exp in self.experiences]
        if self.experiences and not self.experiences[-1].done:
            states.append(self.experiences[-1].next_state)
        return states
    
    def get_state_action_pairs(self) -> List[Tuple[State, Action]]:
        """
        获取所有(状态,动作)对
        Get all (state, action) pairs
        
        这对于Q函数估计很重要
        This is important for Q-function estimation
        """
        return [(exp.state, exp.action) for exp in self.experiences]
    
    def compute_returns(self, gamma: float = 1.0) -> List[float]:
        """
        计算每一步的回报
        Compute return for each step
        
        回报定义：
        Return definition:
        G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... = Σ_{k=0}^{T-t-1} γ^k R_{t+k+1}
        
        这是MC的核心计算！
        This is the core computation of MC!
        
        Args:
            gamma: 折扣因子
                  Discount factor
        
        Returns:
            每一步的回报列表
            List of returns for each step
        
        算法：反向计算更高效
        Algorithm: Backward computation is more efficient
        G_{T-1} = R_T
        G_{t} = R_{t+1} + γG_{t+1}
        """
        if not self.is_complete():
            logger.warning("计算未完成回合的回报")
        
        returns = []
        G = 0  # 初始化为0（终止状态之后）
        
        # 反向遍历计算回报
        for exp in reversed(self.experiences):
            G = exp.reward + gamma * G
            returns.append(G)
        
        # 反转得到正确顺序
        returns.reverse()
        
        return returns
    
    def first_visit_indices(self, state: State) -> List[int]:
        """
        找到状态的首次访问索引
        Find first-visit indices for a state
        
        First-visit MC只使用每个状态的第一次访问
        First-visit MC only uses first visit to each state
        """
        indices = []
        visited = set()
        
        for i, exp in enumerate(self.experiences):
            if exp.state.id not in visited:
                if exp.state.id == state.id:
                    indices.append(i)
                visited.add(exp.state.id)
        
        return indices
    
    def every_visit_indices(self, state: State) -> List[int]:
        """
        找到状态的所有访问索引
        Find all visit indices for a state
        
        Every-visit MC使用每个状态的所有访问
        Every-visit MC uses all visits to each state
        """
        indices = []
        for i, exp in enumerate(self.experiences):
            if exp.state.id == state.id:
                indices.append(i)
        return indices


class Return:
    """
    回报统计
    Return Statistics
    
    管理和更新回报的统计信息
    Manage and update return statistics
    
    这是实现增量MC更新的关键
    This is key to implementing incremental MC updates
    """
    
    def __init__(self):
        """
        初始化回报统计
        Initialize return statistics
        """
        self.returns = []  # 所有观察到的回报
        self.count = 0     # 观察次数
        self.mean = 0.0    # 平均回报
        self.variance = 0.0  # 方差
        self.std = 0.0     # 标准差
    
    def add_return(self, G: float):
        """
        添加一个回报观察
        Add a return observation
        
        使用增量更新公式：
        Using incremental update formula:
        μ_n = μ_{n-1} + (1/n)(G_n - μ_{n-1})
        
        这避免了存储所有回报
        This avoids storing all returns
        """
        self.returns.append(G)
        self.count += 1
        
        # 增量更新均值
        old_mean = self.mean
        self.mean += (G - self.mean) / self.count
        
        # 增量更新方差（Welford's algorithm）
        if self.count > 1:
            self.variance += (G - old_mean) * (G - self.mean)
            self.std = np.sqrt(self.variance / (self.count - 1))
    
    def get_statistics(self) -> Dict[str, float]:
        """
        获取统计信息
        Get statistics
        """
        return {
            'mean': self.mean,
            'std': self.std,
            'count': self.count,
            'min': min(self.returns) if self.returns else 0,
            'max': max(self.returns) if self.returns else 0
        }
    
    def confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """
        计算置信区间
        Compute confidence interval
        
        基于中心极限定理
        Based on Central Limit Theorem
        
        Args:
            confidence: 置信水平
                       Confidence level
        
        Returns:
            (下界, 上界)
            (lower bound, upper bound)
        """
        if self.count < 2:
            return (self.mean, self.mean)
        
        # t分布的临界值
        alpha = 1 - confidence
        df = self.count - 1
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        # 标准误差
        se = self.std / np.sqrt(self.count)
        
        # 置信区间
        margin = t_critical * se
        return (self.mean - margin, self.mean + margin)


# ================================================================================
# 第4.1.2节：大数定律与MC收敛
# Section 4.1.2: Law of Large Numbers and MC Convergence
# ================================================================================

class LawOfLargeNumbers:
    """
    大数定律演示
    Law of Large Numbers Demonstration
    
    这是MC方法的理论基础
    This is the theoretical foundation of MC methods
    
    大数定律说：
    Law of Large Numbers states:
    lim_{n→∞} (1/n)Σᵢ Xᵢ = E[X]  (几乎必然)
    
    对于MC：
    For MC:
    v_π(s) = E[G_t | S_t = s] ≈ (1/n)Σᵢ G_i(s)
    
    关键性质：
    Key properties:
    1. 无偏性：E[estimate] = true_value
       Unbiasedness: E[estimate] = true_value
    2. 一致性：随n增加，estimate → true_value
       Consistency: As n increases, estimate → true_value
    3. 收敛速度：O(1/√n)
       Convergence rate: O(1/√n)
    """
    
    @staticmethod
    def demonstrate_convergence(true_value: float = 5.0,
                              std_dev: float = 2.0,
                              n_samples: int = 1000):
        """
        演示大数定律收敛
        Demonstrate Law of Large Numbers convergence
        
        通过模拟展示样本均值如何收敛到真实期望
        Show how sample mean converges to true expectation through simulation
        
        Args:
            true_value: 真实期望值
                       True expected value
            std_dev: 标准差
                    Standard deviation
            n_samples: 样本数量
                      Number of samples
        """
        print("\n" + "="*60)
        print("大数定律演示")
        print("Law of Large Numbers Demonstration")
        print("="*60)
        
        # 生成样本
        np.random.seed(42)
        samples = np.random.normal(true_value, std_dev, n_samples)
        
        # 计算累积平均
        cumulative_means = np.cumsum(samples) / np.arange(1, n_samples + 1)
        
        # 计算标准误差
        standard_errors = std_dev / np.sqrt(np.arange(1, n_samples + 1))
        
        # 可视化
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 图1：收敛过程
        ax1 = axes[0, 0]
        ax1.plot(cumulative_means, 'b-', alpha=0.7, label='Sample Mean')
        ax1.axhline(y=true_value, color='r', linestyle='--', label=f'True Value = {true_value}')
        
        # 添加置信带
        confidence_band = 1.96 * standard_errors  # 95% confidence
        ax1.fill_between(range(n_samples), 
                         true_value - confidence_band,
                         true_value + confidence_band,
                         alpha=0.2, color='gray', label='95% CI')
        
        ax1.set_xlabel('Number of Samples')
        ax1.set_ylabel('Estimate')
        ax1.set_title('Convergence of Sample Mean')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 图2：误差减小
        ax2 = axes[0, 1]
        errors = np.abs(cumulative_means - true_value)
        ax2.loglog(range(1, n_samples + 1), errors, 'b-', alpha=0.7, label='|Error|')
        
        # 理论误差界限 O(1/√n)
        theoretical_bound = 3 * std_dev / np.sqrt(np.arange(1, n_samples + 1))
        ax2.loglog(range(1, n_samples + 1), theoretical_bound, 'r--', 
                  label='O(1/√n) bound')
        
        ax2.set_xlabel('Number of Samples (log)')
        ax2.set_ylabel('Absolute Error (log)')
        ax2.set_title('Error Decay Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 图3：方差减小
        ax3 = axes[1, 0]
        
        # 计算滑动窗口方差
        window_size = 50
        variances = []
        for i in range(window_size, n_samples):
            window_samples = samples[i-window_size:i]
            variances.append(np.var(window_samples))
        
        ax3.plot(range(window_size, n_samples), variances, 'g-', alpha=0.7)
        ax3.axhline(y=std_dev**2, color='r', linestyle='--', 
                   label=f'True Variance = {std_dev**2:.1f}')
        ax3.set_xlabel('Sample Index')
        ax3.set_ylabel('Variance Estimate')
        ax3.set_title('Variance Estimation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 图4：直方图
        ax4 = axes[1, 1]
        
        # 多个阶段的估计分布
        stages = [10, 100, 1000]
        colors = ['red', 'blue', 'green']
        
        for stage, color in zip(stages, colors):
            if stage <= n_samples:
                # 多次运行获得估计的分布
                estimates = []
                for _ in range(1000):
                    stage_samples = np.random.normal(true_value, std_dev, stage)
                    estimates.append(np.mean(stage_samples))
                
                ax4.hist(estimates, bins=30, alpha=0.3, color=color, 
                        label=f'n={stage}', density=True)
        
        ax4.axvline(x=true_value, color='r', linestyle='--', linewidth=2)
        ax4.set_xlabel('Estimate Value')
        ax4.set_ylabel('Density')
        ax4.set_title('Distribution of Estimates')
        ax4.legend()
        
        plt.suptitle('Law of Large Numbers in Monte Carlo Methods', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # 打印统计
        print(f"\n统计结果 (n={n_samples}):")
        print(f"  真实值: {true_value:.3f}")
        print(f"  最终估计: {cumulative_means[-1]:.3f}")
        print(f"  最终误差: {errors[-1]:.3f}")
        print(f"  理论标准误差: {standard_errors[-1]:.3f}")
        
        # 收敛速度分析
        convergence_point = None
        tolerance = 0.1
        for i, error in enumerate(errors):
            if error < tolerance:
                convergence_point = i + 1
                break
        
        if convergence_point:
            print(f"  收敛到±{tolerance}需要: {convergence_point}个样本")
        
        return fig
    
    @staticmethod
    def analyze_bias_variance_tradeoff():
        """
        分析偏差-方差权衡
        Analyze Bias-Variance Tradeoff
        
        MC估计的关键特性：
        Key properties of MC estimates:
        - 无偏：E[estimate] = true_value
          Unbiased: E[estimate] = true_value
        - 高方差：Var[estimate] = σ²/n
          High variance: Var[estimate] = σ²/n
        
        与TD方法对比（后续章节）：
        Contrast with TD methods (later chapters):
        - TD有偏但低方差
          TD is biased but low variance
        - MC无偏但高方差
          MC is unbiased but high variance
        """
        print("\n" + "="*60)
        print("MC估计的偏差-方差分析")
        print("Bias-Variance Analysis of MC Estimates")
        print("="*60)
        
        print("""
        📊 MC估计的数学性质
        Mathematical Properties of MC Estimates
        ================================
        
        1. 无偏性 Unbiasedness:
           E[Ĝ] = E[G] = v_π(s)
           
           证明：
           Proof:
           MC估计是回报的样本平均
           MC estimate is sample average of returns
           E[(1/n)Σ G_i] = (1/n)Σ E[G_i] = E[G] = v_π(s)
        
        2. 方差 Variance:
           Var[Ĝ] = Var[G]/n = σ²/n
           
           含义：
           Implication:
           - 方差随样本数线性减小
             Variance decreases linearly with samples
           - 标准误差按√n减小
             Standard error decreases as √n
        
        3. 均方误差 Mean Squared Error:
           MSE = Bias² + Variance = 0 + σ²/n = σ²/n
           
           因为MC无偏，MSE完全由方差决定
           Since MC is unbiased, MSE is entirely variance
        
        4. 收敛速度 Convergence Rate:
           P(|Ĝ - v_π(s)| > ε) ≤ 2exp(-2nε²/B²)
           
           这是Hoeffding不等式
           This is Hoeffding's inequality
           
           实践含义：
           Practical implication:
           - 误差以指数速度减小
             Error decreases exponentially
           - 但常数可能很大
             But constant can be large
        
        5. 中心极限定理 Central Limit Theorem:
           √n(Ĝ - v_π(s)) → N(0, σ²)
           
           大样本下，估计近似正态分布
           For large samples, estimate is approximately normal
           
           应用：
           Application:
           - 可以构造置信区间
             Can construct confidence intervals
           - 可以做假设检验
             Can do hypothesis testing
        """)


# ================================================================================
# 第4.1.3节：MC统计与收敛分析
# Section 4.1.3: MC Statistics and Convergence Analysis
# ================================================================================

class MCStatistics:
    """
    MC方法的统计分析
    Statistical Analysis for MC Methods
    
    提供MC估计的各种统计工具
    Provides various statistical tools for MC estimates
    """
    
    def __init__(self):
        """
        初始化统计收集器
        Initialize statistics collector
        """
        # 存储每个状态的回报
        self.state_returns: Dict[str, Return] = defaultdict(Return)
        
        # 存储每个(状态,动作)对的回报
        self.state_action_returns: Dict[Tuple[str, str], Return] = defaultdict(Return)
        
        # 记录访问次数
        self.state_visits: Dict[str, int] = defaultdict(int)
        self.state_action_visits: Dict[Tuple[str, str], int] = defaultdict(int)
        
        # 收敛历史
        self.convergence_history = []
        
        logger.info("初始化MC统计收集器")
    
    def update_state_value(self, state: State, G: float):
        """
        更新状态价值统计
        Update state value statistics
        
        Args:
            state: 状态
            G: 观察到的回报
        """
        state_id = state.id
        self.state_returns[state_id].add_return(G)
        self.state_visits[state_id] += 1
    
    def update_action_value(self, state: State, action: Action, G: float):
        """
        更新动作价值统计
        Update action value statistics
        
        Args:
            state: 状态
            action: 动作
            G: 观察到的回报
        """
        sa_pair = (state.id, action.id)
        self.state_action_returns[sa_pair].add_return(G)
        self.state_action_visits[sa_pair] += 1
    
    def get_state_value_estimate(self, state: State) -> float:
        """
        获取状态价值估计
        Get state value estimate
        
        Returns:
            估计的状态价值
            Estimated state value
        """
        state_id = state.id
        if state_id in self.state_returns:
            return self.state_returns[state_id].mean
        return 0.0
    
    def get_action_value_estimate(self, state: State, action: Action) -> float:
        """
        获取动作价值估计
        Get action value estimate
        
        Returns:
            估计的动作价值
            Estimated action value
        """
        sa_pair = (state.id, action.id)
        if sa_pair in self.state_action_returns:
            return self.state_action_returns[sa_pair].mean
        return 0.0
    
    def get_confidence_intervals(self, confidence: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """
        获取所有状态的置信区间
        Get confidence intervals for all states
        
        Args:
            confidence: 置信水平
                       Confidence level
        
        Returns:
            状态ID到置信区间的映射
            Mapping from state ID to confidence interval
        """
        intervals = {}
        for state_id, returns in self.state_returns.items():
            if returns.count >= 2:
                intervals[state_id] = returns.confidence_interval(confidence)
            else:
                intervals[state_id] = (returns.mean, returns.mean)
        return intervals
    
    def analyze_convergence(self, true_values: Optional[Dict[str, float]] = None):
        """
        分析收敛性
        Analyze convergence
        
        Args:
            true_values: 真实价值（如果已知）
                        True values (if known)
        """
        print("\n" + "="*60)
        print("MC收敛分析")
        print("MC Convergence Analysis")
        print("="*60)
        
        # 统计信息
        print(f"\n访问统计:")
        print(f"  状态数: {len(self.state_visits)}")
        print(f"  总访问次数: {sum(self.state_visits.values())}")
        
        # 访问频率分析
        if self.state_visits:
            visits = list(self.state_visits.values())
            print(f"  平均访问: {np.mean(visits):.1f}")
            print(f"  最少访问: {min(visits)}")
            print(f"  最多访问: {max(visits)}")
        
        # 估计精度
        print(f"\n估计精度:")
        for state_id, returns in self.state_returns.items():
            if returns.count > 0:
                ci = returns.confidence_interval(0.95)
                print(f"  {state_id}: {returns.mean:.3f} ± {(ci[1]-ci[0])/2:.3f} "
                      f"(n={returns.count})")
        
        # 如果有真实值，计算误差
        if true_values:
            print(f"\n误差分析:")
            errors = []
            for state_id, true_value in true_values.items():
                if state_id in self.state_returns:
                    estimate = self.state_returns[state_id].mean
                    error = abs(estimate - true_value)
                    errors.append(error)
                    print(f"  {state_id}: 误差={error:.3f}")
            
            if errors:
                print(f"\n  平均误差: {np.mean(errors):.3f}")
                print(f"  最大误差: {max(errors):.3f}")
                print(f"  RMSE: {np.sqrt(np.mean(np.square(errors))):.3f}")
    
    def plot_convergence(self, state_ids: Optional[List[str]] = None):
        """
        绘制收敛曲线
        Plot convergence curves
        
        Args:
            state_ids: 要绘制的状态ID列表
                      List of state IDs to plot
        """
        if not self.convergence_history:
            logger.warning("没有收敛历史可绘制")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 选择要绘制的状态
        if state_ids is None:
            state_ids = list(self.state_returns.keys())[:4]
        
        # 为每个状态绘制收敛曲线
        for idx, state_id in enumerate(state_ids):
            if idx >= 4:
                break
            
            ax = axes[idx // 2, idx % 2]
            
            if state_id in self.state_returns:
                returns_obj = self.state_returns[state_id]
                
                # 计算累积平均
                if returns_obj.returns:
                    cumulative_means = np.cumsum(returns_obj.returns) / np.arange(1, len(returns_obj.returns) + 1)
                    
                    ax.plot(cumulative_means, 'b-', alpha=0.7)
                    ax.axhline(y=returns_obj.mean, color='r', linestyle='--', 
                              label=f'Final: {returns_obj.mean:.2f}')
                    
                    # 添加置信带
                    if len(returns_obj.returns) > 10:
                        window = 10
                        stds = []
                        for i in range(window, len(returns_obj.returns)):
                            window_returns = returns_obj.returns[i-window:i]
                            stds.append(np.std(window_returns))
                        
                        if stds:
                            upper = cumulative_means[window:] + np.array(stds)
                            lower = cumulative_means[window:] - np.array(stds)
                            ax.fill_between(range(window, len(cumulative_means)), 
                                          lower, upper, alpha=0.2, color='blue')
                    
                    ax.set_xlabel('Visit Number')
                    ax.set_ylabel('Value Estimate')
                    ax.set_title(f'State: {state_id}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
        
        plt.suptitle('MC Value Convergence', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig


# ================================================================================
# 第4.1.4节：MC基础理论展示
# Section 4.1.4: MC Foundations Demonstration
# ================================================================================

class MCFoundations:
    """
    MC基础理论综合
    MC Foundations Integration
    
    整合所有MC基础概念
    Integrate all MC foundation concepts
    """
    
    @staticmethod
    def explain_mc_principles():
        """
        解释MC原理
        Explain MC Principles
        
        教学函数，帮助理解MC的核心思想
        Teaching function to help understand core ideas of MC
        """
        print("\n" + "="*80)
        print("蒙特卡洛方法原理")
        print("Monte Carlo Method Principles")
        print("="*80)
        
        print("""
        📚 1. 什么是蒙特卡洛方法？
        What is Monte Carlo Method?
        ================================
        
        核心思想：通过随机采样来近似计算
        Core idea: Approximate computation through random sampling
        
        在RL中的应用：
        Application in RL:
        v_π(s) = E_π[G_t | S_t = s] ≈ (1/n) Σᵢ G_i(s)
        
        其中G_i(s)是从状态s开始的第i个回合的回报
        Where G_i(s) is the return of i-th episode starting from state s
        
        📚 2. MC vs DP
        ================================
        
        | 方面 Aspect | DP | MC |
        |------------|----|----|
        | 模型 Model | 需要 Required | 不需要 Not required |
        | 更新 Update | 全宽度 Full-width | 采样 Sampling |
        | 偏差 Bias | 无 None | 无 None |
        | 方差 Variance | 无 None | 高 High |
        | 适用 Application | 小空间 Small space | 大空间 Large space |
        | 任务 Tasks | 任意 Any | 回合式 Episodic |
        
        📚 3. First-Visit vs Every-Visit
        ================================
        
        First-Visit MC:
        - 只使用每个状态的首次访问
          Only use first visit to each state
        - 理论性质更好（独立样本）
          Better theoretical properties (independent samples)
        - 收敛到v_π(s)
          Converges to v_π(s)
        
        Every-Visit MC:
        - 使用每个状态的所有访问
          Use all visits to each state
        - 更多数据，可能收敛更快
          More data, may converge faster
        - 也收敛到v_π(s)（但样本相关）
          Also converges to v_π(s) (but samples correlated)
        
        📚 4. 增量实现
        Incremental Implementation
        ================================
        
        避免存储所有回报：
        Avoid storing all returns:
        
        V(s) ← V(s) + α[G - V(s)]
        
        其中：
        Where:
        - α = 1/n(s) 保证收敛到样本均值
          α = 1/n(s) ensures convergence to sample mean
        - α = constant 允许跟踪非平稳问题
          α = constant allows tracking non-stationary problems
        
        这个更新规则贯穿整个RL！
        This update rule permeates all of RL!
        
        📚 5. MC的优势场景
        When MC Shines
        ================================
        
        1. 只关心某些状态的价值
           Only care about value of certain states
           - MC可以只估计这些状态
             MC can estimate only these states
           - DP必须计算所有状态
             DP must compute all states
        
        2. 非马尔可夫环境
           Non-Markovian environments
           - MC不依赖马尔可夫性
             MC doesn't rely on Markov property
           - 只要能生成回合即可
             Just need to generate episodes
        
        3. 模型未知或复杂
           Model unknown or complex
           - MC直接从经验学习
             MC learns directly from experience
           - 不需要转移概率
             No need for transition probabilities
        """)
    
    @staticmethod
    def demonstrate_mc_vs_dp_comparison():
        """
        演示MC与DP的对比
        Demonstrate MC vs DP Comparison
        
        通过简单例子展示两种方法的区别
        Show difference between two methods through simple example
        """
        print("\n" + "="*60)
        print("MC vs DP 对比演示")
        print("MC vs DP Comparison Demo")
        print("="*60)
        
        # 创建一个简单的马尔可夫链
        # Create a simple Markov chain
        states = ['A', 'B', 'C', 'Terminal']
        
        # 转移概率
        P = {
            'A': {'B': 0.5, 'C': 0.5},
            'B': {'Terminal': 1.0},
            'C': {'Terminal': 1.0}
        }
        
        # 奖励
        R = {
            'A': {'B': 0, 'C': 0},
            'B': {'Terminal': 1},
            'C': {'Terminal': 10}
        }
        
        print("\n问题设置:")
        print("  状态: A → {B, C} → Terminal")
        print("  转移: P(B|A)=0.5, P(C|A)=0.5")
        print("  奖励: R(B→T)=1, R(C→T)=10")
        
        # DP解法（精确）
        print("\n1. DP解法（精确）:")
        v_dp = {
            'Terminal': 0,
            'B': 1,
            'C': 10,
            'A': 0.5 * 1 + 0.5 * 10  # = 5.5
        }
        print(f"  V(A) = 0.5 × V(B) + 0.5 × V(C)")
        print(f"       = 0.5 × 1 + 0.5 × 10 = {v_dp['A']}")
        
        # MC解法（模拟）
        print("\n2. MC解法（采样）:")
        np.random.seed(42)
        
        returns_A = []
        n_episodes = 1000
        
        for _ in range(n_episodes):
            # 模拟一个回合
            if np.random.random() < 0.5:
                # A → B → Terminal
                G = 1
            else:
                # A → C → Terminal
                G = 10
            returns_A.append(G)
        
        # 计算MC估计
        mc_estimates = []
        for i in range(1, len(returns_A) + 1):
            mc_estimates.append(np.mean(returns_A[:i]))
        
        print(f"  运行{n_episodes}个回合")
        print(f"  MC估计: {mc_estimates[-1]:.3f}")
        print(f"  真实值: {v_dp['A']}")
        print(f"  误差: {abs(mc_estimates[-1] - v_dp['A']):.3f}")
        
        # 可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 左图：收敛过程
        ax1.plot(mc_estimates, 'b-', alpha=0.7, label='MC Estimate')
        ax1.axhline(y=v_dp['A'], color='r', linestyle='--', label=f"DP Solution = {v_dp['A']}")
        
        # 添加标准误差带
        n_points = len(mc_estimates)
        std_errors = []
        for i in range(1, n_points + 1):
            se = np.std(returns_A[:i]) / np.sqrt(i) if i > 1 else 0
            std_errors.append(se)
        
        upper = np.array(mc_estimates) + 1.96 * np.array(std_errors)
        lower = np.array(mc_estimates) - 1.96 * np.array(std_errors)
        ax1.fill_between(range(n_points), lower, upper, alpha=0.2, color='blue', label='95% CI')
        
        ax1.set_xlabel('Number of Episodes')
        ax1.set_ylabel('Value Estimate')
        ax1.set_title('MC Convergence to DP Solution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右图：回报分布
        ax2.hist(returns_A, bins=20, density=True, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(x=v_dp['A'], color='r', linestyle='--', linewidth=2, label='Expected Value')
        ax2.set_xlabel('Return')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Returns')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Monte Carlo vs Dynamic Programming', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig


# ================================================================================
# 主函数：演示MC基础
# Main Function: Demonstrate MC Foundations
# ================================================================================

def main():
    """
    运行MC基础演示
    Run MC Foundations Demo
    """
    print("\n" + "="*80)
    print("第4.1节：蒙特卡洛方法基础")
    print("Section 4.1: Monte Carlo Foundations")
    print("="*80)
    
    # 1. 解释MC原理
    MCFoundations.explain_mc_principles()
    
    # 2. 演示大数定律
    print("\n演示大数定律...")
    fig1 = LawOfLargeNumbers.demonstrate_convergence()
    
    # 3. 偏差-方差分析
    LawOfLargeNumbers.analyze_bias_variance_tradeoff()
    
    # 4. MC vs DP对比
    print("\n演示MC vs DP...")
    fig2 = MCFoundations.demonstrate_mc_vs_dp_comparison()
    
    # 5. 测试Episode类
    print("\n" + "="*60)
    print("测试Episode类")
    print("Testing Episode Class")
    print("="*60)
    
    # 创建模拟回合
    from src.ch03_finite_mdp.mdp_framework import State, Action
    
    episode = Episode()
    
    # 添加一些经验
    states = [State(f"s{i}", {}) for i in range(4)]
    actions = [Action(f"a{i}", f"Action {i}") for i in range(2)]
    
    # 模拟轨迹: s0 -> s1 -> s2 -> s3(terminal)
    episode.add_experience(Experience(states[0], actions[0], 1.0, states[1], False))
    episode.add_experience(Experience(states[1], actions[1], 2.0, states[2], False))
    episode.add_experience(Experience(states[2], actions[0], 3.0, states[3], True))
    
    print(f"回合长度: {episode.length()}")
    print(f"回合完成: {episode.is_complete()}")
    
    # 计算回报
    returns = episode.compute_returns(gamma=0.9)
    print(f"\n回报 (γ=0.9):")
    for i, G in enumerate(returns):
        print(f"  G_{i} = {G:.3f}")
    
    # 验证回报计算
    print(f"\n验证:")
    print(f"  G_0 = 1 + 0.9×2 + 0.9²×3 = {1 + 0.9*2 + 0.81*3:.3f}")
    print(f"  计算的G_0 = {returns[0]:.3f}")
    
    # 6. 测试统计收集
    print("\n" + "="*60)
    print("测试MC统计")
    print("Testing MC Statistics")
    print("="*60)
    
    stats = MCStatistics()
    
    # 模拟多个回合的回报
    np.random.seed(42)
    for _ in range(100):
        # 为状态s0添加随机回报
        G = np.random.normal(5.0, 2.0)
        stats.update_state_value(states[0], G)
    
    # 获取估计
    estimate = stats.get_state_value_estimate(states[0])
    ci = stats.state_returns[states[0].id].confidence_interval(0.95)
    
    print(f"状态 {states[0].id}:")
    print(f"  估计值: {estimate:.3f}")
    print(f"  95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"  样本数: {stats.state_visits[states[0].id]}")
    
    # 分析收敛
    stats.analyze_convergence({'s0': 5.0})  # 真实值是5.0
    
    print("\n" + "="*80)
    print("MC基础演示完成！")
    print("MC Foundations Demo Complete!")
    print("\n关键要点 Key Takeaways:")
    print("1. MC通过采样估计期望值")
    print("   MC estimates expected value through sampling")
    print("2. 大数定律保证收敛到真实值")
    print("   Law of Large Numbers guarantees convergence to true value")
    print("3. MC估计无偏但高方差")
    print("   MC estimates are unbiased but high variance")
    print("4. 收敛速度是O(1/√n)")
    print("   Convergence rate is O(1/√n)")
    print("5. MC不需要模型，只需要经验")
    print("   MC doesn't need model, only needs experience")
    print("="*80)
    
    plt.show()


if __name__ == "__main__":
    main()