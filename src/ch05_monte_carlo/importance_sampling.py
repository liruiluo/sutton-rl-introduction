"""
================================================================================
第4.4节：重要性采样 - Off-Policy学习的数学基础
Section 4.4: Importance Sampling - Mathematical Foundation of Off-Policy Learning
================================================================================

重要性采样是off-policy学习的核心技术。
Importance sampling is the core technique for off-policy learning.

核心问题：如何用一个分布的样本估计另一个分布的期望？
Core problem: How to estimate expectation under one distribution using samples from another?

数学原理：
Mathematical principle:
E_π[X] = E_b[ρ × X]
其中 ρ = π(·)/b(·) 是重要性采样比率
where ρ = π(·)/b(·) is the importance sampling ratio

两种主要变体：
Two main variants:
1. 普通重要性采样（Ordinary IS）
   - 无偏但高方差
     Unbiased but high variance
   - V^π(s) = (1/n)Σᵢ ρᵢGᵢ

2. 加权重要性采样（Weighted IS）
   - 有偏但低方差
     Biased but lower variance
   - V^π(s) = Σᵢ(ρᵢGᵢ)/Σᵢρᵢ

权衡：
Trade-offs:
- 偏差 vs 方差
  Bias vs Variance
- 收敛速度 vs 稳定性
  Convergence speed vs Stability
- 理论性质 vs 实践性能
  Theoretical properties vs Practical performance

这是通向现代off-policy方法（如Q-learning）的桥梁！
This is the bridge to modern off-policy methods like Q-learning!
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from scipy import stats
import time

# 导入基础组件
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.ch03_finite_mdp.mdp_framework import State, Action, MDPEnvironment
from src.ch03_finite_mdp.policies_and_values import (
    Policy, StateValueFunction, ActionValueFunction,
    StochasticPolicy, DeterministicPolicy
)
from ch04_monte_carlo.mc_foundations import (
    Episode, Experience, Return, MCStatistics
)
from ch04_monte_carlo.mc_control import EpsilonGreedyPolicy

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第4.4.1节：重要性采样基础
# Section 4.4.1: Importance Sampling Fundamentals
# ================================================================================

class ImportanceSamplingTheory:
    """
    重要性采样理论
    Importance Sampling Theory
    
    展示重要性采样的数学原理和性质
    Demonstrate mathematical principles and properties of importance sampling
    
    核心思想：期望的变换
    Core idea: Transformation of expectation
    
    问题设定：
    Problem setup:
    - 想要：E_π[f(X)] 在目标分布π下的期望
      Want: E_π[f(X)] expectation under target distribution π
    - 拥有：来自行为分布b的样本
      Have: Samples from behavior distribution b
    
    解决方案：
    Solution:
    E_π[f(X)] = ∫ f(x)π(x)dx
             = ∫ f(x)[π(x)/b(x)]b(x)dx
             = E_b[f(X) × π(X)/b(X)]
             = E_b[f(X) × ρ(X)]
    
    其中 ρ(X) = π(X)/b(X) 是重要性权重
    where ρ(X) = π(X)/b(X) is importance weight
    
    关键要求：
    Key requirements:
    1. 覆盖条件：b(x) > 0 whenever π(x) > 0
       Coverage: b(x) > 0 whenever π(x) > 0
    2. 已知比率：需要知道π(x)/b(x)
       Known ratio: Need to know π(x)/b(x)
    
    在RL中的应用：
    Application in RL:
    - X是轨迹τ
      X is trajectory τ
    - f(X)是回报G(τ)
      f(X) is return G(τ)
    - π是目标策略
      π is target policy
    - b是行为策略
      b is behavior policy
    """
    
    @staticmethod
    def demonstrate_basic_principle():
        """
        演示重要性采样基本原理
        Demonstrate basic principle of importance sampling
        
        用简单例子展示如何用一个分布估计另一个分布
        Show how to estimate one distribution using another with simple example
        """
        print("\n" + "="*80)
        print("重要性采样基本原理演示")
        print("Importance Sampling Basic Principle Demo")
        print("="*80)
        
        # 简单例子：估计正态分布的期望，用另一个正态分布采样
        # Simple example: Estimate mean of normal, sample from another normal
        
        # 目标分布：N(5, 1)
        # Target distribution: N(5, 1)
        target_mean, target_std = 5.0, 1.0
        
        # 行为分布：N(3, 2)
        # Behavior distribution: N(3, 2)
        behavior_mean, behavior_std = 3.0, 2.0
        
        print(f"\n目标分布 Target: N({target_mean}, {target_std}²)")
        print(f"行为分布 Behavior: N({behavior_mean}, {behavior_std}²)")
        print(f"真实期望 True expectation: {target_mean}")
        
        # 从行为分布采样
        # Sample from behavior distribution
        n_samples = 10000
        np.random.seed(42)
        samples = np.random.normal(behavior_mean, behavior_std, n_samples)
        
        # 方法1：直接平均（错误！）
        # Method 1: Direct average (wrong!)
        naive_estimate = np.mean(samples)
        print(f"\n直接平均 Direct average: {naive_estimate:.3f}")
        print(f"  错误！这是行为分布的期望")
        print(f"  Wrong! This is behavior distribution's mean")
        
        # 方法2：重要性采样（正确！）
        # Method 2: Importance sampling (correct!)
        
        # 计算重要性权重
        # Compute importance weights
        target_pdf = stats.norm.pdf(samples, target_mean, target_std)
        behavior_pdf = stats.norm.pdf(samples, behavior_mean, behavior_std)
        weights = target_pdf / behavior_pdf
        
        # 普通IS估计
        # Ordinary IS estimate
        is_estimate = np.mean(weights * samples)
        print(f"\n普通IS估计 Ordinary IS: {is_estimate:.3f}")
        
        # 加权IS估计
        # Weighted IS estimate
        weighted_is_estimate = np.sum(weights * samples) / np.sum(weights)
        print(f"加权IS估计 Weighted IS: {weighted_is_estimate:.3f}")
        
        # 分析权重
        # Analyze weights
        print(f"\n权重统计 Weight statistics:")
        print(f"  均值 Mean: {np.mean(weights):.3f}")
        print(f"  标准差 Std: {np.std(weights):.3f}")
        print(f"  最小 Min: {np.min(weights):.3f}")
        print(f"  最大 Max: {np.max(weights):.3f}")
        
        # 有效样本大小
        # Effective sample size
        ess = np.sum(weights)**2 / np.sum(weights**2)
        print(f"  有效样本大小 ESS: {ess:.0f} / {n_samples}")
        print(f"  效率 Efficiency: {ess/n_samples:.2%}")
        
        # 可视化
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 图1：分布比较
        # Plot 1: Distribution comparison
        ax1 = axes[0, 0]
        x = np.linspace(-2, 10, 1000)
        target_y = stats.norm.pdf(x, target_mean, target_std)
        behavior_y = stats.norm.pdf(x, behavior_mean, behavior_std)
        
        ax1.plot(x, target_y, 'r-', linewidth=2, label='Target π')
        ax1.plot(x, behavior_y, 'b-', linewidth=2, label='Behavior b')
        ax1.fill_between(x, target_y, alpha=0.3, color='red')
        ax1.fill_between(x, behavior_y, alpha=0.3, color='blue')
        ax1.axvline(x=target_mean, color='red', linestyle='--', alpha=0.5)
        ax1.axvline(x=behavior_mean, color='blue', linestyle='--', alpha=0.5)
        ax1.set_xlabel('x')
        ax1.set_ylabel('Probability Density')
        ax1.set_title('Target vs Behavior Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 图2：重要性权重
        # Plot 2: Importance weights
        ax2 = axes[0, 1]
        weight_func = lambda x: stats.norm.pdf(x, target_mean, target_std) / stats.norm.pdf(x, behavior_mean, behavior_std)
        weights_x = [weight_func(xi) for xi in x]
        ax2.plot(x, weights_x, 'g-', linewidth=2)
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='ρ=1')
        ax2.set_xlabel('x')
        ax2.set_ylabel('Importance Weight ρ(x)')
        ax2.set_title('Importance Weights π(x)/b(x)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 5])
        
        # 图3：估计收敛
        # Plot 3: Estimation convergence
        ax3 = axes[1, 0]
        
        # 计算累积估计
        # Compute cumulative estimates
        cumulative_ordinary = []
        cumulative_weighted = []
        cumulative_naive = []
        
        for i in range(100, n_samples, 100):
            batch_samples = samples[:i]
            batch_weights = target_pdf[:i] / behavior_pdf[:i]
            
            ordinary = np.mean(batch_weights * batch_samples)
            weighted = np.sum(batch_weights * batch_samples) / np.sum(batch_weights)
            naive = np.mean(batch_samples)
            
            cumulative_ordinary.append(ordinary)
            cumulative_weighted.append(weighted)
            cumulative_naive.append(naive)
        
        x_axis = range(100, n_samples, 100)
        ax3.plot(x_axis, cumulative_ordinary, 'g-', alpha=0.7, label='Ordinary IS')
        ax3.plot(x_axis, cumulative_weighted, 'b-', alpha=0.7, label='Weighted IS')
        ax3.plot(x_axis, cumulative_naive, 'gray', alpha=0.5, label='Naive (wrong)')
        ax3.axhline(y=target_mean, color='red', linestyle='--', linewidth=2, label='True value')
        ax3.set_xlabel('Number of Samples')
        ax3.set_ylabel('Estimate')
        ax3.set_title('Convergence of Estimates')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 图4：权重分布
        # Plot 4: Weight distribution
        ax4 = axes[1, 1]
        ax4.hist(weights[:1000], bins=50, alpha=0.7, color='purple', density=True)
        ax4.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='ρ=1')
        ax4.set_xlabel('Weight Value')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Importance Weights')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Importance Sampling Principle Demonstration', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        print("\n" + "="*60)
        print("关键洞察 Key Insights:")
        print("="*60)
        print("""
        1. 重要性权重修正分布差异
           Importance weights correct distribution mismatch
           
        2. 权重方差影响估计质量
           Weight variance affects estimate quality
           
        3. 分布差异越大，效率越低
           Larger distribution difference, lower efficiency
           
        4. 加权IS用偏差换取更低方差
           Weighted IS trades bias for lower variance
        """)
        
        return fig


# ================================================================================
# 第4.4.2节：重要性采样基类
# Section 4.4.2: Importance Sampling Base Class
# ================================================================================

class ImportanceSampling(ABC):
    """
    重要性采样基类
    Importance Sampling Base Class
    
    定义IS方法的共同接口
    Define common interface for IS methods
    
    设计考虑：
    Design considerations:
    1. 支持普通和加权IS
       Support ordinary and weighted IS
    2. 增量和批量更新
       Incremental and batch updates
    3. 诊断和分析工具
       Diagnostic and analysis tools
    4. 方差减少技术
       Variance reduction techniques
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 target_policy: Policy,
                 behavior_policy: Policy,
                 gamma: float = 1.0):
        """
        初始化重要性采样
        Initialize importance sampling
        
        Args:
            env: 环境
            target_policy: 目标策略π
            behavior_policy: 行为策略b
            gamma: 折扣因子
        """
        self.env = env
        self.target_policy = target_policy
        self.behavior_policy = behavior_policy
        self.gamma = gamma
        
        # 价值函数估计
        # Value function estimates
        self.V = StateValueFunction(env.state_space, initial_value=0.0)
        self.Q = ActionValueFunction(env.state_space, env.action_space, initial_value=0.0)
        
        # 统计
        # Statistics
        self.statistics = MCStatistics()
        
        # IS比率记录
        # IS ratio records
        self.is_ratios: List[float] = []
        self.trajectory_ratios: List[List[float]] = []
        
        # 访问计数
        # Visit counts
        self.state_visits = defaultdict(int)
        self.sa_visits = defaultdict(int)
        
        logger.info("初始化重要性采样")
    
    @abstractmethod
    def update_value(self, episode: Episode):
        """
        更新价值估计（子类实现）
        Update value estimate (implemented by subclasses)
        """
        pass
    
    def compute_trajectory_ratio(self, episode: Episode) -> float:
        """
        计算整条轨迹的重要性比率
        Compute importance ratio for entire trajectory
        
        ρ(τ) = ∏_t [π(aₜ|sₜ)/b(aₜ|sₜ)]
        
        这是最基本的IS计算
        This is the most basic IS computation
        """
        ratio = 1.0
        
        for exp in episode.experiences:
            # 获取动作概率
            # Get action probabilities
            target_probs = self.target_policy.get_action_probabilities(
                exp.state
            )
            behavior_probs = self.behavior_policy.get_action_probabilities(
                exp.state
            )
            
            target_prob = target_probs.get(exp.action, 0.0)
            behavior_prob = behavior_probs.get(exp.action, 1e-10)  # 避免除零
            
            ratio *= target_prob / behavior_prob
            
            # 如果比率为0，整条轨迹权重为0
            # If ratio is 0, entire trajectory has weight 0
            if ratio == 0:
                break
        
        return ratio
    
    def compute_per_step_ratios(self, episode: Episode) -> List[float]:
        """
        计算每步的累积重要性比率
        Compute cumulative importance ratios per step
        
        ρₜ:T = ∏_{k=t}^T [π(aₖ|sₖ)/b(aₖ|sₖ)]
        
        用于per-decision IS
        Used for per-decision IS
        """
        ratios = []
        cumulative_ratio = 1.0
        
        # 反向计算（从后向前累积）
        # Backward computation (accumulate from back to front)
        for exp in reversed(episode.experiences):
            target_probs = self.target_policy.get_action_probabilities(
                exp.state
            )
            behavior_probs = self.behavior_policy.get_action_probabilities(
                exp.state
            )
            
            target_prob = target_probs.get(exp.action, 0.0)
            behavior_prob = behavior_probs.get(exp.action, 1e-10)
            
            cumulative_ratio *= target_prob / behavior_prob
            ratios.append(cumulative_ratio)
        
        ratios.reverse()
        return ratios
    
    def diagnose_coverage(self):
        """
        诊断覆盖性条件
        Diagnose coverage condition
        
        检查b是否充分覆盖π
        Check if b adequately covers π
        """
        print("\n" + "="*60)
        print("覆盖性诊断")
        print("Coverage Diagnosis")
        print("="*60)
        
        coverage_violations = 0
        total_checks = 0
        
        for state in self.env.state_space:
            if state.is_terminal:
                continue
            
            target_probs = self.target_policy.get_action_probabilities(
                state
            )
            behavior_probs = self.behavior_policy.get_action_probabilities(
                state
            )
            
            for action in self.env.action_space:
                total_checks += 1
                target_p = target_probs.get(action, 0.0)
                behavior_p = behavior_probs.get(action, 0.0)
                
                if target_p > 0 and behavior_p == 0:
                    coverage_violations += 1
                    print(f"  违反: π({action.id}|{state.id}) = {target_p:.3f}, "
                          f"b({action.id}|{state.id}) = 0")
        
        if coverage_violations > 0:
            print(f"\n⚠️ 发现{coverage_violations}个覆盖性违反！")
            print(f"   Found {coverage_violations} coverage violations!")
            print("   IS估计可能有偏或无限方差")
            print("   IS estimates may be biased or have infinite variance")
        else:
            print("✓ 覆盖性条件满足")
            print("  Coverage condition satisfied")
        
        print(f"\n检查的(s,a)对: {total_checks}")
        print(f"违反比例: {coverage_violations/total_checks:.2%}")
    
    def analyze_variance(self):
        """
        分析IS估计的方差
        Analyze variance of IS estimates
        
        展示IS的主要问题：高方差
        Show main problem of IS: high variance
        """
        if not self.is_ratios:
            print("没有IS比率数据")
            return
        
        ratios = np.array(self.is_ratios)
        
        print("\n" + "="*60)
        print("重要性采样方差分析")
        print("Importance Sampling Variance Analysis")
        print("="*60)
        
        print(f"\nIS比率统计:")
        print(f"  样本数: {len(ratios)}")
        print(f"  均值: {np.mean(ratios):.3f}")
        print(f"  标准差: {np.std(ratios):.3f}")
        print(f"  变异系数(CV): {np.std(ratios)/np.mean(ratios):.3f}")
        
        # 分位数
        # Quantiles
        quantiles = [0, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
        quantile_values = np.quantile(ratios, quantiles)
        
        print(f"\n分位数:")
        for q, v in zip(quantiles, quantile_values):
            print(f"  {q*100:3.0f}%: {v:.3f}")
        
        # 极端值分析
        # Extreme value analysis
        extreme_threshold = np.mean(ratios) + 3 * np.std(ratios)
        extreme_count = np.sum(ratios > extreme_threshold)
        
        print(f"\n极端值 (>均值+3σ): {extreme_count} ({extreme_count/len(ratios):.2%})")
        
        # 有效样本大小
        # Effective sample size
        if len(ratios) > 0:
            sum_w = np.sum(ratios)
            sum_w2 = np.sum(ratios ** 2)
            if sum_w2 > 0:
                ess = (sum_w ** 2) / sum_w2
                print(f"\n有效样本大小(ESS): {ess:.1f} / {len(ratios)}")
                print(f"效率: {ess/len(ratios):.2%}")
        
        print("\n" + "="*40)
        print("方差问题诊断:")
        print("Variance Problem Diagnosis:")
        print("="*40)
        
        cv = np.std(ratios) / np.mean(ratios) if np.mean(ratios) > 0 else float('inf')
        
        if cv > 1:
            print("⚠️ 高变异系数(CV>1): 估计不稳定")
            print("   High CV: Unstable estimates")
        
        if extreme_count / len(ratios) > 0.01:
            print("⚠️ 过多极端值: 少数样本主导估计")
            print("   Too many extremes: Few samples dominate")
        
        efficiency = ess / len(ratios) if len(ratios) > 0 else 0
        if efficiency < 0.1:
            print("⚠️ 低效率(<10%): 大部分样本被浪费")
            print("   Low efficiency: Most samples wasted")


# ================================================================================
# 第4.4.3节：普通重要性采样
# Section 4.4.3: Ordinary Importance Sampling
# ================================================================================

class OrdinaryImportanceSampling(ImportanceSampling):
    """
    普通重要性采样
    Ordinary Importance Sampling
    
    最直接的IS实现
    Most straightforward IS implementation
    
    估计器：
    Estimator:
    V^π(s) = (1/n(s)) Σᵢ ρᵢGᵢ(s)
    
    其中：
    where:
    - n(s)是状态s的访问次数
      n(s) is number of visits to state s
    - ρᵢ是第i次访问的重要性比率
      ρᵢ is importance ratio for i-th visit
    - Gᵢ(s)是从s开始的回报
      Gᵢ(s) is return starting from s
    
    性质：
    Properties:
    - 无偏：E_b[ρG] = E_π[G] = v_π(s)
      Unbiased: E_b[ρG] = E_π[G] = v_π(s)
    - 高方差：Var[ρG]可能很大
      High variance: Var[ρG] can be large
    - 不稳定：少数大权重样本可能主导
      Unstable: Few high-weight samples may dominate
    
    数学证明（无偏性）：
    Mathematical proof (unbiasedness):
    E_b[ρG] = E_b[π(τ)/b(τ) × G(τ)]
            = Σ_τ [π(τ)/b(τ) × G(τ) × b(τ)]
            = Σ_τ π(τ) × G(τ)
            = E_π[G]
    
    这是最纯粹的IS，但实践中常有问题
    This is the purest IS, but often problematic in practice
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 target_policy: Policy,
                 behavior_policy: Policy,
                 gamma: float = 1.0,
                 first_visit: bool = True):
        """
        初始化普通IS
        Initialize ordinary IS
        
        Args:
            env: 环境
            target_policy: 目标策略
            behavior_policy: 行为策略
            gamma: 折扣因子
            first_visit: 是否使用first-visit
        """
        super().__init__(env, target_policy, behavior_policy, gamma)
        
        self.first_visit = first_visit
        
        # 记录所有加权回报（用于分析）
        # Record all weighted returns (for analysis)
        self.weighted_returns: Dict[str, List[float]] = defaultdict(list)
        
        logger.info(f"初始化普通IS: first_visit={first_visit}")
    
    def update_value(self, episode: Episode):
        """
        使用普通IS更新价值
        Update value using ordinary IS
        
        核心：直接平均加权回报
        Core: Direct average of weighted returns
        """
        # 计算回报
        # Compute returns
        returns = episode.compute_returns(self.gamma)
        
        # 计算每步的IS比率
        # Compute per-step IS ratios
        ratios = self.compute_per_step_ratios(episode)
        
        if self.first_visit:
            # First-visit
            visited_states = set()
            
            for t, exp in enumerate(episode.experiences):
                if exp.state.id not in visited_states:
                    visited_states.add(exp.state.id)
                    
                    G = returns[t]
                    rho = ratios[t]
                    
                    # 加权回报
                    # Weighted return
                    weighted_G = rho * G
                    
                    # 记录
                    # Record
                    self.weighted_returns[exp.state.id].append(weighted_G)
                    self.is_ratios.append(rho)
                    self.state_visits[exp.state.id] += 1
                    
                    # 普通IS更新：简单平均
                    # Ordinary IS update: simple average
                    all_weighted = self.weighted_returns[exp.state.id]
                    new_v = np.mean(all_weighted)
                    self.V.set_value(exp.state, new_v)
                    
                    # 更新统计
                    # Update statistics
                    self.statistics.update_state_value(exp.state, weighted_G)
        
        else:
            # Every-visit
            for t, exp in enumerate(episode.experiences):
                G = returns[t]
                rho = ratios[t]
                
                weighted_G = rho * G
                
                self.weighted_returns[exp.state.id].append(weighted_G)
                self.is_ratios.append(rho)
                self.state_visits[exp.state.id] += 1
                
                # 更新价值
                # Update value
                all_weighted = self.weighted_returns[exp.state.id]
                new_v = np.mean(all_weighted)
                self.V.set_value(exp.state, new_v)
                
                self.statistics.update_state_value(exp.state, weighted_G)
    
    def analyze_estimator_properties(self):
        """
        分析普通IS估计器的性质
        Analyze properties of ordinary IS estimator
        
        展示无偏性和高方差
        Show unbiasedness and high variance
        """
        print("\n" + "="*60)
        print("普通IS估计器性质")
        print("Ordinary IS Estimator Properties")
        print("="*60)
        
        print("""
        📊 数学性质 Mathematical Properties
        ====================================
        
        1. 无偏性 Unbiasedness:
           E_b[ρG] = v_π(s) ✓
           
           证明关键：
           Key proof:
           ρ将b的概率测度转换为π的
           ρ transforms b's probability measure to π's
        
        2. 方差 Variance:
           Var[ρG] = E_b[(ρG)²] - (v_π(s))²
                   = E_π[ρG²] - (v_π(s))²
           
           问题：ρ可能很大
           Problem: ρ can be very large
           
           最坏情况：
           Worst case:
           如果π和b很不同，ρ的方差可能无限
           If π and b very different, ρ variance can be infinite
        
        3. 收敛速度 Convergence Rate:
           √n(V̂ - v_π) → N(0, σ²)
           
           其中σ² = Var[ρG]
           where σ² = Var[ρG]
           
           问题：σ²可能非常大
           Problem: σ² can be very large
        
        4. 样本效率 Sample Efficiency:
           有效样本大小 ESS = n × (E[ρ])² / E[ρ²]
           Effective Sample Size
           
           通常ESS << n
           Usually ESS << n
        """)
        
        # 分析实际数据
        # Analyze actual data
        if self.weighted_returns:
            print("\n实际估计分析:")
            print("Actual Estimation Analysis:")
            
            for state_id, weighted_list in list(self.weighted_returns.items())[:3]:
                if len(weighted_list) > 1:
                    mean = np.mean(weighted_list)
                    std = np.std(weighted_list)
                    cv = std / abs(mean) if mean != 0 else float('inf')
                    
                    print(f"\n状态 {state_id}:")
                    print(f"  样本数: {len(weighted_list)}")
                    print(f"  估计值: {mean:.3f}")
                    print(f"  标准差: {std:.3f}")
                    print(f"  变异系数: {cv:.3f}")
                    
                    # 检查极端值影响
                    # Check extreme value impact
                    if len(weighted_list) >= 10:
                        sorted_weights = sorted(weighted_list, reverse=True)
                        top_10_percent = int(len(weighted_list) * 0.1)
                        top_contribution = sum(sorted_weights[:top_10_percent]) / sum(weighted_list)
                        print(f"  前10%样本贡献: {top_contribution:.1%}")


# ================================================================================
# 第4.4.4节：加权重要性采样
# Section 4.4.4: Weighted Importance Sampling
# ================================================================================

class WeightedImportanceSampling(ImportanceSampling):
    """
    加权重要性采样
    Weighted Importance Sampling
    
    用归一化减少方差
    Reduce variance through normalization
    
    估计器：
    Estimator:
    V^π(s) = Σᵢ(ρᵢGᵢ) / Σᵢρᵢ
    
    与普通IS的区别：
    Difference from ordinary IS:
    - 普通：(1/n)Σρᵢ(ρG)ᵢ
      Ordinary: (1/n)Σρᵢ(ρG)ᵢ
    - 加权：Σᵢ(ρᵢGᵢ)/Σᵢρᵢ
      Weighted: Σᵢ(ρᵢGᵢ)/Σᵢρᵢ
    
    性质：
    Properties:
    - 有偏（但渐近无偏）
      Biased (but asymptotically unbiased)
    - 低方差
      Lower variance
    - 更稳定
      More stable
    - 实践中通常更好
      Usually better in practice
    
    偏差分析：
    Bias analysis:
    - 有限样本有偏：E[Σ(ρG)/Σρ] ≠ v_π
      Finite sample biased
    - 渐近无偏：当n→∞时收敛到v_π
      Asymptotically unbiased: converges to v_π as n→∞
    - 偏差随样本数快速减小
      Bias decreases quickly with samples
    
    为什么方差更小？
    Why lower variance?
    归一化使估计对极端权重更鲁棒
    Normalization makes estimate more robust to extreme weights
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 target_policy: Policy,
                 behavior_policy: Policy,
                 gamma: float = 1.0,
                 first_visit: bool = True):
        """
        初始化加权IS
        Initialize weighted IS
        """
        super().__init__(env, target_policy, behavior_policy, gamma)
        
        self.first_visit = first_visit
        
        # 累积分子和分母（用于增量更新）
        # Cumulative numerator and denominator (for incremental update)
        self.C = defaultdict(float)  # 分母：Σρ
        self.weighted_sum = defaultdict(float)  # 分子：Σ(ρG)
        
        logger.info(f"初始化加权IS: first_visit={first_visit}")
    
    def update_value(self, episode: Episode):
        """
        使用加权IS更新价值
        Update value using weighted IS
        
        核心：归一化的加权平均
        Core: Normalized weighted average
        """
        returns = episode.compute_returns(self.gamma)
        ratios = self.compute_per_step_ratios(episode)
        
        if self.first_visit:
            visited_states = set()
            
            for t, exp in enumerate(episode.experiences):
                if exp.state.id not in visited_states:
                    visited_states.add(exp.state.id)
                    
                    G = returns[t]
                    rho = ratios[t]
                    
                    # 更新分子和分母
                    # Update numerator and denominator
                    self.weighted_sum[exp.state.id] += rho * G
                    self.C[exp.state.id] += rho
                    
                    # 记录
                    # Record
                    self.is_ratios.append(rho)
                    self.state_visits[exp.state.id] += 1
                    
                    # 加权IS更新：归一化
                    # Weighted IS update: normalized
                    if self.C[exp.state.id] > 0:
                        new_v = self.weighted_sum[exp.state.id] / self.C[exp.state.id]
                        self.V.set_value(exp.state, new_v)
                    
                    # 统计
                    # Statistics
                    self.statistics.update_state_value(exp.state, G)
        
        else:
            for t, exp in enumerate(episode.experiences):
                G = returns[t]
                rho = ratios[t]
                
                self.weighted_sum[exp.state.id] += rho * G
                self.C[exp.state.id] += rho
                
                self.is_ratios.append(rho)
                self.state_visits[exp.state.id] += 1
                
                if self.C[exp.state.id] > 0:
                    new_v = self.weighted_sum[exp.state.id] / self.C[exp.state.id]
                    self.V.set_value(exp.state, new_v)
                
                self.statistics.update_state_value(exp.state, G)
    
    def compare_with_ordinary(self, ordinary_is: OrdinaryImportanceSampling):
        """
        与普通IS比较
        Compare with ordinary IS
        
        展示偏差-方差权衡
        Show bias-variance tradeoff
        """
        print("\n" + "="*60)
        print("加权IS vs 普通IS")
        print("Weighted IS vs Ordinary IS")
        print("="*60)
        
        # 比较估计值
        # Compare estimates
        print("\n估计值比较:")
        print("Estimate Comparison:")
        
        sample_states = list(self.state_visits.keys())[:5]
        
        print(f"{'State':<10} {'Weighted IS':<12} {'Ordinary IS':<12} {'Difference':<12}")
        print("-" * 46)
        
        for state_id in sample_states:
            # 获取加权IS估计
            # Get weighted IS estimate
            if state_id in self.C and self.C[state_id] > 0:
                weighted_v = self.weighted_sum[state_id] / self.C[state_id]
            else:
                weighted_v = 0.0
            
            # 获取普通IS估计
            # Get ordinary IS estimate
            if state_id in ordinary_is.weighted_returns:
                ordinary_v = np.mean(ordinary_is.weighted_returns[state_id])
            else:
                ordinary_v = 0.0
            
            diff = abs(weighted_v - ordinary_v)
            
            print(f"{state_id:<10} {weighted_v:<12.3f} {ordinary_v:<12.3f} {diff:<12.3f}")
        
        # 比较方差
        # Compare variance
        print("\n方差比较:")
        print("Variance Comparison:")
        
        # 加权IS的有效方差（近似）
        # Effective variance of weighted IS (approximate)
        weighted_vars = []
        ordinary_vars = []
        
        for state_id in sample_states:
            if state_id in ordinary_is.weighted_returns:
                ordinary_var = np.var(ordinary_is.weighted_returns[state_id])
                ordinary_vars.append(ordinary_var)
            
            # 加权IS方差更难直接计算
            # Weighted IS variance harder to compute directly
            # 使用bootstrap或其他方法估计
            # Use bootstrap or other methods to estimate
        
        if ordinary_vars:
            print(f"  普通IS平均方差: {np.mean(ordinary_vars):.3f}")
            print(f"  （加权IS方差通常更小但难以直接计算）")
            print(f"  (Weighted IS variance usually smaller but hard to compute)")
        
        print("\n" + "="*40)
        print("理论比较:")
        print("Theoretical Comparison:")
        print("="*40)
        print("""
        普通IS Ordinary IS:
        ------------------
        ✓ 无偏 Unbiased
        ✗ 高方差 High variance
        ✗ 对极端权重敏感 Sensitive to extreme weights
        
        加权IS Weighted IS:
        ------------------
        ✗ 有限样本有偏 Finite sample biased
        ✓ 低方差 Lower variance
        ✓ 对极端权重鲁棒 Robust to extreme weights
        ✓ 实践中通常更好 Usually better in practice
        
        选择建议 Selection Advice:
        ------------------------
        - 小样本+需要无偏：普通IS
          Small sample + need unbiased: Ordinary IS
        - 大样本+需要稳定：加权IS
          Large sample + need stable: Weighted IS
        - 一般情况：加权IS
          General case: Weighted IS
        """)


# ================================================================================
# 第4.4.5节：增量重要性采样MC
# Section 4.4.5: Incremental Importance Sampling MC
# ================================================================================

class IncrementalISMC(ImportanceSampling):
    """
    增量重要性采样MC
    Incremental Importance Sampling MC
    
    使用增量公式的加权IS
    Weighted IS with incremental formula
    
    这是实践中最常用的形式
    This is the most commonly used form in practice
    
    更新公式：
    Update formula:
    Q(s,a) ← Q(s,a) + (W/C(s,a))[G - Q(s,a)]
    
    其中：
    where:
    - W是重要性权重
      W is importance weight
    - C(s,a)是累积权重
      C(s,a) is cumulative weight
    - G是回报
      G is return
    
    等价于：
    Equivalent to:
    Q(s,a) = Σᵢ(WᵢGᵢ) / Σᵢ Wᵢ
    
    优势：
    Advantages:
    1. 内存效率（不存储历史）
       Memory efficient (no history storage)
    2. 在线学习
       Online learning
    3. 自然的off-policy控制
       Natural off-policy control
    
    这是Q-learning的前身！
    This is the predecessor of Q-learning!
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 target_policy: Policy,
                 behavior_policy: Policy,
                 gamma: float = 1.0):
        """
        初始化增量IS MC
        Initialize incremental IS MC
        """
        super().__init__(env, target_policy, behavior_policy, gamma)
        
        # 累积权重（分母）
        # Cumulative weights (denominator)
        self.C_state = defaultdict(float)
        self.C_sa = defaultdict(float)
        
        logger.info("初始化增量IS MC")
    
    def update_value(self, episode: Episode):
        """
        更新价值（使用增量加权IS）
        Update value (using incremental weighted IS)
        
        注意：IncrementalISMC主要通过learn方法工作
        Note: IncrementalISMC primarily works through learn method
        """
        # 这个方法主要是为了满足基类接口
        # This method is mainly to satisfy base class interface
        # 实际的增量更新在learn方法中实现
        # Actual incremental update is implemented in learn method
        pass
    
    def learn(self, n_episodes: int = 1000, 
             verbose: bool = True) -> Tuple[Policy, ActionValueFunction]:
        """
        学习最优策略
        Learn optimal policy
        
        实现off-policy MC控制
        Implement off-policy MC control
        """
        if verbose:
            print("\n" + "="*60)
            print("增量IS MC学习")
            print("Incremental IS MC Learning")
            print("="*60)
            print(f"  目标策略: {type(self.target_policy).__name__}")
            print(f"  行为策略: {type(self.behavior_policy).__name__}")
            print(f"  回合数: {n_episodes}")
        
        learning_curve = []
        
        for episode_num in range(n_episodes):
            # 用行为策略生成回合
            # Generate episode using behavior policy
            episode = self.generate_episode(self.behavior_policy)
            
            # 计算回报
            # Compute returns
            returns = episode.compute_returns(self.gamma)
            
            # 反向处理（累积重要性权重）
            # Process backward (accumulate importance weights)
            W = 1.0
            
            for t in reversed(range(len(episode.experiences))):
                exp = episode.experiences[t]
                sa_pair = (exp.state.id, exp.action.id)
                G = returns[t]
                
                # 更新Q（增量加权IS）
                # Update Q (incremental weighted IS)
                self.C_sa[sa_pair] += W
                
                if self.C_sa[sa_pair] > 0:
                    old_q = self.Q.get_value(exp.state, exp.action)
                    new_q = old_q + (W / self.C_sa[sa_pair]) * (G - old_q)
                    self.Q.set_value(exp.state, exp.action, new_q)
                
                # 更新V（类似）
                # Update V (similarly)
                self.C_state[exp.state.id] += W
                
                if self.C_state[exp.state.id] > 0:
                    old_v = self.V.get_value(exp.state)
                    new_v = old_v + (W / self.C_state[exp.state.id]) * (G - old_v)
                    self.V.set_value(exp.state, new_v)
                
                # 改进目标策略（贪婪）
                # Improve target policy (greedy)
                if isinstance(self.target_policy, DeterministicPolicy):
                    best_action = None
                    best_value = float('-inf')
                    
                    for action in self.env.action_space:
                        q_value = self.Q.get_value(exp.state, action)
                        if q_value > best_value:
                            best_value = q_value
                            best_action = action
                    
                    if best_action:
                        self.target_policy.policy_map[exp.state] = best_action
                
                # 如果不是目标策略的动作，终止
                # If not target policy action, terminate
                if isinstance(self.target_policy, DeterministicPolicy):
                    if exp.state in self.target_policy.policy_map:
                        if exp.action.id != self.target_policy.policy_map[exp.state].id:
                            break
                
                # 更新W
                # Update W
                behavior_probs = self.behavior_policy.get_action_probabilities(
                    exp.state
                )
                behavior_prob = behavior_probs.get(exp.action, 1e-10)
                
                # 目标策略是确定性的，概率是1
                # Target policy is deterministic, probability is 1
                W = W / behavior_prob
                
                # 记录
                # Record
                self.is_ratios.append(W)
            
            # 记录学习进度
            # Record learning progress
            if returns:
                learning_curve.append(returns[0])
            
            if verbose and (episode_num + 1) % 100 == 0:
                avg_return = np.mean(learning_curve[-100:]) if learning_curve else 0
                print(f"  Episode {episode_num + 1}: 平均回报={avg_return:.2f}")
        
        if verbose:
            print(f"\n学习完成:")
            print(f"  访问的状态: {len(self.C_state)}")
            print(f"  访问的(s,a)对: {len(self.C_sa)}")
            
            # 分析IS比率
            # Analyze IS ratios
            if self.is_ratios:
                print(f"  平均IS比率: {np.mean(self.is_ratios):.3f}")
                print(f"  IS比率标准差: {np.std(self.is_ratios):.3f}")
        
        return self.target_policy, self.Q
    
    def generate_episode(self, policy: Policy, max_steps: int = 1000) -> Episode:
        """
        生成回合
        Generate episode
        """
        episode = Episode()
        state = self.env.reset()
        
        for t in range(max_steps):
            action = policy.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            
            exp = Experience(state, action, reward, next_state, done)
            episode.add_experience(exp)
            
            state = next_state
            
            if done:
                break
        
        return episode
    
    def demonstrate_incremental_update(self):
        """
        演示增量更新
        Demonstrate incremental update
        
        展示增量公式的等价性
        Show equivalence of incremental formula
        """
        print("\n" + "="*60)
        print("增量IS更新演示")
        print("Incremental IS Update Demo")
        print("="*60)
        
        print("""
        📐 增量公式推导
        Incremental Formula Derivation
        ===============================
        
        批量加权IS Batch weighted IS:
        Q_n = Σᵢ₌₁ⁿ(WᵢGᵢ) / Σᵢ₌₁ⁿ Wᵢ
        
        增量形式 Incremental form:
        Q_n = Q_{n-1} + (W_n/C_n)[G_n - Q_{n-1}]
        
        其中 where:
        C_n = Σᵢ₌₁ⁿ Wᵢ = C_{n-1} + W_n
        
        证明等价 Prove equivalence:
        -------------------------------
        令 Let A_n = Σᵢ₌₁ⁿ(WᵢGᵢ), C_n = Σᵢ₌₁ⁿ Wᵢ
        
        则 Then Q_n = A_n / C_n
        
        A_n = A_{n-1} + W_nG_n
            = C_{n-1}Q_{n-1} + W_nG_n
        
        Q_n = A_n / C_n
            = (C_{n-1}Q_{n-1} + W_nG_n) / C_n
            = (C_{n-1}/C_n)Q_{n-1} + (W_n/C_n)G_n
            = Q_{n-1} - (W_n/C_n)Q_{n-1} + (W_n/C_n)G_n
            = Q_{n-1} + (W_n/C_n)[G_n - Q_{n-1}]  ✓
        
        优势 Advantages:
        ----------------
        1. 只需存储Q和C
           Only store Q and C
        2. O(1)更新复杂度
           O(1) update complexity
        3. 自然的在线学习
           Natural online learning
        
        与TD的联系 Connection to TD:
        ---------------------------
        当W=1（on-policy）时：
        When W=1 (on-policy):
        Q ← Q + (1/n)[G - Q]
        
        这就是MC的增量更新！
        This is incremental MC update!
        
        进一步，如果用估计代替G...
        Further, if replace G with estimate...
        → TD方法！
        → TD methods!
        """)


# ================================================================================
# 第4.4.6节：IS可视化器
# Section 4.4.6: IS Visualizer
# ================================================================================

class ISVisualizer:
    """
    重要性采样可视化器
    Importance Sampling Visualizer
    
    提供丰富的可视化来理解IS
    Provides rich visualizations to understand IS
    """
    
    @staticmethod
    def plot_is_comparison(ordinary: OrdinaryImportanceSampling,
                          weighted: WeightedImportanceSampling,
                          true_values: Optional[Dict[str, float]] = None):
        """
        比较普通和加权IS
        Compare ordinary and weighted IS
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 图1：估计值比较
        # Plot 1: Estimate comparison
        ax1 = axes[0, 0]
        
        states = list(set(ordinary.state_visits.keys()) & 
                     set(weighted.state_visits.keys()))[:10]
        
        if states:
            ordinary_estimates = []
            weighted_estimates = []
            
            for state_id in states:
                # 普通IS估计
                # Ordinary IS estimate
                if state_id in ordinary.weighted_returns:
                    ordinary_v = np.mean(ordinary.weighted_returns[state_id])
                else:
                    ordinary_v = 0
                ordinary_estimates.append(ordinary_v)
                
                # 加权IS估计
                # Weighted IS estimate
                if state_id in weighted.C and weighted.C[state_id] > 0:
                    weighted_v = weighted.weighted_sum[state_id] / weighted.C[state_id]
                else:
                    weighted_v = 0
                weighted_estimates.append(weighted_v)
            
            x = np.arange(len(states))
            width = 0.35
            
            ax1.bar(x - width/2, ordinary_estimates, width, 
                   label='Ordinary IS', alpha=0.7, color='blue')
            ax1.bar(x + width/2, weighted_estimates, width,
                   label='Weighted IS', alpha=0.7, color='red')
            
            if true_values:
                true_v = [true_values.get(s, 0) for s in states]
                ax1.plot(x, true_v, 'go-', label='True values', markersize=8)
            
            ax1.set_xticks(x)
            ax1.set_xticklabels(states, rotation=45)
            ax1.set_ylabel('Value Estimate')
            ax1.set_title('Value Estimates Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 图2：IS比率分布
        # Plot 2: IS ratio distribution
        ax2 = axes[0, 1]
        
        if ordinary.is_ratios:
            ax2.hist(np.clip(ordinary.is_ratios, 0, 10), bins=30,
                    alpha=0.5, color='blue', label='Ordinary IS')
        if weighted.is_ratios:
            ax2.hist(np.clip(weighted.is_ratios, 0, 10), bins=30,
                    alpha=0.5, color='red', label='Weighted IS')
        
        ax2.axvline(x=1.0, color='black', linestyle='--', label='ρ=1')
        ax2.set_xlabel('IS Ratio (clipped at 10)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('IS Ratio Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 图3：方差比较
        # Plot 3: Variance comparison
        ax3 = axes[0, 2]
        
        # 计算每个状态的估计方差
        # Compute estimation variance per state
        ordinary_vars = []
        weighted_vars = []
        
        for state_id in states[:5]:
            if state_id in ordinary.weighted_returns:
                if len(ordinary.weighted_returns[state_id]) > 1:
                    ordinary_vars.append(np.var(ordinary.weighted_returns[state_id]))
            
            # 加权IS的方差更复杂，这里用近似
            # Weighted IS variance is complex, use approximation
            if state_id in weighted.statistics.state_returns:
                returns_obj = weighted.statistics.state_returns[state_id]
                if returns_obj.count > 1:
                    weighted_vars.append(returns_obj.variance / returns_obj.count)
        
        if ordinary_vars or weighted_vars:
            labels = ['Ordinary IS', 'Weighted IS']
            variances = [ordinary_vars, weighted_vars]
            
            bp = ax3.boxplot(variances, labels=labels, patch_artist=True)
            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
            
            ax3.set_ylabel('Variance')
            ax3.set_title('Estimation Variance')
            ax3.grid(True, alpha=0.3, axis='y')
        
        # 图4：收敛曲线
        # Plot 4: Convergence curves
        ax4 = axes[1, 0]
        ax4.set_title('Convergence Comparison')
        ax4.set_xlabel('Number of Episodes')
        ax4.set_ylabel('Average Estimation Error')
        ax4.grid(True, alpha=0.3)
        # (需要运行时数据)
        # (Needs runtime data)
        
        # 图5：有效样本大小
        # Plot 5: Effective sample size
        ax5 = axes[1, 1]
        
        # 计算ESS
        # Compute ESS
        if ordinary.is_ratios and weighted.is_ratios:
            # 滑动窗口ESS
            # Sliding window ESS
            window = 100
            ordinary_ess = []
            weighted_ess = []
            
            for i in range(window, min(len(ordinary.is_ratios), 
                                     len(weighted.is_ratios)), 10):
                # 普通IS
                # Ordinary IS
                o_batch = ordinary.is_ratios[i-window:i]
                o_sum_w = np.sum(o_batch)
                o_sum_w2 = np.sum(np.array(o_batch)**2)
                if o_sum_w2 > 0:
                    o_ess = (o_sum_w**2) / o_sum_w2 / window
                    ordinary_ess.append(o_ess)
                
                # 加权IS
                # Weighted IS
                w_batch = weighted.is_ratios[i-window:i]
                w_sum_w = np.sum(w_batch)
                w_sum_w2 = np.sum(np.array(w_batch)**2)
                if w_sum_w2 > 0:
                    w_ess = (w_sum_w**2) / w_sum_w2 / window
                    weighted_ess.append(w_ess)
            
            if ordinary_ess and weighted_ess:
                x_axis = range(len(ordinary_ess))
                ax5.plot(x_axis, ordinary_ess, 'b-', alpha=0.7, label='Ordinary IS')
                ax5.plot(x_axis, weighted_ess, 'r-', alpha=0.7, label='Weighted IS')
                ax5.axhline(y=1.0, color='gray', linestyle='--', label='Perfect')
                ax5.set_xlabel('Episode Batch')
                ax5.set_ylabel('ESS / n')
                ax5.set_title('Effective Sample Size Efficiency')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
        
        # 图6：偏差-方差权衡
        # Plot 6: Bias-variance tradeoff
        ax6 = axes[1, 2]
        ax6.set_title('Bias-Variance Tradeoff')
        
        # 概念图
        # Conceptual plot
        methods = ['Ordinary IS', 'Weighted IS']
        bias = [0, 0.2]  # 普通无偏，加权有小偏差
        variance = [1.0, 0.3]  # 普通高方差，加权低方差
        
        ax6.scatter(bias, variance, s=200, alpha=0.6)
        for i, method in enumerate(methods):
            ax6.annotate(method, (bias[i], variance[i]), 
                        ha='center', va='center')
        
        ax6.set_xlabel('Bias')
        ax6.set_ylabel('Variance')
        ax6.set_xlim([-0.1, 0.5])
        ax6.set_ylim([0, 1.2])
        ax6.grid(True, alpha=0.3)
        
        # 添加MSE等高线（概念性）
        # Add MSE contours (conceptual)
        bias_grid = np.linspace(-0.1, 0.5, 100)
        variance_grid = np.linspace(0, 1.2, 100)
        B, V = np.meshgrid(bias_grid, variance_grid)
        MSE = B**2 + V  # MSE = Bias² + Variance
        
        contour = ax6.contour(B, V, MSE, levels=5, colors='gray', alpha=0.3)
        ax6.clabel(contour, inline=True, fontsize=8, fmt='MSE=%.1f')
        
        plt.suptitle('Importance Sampling Methods Comparison', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig


# ================================================================================
# 第4.4.7节：IS综合演示
# Section 4.4.7: IS Comprehensive Demo
# ================================================================================

def demonstrate_importance_sampling():
    """
    综合演示重要性采样
    Comprehensive demonstration of importance sampling
    """
    print("\n" + "="*80)
    print("重要性采样综合演示")
    print("Importance Sampling Comprehensive Demo")
    print("="*80)
    
    # 1. 基本原理演示
    # 1. Basic principle demo
    print("\n1. 基本原理")
    print("1. Basic Principle")
    fig1 = ImportanceSamplingTheory.demonstrate_basic_principle()
    
    # 2. 在RL环境中的IS
    # 2. IS in RL environment
    print("\n2. 在RL中的重要性采样")
    print("2. Importance Sampling in RL")
    
    # 创建环境
    # Create environment
    from src.ch03_finite_mdp.gridworld import GridWorld
    env = GridWorld(rows=3, cols=3,
                   start_pos=(0,0),
                   goal_pos=(2,2))
    
    # 创建行为策略（探索性）
    # Create behavior policy (exploratory)
    from src.ch03_finite_mdp.policies_and_values import UniformRandomPolicy
    behavior_policy = UniformRandomPolicy(env.action_space)
    
    # 创建目标策略（贪婪）
    # Create target policy (greedy)
    from src.ch03_finite_mdp.policies_and_values import ActionValueFunction, DeterministicPolicy
    Q_init = ActionValueFunction(env.state_space, env.action_space, initial_value=0.0)
    
    # 初始化一个简单的贪婪策略
    # Initialize a simple greedy policy
    policy_map = {}
    for state in env.state_space:
        if not state.is_terminal:
            # 向目标方向的简单启发式
            # Simple heuristic toward goal
            policy_map[state] = env.action_space[1]  # 假设是'right'或'down'
    
    target_policy = DeterministicPolicy(policy_map)
    
    # 运行不同IS方法
    # Run different IS methods
    n_episodes = 500
    
    print(f"\n运行{n_episodes}个回合...")
    print(f"Running {n_episodes} episodes...")
    
    # 3. 普通IS
    # 3. Ordinary IS
    print("\n3. 普通重要性采样")
    print("3. Ordinary Importance Sampling")
    
    ordinary_is = OrdinaryImportanceSampling(
        env, target_policy, behavior_policy, gamma=0.9
    )
    
    # 生成回合并更新
    # Generate episodes and update
    for _ in range(n_episodes):
        # 用行为策略生成回合
        # Generate episode with behavior policy
        episode = ordinary_is.generate_episode(behavior_policy)
        ordinary_is.update_value(episode)
    
    ordinary_is.diagnose_coverage()
    ordinary_is.analyze_variance()
    ordinary_is.analyze_estimator_properties()
    
    # 4. 加权IS
    # 4. Weighted IS
    print("\n4. 加权重要性采样")
    print("4. Weighted Importance Sampling")
    
    weighted_is = WeightedImportanceSampling(
        env, target_policy, behavior_policy, gamma=0.9
    )
    
    for _ in range(n_episodes):
        episode = weighted_is.generate_episode(behavior_policy)
        weighted_is.update_value(episode)
    
    weighted_is.compare_with_ordinary(ordinary_is)
    
    # 5. 增量IS MC
    # 5. Incremental IS MC
    print("\n5. 增量重要性采样MC")
    print("5. Incremental IS MC")
    
    # 创建ε-贪婪行为策略
    # Create ε-greedy behavior policy
    from ch04_monte_carlo.mc_control import EpsilonGreedyPolicy
    behavior_policy_eps = EpsilonGreedyPolicy(Q_init, epsilon=0.3, action_space=env.action_space)
    
    incremental_is = IncrementalISMC(
        env, target_policy, behavior_policy_eps, gamma=0.9
    )
    
    learned_policy, learned_Q = incremental_is.learn(n_episodes, verbose=True)
    incremental_is.demonstrate_incremental_update()
    
    # 可视化比较
    # Visualization comparison
    print("\n生成可视化...")
    print("Generating visualizations...")
    
    fig2 = ISVisualizer.plot_is_comparison(ordinary_is, weighted_is)
    
    # 总结
    # Summary
    print("\n" + "="*80)
    print("关键要点")
    print("Key Takeaways")
    print("="*80)
    print("""
    1. 重要性采样原理:
       Importance Sampling Principle:
       - 用一个分布的样本估计另一个分布
         Estimate one distribution using samples from another
       - 通过重要性权重修正
         Correct through importance weights
    
    2. 普通IS vs 加权IS:
       Ordinary IS vs Weighted IS:
       - 普通：无偏但高方差
         Ordinary: Unbiased but high variance
       - 加权：有偏但低方差
         Weighted: Biased but lower variance
       - 实践中加权通常更好
         Weighted usually better in practice
    
    3. 主要挑战:
       Main Challenges:
       - 方差爆炸
         Variance explosion
       - 覆盖性要求
         Coverage requirement
       - 有效样本大小减少
         Effective sample size reduction
    
    4. 与现代方法的联系:
       Connection to Modern Methods:
       - IS是off-policy学习的基础
         IS is foundation of off-policy learning
       - Q-learning可看作IS的特例
         Q-learning can be seen as special case of IS
       - 现代方法努力减少IS的方差
         Modern methods try to reduce IS variance
    
    5. 实践建议:
       Practical Advice:
       - 保持行为和目标策略接近
         Keep behavior and target policies close
       - 使用加权IS或其他方差减少技术
         Use weighted IS or other variance reduction
       - 监控有效样本大小
         Monitor effective sample size
    """)
    print("="*80)
    
    plt.show()


def generate_episode(env, policy, max_steps=1000):
    """辅助函数：生成回合"""
    from ch04_monte_carlo.mc_foundations import Episode, Experience
    
    episode = Episode()
    state = env.reset()
    
    for _ in range(max_steps):
        action = policy.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        exp = Experience(state, action, reward, next_state, done)
        episode.add_experience(exp)
        
        state = next_state
        if done:
            break
    
    return episode


# 为IS类添加辅助方法
ImportanceSampling.generate_episode = generate_episode


# ================================================================================
# 主函数
# Main Function
# ================================================================================

def main():
    """
    运行IS演示
    Run IS Demo
    """
    demonstrate_importance_sampling()


if __name__ == "__main__":
    main()