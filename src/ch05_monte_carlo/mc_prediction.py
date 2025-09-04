"""
================================================================================
第5.2节：蒙特卡洛预测 - 从经验中估计价值函数
Section 5.2: Monte Carlo Prediction - Estimating Value Functions from Experience
================================================================================

MC预测是MC方法的核心，展示了如何不用模型来估计价值函数。
MC prediction is the core of MC methods, showing how to estimate value functions without a model.

两种主要变体：
Two main variants:
1. First-Visit MC：只使用每个状态的第一次访问
   First-Visit MC: Only use first visit to each state
2. Every-Visit MC：使用每个状态的所有访问
   Every-Visit MC: Use all visits to each state

关键算法：
Key algorithms:
1. 批量MC（存储所有回报）
   Batch MC (stores all returns)
2. 增量MC（只存储平均值）
   Incremental MC (only stores mean)
3. 常数步长MC（适应非平稳）
   Constant-step MC (adapts to non-stationarity)

理论保证：
Theoretical guarantees:
- First-Visit MC：无偏、独立样本、收敛到真实值
  First-Visit MC: Unbiased, independent samples, converges to true value
- Every-Visit MC：无偏、相关样本、也收敛到真实值
  Every-Visit MC: Unbiased, correlated samples, also converges to true value

实践考虑：
Practical considerations:
- First-Visit更理论友好（独立性）
  First-Visit more theory-friendly (independence)
- Every-Visit更数据高效（更多样本）
  Every-Visit more data-efficient (more samples)
- 增量更新节省内存
  Incremental updates save memory
- 常数步长适应变化
  Constant step-size adapts to changes
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time
from abc import ABC, abstractmethod

# 导入基础组件
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.ch03_finite_mdp.mdp_framework import State, Action, MDPEnvironment
from src.ch03_finite_mdp.policies_and_values import (
    Policy, StateValueFunction, ActionValueFunction,
    StochasticPolicy, DeterministicPolicy
)
from .mc_foundations import (
    Episode, Experience, Return, MCStatistics
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第5.2.1节：MC预测基类
# Section 5.2.1: MC Prediction Base Class
# ================================================================================

class MCPrediction(ABC):
    """
    蒙特卡洛预测基类
    Monte Carlo Prediction Base Class
    
    定义了所有MC预测算法的共同接口
    Defines common interface for all MC prediction algorithms
    
    设计原则：
    Design principles:
    1. 统一的估计接口
       Unified estimation interface
    2. 灵活的访问策略（first/every）
       Flexible visit strategy (first/every)
    3. 支持增量和批量更新
       Support incremental and batch updates
    4. 完整的统计跟踪
       Complete statistics tracking
    
    为什么需要基类？
    Why need base class?
    - 确保所有MC算法有一致的接口
      Ensure all MC algorithms have consistent interface
    - 共享通用功能（采样、统计）
      Share common functionality (sampling, statistics)
    - 方便算法比较和切换
      Easy algorithm comparison and switching
    """
    
    def __init__(self, 
                 env: MDPEnvironment,
                 gamma: float = 1.0,
                 visit_type: str = 'first'):
        """
        初始化MC预测
        Initialize MC Prediction
        
        Args:
            env: MDP环境
                MDP environment
            gamma: 折扣因子
                  Discount factor
            visit_type: 访问类型 'first' 或 'every'
                       Visit type 'first' or 'every'
        
        设计考虑：
        Design considerations:
        - gamma=1.0是MC的常见选择（无折扣）
          gamma=1.0 is common for MC (undiscounted)
        - visit_type影响收敛速度和方差
          visit_type affects convergence speed and variance
        """
        self.env = env
        self.gamma = gamma
        self.visit_type = visit_type
        
        # 初始化价值函数
        # Initialize value functions
        self.V = StateValueFunction(env.state_space, initial_value=0.0)
        self.Q = ActionValueFunction(env.state_space, env.action_space, initial_value=0.0)
        
        # 统计收集
        # Statistics collection
        self.statistics = MCStatistics()
        
        # 记录所有回合
        # Record all episodes
        self.episodes: List[Episode] = []
        
        # 访问计数
        # Visit counts
        self.state_visits: Dict[str, int] = defaultdict(int)
        self.state_action_visits: Dict[Tuple[str, str], int] = defaultdict(int)
        
        # 收敛历史
        # Convergence history
        self.convergence_history: List[float] = []
        
        logger.info(f"初始化{visit_type}-visit MC预测, γ={gamma}")
    
    @abstractmethod
    def update_value(self, episode: Episode):
        """
        更新价值函数（子类实现）
        Update value function (implemented by subclasses)
        
        这是MC算法的核心差异点
        This is the key difference point of MC algorithms
        """
        pass
    
    def generate_episode(self, policy: Policy, max_steps: int = 1000) -> Episode:
        """
        生成一个回合
        Generate an episode
        
        遵循策略π直到终止
        Follow policy π until termination
        
        Args:
            policy: 要评估的策略
                   Policy to evaluate
            max_steps: 最大步数（避免无限循环）
                      Maximum steps (avoid infinite loop)
        
        Returns:
            完整的回合
            Complete episode
        
        实现细节：
        Implementation details:
        - 使用环境的step函数模拟
          Use environment's step function to simulate
        - 记录完整轨迹
          Record complete trajectory
        - 处理最大步数限制
          Handle maximum steps limit
        """
        episode = Episode()
        state = self.env.reset()
        
        for t in range(max_steps):
            # 根据策略选择动作
            # Select action according to policy
            action = policy.select_action(state)
            
            # 执行动作
            # Execute action
            next_state, reward, done, info = self.env.step(action)
            
            # 记录经验
            # Record experience
            exp = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            )
            episode.add_experience(exp)
            
            # 更新状态
            # Update state
            state = next_state
            
            if done:
                break
        
        if not episode.is_complete():
            logger.warning(f"回合未正常结束（达到最大步数{max_steps}）")
        
        return episode
    
    def estimate_V(self, 
                   policy: Policy,
                   n_episodes: int = 1000,
                   verbose: bool = True) -> StateValueFunction:
        """
        估计状态价值函数 V^π
        Estimate state value function V^π
        
        这是MC预测的主函数
        This is the main function of MC prediction
        
        Args:
            policy: 要评估的策略
                   Policy to evaluate
            n_episodes: 采样回合数
                       Number of episodes to sample
            verbose: 是否输出进度
                    Whether to output progress
        
        Returns:
            估计的状态价值函数
            Estimated state value function
        
        算法流程：
        Algorithm flow:
        1. 生成回合
           Generate episode
        2. 计算回报
           Compute returns
        3. 更新估计
           Update estimates
        4. 重复直到收敛
           Repeat until convergence
        """
        if verbose:
            print(f"\n开始{self.visit_type}-visit MC估计V^π")
            print(f"Starting {self.visit_type}-visit MC estimation of V^π")
            print(f"  环境: {self.env.name}")
            print(f"  回合数: {n_episodes}")
            print(f"  折扣因子: {self.gamma}")
        
        start_time = time.time()
        
        for episode_num in range(n_episodes):
            # 生成回合
            # Generate episode
            episode = self.generate_episode(policy)
            self.episodes.append(episode)
            
            # 更新价值估计
            # Update value estimates
            self.update_value(episode)
            
            # 记录收敛历史（每10个回合）
            # Record convergence history (every 10 episodes)
            if (episode_num + 1) % 10 == 0:
                # 计算最大变化
                # Compute maximum change
                max_change = self._compute_max_change()
                self.convergence_history.append(max_change)
                
                if verbose and (episode_num + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Episode {episode_num + 1}/{n_episodes}: "
                          f"max_change={max_change:.6f}, "
                          f"time={elapsed:.1f}s")
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\nMC预测完成:")
            print(f"  总时间: {total_time:.2f}秒")
            print(f"  平均每回合: {total_time/n_episodes*1000:.1f}毫秒")
            print(f"  访问状态数: {len(self.state_visits)}")
            print(f"  平均访问次数: {np.mean(list(self.state_visits.values())):.1f}")
        
        return self.V
    
    def estimate_Q(self,
                   policy: Policy,
                   n_episodes: int = 1000,
                   verbose: bool = True) -> ActionValueFunction:
        """
        估计动作价值函数 Q^π
        Estimate action value function Q^π
        
        与V估计类似，但追踪(s,a)对
        Similar to V estimation but tracks (s,a) pairs
        
        Q函数对控制更重要！
        Q function is more important for control!
        """
        if verbose:
            print(f"\n开始{self.visit_type}-visit MC估计Q^π")
            print(f"Starting {self.visit_type}-visit MC estimation of Q^π")
        
        start_time = time.time()
        
        for episode_num in range(n_episodes):
            # 生成回合
            episode = self.generate_episode(policy)
            self.episodes.append(episode)
            
            # 更新Q值估计
            self._update_Q_values(episode)
            
            if verbose and (episode_num + 1) % 100 == 0:
                print(f"  Episode {episode_num + 1}/{n_episodes}")
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\nQ估计完成: {total_time:.2f}秒")
            print(f"  访问(s,a)对: {len(self.state_action_visits)}")
        
        return self.Q
    
    def _update_Q_values(self, episode: Episode):
        """
        更新Q值
        Update Q values
        
        处理(状态,动作)对的回报
        Process returns for (state,action) pairs
        """
        # 计算回报
        returns = episode.compute_returns(self.gamma)
        
        # 根据访问类型获取索引
        if self.visit_type == 'first':
            # First-visit: 只使用首次访问
            sa_pairs_seen = set()
            for t, exp in enumerate(episode.experiences):
                sa_pair = (exp.state.id, exp.action.id)
                if sa_pair not in sa_pairs_seen:
                    sa_pairs_seen.add(sa_pair)
                    G = returns[t]
                    self.statistics.update_action_value(exp.state, exp.action, G)
                    self.state_action_visits[sa_pair] += 1
                    
                    # 增量更新Q
                    old_q = self.Q.get_value(exp.state, exp.action)
                    n = self.state_action_visits[sa_pair]
                    new_q = old_q + (G - old_q) / n
                    self.Q.set_value(exp.state, exp.action, new_q)
        
        else:  # every-visit
            # Every-visit: 使用所有访问
            for t, exp in enumerate(episode.experiences):
                sa_pair = (exp.state.id, exp.action.id)
                G = returns[t]
                self.statistics.update_action_value(exp.state, exp.action, G)
                self.state_action_visits[sa_pair] += 1
                
                # 增量更新Q
                old_q = self.Q.get_value(exp.state, exp.action)
                n = self.state_action_visits[sa_pair]
                new_q = old_q + (G - old_q) / n
                self.Q.set_value(exp.state, exp.action, new_q)
    
    def _compute_max_change(self) -> float:
        """
        计算最大价值变化
        Compute maximum value change
        
        用于监控收敛
        Used to monitor convergence
        """
        if len(self.episodes) < 2:
            return float('inf')
        
        # 简化：返回最近的平均回报变化
        # Simplified: return recent average return change
        recent_returns = []
        for episode in self.episodes[-10:]:
            if episode.experiences:
                returns = episode.compute_returns(self.gamma)
                if returns:
                    recent_returns.append(returns[0])
        
        if len(recent_returns) < 2:
            return float('inf')
        
        return np.std(recent_returns)


# ================================================================================
# 第5.2.2节：First-Visit MC预测
# Section 5.2.2: First-Visit MC Prediction
# ================================================================================

class FirstVisitMC(MCPrediction):
    """
    First-Visit蒙特卡洛预测
    First-Visit Monte Carlo Prediction
    
    只使用每个状态的首次访问来更新估计
    Only use first visit to each state to update estimates
    
    理论性质：
    Theoretical properties:
    - 每个回合对每个状态最多贡献一个样本
      Each episode contributes at most one sample per state
    - 样本之间独立（重要！）
      Samples are independent (important!)
    - 收敛到真实价值（大数定律）
      Converges to true value (law of large numbers)
    
    算法步骤：
    Algorithm steps:
    1. 生成回合 τ = (S₀, A₀, R₁, S₁, ..., Sₜ)
       Generate episode τ = (S₀, A₀, R₁, S₁, ..., Sₜ)
    2. 对每个出现的状态s:
       For each state s appearing in episode:
       - 找到s的首次出现时间t
         Find first occurrence time t of s
       - 计算从t开始的回报G_t
         Compute return G_t from time t
       - 更新: V(s) ← V(s) + α[G_t - V(s)]
         Update: V(s) ← V(s) + α[G_t - V(s)]
    
    为什么叫"First-Visit"？
    Why called "First-Visit"?
    因为如果一个状态在回合中出现多次，只使用第一次
    Because if a state appears multiple times, only use the first
    
    类比：第一印象
    Analogy: First impression
    就像只用第一印象来判断一个人，忽略后续的接触
    Like judging a person only by first impression, ignoring later encounters
    """
    
    def __init__(self, env: MDPEnvironment, gamma: float = 1.0):
        """
        初始化First-Visit MC
        Initialize First-Visit MC
        """
        super().__init__(env, gamma, visit_type='first')
        
        # First-visit特有：记录每个状态的所有首次回报
        # First-visit specific: record all first-visit returns for each state
        self.first_returns: Dict[str, List[float]] = defaultdict(list)
        
        logger.info("初始化First-Visit MC预测")
    
    def update_value(self, episode: Episode):
        """
        使用first-visit更新价值
        Update value using first-visit
        
        核心逻辑：只处理每个状态的第一次出现
        Core logic: only process first occurrence of each state
        
        数学更新：
        Mathematical update:
        V(s) = (1/n(s)) Σᵢ Gᵢ(s)
        其中Gᵢ(s)是第i个回合中s首次出现的回报
        where Gᵢ(s) is return from first occurrence of s in episode i
        """
        # 计算整个回合的回报
        # Compute returns for entire episode
        returns = episode.compute_returns(self.gamma)
        
        # 记录已访问状态（用于first-visit）
        # Track visited states (for first-visit)
        visited_states = set()
        
        # 遍历回合中的每一步
        # Iterate through each step in episode
        for t, exp in enumerate(episode.experiences):
            state = exp.state
            
            # First-visit: 只处理首次访问
            # First-visit: only process first visit
            if state.id not in visited_states:
                visited_states.add(state.id)
                
                # 获取从时间t开始的回报
                # Get return starting from time t
                G = returns[t]
                
                # 记录这个回报
                # Record this return
                self.first_returns[state.id].append(G)
                
                # 更新统计
                # Update statistics
                self.statistics.update_state_value(state, G)
                self.state_visits[state.id] += 1
                
                # 增量更新价值估计
                # Incremental value update
                # V(s) ← V(s) + (1/n)[G - V(s)]
                n = self.state_visits[state.id]
                old_v = self.V.get_value(state)
                new_v = old_v + (G - old_v) / n
                self.V.set_value(state, new_v)
                
                # 详细日志（调试用）
                # Detailed logging (for debugging)
                if logger.level == logging.DEBUG:
                    logger.debug(f"First-visit更新: {state.id}")
                    logger.debug(f"  回报G = {G:.3f}")
                    logger.debug(f"  旧V = {old_v:.3f}")
                    logger.debug(f"  新V = {new_v:.3f}")
                    logger.debug(f"  访问次数 = {n}")
    
    def analyze_convergence(self, true_values: Optional[Dict[str, float]] = None):
        """
        分析first-visit收敛性
        Analyze first-visit convergence
        
        展示独立样本的优势
        Show advantage of independent samples
        """
        print("\n" + "="*60)
        print("First-Visit MC收敛分析")
        print("First-Visit MC Convergence Analysis")
        print("="*60)
        
        # 统计每个状态的信息
        # Statistics for each state
        for state_id, returns_list in self.first_returns.items():
            if len(returns_list) > 0:
                mean = np.mean(returns_list)
                std = np.std(returns_list) if len(returns_list) > 1 else 0
                n = len(returns_list)
                
                print(f"\n状态 {state_id}:")
                print(f"  首次访问次数: {n}")
                print(f"  平均回报: {mean:.3f}")
                print(f"  标准差: {std:.3f}")
                
                if n > 1:
                    # 计算标准误差
                    # Compute standard error
                    se = std / np.sqrt(n)
                    
                    # 95%置信区间
                    # 95% confidence interval
                    ci_lower = mean - 1.96 * se
                    ci_upper = mean + 1.96 * se
                    
                    print(f"  标准误差: {se:.3f}")
                    print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
                
                # 如果有真实值，计算误差
                # If true values available, compute error
                if true_values and state_id in true_values:
                    true_v = true_values[state_id]
                    error = abs(mean - true_v)
                    print(f"  真实值: {true_v:.3f}")
                    print(f"  误差: {error:.3f}")
                    
                    # 检验是否在置信区间内
                    # Check if true value in confidence interval
                    if n > 1:
                        if ci_lower <= true_v <= ci_upper:
                            print(f"  ✓ 真实值在95% CI内")
                        else:
                            print(f"  ✗ 真实值不在95% CI内")
        
        # 展示样本独立性的好处
        # Show benefit of sample independence
        print("\n" + "="*40)
        print("First-Visit的独立性优势:")
        print("Independence Advantage of First-Visit:")
        print("="*40)
        print("""
        1. 样本独立 → 方差公式简单
           Independent samples → Simple variance formula
           Var[mean] = Var[G]/n
        
        2. 中心极限定理直接适用
           Central Limit Theorem directly applies
           mean ~ N(μ, σ²/n)
        
        3. 置信区间构造简单
           Simple confidence interval construction
           CI = mean ± z_{α/2} × SE
        
        4. 统计检验更可靠
           More reliable statistical tests
        """)


# ================================================================================
# 第5.2.3节：Every-Visit MC预测
# Section 5.2.3: Every-Visit MC Prediction
# ================================================================================

class EveryVisitMC(MCPrediction):
    """
    Every-Visit蒙特卡洛预测
    Every-Visit Monte Carlo Prediction
    
    使用每个状态的所有访问来更新估计
    Use all visits to each state to update estimates
    
    理论性质：
    Theoretical properties:
    - 每个回合可能贡献多个样本
      Each episode may contribute multiple samples
    - 样本之间可能相关（注意！）
      Samples may be correlated (note!)
    - 仍然收敛到真实价值（但理论更复杂）
      Still converges to true value (but theory more complex)
    - 实践中常常收敛更快（更多数据）
      Often converges faster in practice (more data)
    
    算法步骤：
    Algorithm steps:
    1. 生成回合 τ
       Generate episode τ
    2. 对每个(s,t)，其中s在时间t出现:
       For each (s,t) where s appears at time t:
       - 计算从t开始的回报G_t
         Compute return G_t from time t
       - 更新: V(s) ← V(s) + α[G_t - V(s)]
         Update: V(s) ← V(s) + α[G_t - V(s)]
    
    为什么用Every-Visit？
    Why use Every-Visit?
    - 更多数据 → 潜在更快收敛
      More data → Potentially faster convergence
    - 某些环境中状态访问稀疏
      State visits sparse in some environments
    - 实践中表现常常很好
      Often performs well in practice
    
    类比：多次采样
    Analogy: Multiple sampling
    就像多次品尝同一道菜来评价味道
    Like tasting the same dish multiple times to evaluate taste
    """
    
    def __init__(self, env: MDPEnvironment, gamma: float = 1.0):
        """
        初始化Every-Visit MC
        Initialize Every-Visit MC
        """
        super().__init__(env, gamma, visit_type='every')
        
        # Every-visit特有：记录所有回报（包括重复访问）
        # Every-visit specific: record all returns (including repeated visits)
        self.all_returns: Dict[str, List[float]] = defaultdict(list)
        
        # 记录访问模式（用于分析相关性）
        # Record visit patterns (for correlation analysis)
        self.visit_patterns: List[List[str]] = []
        
        logger.info("初始化Every-Visit MC预测")
    
    def update_value(self, episode: Episode):
        """
        使用every-visit更新价值
        Update value using every-visit
        
        核心区别：处理所有访问，不只是第一次
        Key difference: process all visits, not just first
        
        数学更新：
        Mathematical update:
        V(s) = (1/N(s)) Σᵢ Σₜ Gᵢ,ₜ(s)
        其中Gᵢ,ₜ(s)是第i个回合中s在时间t出现的回报
        where Gᵢ,ₜ(s) is return from occurrence of s at time t in episode i
        """
        # 计算整个回合的回报
        # Compute returns for entire episode
        returns = episode.compute_returns(self.gamma)
        
        # 记录这个回合的访问模式
        # Record visit pattern for this episode
        visit_pattern = []
        
        # 遍历回合中的每一步
        # Iterate through each step in episode
        for t, exp in enumerate(episode.experiences):
            state = exp.state
            visit_pattern.append(state.id)
            
            # Every-visit: 处理所有访问
            # Every-visit: process all visits
            G = returns[t]
            
            # 记录这个回报
            # Record this return
            self.all_returns[state.id].append(G)
            
            # 更新统计
            # Update statistics
            self.statistics.update_state_value(state, G)
            self.state_visits[state.id] += 1
            
            # 增量更新价值估计
            # Incremental value update
            n = self.state_visits[state.id]
            old_v = self.V.get_value(state)
            new_v = old_v + (G - old_v) / n
            self.V.set_value(state, new_v)
            
            # 详细日志
            # Detailed logging
            if logger.level == logging.DEBUG:
                logger.debug(f"Every-visit更新: {state.id}")
                logger.debug(f"  第{self.state_visits[state.id]}次访问")
                logger.debug(f"  回报G = {G:.3f}")
                logger.debug(f"  更新: {old_v:.3f} → {new_v:.3f}")
        
        # 记录访问模式
        # Record visit pattern
        self.visit_patterns.append(visit_pattern)
    
    def analyze_correlation(self):
        """
        分析every-visit的样本相关性
        Analyze sample correlation in every-visit
        
        展示为什么理论分析更复杂
        Show why theoretical analysis is more complex
        """
        print("\n" + "="*60)
        print("Every-Visit样本相关性分析")
        print("Every-Visit Sample Correlation Analysis")
        print("="*60)
        
        # 找出重复访问的状态
        # Find states with repeated visits
        repeated_states = {}
        for pattern in self.visit_patterns[-10:]:  # 看最近10个回合
            state_counts = defaultdict(int)
            for state_id in pattern:
                state_counts[state_id] += 1
            
            for state_id, count in state_counts.items():
                if count > 1:
                    if state_id not in repeated_states:
                        repeated_states[state_id] = []
                    repeated_states[state_id].append(count)
        
        if repeated_states:
            print("\n重复访问的状态:")
            print("States with Repeated Visits:")
            for state_id, counts in repeated_states.items():
                avg_repeats = np.mean(counts)
                max_repeats = max(counts)
                print(f"  {state_id}: 平均重复{avg_repeats:.1f}次, 最多{max_repeats}次")
            
            print("\n相关性影响:")
            print("Correlation Impact:")
            print("""
            1. 同一回合内的回报相关
               Returns within same episode are correlated
               因为共享未来轨迹
               Because they share future trajectory
            
            2. 有效样本数 < 总样本数
               Effective sample size < Total sample size
               n_eff = n / (1 + 2Σρ)
               其中ρ是自相关系数
               where ρ is autocorrelation coefficient
            
            3. 标准误差估计需要调整
               Standard error estimation needs adjustment
               简单的SE = σ/√n会低估真实误差
               Simple SE = σ/√n underestimates true error
            
            4. 但实践中仍然有效！
               But still effective in practice!
               更多数据通常补偿了相关性
               More data usually compensates for correlation
            """)
        else:
            print("近期回合中没有重复访问")
            print("No repeated visits in recent episodes")
    
    def compare_with_first_visit(self, first_visit_mc: 'FirstVisitMC'):
        """
        与First-Visit比较
        Compare with First-Visit
        
        展示两种方法的差异
        Show differences between two methods
        """
        print("\n" + "="*60)
        print("First-Visit vs Every-Visit比较")
        print("First-Visit vs Every-Visit Comparison")
        print("="*60)
        
        # 比较样本数
        # Compare sample counts
        print("\n样本数比较:")
        print("Sample Count Comparison:")
        
        for state_id in self.state_visits.keys():
            every_count = self.state_visits[state_id]
            first_count = first_visit_mc.state_visits.get(state_id, 0)
            
            if first_count > 0:
                ratio = every_count / first_count
                print(f"  {state_id}: Every={every_count}, First={first_count}, "
                      f"比率={ratio:.2f}")
        
        # 比较收敛速度（通过方差）
        # Compare convergence speed (through variance)
        print("\n估计方差比较:")
        print("Estimation Variance Comparison:")
        
        for state_id in self.all_returns.keys():
            if state_id in first_visit_mc.first_returns:
                every_returns = self.all_returns[state_id]
                first_returns = first_visit_mc.first_returns[state_id]
                
                if len(every_returns) > 1 and len(first_returns) > 1:
                    every_var = np.var(every_returns)
                    first_var = np.var(first_returns)
                    
                    # 注意：这不是完全公平的比较（样本数不同）
                    # Note: not entirely fair comparison (different sample sizes)
                    print(f"  {state_id}:")
                    print(f"    Every-visit方差: {every_var:.3f}")
                    print(f"    First-visit方差: {first_var:.3f}")
                    
                    # 调整后的比较（考虑样本数）
                    # Adjusted comparison (considering sample size)
                    every_se = np.sqrt(every_var / len(every_returns))
                    first_se = np.sqrt(first_var / len(first_returns))
                    print(f"    Every-visit SE: {every_se:.3f}")
                    print(f"    First-visit SE: {first_se:.3f}")


# ================================================================================
# 第5.2.4节：增量MC预测
# Section 5.2.4: Incremental MC Prediction
# ================================================================================

class IncrementalMC(MCPrediction):
    """
    增量蒙特卡洛预测
    Incremental Monte Carlo Prediction
    
    使用增量更新公式，不存储所有回报
    Use incremental update formula, don't store all returns
    
    核心思想：运行平均
    Core idea: Running average
    V_{n+1} = V_n + (1/(n+1))[G_{n+1} - V_n]
    
    可以改写为：
    Can be rewritten as:
    V_{n+1} = V_n + α_n[G_{n+1} - V_n]
    其中 α_n = 1/(n+1)
    where α_n = 1/(n+1)
    
    优势：
    Advantages:
    1. 内存高效（O(|S|)而非O(|S|×n)）
       Memory efficient (O(|S|) not O(|S|×n))
    2. 计算高效（O(1)更新）
       Computationally efficient (O(1) update)
    3. 适合在线学习
       Suitable for online learning
    
    变体：
    Variants:
    1. 递减步长：α_n = 1/n → 收敛到真实值
       Decreasing step-size: α_n = 1/n → converges to true value
    2. 常数步长：α_n = α → 跟踪非平稳
       Constant step-size: α_n = α → tracks non-stationarity
    
    这是TD方法的前身！
    This is the predecessor of TD methods!
    """
    
    def __init__(self, 
                 env: MDPEnvironment,
                 gamma: float = 1.0,
                 alpha: Optional[float] = None,
                 visit_type: str = 'first'):
        """
        初始化增量MC
        Initialize Incremental MC
        
        Args:
            env: 环境
            gamma: 折扣因子
            alpha: 步长（None表示用1/n）
                  Step-size (None means use 1/n)
            visit_type: 'first' 或 'every'
        
        设计选择：
        Design choices:
        - alpha=None: 保证收敛到真实值
          alpha=None: Guarantees convergence to true value
        - alpha=常数: 适应非平稳环境
          alpha=constant: Adapts to non-stationary environment
        """
        super().__init__(env, gamma, visit_type)
        
        self.alpha = alpha  # 固定步长（如果指定）
        self.use_constant_alpha = (alpha is not None)
        
        # 不需要存储所有回报！
        # No need to store all returns!
        # 这就是"增量"的含义
        # This is what "incremental" means
        
        logger.info(f"初始化增量MC: α={'constant='+str(alpha) if alpha else '1/n'}")
    
    def update_value(self, episode: Episode):
        """
        增量更新价值
        Incremental value update
        
        关键：不存储历史，直接更新运行平均
        Key: Don't store history, directly update running average
        
        更新规则：
        Update rule:
        V(s) ← V(s) + α[G - V(s)]
        
        其中α是学习率：
        where α is learning rate:
        - 1/n(s): 精确平均
          1/n(s): Exact average
        - 常数: 指数加权平均
          constant: Exponentially weighted average
        """
        # 计算回报
        returns = episode.compute_returns(self.gamma)
        
        if self.visit_type == 'first':
            # First-visit增量更新
            visited_states = set()
            
            for t, exp in enumerate(episode.experiences):
                state = exp.state
                
                if state.id not in visited_states:
                    visited_states.add(state.id)
                    G = returns[t]
                    
                    # 更新访问计数
                    self.state_visits[state.id] += 1
                    n = self.state_visits[state.id]
                    
                    # 确定步长
                    if self.use_constant_alpha:
                        alpha = self.alpha
                    else:
                        alpha = 1.0 / n
                    
                    # 增量更新
                    old_v = self.V.get_value(state)
                    new_v = old_v + alpha * (G - old_v)
                    self.V.set_value(state, new_v)
                    
                    # 更新统计（用于分析）
                    self.statistics.update_state_value(state, G)
        
        else:  # every-visit
            # Every-visit增量更新
            for t, exp in enumerate(episode.experiences):
                state = exp.state
                G = returns[t]
                
                # 更新访问计数
                self.state_visits[state.id] += 1
                n = self.state_visits[state.id]
                
                # 确定步长
                if self.use_constant_alpha:
                    alpha = self.alpha
                else:
                    alpha = 1.0 / n
                
                # 增量更新
                old_v = self.V.get_value(state)
                new_v = old_v + alpha * (G - old_v)
                self.V.set_value(state, new_v)
                
                # 更新统计
                self.statistics.update_state_value(state, G)
    
    def demonstrate_step_size_effect(self, policy: Policy, n_episodes: int = 1000):
        """
        演示步长的影响
        Demonstrate effect of step-size
        
        比较递减步长vs常数步长
        Compare decreasing vs constant step-size
        """
        print("\n" + "="*60)
        print("步长影响演示")
        print("Step-size Effect Demonstration")
        print("="*60)
        
        # 创建两个版本
        # Create two versions
        mc_decreasing = IncrementalMC(self.env, self.gamma, alpha=None, visit_type=self.visit_type)
        mc_constant = IncrementalMC(self.env, self.gamma, alpha=0.1, visit_type=self.visit_type)
        
        # 记录学习曲线
        # Record learning curves
        decreasing_curve = []
        constant_curve = []
        
        for ep in range(n_episodes):
            # 生成相同的回合
            episode = self.generate_episode(policy)
            
            # 两种方法都更新
            mc_decreasing.update_value(episode)
            mc_constant.update_value(episode)
            
            # 记录价值（选择一个代表性状态）
            if self.env.state_space:
                sample_state = self.env.state_space[0]
                decreasing_curve.append(mc_decreasing.V.get_value(sample_state))
                constant_curve.append(mc_constant.V.get_value(sample_state))
        
        # 可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 左图：学习曲线
        ax1.plot(decreasing_curve, 'b-', alpha=0.7, label='α=1/n (decreasing)')
        ax1.plot(constant_curve, 'r-', alpha=0.7, label='α=0.1 (constant)')
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Value Estimate')
        ax1.set_title('Learning Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右图：步长变化
        n_values = np.arange(1, 101)
        decreasing_alphas = 1.0 / n_values
        constant_alphas = np.ones_like(n_values) * 0.1
        
        ax2.plot(n_values, decreasing_alphas, 'b-', label='α=1/n')
        ax2.plot(n_values, constant_alphas, 'r-', label='α=0.1')
        ax2.set_xlabel('Visit Count n')
        ax2.set_ylabel('Step-size α')
        ax2.set_title('Step-size Schedules')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 0.5])
        
        plt.suptitle('Incremental MC: Step-size Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        print("\n关键观察:")
        print("Key Observations:")
        print("""
        1. 递减步长 (α=1/n):
           Decreasing step-size (α=1/n):
           - 保证收敛到真实值
             Guarantees convergence to true value
           - 后期学习变慢
             Learning slows down later
           - 适合平稳环境
             Suitable for stationary environments
        
        2. 常数步长 (α=constant):
           Constant step-size (α=constant):
           - 持续学习和适应
             Continues to learn and adapt
           - 可能在真实值附近震荡
             May oscillate around true value
           - 适合非平稳环境
             Suitable for non-stationary environments
        
        3. 权衡：
           Trade-off:
           - 收敛精度 vs 适应能力
             Convergence accuracy vs Adaptability
           - 理论保证 vs 实践性能
             Theoretical guarantee vs Practical performance
        """)
        
        return fig
    
    def explain_incremental_formula(self):
        """
        解释增量公式
        Explain incremental formula
        
        展示为什么增量更新等价于批量平均
        Show why incremental update equals batch average
        """
        print("\n" + "="*60)
        print("增量公式推导")
        print("Incremental Formula Derivation")
        print("="*60)
        
        print("""
        📐 数学推导 Mathematical Derivation
        =====================================
        
        目标：计算n个回报的平均值
        Goal: Compute average of n returns
        
        批量方法 Batch method:
        V_n = (1/n) Σᵢ₌₁ⁿ Gᵢ
        
        增量方法 Incremental method:
        V_n = V_{n-1} + (1/n)[G_n - V_{n-1}]
        
        证明等价性 Prove equivalence:
        ----------------------------------------
        V_n = (1/n) Σᵢ₌₁ⁿ Gᵢ
            = (1/n)[G_n + Σᵢ₌₁ⁿ⁻¹ Gᵢ]
            = (1/n)[G_n + (n-1)V_{n-1}]
            = (1/n)G_n + ((n-1)/n)V_{n-1}
            = V_{n-1} + (1/n)[G_n - V_{n-1}]  ✓
        
        💡 关键洞察 Key Insights
        ========================
        
        1. 误差项 Error term:
           [G_n - V_{n-1}] 
           = 新样本与当前估计的差
           = Difference between new sample and current estimate
           = "惊喜"或"预测误差"
           = "Surprise" or "Prediction error"
        
        2. 学习率 Learning rate:
           α = 1/n
           = 随着样本增加而减小
           = Decreases as samples increase
           = 新样本的影响逐渐降低
           = New samples have decreasing influence
        
        3. 更新方向 Update direction:
           如果 G_n > V_{n-1}: 向上调整
           If G_n > V_{n-1}: Adjust upward
           如果 G_n < V_{n-1}: 向下调整
           If G_n < V_{n-1}: Adjust downward
        
        4. 这是RL中通用的更新模式！
           This is the universal update pattern in RL!
           新估计 = 旧估计 + 步长 × [目标 - 旧估计]
           New estimate = Old estimate + StepSize × [Target - Old estimate]
        
        🔄 常数步长的含义
        Meaning of Constant Step-size
        ==============================
        
        当 α = 常数（如0.1）:
        When α = constant (e.g., 0.1):
        
        V_n = V_{n-1} + α[G_n - V_{n-1}]
            = (1-α)V_{n-1} + αG_n
            = α G_n + (1-α)α G_{n-1} + (1-α)²α G_{n-2} + ...
            = α Σᵢ₌₁ⁿ (1-α)^{n-i} Gᵢ
        
        这是指数加权移动平均！
        This is exponentially weighted moving average!
        
        - 最近的样本权重最大
          Recent samples have highest weight
        - 旧样本权重指数衰减
          Old samples decay exponentially
        - 永远不会完全"忘记"
          Never completely "forgets"
        """)


# ================================================================================
# 第5.2.5节：MC预测可视化器
# Section 5.2.5: MC Prediction Visualizer
# ================================================================================

class MCPredictionVisualizer:
    """
    MC预测可视化器
    MC Prediction Visualizer
    
    提供丰富的可视化来理解MC预测
    Provides rich visualizations to understand MC prediction
    
    可视化内容：
    Visualization contents:
    1. 收敛曲线
       Convergence curves
    2. 置信区间
       Confidence intervals
    3. First-visit vs Every-visit比较
       First-visit vs Every-visit comparison
    4. 访问频率热力图
       Visit frequency heatmap
    5. 回报分布
       Return distributions
    """
    
    @staticmethod
    def plot_convergence_comparison(mc_methods: Dict[str, MCPrediction],
                                   true_values: Optional[Dict[str, float]] = None):
        """
        比较不同MC方法的收敛
        Compare convergence of different MC methods
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 准备颜色
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        # 图1：价值估计演化
        ax1 = axes[0, 0]
        ax1.set_title('Value Estimate Evolution')
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Value Estimate')
        
        for idx, (name, mc) in enumerate(mc_methods.items()):
            if mc.convergence_history:
                ax1.plot(mc.convergence_history, 
                        color=colors[idx % len(colors)],
                        label=name, alpha=0.7)
        
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 图2：最终估计比较
        ax2 = axes[0, 1]
        ax2.set_title('Final Value Estimates')
        
        # 选择一些状态来比较
        sample_states = []
        for mc in mc_methods.values():
            for state in mc.env.state_space[:5]:  # 最多5个状态
                if state not in sample_states:
                    sample_states.append(state)
        
        x = np.arange(len(sample_states))
        width = 0.8 / len(mc_methods)
        
        for idx, (name, mc) in enumerate(mc_methods.items()):
            values = [mc.V.get_value(s) for s in sample_states]
            offset = (idx - len(mc_methods)/2) * width
            ax2.bar(x + offset, values, width, 
                   label=name, alpha=0.7,
                   color=colors[idx % len(colors)])
        
        if true_values:
            true_vals = [true_values.get(s.id, 0) for s in sample_states]
            ax2.plot(x, true_vals, 'k*', markersize=10, label='True Values')
        
        ax2.set_xticks(x)
        ax2.set_xticklabels([s.id for s in sample_states], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 图3：访问频率比较
        ax3 = axes[1, 0]
        ax3.set_title('State Visit Frequencies')
        
        for idx, (name, mc) in enumerate(mc_methods.items()):
            visits = list(mc.state_visits.values())
            if visits:
                ax3.hist(visits, bins=20, alpha=0.5, 
                        label=name, color=colors[idx % len(colors)])
        
        ax3.set_xlabel('Visit Count')
        ax3.set_ylabel('Number of States')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 图4：收敛速度（标准差）
        ax4 = axes[1, 1]
        ax4.set_title('Convergence Speed (Std of Returns)')
        
        for idx, (name, mc) in enumerate(mc_methods.items()):
            # 计算每个状态回报的标准差
            stds = []
            for state_id in mc.statistics.state_returns:
                returns_obj = mc.statistics.state_returns[state_id]
                if returns_obj.count > 1:
                    stds.append(returns_obj.std)
            
            if stds:
                ax4.boxplot(stds, positions=[idx], labels=[name],
                           patch_artist=True, 
                           boxprops=dict(facecolor=colors[idx % len(colors)], alpha=0.5))
        
        ax4.set_ylabel('Standard Deviation')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('MC Prediction Methods Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_return_distributions(mc: MCPrediction, 
                                 states_to_plot: Optional[List[State]] = None):
        """
        绘制回报分布
        Plot return distributions
        
        展示MC估计的分布特性
        Show distributional properties of MC estimates
        """
        if states_to_plot is None:
            # 选择访问最频繁的状态
            # Select most frequently visited states
            sorted_states = sorted(mc.state_visits.items(), 
                                 key=lambda x: x[1], reverse=True)
            states_to_plot = [mc.env.get_state_by_id(s_id) 
                            for s_id, _ in sorted_states[:4]]
        
        n_states = len(states_to_plot)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, state in enumerate(states_to_plot):
            if idx >= 4:
                break
            
            ax = axes[idx]
            
            # 获取该状态的所有回报
            # Get all returns for this state
            returns_obj = mc.statistics.state_returns.get(state.id)
            
            if returns_obj and returns_obj.returns:
                returns = returns_obj.returns
                
                # 直方图
                # Histogram
                n, bins, patches = ax.hist(returns, bins=20, density=True,
                                          alpha=0.7, color='steelblue',
                                          edgecolor='black')
                
                # 拟合正态分布
                # Fit normal distribution
                mu, sigma = np.mean(returns), np.std(returns)
                x = np.linspace(min(returns), max(returns), 100)
                normal_pdf = stats.norm.pdf(x, mu, sigma)
                ax.plot(x, normal_pdf, 'r-', linewidth=2, 
                       label=f'N({mu:.2f}, {sigma:.2f}²)')
                
                # 添加均值线
                # Add mean line
                ax.axvline(x=mu, color='red', linestyle='--', 
                          linewidth=2, alpha=0.7, label=f'Mean={mu:.2f}')
                
                # 添加置信区间
                # Add confidence interval
                ci = returns_obj.confidence_interval(0.95)
                ax.axvspan(ci[0], ci[1], alpha=0.2, color='yellow',
                          label=f'95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]')
                
                ax.set_xlabel('Return G')
                ax.set_ylabel('Density')
                ax.set_title(f'State: {state.id} (n={len(returns)})')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'No data for {state.id}',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'State: {state.id}')
        
        plt.suptitle('Return Distributions by State', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_learning_curves(mc_methods: Dict[str, MCPrediction],
                           metric: str = 'mean_squared_error',
                           true_values: Optional[Dict[str, float]] = None):
        """
        绘制学习曲线
        Plot learning curves
        
        展示不同指标随时间的变化
        Show how different metrics change over time
        """
        if not true_values:
            print("警告：没有真实值，使用估计值的变化作为代理")
            print("Warning: No true values, using estimate changes as proxy")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for idx, (name, mc) in enumerate(mc_methods.items()):
            if metric == 'mean_squared_error' and true_values:
                # 计算MSE
                # Compute MSE
                mse_history = []
                for ep_idx in range(0, len(mc.episodes), 10):
                    mse = 0
                    count = 0
                    for state in mc.env.state_space:
                        if state.id in true_values:
                            estimate = mc.V.get_value(state)
                            true_val = true_values[state.id]
                            mse += (estimate - true_val) ** 2
                            count += 1
                    if count > 0:
                        mse_history.append(mse / count)
                
                if mse_history:
                    x = np.arange(0, len(mc.episodes), 10)
                    ax.plot(x, mse_history, color=colors[idx % len(colors)],
                           label=name, linewidth=2, alpha=0.7)
            
            elif metric == 'max_change':
                # 使用收敛历史
                # Use convergence history
                if mc.convergence_history:
                    x = np.arange(len(mc.convergence_history)) * 10
                    ax.plot(x, mc.convergence_history,
                           color=colors[idx % len(colors)],
                           label=name, linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Episodes')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Learning Curves: {metric}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 对数尺度可能更有用
        # Log scale might be more useful
        ax.set_yscale('log')
        
        plt.tight_layout()
        return fig


# ================================================================================
# 第5.2.6节：MC预测综合演示
# Section 5.2.6: MC Prediction Comprehensive Demo
# ================================================================================

def demonstrate_mc_prediction():
    """
    综合演示MC预测方法
    Comprehensive demonstration of MC prediction methods
    
    展示所有MC预测算法的特点和比较
    Show characteristics and comparison of all MC prediction algorithms
    """
    print("\n" + "="*80)
    print("蒙特卡洛预测方法综合演示")
    print("Monte Carlo Prediction Methods Comprehensive Demo")
    print("="*80)
    
    # 创建一个简单的网格世界用于测试
    # Create a simple grid world for testing
    from src.ch03_finite_mdp.gridworld import GridWorld
    from src.ch03_finite_mdp.policies_and_values import UniformRandomPolicy
    
    env = GridWorld(rows=4, cols=4, 
                   start_pos=(0,0), 
                   goal_pos=(3,3),
                   obstacles={(1,1), (2,2)})
    
    print(f"\n测试环境: {env.name}")
    print(f"  状态数: {len(env.state_space)}")
    print(f"  动作数: {len(env.action_space)}")
    
    # 创建要评估的策略（随机策略）
    # Create policy to evaluate (random policy)
    policy = UniformRandomPolicy(env.action_space)
    print(f"\n评估策略: 均匀随机策略")
    
    # 运行不同的MC方法
    # Run different MC methods
    n_episodes = 500
    print(f"\n运行{n_episodes}个回合...")
    
    # 1. First-Visit MC
    print("\n1. First-Visit MC")
    first_visit_mc = FirstVisitMC(env, gamma=0.9)
    V_first = first_visit_mc.estimate_V(policy, n_episodes, verbose=True)
    
    # 2. Every-Visit MC
    print("\n2. Every-Visit MC")
    every_visit_mc = EveryVisitMC(env, gamma=0.9)
    V_every = every_visit_mc.estimate_V(policy, n_episodes, verbose=True)
    
    # 3. Incremental MC (递减步长)
    print("\n3. Incremental MC (α=1/n)")
    incremental_mc = IncrementalMC(env, gamma=0.9, alpha=None, visit_type='first')
    V_inc = incremental_mc.estimate_V(policy, n_episodes, verbose=True)
    
    # 4. Incremental MC (常数步长)
    print("\n4. Incremental MC (α=0.1)")
    constant_mc = IncrementalMC(env, gamma=0.9, alpha=0.1, visit_type='first')
    V_const = constant_mc.estimate_V(policy, n_episodes, verbose=True)
    
    # 分析和比较
    # Analysis and comparison
    print("\n" + "="*60)
    print("方法比较")
    print("Method Comparison")
    print("="*60)
    
    # 比较特定状态的估计
    # Compare estimates for specific states
    sample_states = [env.state_space[0], env.state_space[-1]]  # 起点和终点
    
    print("\n价值估计比较:")
    print("Value Estimate Comparison:")
    print(f"{'State':<15} {'First-Visit':<12} {'Every-Visit':<12} "
          f"{'Incremental':<12} {'Constant-α':<12}")
    print("-" * 63)
    
    for state in sample_states:
        if not state.is_terminal:
            v_first = first_visit_mc.V.get_value(state)
            v_every = every_visit_mc.V.get_value(state)
            v_inc = incremental_mc.V.get_value(state)
            v_const = constant_mc.V.get_value(state)
            
            print(f"{state.id:<15} {v_first:<12.3f} {v_every:<12.3f} "
                  f"{v_inc:<12.3f} {v_const:<12.3f}")
    
    # First-visit分析
    print("\n" + "="*60)
    first_visit_mc.analyze_convergence()
    
    # Every-visit相关性分析
    print("\n" + "="*60)
    every_visit_mc.analyze_correlation()
    
    # 比较First和Every
    print("\n" + "="*60)
    every_visit_mc.compare_with_first_visit(first_visit_mc)
    
    # 步长影响演示
    print("\n" + "="*60)
    fig_stepsize = incremental_mc.demonstrate_step_size_effect(policy, 200)
    
    # 增量公式解释
    incremental_mc.explain_incremental_formula()
    
    # 可视化比较
    print("\n生成可视化...")
    
    mc_methods = {
        'First-Visit': first_visit_mc,
        'Every-Visit': every_visit_mc,
        'Incremental(1/n)': incremental_mc,
        'Incremental(α=0.1)': constant_mc
    }
    
    # 收敛比较
    fig_conv = MCPredictionVisualizer.plot_convergence_comparison(mc_methods)
    
    # 回报分布
    fig_dist = MCPredictionVisualizer.plot_return_distributions(first_visit_mc)
    
    # 学习曲线
    fig_learn = MCPredictionVisualizer.plot_learning_curves(
        mc_methods, metric='max_change')
    
    print("\n" + "="*80)
    print("MC预测演示完成！")
    print("MC Prediction Demo Complete!")
    print("\n关键要点 Key Takeaways:")
    print("1. First-Visit: 理论性质好，样本独立")
    print("   First-Visit: Good theoretical properties, independent samples")
    print("2. Every-Visit: 数据效率高，收敛可能更快")
    print("   Every-Visit: Data efficient, may converge faster")
    print("3. 增量更新: 内存高效，适合在线学习")
    print("   Incremental: Memory efficient, suitable for online learning")
    print("4. 常数步长: 适应非平稳，但可能不收敛到精确值")
    print("   Constant step-size: Adapts to non-stationarity, but may not converge exactly")
    print("5. MC是无模型学习的基础，为TD方法铺路")
    print("   MC is foundation of model-free learning, paving way for TD methods")
    print("="*80)
    
    plt.show()


# ================================================================================
# 主函数
# Main Function
# ================================================================================

def main():
    """
    运行MC预测演示
    Run MC Prediction Demo
    """
    # 完整演示
    demonstrate_mc_prediction()


if __name__ == "__main__":
    main()