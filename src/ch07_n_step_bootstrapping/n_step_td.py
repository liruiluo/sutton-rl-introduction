"""
================================================================================
第7.1节：n步TD预测 - TD和MC之间的谱
Section 7.1: n-step TD Prediction - The Spectrum between TD and MC
================================================================================

n步方法是强化学习的统一框架！
n-step methods are a unifying framework in RL!

核心思想 Core Idea:
不必在TD(0)的一步更新和MC的完整回报之间二选一
We don't have to choose between one-step TD and complete MC returns

n步回报 n-step Return:
G_t^(n) = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γ^n V(S_{t+n})

其中 where:
- n=1: TD(0)
- n=∞: Monte Carlo
- 1<n<∞: 介于两者之间 Between the two

更新规则 Update Rule:
V(S_t) ← V(S_t) + α[G_t^(n) - V(S_t)]

优势 Advantages:
1. 更快的传播信用
   Faster credit propagation than TD(0)
2. 更低的方差
   Lower variance than MC
3. 可以在回合中更新
   Can update during episode
4. 灵活的偏差-方差权衡
   Flexible bias-variance tradeoff

挑战 Challenges:
1. 需要存储n步历史
   Need to store n-step history
2. 延迟n步才能更新
   n-step delay before update
3. n的选择是超参数
   Choice of n is hyperparameter

实践建议 Practical Tips:
- n=4-8通常效果好
  n=4-8 often works well
- 可以动态调整n
  Can adapt n dynamically
- 考虑使用n步的平均
  Consider averaging over n
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union, Deque
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
# Import base components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from ch02_mdp.mdp_framework import State, Action, MDPEnvironment, MDPAgent
from ch02_mdp.policies_and_values import (
    Policy, StateValueFunction, ActionValueFunction,
    StochasticPolicy, DeterministicPolicy
)
from ch05_temporal_difference.td_foundations import TDError, TDErrorAnalyzer

# 设置日志
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第7.1.1节：n步回报计算
# Section 7.1.1: n-step Return Computation
# ================================================================================

class NStepReturn:
    """
    n步回报计算器
    n-step Return Calculator
    
    计算各种n步回报
    Computes various n-step returns
    
    关键公式 Key Formula:
    G_t:t+n = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γ^n V(S_{t+n})
    
    特殊情况 Special Cases:
    - 如果t+n≥T（终止），则不需要bootstrap
      If t+n≥T (terminal), no bootstrap needed
    - n=1: 标准TD回报
      Standard TD return
    - n=∞: MC回报
      MC return
    """
    
    @staticmethod
    def compute_n_step_return(
        rewards: List[float],
        values: List[float],
        n: int,
        gamma: float,
        t: int = 0
    ) -> float:
        """
        计算n步回报
        Compute n-step return
        
        Args:
            rewards: 奖励序列 R_1, R_2, ..., R_T
                    Reward sequence
            values: 状态价值 V(S_0), V(S_1), ..., V(S_T)
                   State values (注意比rewards多一个)
            n: 步数
               Number of steps
            gamma: 折扣因子
                  Discount factor
            t: 起始时刻
               Starting time
        
        Returns:
            n步回报 G_t:t+n
            n-step return
        """
        T = len(rewards)  # 终止时间
        
        # 确保n合理
        # Ensure n is reasonable
        n = min(n, T - t)
        
        if n <= 0:
            return 0.0
        
        # 计算n步奖励的折扣和
        # Compute discounted sum of n-step rewards
        g_n = 0.0
        for k in range(n):
            if t + k < T:
                g_n += (gamma ** k) * rewards[t + k]
        
        # 如果没到终点，添加bootstrap值
        # Add bootstrap value if not terminal
        if t + n < T and t + n < len(values):
            g_n += (gamma ** n) * values[t + n]
        
        return g_n
    
    @staticmethod
    def compute_all_n_step_returns(
        rewards: List[float],
        values: List[float],
        max_n: int,
        gamma: float
    ) -> Dict[int, List[float]]:
        """
        计算所有时刻的所有n步回报
        Compute all n-step returns for all times
        
        用于分析和比较
        For analysis and comparison
        
        Args:
            rewards: 奖励序列
                    Reward sequence
            values: 状态价值序列
                   State value sequence
            max_n: 最大步数
                  Maximum steps
            gamma: 折扣因子
                  Discount factor
        
        Returns:
            {n: [G_0^n, G_1^n, ...]} 每个n的回报序列
            Returns for each n
        """
        T = len(rewards)
        returns = {n: [] for n in range(1, max_n + 1)}
        
        for n in range(1, max_n + 1):
            for t in range(T):
                g_n = NStepReturn.compute_n_step_return(
                    rewards, values, n, gamma, t
                )
                returns[n].append(g_n)
        
        return returns
    
    @staticmethod
    def analyze_n_step_returns(
        rewards: List[float],
        values: List[float],
        gamma: float = 0.9
    ):
        """
        分析n步回报的特性
        Analyze characteristics of n-step returns
        
        展示不同n值的效果
        Show effects of different n values
        """
        print("\n" + "="*60)
        print("n步回报分析 n-step Return Analysis")
        print("="*60)
        
        T = len(rewards)
        max_n = min(T, 10)  # 最多分析10步
        
        # 计算所有n步回报
        # Compute all n-step returns
        all_returns = NStepReturn.compute_all_n_step_returns(
            rewards, values, max_n, gamma
        )
        
        # 分析每个n
        # Analyze each n
        print(f"\n奖励序列 Reward sequence: {rewards}")
        print(f"折扣因子 Discount factor: γ={gamma}")
        print(f"\n{'n':<5} {'平均回报':<15} {'标准差':<15} {'说明':<20}")
        print("-" * 55)
        
        for n in sorted(all_returns.keys()):
            returns = all_returns[n]
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # 说明
            # Description
            if n == 1:
                desc = "TD(0)"
            elif n >= T:
                desc = "MC (完整回报)"
            else:
                desc = f"{n}步TD"
            
            print(f"{n:<5} {mean_return:<15.3f} {std_return:<15.3f} {desc:<20}")
        
        # 关键洞察
        # Key insights
        print("\n关键洞察 Key Insights:")
        print("-" * 40)
        print("""
        1. n增加，偏差减小但方差增加
           As n increases, bias decreases but variance increases
           
        2. 小n快速但可能不准
           Small n is fast but may be inaccurate
           
        3. 大n准确但慢
           Large n is accurate but slow
           
        4. 最优n取决于具体问题
           Optimal n depends on the problem
        """)


# ================================================================================
# 第7.1.2节：n步缓冲区
# Section 7.1.2: n-step Buffer
# ================================================================================

@dataclass
class NStepExperience:
    """
    n步经验元组
    n-step Experience Tuple
    
    存储n步方法需要的信息
    Stores information needed for n-step methods
    """
    state: State
    action: Optional[Action]
    reward: float
    next_state: State
    done: bool
    value: Optional[float] = None  # 用于存储V(S)
    q_value: Optional[float] = None  # 用于存储Q(S,A)
    
class NStepBuffer:
    """
    n步缓冲区
    n-step Buffer
    
    存储最近n步的经验
    Stores recent n-step experiences
    
    关键功能 Key Features:
    1. 循环缓冲区，自动覆盖旧数据
       Circular buffer, auto-overwrites old data
    2. 支持可变n
       Supports variable n
    3. 处理回合边界
       Handles episode boundaries
    
    实现技巧 Implementation Tricks:
    - 使用deque实现高效的FIFO
      Use deque for efficient FIFO
    - 延迟n步后才能计算回报
      n-step delay before computing returns
    - 回合结束时flush剩余数据
      Flush remaining data at episode end
    """
    
    def __init__(self, n: int):
        """
        初始化n步缓冲区
        Initialize n-step buffer
        
        Args:
            n: 步数
               Number of steps
        """
        self.n = n
        self.buffer: Deque[NStepExperience] = deque(maxlen=n+1)
        self.gamma_powers = [1.0]  # 预计算γ的幂
        
        logger.info(f"初始化{n}步缓冲区")
    
    def add(self, experience: NStepExperience):
        """
        添加经验
        Add experience
        
        Args:
            experience: 经验元组
                       Experience tuple
        """
        self.buffer.append(experience)
    
    def is_ready(self) -> bool:
        """
        检查是否可以计算n步回报
        Check if ready to compute n-step return
        
        Returns:
            是否有足够的经验
            Whether have enough experiences
        """
        return len(self.buffer) >= self.n + 1
    
    def compute_n_step_return(self, gamma: float) -> float:
        """
        计算n步回报
        Compute n-step return
        
        使用缓冲区中的经验
        Using experiences in buffer
        
        Args:
            gamma: 折扣因子
                  Discount factor
        
        Returns:
            n步回报
            n-step return
        """
        if not self.is_ready() and not self.buffer[-1].done:
            return 0.0
        
        # 预计算gamma的幂（优化）
        # Precompute gamma powers (optimization)
        while len(self.gamma_powers) < len(self.buffer):
            self.gamma_powers.append(self.gamma_powers[-1] * gamma)
        
        # 计算n步折扣奖励和
        # Compute n-step discounted reward sum
        g_n = 0.0
        n_actual = min(self.n, len(self.buffer) - 1)
        
        for i in range(n_actual):
            if i < len(self.buffer) - 1:
                g_n += self.gamma_powers[i] * self.buffer[i].reward
        
        # 添加bootstrap值（如果需要）
        # Add bootstrap value (if needed)
        if n_actual < len(self.buffer) - 1 and not self.buffer[n_actual].done:
            if self.buffer[n_actual].value is not None:
                g_n += self.gamma_powers[n_actual] * self.buffer[n_actual].value
        
        return g_n
    
    def get_update_data(self, gamma: float) -> Optional[Tuple[State, float]]:
        """
        获取更新数据
        Get update data
        
        Returns:
            (要更新的状态, n步回报) 或 None
            (state to update, n-step return) or None
        """
        if not self.is_ready() and not (self.buffer and self.buffer[-1].done):
            return None
        
        g_n = self.compute_n_step_return(gamma)
        return (self.buffer[0].state, g_n)
    
    def clear(self):
        """
        清空缓冲区
        Clear buffer
        """
        self.buffer.clear()
        self.gamma_powers = [1.0]
    
    def flush(self, gamma: float) -> List[Tuple[State, float]]:
        """
        刷新缓冲区（回合结束时）
        Flush buffer (at episode end)
        
        返回所有剩余的更新
        Return all remaining updates
        
        Args:
            gamma: 折扣因子
                  Discount factor
        
        Returns:
            [(状态, n步回报), ...]
            [(state, n-step return), ...]
        """
        updates = []
        
        while len(self.buffer) > 1:
            if data := self.get_update_data(gamma):
                updates.append(data)
            self.buffer.popleft()
        
        return updates


# ================================================================================
# 第7.1.3节：n步TD算法
# Section 7.1.3: n-step TD Algorithm
# ================================================================================

class NStepTD:
    """
    n步TD预测算法
    n-step TD Prediction Algorithm
    
    算法步骤 Algorithm Steps:
    1. 初始化V(s)任意，V(terminal)=0
       Initialize V(s) arbitrarily, V(terminal)=0
    2. 对每个回合：
       For each episode:
       a. 初始化S_0
          Initialize S_0
       b. T = ∞
       c. 对t = 0, 1, 2, ...：
          For t = 0, 1, 2, ...:
          - 如果t < T：
            If t < T:
            执行动作，观察R_{t+1}, S_{t+1}
            Take action, observe R_{t+1}, S_{t+1}
            如果S_{t+1}终止：T = t + 1
            If S_{t+1} terminal: T = t + 1
          - τ = t - n + 1（要更新的时刻）
            τ = t - n + 1 (time to update)
          - 如果τ ≥ 0：
            If τ ≥ 0:
            G = Σ_{i=τ+1}^{min(τ+n,T)} γ^{i-τ-1} R_i
            如果τ + n < T: G = G + γ^n V(S_{τ+n})
            If τ + n < T: G = G + γ^n V(S_{τ+n})
            V(S_τ) ← V(S_τ) + α[G - V(S_τ)]
       直到τ = T - 1
       Until τ = T - 1
    
    优势 Advantages:
    1. 比TD(0)更快的信用传播
       Faster credit propagation than TD(0)
    2. 比MC更低的方差
       Lower variance than MC
    3. 在线更新
       Online updates
    4. 统一的框架
       Unified framework
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 n: int = 4,
                 gamma: float = 0.99,
                 alpha: Union[float, Callable] = 0.1):
        """
        初始化n步TD
        Initialize n-step TD
        
        Args:
            env: 环境
                Environment
            n: 步数
               Number of steps
            gamma: 折扣因子
                  Discount factor
            alpha: 学习率
                  Learning rate
        """
        self.env = env
        self.n = n
        self.gamma = gamma
        
        # 学习率
        # Learning rate
        if callable(alpha):
            self.alpha_func = alpha
        else:
            self.alpha_func = lambda t: alpha
        
        # 价值函数
        # Value function
        self.V = StateValueFunction(env.state_space, initial_value=0.0)
        
        # n步缓冲区
        # n-step buffer
        self.buffer = NStepBuffer(n)
        
        # TD误差分析
        # TD error analysis
        self.td_analyzer = TDErrorAnalyzer()
        
        # 统计
        # Statistics
        self.episode_count = 0
        self.step_count = 0
        self.episode_returns = []
        self.n_step_returns_history = []
        
        logger.info(f"初始化{n}步TD: γ={gamma}, α={alpha}")
    
    def learn_episode(self, policy: Policy) -> float:
        """
        学习一个回合
        Learn one episode
        
        使用n步更新
        Using n-step updates
        
        Args:
            policy: 策略
                   Policy
        
        Returns:
            回合回报
            Episode return
        """
        # 初始化
        # Initialize
        state = self.env.reset()
        self.buffer.clear()
        
        episode_return = 0.0
        t = 0
        T = float('inf')  # 终止时间
        
        # 存储轨迹用于n步计算
        # Store trajectory for n-step computation
        states = [state]
        rewards = []
        
        # 主循环
        # Main loop
        while True:
            # 时间步τ需要更新（延迟n-1步）
            # Time step τ to update (n-1 step delay)
            tau = t - self.n + 1
            
            if t < T:
                # 选择并执行动作
                # Select and execute action
                action = policy.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                # 存储经验
                # Store experience
                states.append(next_state)
                rewards.append(reward)
                
                # 添加到缓冲区
                # Add to buffer
                exp = NStepExperience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    value=self.V.get_value(next_state) if not done else 0.0
                )
                self.buffer.add(exp)
                
                # 累积回报
                # Accumulate return
                episode_return += reward * (self.gamma ** t)
                
                # 检查终止
                # Check termination
                if done:
                    T = t + 1
                
                # 更新状态
                # Update state
                state = next_state
            
            # 如果可以更新τ时刻的状态
            # If can update state at time τ
            if tau >= 0:
                # 计算n步回报
                # Compute n-step return
                n_actual = min(self.n, T - tau)
                g_n = 0.0
                
                # 累积n步奖励
                # Accumulate n-step rewards
                for i in range(n_actual):
                    if tau + i < len(rewards):
                        g_n += (self.gamma ** i) * rewards[tau + i]
                
                # 添加bootstrap（如果需要）
                # Add bootstrap (if needed)
                if tau + n_actual < T and tau + n_actual < len(states):
                    g_n += (self.gamma ** n_actual) * self.V.get_value(states[tau + n_actual])
                
                # 更新价值函数
                # Update value function
                if tau < len(states):
                    update_state = states[tau]
                    old_v = self.V.get_value(update_state)
                    alpha = self.alpha_func(self.step_count)
                    new_v = old_v + alpha * (g_n - old_v)
                    self.V.set_value(update_state, new_v)
                    
                    # 记录TD误差
                    # Record TD error
                    td_error = g_n - old_v
                    td_err_obj = TDError(
                        value=td_error,
                        timestep=self.step_count,
                        state=update_state,
                        next_state=states[tau + 1] if tau + 1 < len(states) else None,
                        reward=rewards[tau] if tau < len(rewards) else 0,
                        state_value=old_v,
                        next_state_value=g_n
                    )
                    self.td_analyzer.add_error(td_err_obj)
                    
                    self.step_count += 1
                    self.n_step_returns_history.append(g_n)
            
            # 增加时间步
            # Increment time step
            t += 1
            
            # 检查是否结束
            # Check if done
            if tau == T - 1:
                break
        
        # 记录统计
        # Record statistics
        self.episode_count += 1
        self.episode_returns.append(episode_return)
        
        return episode_return
    
    def learn(self,
             policy: Policy,
             n_episodes: int = 1000,
             verbose: bool = True) -> StateValueFunction:
        """
        学习价值函数
        Learn value function
        
        Args:
            policy: 策略
                   Policy
            n_episodes: 回合数
                       Number of episodes
            verbose: 是否输出进度
                    Whether to output progress
        
        Returns:
            学习的价值函数
            Learned value function
        """
        if verbose:
            print(f"\n开始{self.n}步TD学习: {n_episodes}回合")
            print(f"Starting {self.n}-step TD learning: {n_episodes} episodes")
            print(f"  参数: γ={self.gamma}, n={self.n}")
        
        for episode in range(n_episodes):
            episode_return = self.learn_episode(policy)
            
            if verbose and (episode + 1) % max(1, n_episodes // 10) == 0:
                avg_return = np.mean(self.episode_returns[-100:]) \
                           if len(self.episode_returns) >= 100 \
                           else np.mean(self.episode_returns)
                
                stats = self.td_analyzer.get_statistics()
                
                print(f"Episode {episode + 1}/{n_episodes}: "
                      f"Return={episode_return:.2f}, "
                      f"Avg Return={avg_return:.2f}, "
                      f"TD Error={stats.get('recent_abs_mean', 0):.4f}")
        
        if verbose:
            print(f"\n学习完成!")
            print(f"  最终平均回报: {np.mean(self.episode_returns[-100:]):.2f}")
            if self.n_step_returns_history:
                print(f"  平均n步回报: {np.mean(self.n_step_returns_history):.2f}")
        
        return self.V


# ================================================================================
# 第7.1.4节：n步TD比较分析
# Section 7.1.4: n-step TD Comparison Analysis
# ================================================================================

class NStepTDComparator:
    """
    n步TD比较器
    n-step TD Comparator
    
    系统比较不同n值的效果
    Systematically compare effects of different n values
    
    关键指标 Key Metrics:
    1. 收敛速度
       Convergence speed
    2. 最终性能
       Final performance
    3. 学习稳定性
       Learning stability
    4. 计算成本
       Computational cost
    """
    
    def __init__(self, env: MDPEnvironment):
        """
        初始化比较器
        Initialize comparator
        
        Args:
            env: 环境
                Environment
        """
        self.env = env
        self.results = {}
        
        logger.info("初始化n步TD比较器")
    
    def compare_n_values(self,
                        n_values: List[int] = [1, 2, 4, 8, 16],
                        n_episodes: int = 500,
                        n_runs: int = 10,
                        gamma: float = 0.99,
                        alpha: float = 0.1,
                        verbose: bool = True) -> Dict[str, Any]:
        """
        比较不同n值
        Compare different n values
        
        Args:
            n_values: 要比较的n值列表
                     List of n values to compare
            n_episodes: 每次运行的回合数
                       Episodes per run
            n_runs: 运行次数
                   Number of runs
            gamma: 折扣因子
                  Discount factor
            alpha: 学习率
                  Learning rate
            verbose: 是否输出进度
                    Whether to output progress
        
        Returns:
            比较结果
            Comparison results
        """
        if verbose:
            print("\n" + "="*80)
            print("n步TD比较实验")
            print("n-step TD Comparison Experiment")
            print("="*80)
            print(f"比较n值: {n_values}")
            print(f"参数: γ={gamma}, α={alpha}")
            print(f"实验: {n_episodes}回合 × {n_runs}次运行")
        
        from ch02_mdp.policies_and_values import UniformRandomPolicy
        policy = UniformRandomPolicy(self.env.action_space)
        
        # 存储结果
        # Store results
        results = {n: {
            'returns': [],
            'final_returns': [],
            'convergence_episodes': [],
            'computation_times': []
        } for n in n_values}
        
        # 运行实验
        # Run experiments
        for run in range(n_runs):
            if verbose:
                print(f"\n运行 {run + 1}/{n_runs}:")
            
            for n in n_values:
                # 计时开始
                # Start timing
                start_time = time.time()
                
                # 创建n步TD算法
                # Create n-step TD algorithm
                n_step_td = NStepTD(
                    self.env,
                    n=n,
                    gamma=gamma,
                    alpha=alpha
                )
                
                # 学习
                # Learn
                for episode in range(n_episodes):
                    episode_return = n_step_td.learn_episode(policy)
                
                # 计时结束
                # End timing
                elapsed_time = time.time() - start_time
                
                # 记录结果
                # Record results
                results[n]['returns'].append(n_step_td.episode_returns)
                results[n]['final_returns'].append(n_step_td.episode_returns[-1])
                results[n]['computation_times'].append(elapsed_time)
                
                # 计算收敛
                # Compute convergence
                convergence_ep = self._find_convergence_episode(n_step_td.episode_returns)
                results[n]['convergence_episodes'].append(convergence_ep)
                
                if verbose:
                    print(f"  n={n}: 最终回报={n_step_td.episode_returns[-1]:.2f}, "
                          f"收敛={convergence_ep}, 时间={elapsed_time:.2f}s")
        
        # 分析结果
        # Analyze results
        self.results = self._analyze_results(results)
        
        if verbose:
            self._print_comparison_summary()
        
        return self.results
    
    def _find_convergence_episode(self, returns: List[float],
                                  window: int = 50,
                                  threshold: float = 0.1) -> int:
        """
        找到收敛回合
        Find convergence episode
        """
        if len(returns) < window:
            return len(returns)
        
        for i in range(window, len(returns)):
            recent = returns[i-window:i]
            if len(recent) > 1:
                mean = np.mean(recent)
                std = np.std(recent)
                if std / (abs(mean) + 1e-10) < threshold:
                    return i
        
        return len(returns)
    
    def _analyze_results(self, results: Dict) -> Dict:
        """
        分析结果
        Analyze results
        """
        analyzed = {}
        
        for n, data in results.items():
            returns_array = np.array(data['returns'])
            
            analyzed[n] = {
                'mean_returns': np.mean(returns_array, axis=0),
                'std_returns': np.std(returns_array, axis=0),
                'final_return_mean': np.mean(data['final_returns']),
                'final_return_std': np.std(data['final_returns']),
                'convergence_mean': np.mean(data['convergence_episodes']),
                'convergence_std': np.std(data['convergence_episodes']),
                'computation_time_mean': np.mean(data['computation_times']),
                'computation_time_std': np.std(data['computation_times'])
            }
        
        return analyzed
    
    def _print_comparison_summary(self):
        """
        打印比较摘要
        Print comparison summary
        """
        print("\n" + "="*80)
        print("n步TD比较结果摘要")
        print("n-step TD Comparison Results Summary")
        print("="*80)
        
        print(f"\n{'n值':<10} {'最终回报':<20} {'收敛回合':<20} {'计算时间(s)':<15}")
        print("-" * 65)
        
        for n, data in sorted(self.results.items()):
            final_return = f"{data['final_return_mean']:.2f} ± {data['final_return_std']:.2f}"
            convergence = f"{data['convergence_mean']:.0f} ± {data['convergence_std']:.0f}"
            comp_time = f"{data['computation_time_mean']:.2f}"
            
            # 特殊标记
            # Special marks
            if n == 1:
                n_str = f"{n} (TD)"
            else:
                n_str = f"{n}"
            
            print(f"{n_str:<10} {final_return:<20} {convergence:<20} {comp_time:<15}")
        
        # 找最佳n
        # Find best n
        best_convergence = min(self.results.items(),
                              key=lambda x: x[1]['convergence_mean'])
        best_return = max(self.results.items(),
                         key=lambda x: x[1]['final_return_mean'])
        
        print(f"\n最快收敛: n={best_convergence[0]}")
        print(f"最高回报: n={best_return[0]}")
        
        print("""
        典型观察 Typical Observations:
        - n=1 (TD): 快速但高方差
                   Fast but high variance
        - n=4-8: 通常最佳平衡
                Usually best balance
        - 大n: 稳定但慢
              Stable but slow
        """)


# ================================================================================
# 主函数：演示n步TD
# Main Function: Demonstrate n-step TD
# ================================================================================

def demonstrate_n_step_td():
    """
    演示n步TD算法
    Demonstrate n-step TD algorithm
    """
    print("\n" + "="*80)
    print("第7.1节：n步TD预测")
    print("Section 7.1: n-step TD Prediction")
    print("="*80)
    
    from ch02_mdp.gridworld import GridWorld
    from ch02_mdp.policies_and_values import UniformRandomPolicy
    
    # 创建环境
    # Create environment
    env = GridWorld(rows=4, cols=4,
                   start_pos=(0,0),
                   goal_pos=(3,3))
    
    print(f"\n创建4×4 GridWorld")
    print(f"  起点: (0,0)")
    print(f"  终点: (3,3)")
    
    # 创建策略
    # Create policy
    policy = UniformRandomPolicy(env.action_space)
    
    # 1. 演示n步回报计算
    # 1. Demonstrate n-step return computation
    print("\n" + "="*60)
    print("1. n步回报计算演示")
    print("1. n-step Return Computation Demo")
    print("="*60)
    
    # 示例数据
    # Example data
    rewards = [0, 0, 1, 0, 0]
    values = [0.1, 0.2, 0.5, 0.3, 0.1, 0]
    gamma = 0.9
    
    print(f"\n奖励序列: {rewards}")
    print(f"价值序列: {values}")
    print(f"折扣因子: γ={gamma}")
    
    NStepReturn.analyze_n_step_returns(rewards, values, gamma)
    
    # 2. 测试不同n值的TD学习
    # 2. Test TD learning with different n values
    print("\n" + "="*60)
    print("2. 不同n值的TD学习")
    print("2. TD Learning with Different n Values")
    print("="*60)
    
    n_values = [1, 2, 4, 8]
    
    for n in n_values:
        print(f"\n测试{n}步TD:")
        n_step_td = NStepTD(env, n=n, gamma=0.9, alpha=0.1)
        
        V = n_step_td.learn(policy, n_episodes=100, verbose=False)
        
        # 显示结果
        # Show results
        avg_return = np.mean(n_step_td.episode_returns[-20:])
        print(f"  平均回报: {avg_return:.2f}")
        
        # 显示一些价值
        # Show some values
        sample_state = env.state_space[0]
        if not sample_state.is_terminal:
            value = V.get_value(sample_state)
            print(f"  V(s0) = {value:.3f}")
    
    # 3. 系统比较不同n值
    # 3. Systematic comparison of different n values
    print("\n" + "="*60)
    print("3. 系统比较不同n值")
    print("3. Systematic Comparison of Different n Values")
    print("="*60)
    
    comparator = NStepTDComparator(env)
    results = comparator.compare_n_values(
        n_values=[1, 2, 4, 8, 16],
        n_episodes=200,
        n_runs=5,
        gamma=0.9,
        alpha=0.1,
        verbose=True
    )
    
    # 总结
    # Summary
    print("\n" + "="*80)
    print("n步TD预测总结")
    print("n-step TD Prediction Summary")
    print("="*80)
    
    print("""
    关键要点 Key Takeaways:
    =======================
    
    1. n步方法统一TD和MC
       n-step methods unify TD and MC
       
    2. n控制偏差-方差权衡
       n controls bias-variance tradeoff
       
    3. 中等n值(4-8)通常最优
       Moderate n (4-8) often optimal
       
    4. n步延迟是代价
       n-step delay is the cost
       
    5. 可以动态调整n
       Can adapt n dynamically
    """)


if __name__ == "__main__":
    demonstrate_n_step_td()