"""
================================================================================
第5.5节：n-step TD方法 - MC和TD的统一
Section 5.5: n-step TD Methods - Unification of MC and TD
================================================================================

n-step TD是MC和TD(0)之间的桥梁！
n-step TD is the bridge between MC and TD(0)!

谱系 Spectrum:
TD(0) → n-step TD → Monte Carlo
n=1      n=2,3,...    n=∞

更新目标 Update targets:
- TD(0):    R_{t+1} + γV(S_{t+1})
- 2-step:   R_{t+1} + γR_{t+2} + γ²V(S_{t+2})
- 3-step:   R_{t+1} + γR_{t+2} + γ²R_{t+3} + γ³V(S_{t+3})
- ...
- MC:       R_{t+1} + γR_{t+2} + ... + γ^{T-t-1}R_T

n-step回报 n-step return:
G_t^(n) = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γ^n V(S_{t+n})

偏差-方差权衡 Bias-Variance Tradeoff:
- 小n: 低方差，高偏差（更像TD）
  Small n: Low variance, high bias (more like TD)
- 大n: 高方差，低偏差（更像MC）
  Large n: High variance, low bias (more like MC)

最优n的选择：
Choosing optimal n:
- 取决于问题特性
  Depends on problem characteristics
- 通常n=3到n=10效果好
  Usually n=3 to n=10 works well
- 可以动态调整
  Can be dynamically adjusted

资格迹（Eligibility Traces）的预览：
Preview of Eligibility Traces:
TD(λ)实际上是所有n-step回报的加权平均！
TD(λ) is actually weighted average of all n-step returns!
G^λ = (1-λ)Σ_{n=1}^∞ λ^{n-1} G_t^(n)

应用：
Applications:
- A3C使用n-step returns
- Rainbow DQN包含n-step
- 许多现代算法的基础
  Foundation of many modern algorithms
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
from ch04_monte_carlo.mc_control import EpsilonGreedyPolicy
from ch05_temporal_difference.td_foundations import TDError, TDErrorAnalyzer

# 设置日志
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第5.5.1节：n-step TD预测
# Section 5.5.1: n-step TD Prediction
# ================================================================================

@dataclass
class NStepExperience:
    """
    n-step经验
    n-step Experience
    
    存储n步的转移序列
    Store n-step transition sequence
    """
    states: List[State]      # S_t, S_{t+1}, ..., S_{t+n}
    actions: List[Action]     # A_t, A_{t+1}, ..., A_{t+n-1}
    rewards: List[float]      # R_{t+1}, R_{t+2}, ..., R_{t+n}
    
    @property
    def n(self) -> int:
        """步数"""
        return len(self.rewards)
    
    def compute_n_step_return(self, gamma: float, 
                             final_value: float = 0.0) -> float:
        """
        计算n-step回报
        Compute n-step return
        
        G_t^(n) = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γ^n V(S_{t+n})
        
        Args:
            gamma: 折扣因子
                  Discount factor
            final_value: V(S_{t+n})的值
                        Value of V(S_{t+n})
        
        Returns:
            n-step回报
            n-step return
        """
        g = 0.0
        for i, r in enumerate(self.rewards):
            g += (gamma ** i) * r
        g += (gamma ** self.n) * final_value
        return g


class NStepTD:
    """
    n-step TD预测
    n-step TD Prediction
    
    学习状态价值函数V^π
    Learn state value function V^π
    
    算法步骤 Algorithm steps:
    1. 初始化V(s)任意，V(terminal)=0
       Initialize V(s) arbitrarily, V(terminal)=0
    2. 重复每个回合：
       Repeat for each episode:
       存储S_0
       Store S_0
       T ← ∞
       For t = 0, 1, 2, ...:
         If t < T:
           执行动作，观察并存储S_{t+1}, R_{t+1}
           Take action, observe and store S_{t+1}, R_{t+1}
           如果S_{t+1}终止，则T ← t+1
           If S_{t+1} terminal, then T ← t+1
         τ ← t - n + 1 (要更新的时间)
         τ ← t - n + 1 (time to update)
         If τ ≥ 0:
           G ← Σ_{i=τ+1}^{min(τ+n,T)} γ^{i-τ-1} R_i
           If τ + n < T:
             G ← G + γ^n V(S_{τ+n})
           V(S_τ) ← V(S_τ) + α[G - V(S_τ)]
       Until τ = T - 1
    
    特性 Properties:
    1. n=1时退化为TD(0)
       Reduces to TD(0) when n=1
    2. n=∞时退化为MC
       Reduces to MC when n=∞  
    3. 延迟n步更新
       Delayed update by n steps
    4. 需要存储n步历史
       Need to store n-step history
    
    优势 Advantages:
    1. 更快的信用分配
       Faster credit assignment
    2. 更灵活的偏差-方差权衡
       More flexible bias-variance tradeoff
    3. 通常比TD(0)和MC都好
       Often better than both TD(0) and MC
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 n: int = 3,
                 gamma: float = 1.0,
                 alpha: Union[float, Callable] = 0.1):
        """
        初始化n-step TD
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
        
        # TD误差分析
        # TD error analysis
        self.td_analyzer = TDErrorAnalyzer()
        
        # 统计
        # Statistics
        self.episode_count = 0
        self.step_count = 0
        self.episode_returns = []
        
        logger.info(f"初始化{n}-step TD: γ={gamma}, α={alpha}")
    
    def learn_episode(self, policy: Policy) -> float:
        """
        学习一个回合
        Learn one episode
        
        使用n-step更新
        Use n-step updates
        
        Args:
            policy: 策略
                   Policy
        
        Returns:
            回合回报
            Episode return
        """
        # 初始化存储
        # Initialize storage
        states = [self.env.reset()]
        actions = []
        rewards = [0]  # R_0未使用，为了索引对齐
        
        T = float('inf')  # 终止时间
        t = 0
        
        episode_return = 0.0
        
        while True:
            # 第1阶段：执行动作
            # Phase 1: Take action
            if t < T:
                # 选择并执行动作
                # Select and execute action
                action = policy.select_action(states[t])
                next_state, reward, done, _ = self.env.step(action)
                
                # 存储
                # Store
                actions.append(action)
                rewards.append(reward)
                states.append(next_state)
                
                # 累积真实回报（用于统计）
                # Accumulate actual return (for statistics)
                episode_return += reward * (self.gamma ** t)
                
                if done:
                    T = t + 1
            
            # 第2阶段：更新价值
            # Phase 2: Update value
            tau = t - self.n + 1  # 要更新的时间
            
            if tau >= 0:
                # 计算n-step回报
                # Compute n-step return
                G = 0.0
                for i in range(tau + 1, min(tau + self.n, T) + 1):
                    G += (self.gamma ** (i - tau - 1)) * rewards[i]
                
                if tau + self.n < T:
                    # 添加bootstrap项
                    # Add bootstrap term
                    G += (self.gamma ** self.n) * self.V.get_value(states[tau + self.n])
                
                # 更新V(S_tau)
                # Update V(S_tau)
                if not states[tau].is_terminal:
                    old_v = self.V.get_value(states[tau])
                    alpha = self.alpha_func(self.step_count)
                    td_error = G - old_v
                    new_v = old_v + alpha * td_error
                    self.V.set_value(states[tau], new_v)
                    
                    # 记录TD误差
                    # Record TD error
                    td_err_obj = TDError(
                        value=td_error,
                        timestep=self.step_count,
                        state=states[tau],
                        next_state=states[min(tau + self.n, T)] if tau + self.n <= T else None,
                        reward=rewards[tau + 1] if tau + 1 < len(rewards) else 0,
                        state_value=old_v,
                        next_state_value=G
                    )
                    self.td_analyzer.add_error(td_err_obj)
                    
                    self.step_count += 1
            
            if tau == T - 1:
                break
            
            t += 1
        
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
            print(f"\n开始{self.n}-step TD学习: {n_episodes}回合")
            print(f"Starting {self.n}-step TD learning: {n_episodes} episodes")
        
        for episode in range(n_episodes):
            episode_return = self.learn_episode(policy)
            
            if verbose and (episode + 1) % max(1, n_episodes // 10) == 0:
                avg_return = np.mean(self.episode_returns[-100:]) if self.episode_returns else 0
                stats = self.td_analyzer.get_statistics()
                
                print(f"Episode {episode + 1}/{n_episodes}: "
                      f"Return={episode_return:.2f}, "
                      f"Avg Return={avg_return:.2f}, "
                      f"TD Error={stats.get('recent_abs_mean', 0):.4f}")
        
        if verbose:
            print(f"\n{self.n}-step TD学习完成!")
            print(f"{self.n}-step TD learning complete!")
        
        return self.V


# ================================================================================
# 第5.5.2节：n-step SARSA
# Section 5.5.2: n-step SARSA
# ================================================================================

class NStepSARSA:
    """
    n-step SARSA
    
    将SARSA扩展到n步
    Extend SARSA to n steps
    
    更新目标 Update target:
    G_t^(n) = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γ^n Q(S_{t+n}, A_{t+n})
    
    特性 Properties:
    1. n=1时是普通SARSA
       Regular SARSA when n=1
    2. On-policy控制
       On-policy control
    3. 需要存储n步的(S,A,R)序列
       Need to store n-step (S,A,R) sequence
    
    Expected n-step SARSA:
    可以使用期望而不是采样的A_{t+n}
    Can use expectation instead of sampled A_{t+n}
    G_t^(n) = R_{t+1} + ... + γ^n Σ_a π(a|S_{t+n}) Q(S_{t+n}, a)
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 n: int = 3,
                 gamma: float = 0.99,
                 alpha: Union[float, Callable] = 0.1,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        初始化n-step SARSA
        Initialize n-step SARSA
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
        
        # Q函数
        # Q function
        self.Q = ActionValueFunction(
            env.state_space,
            env.action_space,
            initial_value=0.0
        )
        
        # ε-贪婪策略
        # ε-greedy policy
        self.policy = EpsilonGreedyPolicy(
            self.Q,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            action_space=env.action_space
        )
        
        # TD误差分析
        # TD error analysis
        self.td_analyzer = TDErrorAnalyzer()
        
        # 统计
        # Statistics
        self.episode_count = 0
        self.step_count = 0
        self.episode_returns = []
        self.episode_lengths = []
        
        logger.info(f"初始化{n}-step SARSA: γ={gamma}, α={alpha}, ε={epsilon}")
    
    def learn_episode(self) -> Tuple[float, int]:
        """
        学习一个回合
        Learn one episode
        """
        # 初始化存储
        # Initialize storage
        states = [self.env.reset()]
        actions = [self.policy.select_action(states[0])]
        rewards = [0]  # R_0未使用
        
        T = float('inf')
        t = 0
        
        episode_return = 0.0
        episode_length = 0
        
        while True:
            # 第1阶段：执行动作
            # Phase 1: Take action
            if t < T:
                # 执行当前动作
                # Execute current action
                next_state, reward, done, _ = self.env.step(actions[t])
                
                # 存储
                # Store
                rewards.append(reward)
                states.append(next_state)
                
                episode_return += reward * (self.gamma ** t)
                episode_length += 1
                
                if done:
                    T = t + 1
                else:
                    # 选择下一个动作
                    # Select next action
                    next_action = self.policy.select_action(next_state)
                    actions.append(next_action)
            
            # 第2阶段：更新Q
            # Phase 2: Update Q
            tau = t - self.n + 1
            
            if tau >= 0:
                # 计算n-step回报
                # Compute n-step return
                G = 0.0
                for i in range(tau + 1, min(tau + self.n, T) + 1):
                    G += (self.gamma ** (i - tau - 1)) * rewards[i]
                
                if tau + self.n < T:
                    # 添加Q(S_{tau+n}, A_{tau+n})
                    # Add Q(S_{tau+n}, A_{tau+n})
                    G += (self.gamma ** self.n) * self.Q.get_value(
                        states[tau + self.n], actions[tau + self.n]
                    )
                
                # 更新Q(S_tau, A_tau)
                # Update Q(S_tau, A_tau)
                if not states[tau].is_terminal:
                    old_q = self.Q.get_value(states[tau], actions[tau])
                    alpha = self.alpha_func(self.step_count)
                    td_error = G - old_q
                    new_q = old_q + alpha * td_error
                    self.Q.set_value(states[tau], actions[tau], new_q)
                    
                    # 记录TD误差
                    # Record TD error
                    td_err_obj = TDError(
                        value=td_error,
                        timestep=self.step_count,
                        state=states[tau],
                        next_state=states[min(tau + self.n, len(states) - 1)],
                        reward=rewards[tau + 1] if tau + 1 < len(rewards) else 0,
                        state_value=old_q,
                        next_state_value=G
                    )
                    self.td_analyzer.add_error(td_err_obj)
                    
                    self.step_count += 1
            
            if tau == T - 1:
                break
            
            t += 1
        
        # 衰减ε
        # Decay ε
        self.policy.decay_epsilon()
        
        self.episode_count += 1
        self.episode_returns.append(episode_return)
        self.episode_lengths.append(episode_length)
        
        return episode_return, episode_length
    
    def learn(self,
             n_episodes: int = 1000,
             verbose: bool = True) -> ActionValueFunction:
        """
        学习Q函数
        Learn Q function
        """
        if verbose:
            print(f"\n开始{self.n}-step SARSA学习: {n_episodes}回合")
            print(f"Starting {self.n}-step SARSA: {n_episodes} episodes")
            print(f"  初始ε: {self.policy.epsilon:.3f}")
        
        for episode in range(n_episodes):
            episode_return, episode_length = self.learn_episode()
            
            if verbose and (episode + 1) % max(1, n_episodes // 10) == 0:
                avg_return = np.mean(self.episode_returns[-100:]) if len(self.episode_returns) >= 100 else np.mean(self.episode_returns)
                avg_length = np.mean(self.episode_lengths[-100:]) if len(self.episode_lengths) >= 100 else np.mean(self.episode_lengths)
                
                print(f"Episode {episode + 1}/{n_episodes}: "
                      f"Return={episode_return:.2f}, "
                      f"Avg Return={avg_return:.2f}, "
                      f"Avg Length={avg_length:.1f}, "
                      f"ε={self.policy.epsilon:.3f}")
        
        if verbose:
            print(f"\n{self.n}-step SARSA学习完成!")
            print(f"  最终ε: {self.policy.epsilon:.3f}")
            print(f"  总步数: {self.step_count}")
        
        return self.Q


# ================================================================================
# 第5.5.3节：n值比较实验
# Section 5.5.3: n Value Comparison Experiment
# ================================================================================

class NStepComparator:
    """
    n步方法比较器
    n-step Methods Comparator
    
    系统比较不同n值的效果
    Systematically compare effects of different n values
    
    实验维度：
    Experimental dimensions:
    1. 收敛速度
       Convergence speed
    2. 最终性能
       Final performance
    3. 稳定性
       Stability
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
        
        logger.info("初始化n-step比较器")
    
    def compare_n_values(self,
                        n_values: List[int] = [1, 2, 3, 5, 10],
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
            print("n-step TD方法比较实验")
            print("n-step TD Methods Comparison")
            print("="*80)
            print(f"比较n值: {n_values}")
            print(f"参数: γ={gamma}, α={alpha}")
            print(f"实验: {n_episodes}回合 × {n_runs}次运行")
        
        from ch02_mdp.policies_and_values import UniformRandomPolicy
        policy = UniformRandomPolicy(self.env.action_space)
        
        results = {n: {
            'returns': [],
            'final_returns': [],
            'convergence_episodes': [],
            'td_errors': []
        } for n in n_values}
        
        for run in range(n_runs):
            if verbose:
                print(f"\n运行 {run + 1}/{n_runs}:")
            
            for n in n_values:
                # 创建n-step TD
                # Create n-step TD
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
                
                # 记录结果
                # Record results
                results[n]['returns'].append(n_step_td.episode_returns)
                results[n]['final_returns'].append(n_step_td.episode_returns[-1])
                
                # 计算收敛回合
                # Compute convergence episode
                convergence_ep = self._find_convergence_episode(n_step_td.episode_returns)
                results[n]['convergence_episodes'].append(convergence_ep)
                
                # 记录最终TD误差
                # Record final TD errors
                stats = n_step_td.td_analyzer.get_statistics()
                results[n]['td_errors'].append(stats.get('recent_abs_mean', 0))
                
                if verbose:
                    print(f"  n={n}: 最终回报={n_step_td.episode_returns[-1]:.2f}, "
                          f"收敛回合={convergence_ep}, "
                          f"TD误差={stats.get('recent_abs_mean', 0):.4f}")
        
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
                'td_error_mean': np.mean(data['td_errors']),
                'td_error_std': np.std(data['td_errors'])
            }
        
        return analyzed
    
    def _print_comparison_summary(self):
        """
        打印比较摘要
        Print comparison summary
        """
        print("\n" + "="*80)
        print("n-step比较结果摘要")
        print("n-step Comparison Results Summary")
        print("="*80)
        
        print(f"\n{'n值':<10} {'最终回报 Final Return':<25} "
              f"{'收敛回合 Convergence':<25} {'TD误差 TD Error':<20}")
        print("-" * 80)
        
        for n, data in sorted(self.results.items()):
            final_return = f"{data['final_return_mean']:.2f} ± {data['final_return_std']:.2f}"
            convergence = f"{data['convergence_mean']:.0f} ± {data['convergence_std']:.0f}"
            td_error = f"{data['td_error_mean']:.4f} ± {data['td_error_std']:.4f}"
            
            print(f"{n:<10} {final_return:<25} {convergence:<25} {td_error:<20}")
        
        # 找最佳n
        # Find best n
        best_n = min(self.results.items(),
                    key=lambda x: x[1]['convergence_mean'])[0]
        
        print(f"\n最快收敛的n值: {best_n}")
        print(f"Fastest converging n: {best_n}")
        
        print("""
        典型观察 Typical Observations:
        - n=1 (TD(0)): 快速但可能不稳定
                       Fast but may be unstable
        - n=2-5: 通常最佳平衡
                Usually best balance
        - n=10+: 接近MC，高方差
                Approaching MC, high variance
        - 最优n取决于具体问题
          Optimal n depends on specific problem
        """)
    
    def plot_comparison(self):
        """
        绘制比较图
        Plot comparison
        """
        if not self.results:
            print("请先运行比较实验")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 设置颜色映射
        # Set color map
        n_values = sorted(self.results.keys())
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(n_values)))
        
        # 图1：学习曲线
        # Plot 1: Learning curves
        ax1 = axes[0, 0]
        for i, n in enumerate(n_values):
            data = self.results[n]
            mean_returns = data['mean_returns']
            episodes = range(len(mean_returns))
            
            ax1.plot(episodes, mean_returns,
                    color=colors[i], label=f'n={n}', linewidth=2)
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Return')
        ax1.set_title('Learning Curves for Different n')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 图2：收敛速度vs n
        # Plot 2: Convergence speed vs n
        ax2 = axes[0, 1]
        
        conv_means = [self.results[n]['convergence_mean'] for n in n_values]
        conv_stds = [self.results[n]['convergence_std'] for n in n_values]
        
        ax2.errorbar(n_values, conv_means, yerr=conv_stds,
                    marker='o', markersize=8, linewidth=2, capsize=5)
        ax2.set_xlabel('n (steps)')
        ax2.set_ylabel('Convergence Episode')
        ax2.set_title('Convergence Speed vs n')
        ax2.grid(True, alpha=0.3)
        
        # 图3：最终性能vs n
        # Plot 3: Final performance vs n
        ax3 = axes[1, 0]
        
        final_means = [self.results[n]['final_return_mean'] for n in n_values]
        final_stds = [self.results[n]['final_return_std'] for n in n_values]
        
        ax3.errorbar(n_values, final_means, yerr=final_stds,
                    marker='s', markersize=8, linewidth=2, capsize=5,
                    color='red')
        ax3.set_xlabel('n (steps)')
        ax3.set_ylabel('Final Return')
        ax3.set_title('Final Performance vs n')
        ax3.grid(True, alpha=0.3)
        
        # 图4：TD误差vs n
        # Plot 4: TD error vs n
        ax4 = axes[1, 1]
        
        td_means = [self.results[n]['td_error_mean'] for n in n_values]
        td_stds = [self.results[n]['td_error_std'] for n in n_values]
        
        ax4.errorbar(n_values, td_means, yerr=td_stds,
                    marker='^', markersize=8, linewidth=2, capsize=5,
                    color='green')
        ax4.set_xlabel('n (steps)')
        ax4.set_ylabel('Final TD Error')
        ax4.set_title('TD Error vs n')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('n-step TD Methods Comparison',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig


# ================================================================================
# 第5.5.4节：树备份(Tree Backup)算法预览
# Section 5.5.4: Tree Backup Algorithm Preview
# ================================================================================

class TreeBackup:
    """
    树备份算法 - Off-Policy n-step方法
    Tree Backup Algorithm - Off-Policy n-step Method
    
    这是n-step Expected SARSA的推广
    This is generalization of n-step Expected SARSA
    
    核心思想：
    Core idea:
    使用期望而不是采样的动作
    Use expectation instead of sampled actions
    
    更新目标：
    Update target:
    考虑所有可能的动作分支，像树一样
    Consider all possible action branches, like a tree
    
    这是通向TD(λ)的桥梁！
    This is bridge to TD(λ)!
    
    注意：这里只是预览，完整实现在高级章节
    Note: This is just preview, full implementation in advanced chapters
    """
    
    @staticmethod
    def explain_concept():
        """
        解释树备份概念
        Explain tree backup concept
        """
        print("\n" + "="*80)
        print("树备份算法概念")
        print("Tree Backup Algorithm Concept")
        print("="*80)
        
        print("""
        树备份算法 Tree Backup:
        ========================
        
        1. 动机 Motivation:
        ------------------
        - n-step SARSA是on-policy
          n-step SARSA is on-policy
        - 我们想要off-policy的n-step方法
          We want off-policy n-step method
        - 解决方案：使用期望而不是采样
          Solution: Use expectation instead of sampling
        
        2. 核心思想 Core Idea:
        ---------------------
        考虑动作树的所有分支：
        Consider all branches of action tree:
        
                    S_t
                 /   |   \\
              A1    A2    A3
              |     |     |
            S'1   S'2   S'3
            ...   ...   ...
        
        使用策略概率加权每个分支
        Weight each branch by policy probability
        
        3. 更新公式 Update Formula:
        --------------------------
        复杂但优雅的递归公式
        Complex but elegant recursive formula
        
        G_{t:t+n} = R_{t+1} + γ Σ_a π(a|S_{t+1}) Q_{t+n-1}(S_{t+1}, a)
                    如果选择的动作不是a
                    if chosen action is not a
        
        4. 优势 Advantages:
        -------------------
        - Off-policy学习
          Off-policy learning
        - 无需重要性采样
          No importance sampling needed
        - 低方差
          Low variance
        
        5. 与其他方法的关系：
        Relation to other methods:
        - n=1时退化为Expected SARSA
          Reduces to Expected SARSA when n=1
        - 是Q(σ)算法的特例
          Special case of Q(σ) algorithm
        - TD(λ)的基础之一
          One foundation of TD(λ)
        """)


# ================================================================================
# 主函数：演示n-step TD
# Main Function: Demonstrate n-step TD
# ================================================================================

def demonstrate_n_step_td():
    """
    演示n-step TD方法
    Demonstrate n-step TD methods
    """
    print("\n" + "="*80)
    print("第5.5节：n-step TD方法")
    print("Section 5.5: n-step TD Methods")
    print("="*80)
    
    from ch02_mdp.gridworld import GridWorld
    from ch02_mdp.policies_and_values import UniformRandomPolicy
    
    # 创建环境
    # Create environment
    env = GridWorld(rows=5, cols=5,
                   start_pos=(0,0),
                   goal_pos=(4,4))
    
    print(f"\n创建5×5 GridWorld")
    print(f"  起点: (0,0)")
    print(f"  终点: (4,4)")
    
    # 1. 测试不同n值的TD预测
    # 1. Test TD prediction with different n
    print("\n" + "="*60)
    print("1. n-step TD预测比较")
    print("1. n-step TD Prediction Comparison")
    print("="*60)
    
    policy = UniformRandomPolicy(env.action_space)
    
    # 比较不同n值
    # Compare different n values
    n_values = [1, 2, 3, 5, 10]
    
    for n in n_values[:3]:  # 演示前3个
        print(f"\n测试 {n}-step TD:")
        n_step_td = NStepTD(env, n=n, gamma=0.9, alpha=0.1)
        V = n_step_td.learn(policy, n_episodes=100, verbose=False)
        
        # 显示部分价值
        # Show some values
        sample_states = env.state_space[:5]
        print("  学习的价值（部分）：")
        for state in sample_states:
            if not state.is_terminal:
                value = V.get_value(state)
                print(f"    V({state.id}) = {value:.3f}")
    
    # 2. n值比较实验
    # 2. n value comparison experiment
    print("\n" + "="*60)
    print("2. 系统比较不同n值")
    print("2. Systematic Comparison of n Values")
    print("="*60)
    
    comparator = NStepComparator(env)
    results = comparator.compare_n_values(
        n_values=[1, 2, 3, 5, 10],
        n_episodes=200,
        n_runs=5,
        gamma=0.9,
        alpha=0.1,
        verbose=True
    )
    
    # 绘制比较图
    # Plot comparison
    fig1 = comparator.plot_comparison()
    
    # 3. n-step SARSA测试
    # 3. n-step SARSA test
    print("\n" + "="*60)
    print("3. n-step SARSA控制")
    print("3. n-step SARSA Control")
    print("="*60)
    
    # 测试3-step SARSA
    # Test 3-step SARSA
    n_step_sarsa = NStepSARSA(env, n=3, gamma=0.99, alpha=0.1, epsilon=0.1)
    Q = n_step_sarsa.learn(n_episodes=100, verbose=True)
    
    # 4. 树备份概念
    # 4. Tree backup concept
    TreeBackup.explain_concept()
    
    # 总结
    # Summary
    print("\n" + "="*80)
    print("n-step TD方法总结")
    print("n-step TD Methods Summary")
    print("="*80)
    
    print("""
    关键要点 Key Takeaways:
    =======================
    
    1. n-step TD统一了TD(0)和MC
       n-step TD unifies TD(0) and MC
       
    2. 偏差-方差权衡：
       Bias-Variance Tradeoff:
       - 小n → 低方差，高偏差
         Small n → Low variance, high bias
       - 大n → 高方差，低偏差
         Large n → High variance, low bias
    
    3. 实践建议：
       Practical Advice:
       - 通常n=3到n=5效果最好
         Usually n=3 to n=5 works best
       - 可以动态调整n
         Can dynamically adjust n
       - 考虑计算和内存成本
         Consider computational and memory cost
    
    4. 现代应用：
       Modern Applications:
       - A3C使用n-step returns
       - Rainbow DQN包含n-step
       - 许多算法的基础组件
         Foundation component of many algorithms
    
    5. 下一步：
       Next Steps:
       - TD(λ)：所有n的加权组合
         TD(λ): Weighted combination of all n
       - 资格迹：高效实现
         Eligibility traces: Efficient implementation
    """)
    
    print("="*80)
    
    plt.show()


if __name__ == "__main__":
    demonstrate_n_step_td()