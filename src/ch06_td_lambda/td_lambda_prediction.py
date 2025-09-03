"""
================================================================================
第6.2节：TD(λ)预测算法
Section 6.2: TD(λ) Prediction Algorithms
================================================================================

TD(λ)预测 = 资格迹 + TD误差
TD(λ) Prediction = Eligibility Traces + TD Error

核心更新规则 Core Update Rule:
δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)    # TD误差 TD error
e_t(s) = γλe_{t-1}(s) + 1(S_t = s)      # 资格迹 Eligibility trace
V(s) ← V(s) + αδ_t e_t(s), ∀s           # 价值更新 Value update

这个简单的规则产生了强大的学习算法！
This simple rule produces a powerful learning algorithm!

TD(λ)的优势 Advantages of TD(λ):
1. 更快的学习速度
   Faster learning speed
2. 更好的信用分配
   Better credit assignment
3. 在线学习能力
   Online learning capability
4. 统一的算法框架
   Unified algorithmic framework

参数选择 Parameter Selection:
- λ = 0: TD(0)，最小方差，可能慢
         Minimum variance, may be slow
- λ = 0.8-0.95: 通常的好选择
                Typically good choice
- λ = 1: MC，无偏但高方差
         Unbiased but high variance

实现细节 Implementation Details:
1. 迹的稀疏表示节省内存
   Sparse representation of traces saves memory
2. 迹截断避免数值问题
   Trace truncation avoids numerical issues
3. 迹重置在新回合开始
   Trace reset at episode start
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union, Set
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
from ch06_td_lambda.eligibility_traces import EligibilityTrace, LambdaReturn

# 设置日志
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第6.2.1节：离线TD(λ)预测
# Section 6.2.1: Offline TD(λ) Prediction
# ================================================================================

class OfflineTDLambda:
    """
    离线TD(λ)预测
    Offline TD(λ) Prediction
    
    使用λ-回报的前向视角算法
    Forward view algorithm using λ-return
    
    算法步骤 Algorithm Steps:
    1. 生成完整回合
       Generate complete episode
    2. 计算每个状态的λ-回报
       Compute λ-return for each state
    3. 批量更新价值函数
       Batch update value function
    
    特点 Characteristics:
    - 理论上清晰
      Theoretically clear
    - 需要存储完整轨迹
      Needs to store complete trajectory
    - 不能在线学习
      Cannot learn online
    - 主要用于理解和分析
      Mainly for understanding and analysis
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 gamma: float = 0.99,
                 lambda_: float = 0.9,
                 alpha: Union[float, Callable] = 0.1):
        """
        初始化离线TD(λ)
        Initialize offline TD(λ)
        
        Args:
            env: 环境
                Environment
            gamma: 折扣因子
                  Discount factor
            lambda_: λ参数，控制n步回报的权重
                    Lambda parameter, controls n-step return weighting
            alpha: 学习率
                  Learning rate
        """
        self.env = env
        self.gamma = gamma
        self.lambda_ = lambda_
        
        # 学习率
        # Learning rate
        if callable(alpha):
            self.alpha_func = alpha
        else:
            self.alpha_func = lambda t: alpha
        
        # 价值函数
        # Value function
        self.V = StateValueFunction(env.state_space, initial_value=0.0)
        
        # 统计
        # Statistics
        self.episode_count = 0
        self.step_count = 0
        self.episode_returns = []
        self.lambda_returns_history = []
        
        logger.info(f"初始化离线TD(λ): γ={gamma}, λ={lambda_}, α={alpha}")
    
    def learn_episode(self, policy: Policy) -> float:
        """
        学习一个回合（离线）
        Learn one episode (offline)
        
        Args:
            policy: 策略
                   Policy
        
        Returns:
            回合回报
            Episode return
        """
        # 生成完整回合
        # Generate complete episode
        states = []
        rewards = []
        
        state = self.env.reset()
        states.append(state)
        
        done = False
        while not done:
            action = policy.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            
            rewards.append(reward)
            states.append(next_state)
            
            state = next_state
        
        # 计算真实回报（用于统计）
        # Compute actual return (for statistics)
        episode_return = sum(r * (self.gamma ** t) for t, r in enumerate(rewards))
        
        # 计算λ-回报
        # Compute λ-returns
        T = len(rewards)
        values = [self.V.get_value(s) for s in states]
        
        lambda_returns = []
        for t in range(T):
            # 计算G_t^λ
            g_lambda = 0.0
            
            # 计算所有n步回报的加权和
            # Compute weighted sum of all n-step returns
            for n in range(1, T - t + 1):
                # n步回报
                # n-step return
                g_n = sum((self.gamma ** k) * rewards[t + k] 
                         for k in range(min(n, T - t)))
                
                # Bootstrap值
                # Bootstrap value
                if t + n < T:
                    g_n += (self.gamma ** n) * values[t + n]
                
                # λ加权
                # λ weighting
                if n < T - t:
                    weight = (1 - self.lambda_) * (self.lambda_ ** (n - 1))
                else:
                    # 最后一项
                    # Last term
                    weight = self.lambda_ ** (T - t - 1)
                
                g_lambda += weight * g_n
            
            lambda_returns.append(g_lambda)
        
        # 更新价值函数
        # Update value function
        alpha = self.alpha_func(self.step_count)
        
        for t in range(T):
            if not states[t].is_terminal:
                old_v = self.V.get_value(states[t])
                new_v = old_v + alpha * (lambda_returns[t] - old_v)
                self.V.set_value(states[t], new_v)
                self.step_count += 1
        
        # 记录统计
        # Record statistics
        self.episode_count += 1
        self.episode_returns.append(episode_return)
        self.lambda_returns_history.append(lambda_returns)
        
        return episode_return


# ================================================================================
# 第6.2.2节：在线TD(λ)预测（资格迹）
# Section 6.2.2: Online TD(λ) Prediction (Eligibility Traces)
# ================================================================================

class OnlineTDLambda:
    """
    在线TD(λ)预测
    Online TD(λ) Prediction
    
    使用资格迹的后向视角算法
    Backward view algorithm using eligibility traces
    
    核心算法 Core Algorithm:
    对每个时间步 For each time step:
    1. 观察转移(S,R,S')
       Observe transition (S,R,S')
    2. 计算TD误差: δ = R + γV(S') - V(S)
       Compute TD error
    3. 更新迹: e(S) ← e(S) + 1
       Update trace
    4. 更新所有状态: V(s) ← V(s) + αδe(s), ∀s
       Update all states
    5. 衰减迹: e(s) ← γλe(s), ∀s
       Decay traces
    
    这是TD(λ)最常用的实现！
    This is the most common implementation of TD(λ)!
    
    优势 Advantages:
    - 真正的在线学习
      True online learning
    - 不需要存储轨迹
      No need to store trajectory
    - 计算效率高（稀疏迹）
      Computationally efficient (sparse traces)
    - 实时更新
      Real-time updates
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 gamma: float = 0.99,
                 lambda_: float = 0.9,
                 alpha: Union[float, Callable] = 0.1,
                 trace_type: str = "accumulating",
                 trace_threshold: float = 1e-4):
        """
        初始化在线TD(λ)
        Initialize online TD(λ)
        
        Args:
            env: 环境
                Environment
            gamma: 折扣因子
                  Discount factor
            lambda_: λ参数
                    Lambda parameter
            alpha: 学习率
                  Learning rate
            trace_type: 迹类型 ("accumulating", "replacing", "dutch")
                       Trace type
            trace_threshold: 迹截断阈值
                           Trace truncation threshold
        """
        self.env = env
        self.gamma = gamma
        self.lambda_ = lambda_
        self.trace_type = trace_type
        self.trace_threshold = trace_threshold
        
        # 学习率
        # Learning rate
        if callable(alpha):
            self.alpha_func = alpha
        else:
            self.alpha_func = lambda t: alpha
        
        # 价值函数
        # Value function
        self.V = StateValueFunction(env.state_space, initial_value=0.0)
        
        # 资格迹（稀疏表示）
        # Eligibility traces (sparse representation)
        self.traces = {}
        
        # TD误差分析
        # TD error analysis
        self.td_analyzer = TDErrorAnalyzer()
        
        # 统计
        # Statistics
        self.episode_count = 0
        self.step_count = 0
        self.episode_returns = []
        self.active_traces_history = []
        self.max_traces_per_episode = []
        
        logger.info(f"初始化在线TD(λ): γ={gamma}, λ={lambda_}, "
                   f"trace_type={trace_type}")
    
    def reset_traces(self):
        """
        重置资格迹
        Reset eligibility traces
        
        新回合开始时调用
        Called at start of new episode
        """
        self.traces.clear()
    
    def update_traces(self, state: State, alpha: float = None):
        """
        更新资格迹
        Update eligibility traces
        
        Args:
            state: 当前状态
                  Current state
            alpha: 学习率（Dutch迹需要）
                  Learning rate (needed for Dutch traces)
        """
        # 先衰减所有迹
        # First decay all traces
        decay = self.gamma * self.lambda_
        
        # 使用列表避免运行时修改
        # Use list to avoid runtime modification
        for s in list(self.traces.keys()):
            self.traces[s] *= decay
            
            # 截断小迹
            # Truncate small traces
            if self.traces[s] < self.trace_threshold:
                del self.traces[s]
        
        # 更新当前状态的迹
        # Update trace for current state
        if self.trace_type == "accumulating":
            # 累积迹
            # Accumulating trace
            self.traces[state] = self.traces.get(state, 0.0) + 1.0
            
        elif self.trace_type == "replacing":
            # 替换迹
            # Replacing trace
            self.traces[state] = 1.0
            
        elif self.trace_type == "dutch":
            # Dutch迹
            # Dutch trace
            if alpha is not None:
                old_trace = self.traces.get(state, 0.0)
                self.traces[state] = old_trace + alpha * (1.0 - old_trace)
            else:
                self.traces[state] = 1.0
    
    def learn_episode(self, policy: Policy) -> float:
        """
        学习一个回合（在线）
        Learn one episode (online)
        
        每步实时更新
        Real-time update every step
        
        Args:
            policy: 策略
                   Policy
        
        Returns:
            回合回报
            Episode return
        """
        # 重置迹
        # Reset traces
        self.reset_traces()
        
        # 初始化
        # Initialize
        state = self.env.reset()
        episode_return = 0.0
        episode_steps = 0
        max_active_traces = 0
        
        while True:
            # 选择动作
            # Select action
            action = policy.select_action(state)
            
            # 执行动作
            # Execute action
            next_state, reward, done, _ = self.env.step(action)
            
            # 计算TD误差
            # Compute TD error
            v_current = self.V.get_value(state)
            v_next = self.V.get_value(next_state) if not done else 0.0
            td_error = reward + self.gamma * v_next - v_current
            
            # 获取学习率
            # Get learning rate
            alpha = self.alpha_func(self.step_count)
            
            # 更新迹（先更新当前状态）
            # Update traces (first update current state)
            if self.trace_type == "dutch":
                self.update_traces(state, alpha)
            else:
                self.update_traces(state)
            
            # 更新所有有迹的状态
            # Update all states with traces
            for s, e in list(self.traces.items()):
                if not s.is_terminal:
                    old_v = self.V.get_value(s)
                    new_v = old_v + alpha * td_error * e
                    self.V.set_value(s, new_v)
            
            # 记录TD误差
            # Record TD error
            td_err_obj = TDError(
                value=td_error,
                timestep=self.step_count,
                state=state,
                next_state=next_state,
                reward=reward,
                state_value=v_current,
                next_state_value=v_next
            )
            self.td_analyzer.add_error(td_err_obj)
            
            # 更新统计
            # Update statistics
            episode_return += reward * (self.gamma ** episode_steps)
            episode_steps += 1
            self.step_count += 1
            max_active_traces = max(max_active_traces, len(self.traces))
            
            # 转移状态
            # Transition state
            state = next_state
            
            if done:
                break
        
        # 记录统计
        # Record statistics
        self.episode_count += 1
        self.episode_returns.append(episode_return)
        self.max_traces_per_episode.append(max_active_traces)
        self.active_traces_history.append(len(self.traces))
        
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
            print(f"\n开始在线TD(λ)学习: {n_episodes}回合")
            print(f"Starting online TD(λ) learning: {n_episodes} episodes")
            print(f"  参数: γ={self.gamma}, λ={self.lambda_}")
            print(f"  迹类型: {self.trace_type}")
        
        for episode in range(n_episodes):
            episode_return = self.learn_episode(policy)
            
            if verbose and (episode + 1) % max(1, n_episodes // 10) == 0:
                avg_return = np.mean(self.episode_returns[-100:]) \
                           if len(self.episode_returns) >= 100 \
                           else np.mean(self.episode_returns)
                
                stats = self.td_analyzer.get_statistics()
                avg_traces = np.mean(self.max_traces_per_episode[-100:]) \
                           if len(self.max_traces_per_episode) >= 100 \
                           else np.mean(self.max_traces_per_episode)
                
                print(f"Episode {episode + 1}/{n_episodes}: "
                      f"Return={episode_return:.2f}, "
                      f"Avg Return={avg_return:.2f}, "
                      f"TD Error={stats.get('recent_abs_mean', 0):.4f}, "
                      f"Avg Traces={avg_traces:.1f}")
        
        if verbose:
            print(f"\n学习完成!")
            print(f"  最终平均回报: {np.mean(self.episode_returns[-100:]):.2f}")
            print(f"  平均活跃迹数: {np.mean(self.max_traces_per_episode):.1f}")
        
        return self.V


# ================================================================================
# 第6.2.3节：TD(λ)与TD(0)、MC的比较
# Section 6.2.3: Comparison of TD(λ) with TD(0) and MC
# ================================================================================

class TDLambdaComparator:
    """
    TD(λ)比较器
    TD(λ) Comparator
    
    系统比较不同λ值的效果
    Systematically compare effects of different λ values
    
    关键观察 Key Observations:
    1. λ=0 → TD(0): 快速但可能不准
       Fast but may be inaccurate
    2. λ=1 → MC: 准确但慢
       Accurate but slow
    3. 中间值通常最优
       Intermediate values often optimal
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
        
        logger.info("初始化TD(λ)比较器")
    
    def compare_lambda_values(self,
                             lambda_values: List[float] = [0.0, 0.5, 0.9, 0.95, 1.0],
                             n_episodes: int = 500,
                             n_runs: int = 10,
                             gamma: float = 0.99,
                             alpha: float = 0.1,
                             verbose: bool = True) -> Dict[str, Any]:
        """
        比较不同λ值
        Compare different λ values
        
        Args:
            lambda_values: 要比较的λ值
                         Lambda values to compare
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
            print("TD(λ)参数比较实验")
            print("TD(λ) Parameter Comparison Experiment")
            print("="*80)
            print(f"比较λ值: {lambda_values}")
            print(f"参数: γ={gamma}, α={alpha}")
            print(f"实验: {n_episodes}回合 × {n_runs}次运行")
        
        from ch02_mdp.policies_and_values import UniformRandomPolicy
        policy = UniformRandomPolicy(self.env.action_space)
        
        # 存储结果
        # Store results
        results = {lam: {
            'returns': [],
            'final_returns': [],
            'convergence_episodes': [],
            'final_td_errors': [],
            'avg_traces': []
        } for lam in lambda_values}
        
        # 运行实验
        # Run experiments
        for run in range(n_runs):
            if verbose:
                print(f"\n运行 {run + 1}/{n_runs}:")
            
            for lam in lambda_values:
                # 创建TD(λ)算法
                # Create TD(λ) algorithm
                td_lambda = OnlineTDLambda(
                    self.env,
                    gamma=gamma,
                    lambda_=lam,
                    alpha=alpha
                )
                
                # 学习
                # Learn
                for episode in range(n_episodes):
                    episode_return = td_lambda.learn_episode(policy)
                
                # 记录结果
                # Record results
                results[lam]['returns'].append(td_lambda.episode_returns)
                results[lam]['final_returns'].append(td_lambda.episode_returns[-1])
                
                # 计算收敛速度
                # Compute convergence speed
                convergence_ep = self._find_convergence_episode(td_lambda.episode_returns)
                results[lam]['convergence_episodes'].append(convergence_ep)
                
                # TD误差
                # TD error
                stats = td_lambda.td_analyzer.get_statistics()
                results[lam]['final_td_errors'].append(stats.get('recent_abs_mean', 0))
                
                # 平均迹数
                # Average traces
                avg_traces = np.mean(td_lambda.max_traces_per_episode) \
                           if td_lambda.max_traces_per_episode else 0
                results[lam]['avg_traces'].append(avg_traces)
                
                if verbose:
                    print(f"  λ={lam}: 最终回报={td_lambda.episode_returns[-1]:.2f}, "
                          f"收敛={convergence_ep}, 平均迹={avg_traces:.1f}")
        
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
        
        for lam, data in results.items():
            returns_array = np.array(data['returns'])
            
            analyzed[lam] = {
                'mean_returns': np.mean(returns_array, axis=0),
                'std_returns': np.std(returns_array, axis=0),
                'final_return_mean': np.mean(data['final_returns']),
                'final_return_std': np.std(data['final_returns']),
                'convergence_mean': np.mean(data['convergence_episodes']),
                'convergence_std': np.std(data['convergence_episodes']),
                'td_error_mean': np.mean(data['final_td_errors']),
                'td_error_std': np.std(data['final_td_errors']),
                'avg_traces_mean': np.mean(data['avg_traces']),
                'avg_traces_std': np.std(data['avg_traces'])
            }
        
        return analyzed
    
    def _print_comparison_summary(self):
        """
        打印比较摘要
        Print comparison summary
        """
        print("\n" + "="*80)
        print("TD(λ)比较结果摘要")
        print("TD(λ) Comparison Results Summary")
        print("="*80)
        
        print(f"\n{'λ值':<10} {'最终回报':<20} {'收敛回合':<20} "
              f"{'TD误差':<15} {'平均迹数':<15}")
        print("-" * 80)
        
        for lam, data in sorted(self.results.items()):
            final_return = f"{data['final_return_mean']:.2f} ± {data['final_return_std']:.2f}"
            convergence = f"{data['convergence_mean']:.0f} ± {data['convergence_std']:.0f}"
            td_error = f"{data['td_error_mean']:.4f}"
            avg_traces = f"{data['avg_traces_mean']:.1f}"
            
            # 特殊标记
            # Special marks
            if lam == 0.0:
                lam_str = f"{lam:.1f} (TD)"
            elif lam == 1.0:
                lam_str = f"{lam:.1f} (MC)"
            else:
                lam_str = f"{lam:.1f}"
            
            print(f"{lam_str:<10} {final_return:<20} {convergence:<20} "
                  f"{td_error:<15} {avg_traces:<15}")
        
        # 找最佳λ
        # Find best λ
        best_convergence = min(self.results.items(),
                              key=lambda x: x[1]['convergence_mean'])
        best_return = max(self.results.items(),
                         key=lambda x: x[1]['final_return_mean'])
        
        print(f"\n最快收敛: λ={best_convergence[0]}")
        print(f"最高回报: λ={best_return[0]}")
        
        print("""
        典型观察 Typical Observations:
        - λ=0 (TD): 快速收敛，低计算成本
                   Fast convergence, low computational cost
        - λ=0.8-0.95: 通常最佳平衡
                     Usually best balance
        - λ=1 (MC): 准确但慢，需要更多迹
                   Accurate but slow, needs more traces
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
        
        lambda_values = sorted(self.results.keys())
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(lambda_values)))
        
        # 图1：学习曲线
        # Plot 1: Learning curves
        ax1 = axes[0, 0]
        for lam, color in zip(lambda_values, colors):
            data = self.results[lam]
            mean_returns = data['mean_returns']
            std_returns = data['std_returns']
            episodes = range(len(mean_returns))
            
            label = f'λ={lam}'
            if lam == 0.0:
                label += ' (TD)'
            elif lam == 1.0:
                label += ' (MC)'
            
            ax1.plot(episodes, mean_returns, color=color,
                    label=label, linewidth=2)
            ax1.fill_between(episodes,
                            mean_returns - std_returns,
                            mean_returns + std_returns,
                            color=color, alpha=0.1)
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Return')
        ax1.set_title('Learning Curves for Different λ')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 图2：收敛速度
        # Plot 2: Convergence speed
        ax2 = axes[0, 1]
        
        conv_means = [self.results[lam]['convergence_mean'] for lam in lambda_values]
        conv_stds = [self.results[lam]['convergence_std'] for lam in lambda_values]
        
        ax2.errorbar(lambda_values, conv_means, yerr=conv_stds,
                    marker='o', markersize=8, linewidth=2, capsize=5)
        ax2.set_xlabel('λ')
        ax2.set_ylabel('Convergence Episode')
        ax2.set_title('Convergence Speed vs λ')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(lambda_values)
        
        # 图3：最终性能
        # Plot 3: Final performance
        ax3 = axes[1, 0]
        
        final_means = [self.results[lam]['final_return_mean'] for lam in lambda_values]
        final_stds = [self.results[lam]['final_return_std'] for lam in lambda_values]
        
        ax3.bar(range(len(lambda_values)), final_means,
               yerr=final_stds, capsize=5)
        ax3.set_xticks(range(len(lambda_values)))
        ax3.set_xticklabels([f'{lam:.1f}' for lam in lambda_values])
        ax3.set_xlabel('λ')
        ax3.set_ylabel('Final Return')
        ax3.set_title('Final Performance vs λ')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 图4：计算成本（平均迹数）
        # Plot 4: Computational cost (average traces)
        ax4 = axes[1, 1]
        
        trace_means = [self.results[lam]['avg_traces_mean'] for lam in lambda_values]
        trace_stds = [self.results[lam]['avg_traces_std'] for lam in lambda_values]
        
        ax4.errorbar(lambda_values, trace_means, yerr=trace_stds,
                    marker='s', markersize=8, linewidth=2, capsize=5,
                    color='green')
        ax4.set_xlabel('λ')
        ax4.set_ylabel('Average Active Traces')
        ax4.set_title('Computational Cost vs λ')
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(lambda_values)
        
        plt.suptitle('TD(λ) Parameter Comparison',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig


# ================================================================================
# 主函数：演示TD(λ)预测
# Main Function: Demonstrate TD(λ) Prediction
# ================================================================================

def demonstrate_td_lambda_prediction():
    """
    演示TD(λ)预测算法
    Demonstrate TD(λ) prediction algorithms
    """
    print("\n" + "="*80)
    print("第6.2节：TD(λ)预测算法")
    print("Section 6.2: TD(λ) Prediction Algorithms")
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
    
    # 1. 离线TD(λ)测试
    # 1. Offline TD(λ) test
    print("\n" + "="*60)
    print("1. 离线TD(λ)预测")
    print("1. Offline TD(λ) Prediction")
    print("="*60)
    
    offline_td = OfflineTDLambda(env, gamma=0.9, lambda_=0.8, alpha=0.1)
    
    print("\n学习100回合...")
    for episode in range(100):
        episode_return = offline_td.learn_episode(policy)
        if (episode + 1) % 20 == 0:
            print(f"  Episode {episode + 1}: Return = {episode_return:.2f}")
    
    print("\n学习的价值（部分）:")
    for i in range(min(5, len(env.state_space))):
        state = env.state_space[i]
        if not state.is_terminal:
            value = offline_td.V.get_value(state)
            print(f"  V({state.id}) = {value:.3f}")
    
    # 2. 在线TD(λ)测试
    # 2. Online TD(λ) test
    print("\n" + "="*60)
    print("2. 在线TD(λ)预测（资格迹）")
    print("2. Online TD(λ) Prediction (Eligibility Traces)")
    print("="*60)
    
    # 测试不同迹类型
    # Test different trace types
    trace_types = ["accumulating", "replacing", "dutch"]
    
    for trace_type in trace_types:
        print(f"\n测试{trace_type}迹:")
        online_td = OnlineTDLambda(
            env, gamma=0.9, lambda_=0.8, alpha=0.1,
            trace_type=trace_type
        )
        
        V = online_td.learn(policy, n_episodes=100, verbose=False)
        
        # 显示结果
        # Show results
        avg_return = np.mean(online_td.episode_returns[-20:])
        avg_traces = np.mean(online_td.max_traces_per_episode)
        
        print(f"  平均回报: {avg_return:.2f}")
        print(f"  平均活跃迹: {avg_traces:.1f}")
        
        # 显示一些价值
        # Show some values
        sample_state = env.state_space[0]
        if not sample_state.is_terminal:
            value = V.get_value(sample_state)
            print(f"  V(s0) = {value:.3f}")
    
    # 3. λ参数比较
    # 3. Lambda parameter comparison
    print("\n" + "="*60)
    print("3. λ参数系统比较")
    print("3. Systematic Lambda Parameter Comparison")
    print("="*60)
    
    comparator = TDLambdaComparator(env)
    results = comparator.compare_lambda_values(
        lambda_values=[0.0, 0.5, 0.8, 0.9, 0.95, 1.0],
        n_episodes=200,
        n_runs=5,
        gamma=0.9,
        alpha=0.1,
        verbose=True
    )
    
    # 绘图
    # Plot
    fig = comparator.plot_comparison()
    
    # 总结
    # Summary
    print("\n" + "="*80)
    print("TD(λ)预测算法总结")
    print("TD(λ) Prediction Algorithm Summary")
    print("="*80)
    
    print("""
    关键要点 Key Takeaways:
    =======================
    
    1. TD(λ)统一了TD和MC
       TD(λ) unifies TD and MC
       
    2. λ控制偏差-方差权衡
       λ controls bias-variance tradeoff
       
    3. 资格迹实现高效在线学习
       Eligibility traces enable efficient online learning
       
    4. 中间λ值通常最优（0.8-0.95）
       Intermediate λ values often optimal (0.8-0.95)
       
    5. 不同迹类型适合不同场景
       Different trace types suit different scenarios
       
    6. 稀疏迹表示节省计算
       Sparse trace representation saves computation
    """)
    
    plt.show()


if __name__ == "__main__":
    demonstrate_td_lambda_prediction()