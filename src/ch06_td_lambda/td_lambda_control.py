"""
================================================================================
第6.3节：TD(λ)控制算法 - SARSA(λ)和Q(λ)
Section 6.3: TD(λ) Control Algorithms - SARSA(λ) and Q(λ)
================================================================================

将资格迹扩展到控制问题！
Extending eligibility traces to control problems!

核心思想 Core Idea:
从状态迹e(s)扩展到状态-动作迹e(s,a)
Extend from state traces e(s) to state-action traces e(s,a)

SARSA(λ)更新规则 Update Rule:
δ_t = R_{t+1} + γQ(S_{t+1}, A_{t+1}) - Q(S_t, A_t)
e_t(s,a) = γλe_{t-1}(s,a) + 1(S_t=s, A_t=a)
Q(s,a) ← Q(s,a) + αδ_t e_t(s,a), ∀(s,a)

Q(λ)的挑战 Challenges of Q(λ):
Off-policy学习需要重要性采样修正！
Off-policy learning needs importance sampling corrections!

Watkins's Q(λ): 探索时截断迹
                Truncate traces on exploration
Peng's Q(λ): 使用重要性采样
            Use importance sampling

实践建议 Practical Advice:
1. SARSA(λ)通常比Q(λ)稳定
   SARSA(λ) usually more stable than Q(λ)
2. λ=0.8-0.95效果好
   λ=0.8-0.95 works well
3. 替换迹在控制中常用
   Replacing traces common in control
4. 注意迹的内存管理
   Mind trace memory management
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
from ch04_monte_carlo.mc_control import EpsilonGreedyPolicy
from ch05_temporal_difference.td_foundations import TDError, TDErrorAnalyzer
from ch06_td_lambda.eligibility_traces import EligibilityTrace

# 设置日志
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第6.3.1节：SARSA(λ)算法
# Section 6.3.1: SARSA(λ) Algorithm
# ================================================================================

class SARSALambda:
    """
    SARSA(λ)算法 - On-Policy TD控制with资格迹
    SARSA(λ) Algorithm - On-Policy TD Control with Eligibility Traces
    
    SARSA(λ)是SARSA的自然扩展
    SARSA(λ) is natural extension of SARSA
    
    算法步骤 Algorithm Steps:
    1. 初始化Q(s,a)任意，Q(terminal,·)=0
       Initialize Q(s,a) arbitrarily, Q(terminal,·)=0
    2. 重复每个回合：
       Repeat for each episode:
       a. 初始化S，选择A（ε-贪婪）
          Initialize S, choose A (ε-greedy)
       b. 初始化迹e(s,a)=0, ∀(s,a)
          Initialize traces e(s,a)=0
       c. 重复每步：
          Repeat for each step:
          - 执行A，观察R,S'
            Take A, observe R,S'
          - 选择A'（ε-贪婪）
            Choose A' (ε-greedy)
          - δ = R + γQ(S',A') - Q(S,A)
          - e(S,A) = e(S,A) + 1（或=1替换迹）
          - 对所有(s,a):
            For all (s,a):
            Q(s,a) = Q(s,a) + αδe(s,a)
            e(s,a) = γλe(s,a)
          - S = S', A = A'
       直到S是终止状态
       Until S is terminal
    
    优势 Advantages:
    1. 比SARSA更快的信用分配
       Faster credit assignment than SARSA
    2. 在线学习
       Online learning
    3. 理论保证（on-policy）
       Theoretical guarantees (on-policy)
    
    应用 Applications:
    - 机器人控制
      Robot control
    - 游戏AI
      Game AI
    - 实时决策系统
      Real-time decision systems
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 gamma: float = 0.99,
                 lambda_: float = 0.9,
                 alpha: Union[float, Callable] = 0.1,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 trace_type: str = "replacing",
                 trace_threshold: float = 1e-4):
        """
        初始化SARSA(λ)
        Initialize SARSA(λ)
        
        Args:
            env: 环境
                Environment
            gamma: 折扣因子
                  Discount factor
            lambda_: λ参数
                    Lambda parameter
            alpha: 学习率
                  Learning rate
            epsilon: 探索率
                    Exploration rate
            epsilon_decay: ε衰减率
                         ε decay rate
            epsilon_min: 最小ε
                        Minimum ε
            trace_type: 迹类型
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
        
        # 资格迹（稀疏表示）
        # Eligibility traces (sparse representation)
        # Key: (state, action) tuple
        self.traces = {}
        
        # TD误差分析
        # TD error analysis
        self.td_analyzer = TDErrorAnalyzer()
        
        # 统计
        # Statistics
        self.episode_count = 0
        self.step_count = 0
        self.episode_returns = []
        self.episode_lengths = []
        self.max_traces_per_episode = []
        
        logger.info(f"初始化SARSA(λ): γ={gamma}, λ={lambda_}, "
                   f"ε={epsilon}, trace_type={trace_type}")
    
    def reset_traces(self):
        """
        重置资格迹
        Reset eligibility traces
        """
        self.traces.clear()
    
    def update_trace(self, state: State, action: Action):
        """
        更新资格迹
        Update eligibility trace
        
        Args:
            state: 状态
                  State
            action: 动作
                   Action
        """
        key = (state, action)
        
        if self.trace_type == "accumulating":
            # 累积迹
            # Accumulating trace
            self.traces[key] = self.traces.get(key, 0.0) + 1.0
            
        elif self.trace_type == "replacing":
            # 替换迹（控制中更常用）
            # Replacing trace (more common in control)
            self.traces[key] = 1.0
            
        elif self.trace_type == "dutch":
            # Dutch迹
            # Dutch trace
            alpha = self.alpha_func(self.step_count)
            old_trace = self.traces.get(key, 0.0)
            self.traces[key] = old_trace + alpha * (1.0 - old_trace)
    
    def decay_traces(self):
        """
        衰减所有迹
        Decay all traces
        """
        decay_factor = self.gamma * self.lambda_
        
        # 使用列表复制避免运行时修改
        # Use list copy to avoid runtime modification
        for key in list(self.traces.keys()):
            self.traces[key] *= decay_factor
            
            # 截断小迹
            # Truncate small traces
            if self.traces[key] < self.trace_threshold:
                del self.traces[key]
    
    def learn_episode(self) -> Tuple[float, int]:
        """
        学习一个回合
        Learn one episode
        
        Returns:
            (回合回报, 回合长度)
            (episode return, episode length)
        """
        # 重置迹
        # Reset traces
        self.reset_traces()
        
        # 初始化S
        # Initialize S
        state = self.env.reset()
        
        # 选择A（从S使用策略）
        # Choose A (from S using policy)
        action = self.policy.select_action(state)
        
        episode_return = 0.0
        episode_length = 0
        max_active_traces = 0
        
        while True:
            # 执行动作A，观察R和S'
            # Take action A, observe R and S'
            next_state, reward, done, _ = self.env.step(action)
            
            # 选择A'（从S'使用策略）
            # Choose A' (from S' using policy)
            if not done:
                next_action = self.policy.select_action(next_state)
            else:
                next_action = None
            
            # 计算TD误差
            # Compute TD error
            q_current = self.Q.get_value(state, action)
            if done:
                td_target = reward
                q_next = 0.0
            else:
                q_next = self.Q.get_value(next_state, next_action)
                td_target = reward + self.gamma * q_next
            
            td_error = td_target - q_current
            
            # 更新当前(s,a)的迹
            # Update trace for current (s,a)
            self.update_trace(state, action)
            
            # 获取学习率
            # Get learning rate
            alpha = self.alpha_func(self.step_count)
            
            # 更新所有有迹的(s,a)对
            # Update all (s,a) pairs with traces
            for (s, a), e in list(self.traces.items()):
                if not s.is_terminal:
                    old_q = self.Q.get_value(s, a)
                    new_q = old_q + alpha * td_error * e
                    self.Q.set_value(s, a, new_q)
            
            # 衰减所有迹
            # Decay all traces
            self.decay_traces()
            
            # 记录TD误差
            # Record TD error
            td_err_obj = TDError(
                value=td_error,
                timestep=self.step_count,
                state=state,
                next_state=next_state,
                reward=reward,
                state_value=q_current,
                next_state_value=q_next
            )
            self.td_analyzer.add_error(td_err_obj)
            
            # 累积回报和统计
            # Accumulate return and statistics
            episode_return += reward * (self.gamma ** episode_length)
            episode_length += 1
            self.step_count += 1
            max_active_traces = max(max_active_traces, len(self.traces))
            
            # S ← S'; A ← A'
            state = next_state
            action = next_action
            
            if done:
                break
        
        # 衰减ε
        # Decay ε
        self.policy.decay_epsilon()
        
        # 记录统计
        # Record statistics
        self.episode_count += 1
        self.episode_returns.append(episode_return)
        self.episode_lengths.append(episode_length)
        self.max_traces_per_episode.append(max_active_traces)
        
        return episode_return, episode_length
    
    def learn(self,
             n_episodes: int = 1000,
             verbose: bool = True) -> ActionValueFunction:
        """
        学习Q函数
        Learn Q function
        
        Args:
            n_episodes: 回合数
                       Number of episodes
            verbose: 是否输出进度
                    Whether to output progress
        
        Returns:
            学习的Q函数
            Learned Q function
        """
        if verbose:
            print(f"\n开始SARSA(λ)学习: {n_episodes}回合")
            print(f"Starting SARSA(λ) learning: {n_episodes} episodes")
            print(f"  参数: γ={self.gamma}, λ={self.lambda_}")
            print(f"  初始ε: {self.policy.epsilon:.3f}")
            print(f"  迹类型: {self.trace_type}")
        
        for episode in range(n_episodes):
            episode_return, episode_length = self.learn_episode()
            
            if verbose and (episode + 1) % max(1, n_episodes // 10) == 0:
                avg_return = np.mean(self.episode_returns[-100:]) \
                           if len(self.episode_returns) >= 100 \
                           else np.mean(self.episode_returns)
                avg_length = np.mean(self.episode_lengths[-100:]) \
                           if len(self.episode_lengths) >= 100 \
                           else np.mean(self.episode_lengths)
                avg_traces = np.mean(self.max_traces_per_episode[-100:]) \
                           if len(self.max_traces_per_episode) >= 100 \
                           else np.mean(self.max_traces_per_episode)
                
                stats = self.td_analyzer.get_statistics()
                
                print(f"Episode {episode + 1}/{n_episodes}: "
                      f"Return={episode_return:.2f}, "
                      f"Avg Return={avg_return:.2f}, "
                      f"Avg Length={avg_length:.1f}, "
                      f"ε={self.policy.epsilon:.3f}, "
                      f"Avg Traces={avg_traces:.1f}")
        
        if verbose:
            print(f"\nSARSA(λ)学习完成!")
            print(f"  最终ε: {self.policy.epsilon:.3f}")
            print(f"  总步数: {self.step_count}")
            print(f"  平均活跃迹: {np.mean(self.max_traces_per_episode):.1f}")
        
        return self.Q


# ================================================================================
# 第6.3.2节：Watkins's Q(λ)算法
# Section 6.3.2: Watkins's Q(λ) Algorithm
# ================================================================================

class WatkinsQLambda:
    """
    Watkins's Q(λ)算法 - Off-Policy TD控制with截断迹
    Watkins's Q(λ) Algorithm - Off-Policy TD Control with Truncated Traces
    
    关键创新 Key Innovation:
    当执行探索动作时截断迹！
    Truncate traces when taking exploratory actions!
    
    原理 Rationale:
    - Q-learning学习贪婪策略
      Q-learning learns greedy policy
    - 探索动作不属于目标策略
      Exploratory actions not from target policy
    - 截断迹避免错误的信用分配
      Truncating traces avoids incorrect credit assignment
    
    更新规则 Update Rule:
    δ = R + γ max_a Q(S',a) - Q(S,A)
    
    如果A是贪婪动作 If A is greedy action:
      e(S,A) = e(S,A) + 1
      对所有(s,a): Q(s,a) = Q(s,a) + αδe(s,a)
                  e(s,a) = γλe(s,a)
    否则 Otherwise:
      对所有(s,a): Q(s,a) = Q(s,a) + αδ1(s=S,a=A)
                  e(s,a) = 0（截断）
    
    特点 Characteristics:
    - 保持Q-learning的off-policy性质
      Maintains off-policy nature of Q-learning
    - 迹只在贪婪路径上传播
      Traces only propagate on greedy path
    - 收敛到Q*（在某些条件下）
      Converges to Q* (under conditions)
    
    局限 Limitations:
    - 探索时迹被截断，学习可能慢
      Traces truncated on exploration, may learn slowly
    - 不充分利用所有经验
      Doesn't fully utilize all experience
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 gamma: float = 0.99,
                 lambda_: float = 0.9,
                 alpha: Union[float, Callable] = 0.1,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 trace_threshold: float = 1e-4):
        """
        初始化Watkins's Q(λ)
        Initialize Watkins's Q(λ)
        """
        self.env = env
        self.gamma = gamma
        self.lambda_ = lambda_
        self.trace_threshold = trace_threshold
        
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
        
        # ε-贪婪策略（行为策略）
        # ε-greedy policy (behavior policy)
        self.behavior_policy = EpsilonGreedyPolicy(
            self.Q,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            action_space=env.action_space
        )
        
        # 资格迹
        # Eligibility traces
        self.traces = {}
        
        # TD误差分析
        # TD error analysis
        self.td_analyzer = TDErrorAnalyzer()
        
        # 统计
        # Statistics
        self.episode_count = 0
        self.step_count = 0
        self.episode_returns = []
        self.episode_lengths = []
        self.greedy_steps = []  # 贪婪步数
        self.trace_truncations = []  # 迹截断次数
        
        logger.info(f"初始化Watkins's Q(λ): γ={gamma}, λ={lambda_}, ε={epsilon}")
    
    def reset_traces(self):
        """
        重置资格迹
        Reset eligibility traces
        """
        self.traces.clear()
    
    def truncate_traces(self):
        """
        截断所有迹（探索时调用）
        Truncate all traces (called on exploration)
        """
        self.traces.clear()
    
    def is_greedy_action(self, state: State, action: Action) -> bool:
        """
        检查动作是否贪婪
        Check if action is greedy
        
        Args:
            state: 状态
                  State
            action: 动作
                   Action
        
        Returns:
            是否贪婪动作
            Whether greedy action
        """
        # 获取所有Q值
        # Get all Q values
        q_values = [self.Q.get_value(state, a) for a in self.env.action_space]
        max_q = max(q_values)
        
        # 检查是否最大Q值动作
        # Check if max Q value action
        action_q = self.Q.get_value(state, action)
        
        # 考虑数值误差
        # Consider numerical error
        return abs(action_q - max_q) < 1e-10
    
    def learn_episode(self) -> Tuple[float, int]:
        """
        学习一个回合
        Learn one episode
        
        Returns:
            (回合回报, 回合长度)
            (episode return, episode length)
        """
        # 重置迹
        # Reset traces
        self.reset_traces()
        
        # 初始化
        # Initialize
        state = self.env.reset()
        
        episode_return = 0.0
        episode_length = 0
        greedy_count = 0
        truncation_count = 0
        
        while True:
            # 选择动作（ε-贪婪）
            # Choose action (ε-greedy)
            action = self.behavior_policy.select_action(state)
            
            # 检查是否贪婪动作
            # Check if greedy action
            is_greedy = self.is_greedy_action(state, action)
            if is_greedy:
                greedy_count += 1
            
            # 执行动作
            # Execute action
            next_state, reward, done, _ = self.env.step(action)
            
            # 计算TD误差（Q-learning风格）
            # Compute TD error (Q-learning style)
            q_current = self.Q.get_value(state, action)
            
            if done:
                td_target = reward
                max_q_next = 0.0
            else:
                # 找最大Q值
                # Find max Q value
                q_values_next = [self.Q.get_value(next_state, a)
                               for a in self.env.action_space]
                max_q_next = max(q_values_next)
                td_target = reward + self.gamma * max_q_next
            
            td_error = td_target - q_current
            
            # 获取学习率
            # Get learning rate
            alpha = self.alpha_func(self.step_count)
            
            if is_greedy:
                # 贪婪动作：正常TD(λ)更新
                # Greedy action: normal TD(λ) update
                
                # 更新当前(s,a)的迹
                # Update trace for current (s,a)
                key = (state, action)
                self.traces[key] = self.traces.get(key, 0.0) + 1.0
                
                # 更新所有有迹的(s,a)
                # Update all (s,a) with traces
                for (s, a), e in list(self.traces.items()):
                    if not s.is_terminal:
                        old_q = self.Q.get_value(s, a)
                        new_q = old_q + alpha * td_error * e
                        self.Q.set_value(s, a, new_q)
                
                # 衰减迹
                # Decay traces
                decay_factor = self.gamma * self.lambda_
                for key in list(self.traces.keys()):
                    self.traces[key] *= decay_factor
                    if self.traces[key] < self.trace_threshold:
                        del self.traces[key]
            
            else:
                # 探索动作：只更新当前(s,a)，截断迹
                # Exploratory action: only update current (s,a), truncate traces
                
                # 只更新当前(s,a)
                # Only update current (s,a)
                old_q = self.Q.get_value(state, action)
                new_q = old_q + alpha * td_error
                self.Q.set_value(state, action, new_q)
                
                # 截断所有迹
                # Truncate all traces
                self.truncate_traces()
                truncation_count += 1
            
            # 记录TD误差
            # Record TD error
            td_err_obj = TDError(
                value=td_error,
                timestep=self.step_count,
                state=state,
                next_state=next_state,
                reward=reward,
                state_value=q_current,
                next_state_value=max_q_next
            )
            self.td_analyzer.add_error(td_err_obj)
            
            # 累积回报
            # Accumulate return
            episode_return += reward * (self.gamma ** episode_length)
            episode_length += 1
            self.step_count += 1
            
            # 转移状态
            # Transition state
            state = next_state
            
            if done:
                break
        
        # 衰减ε
        # Decay ε
        self.behavior_policy.decay_epsilon()
        
        # 记录统计
        # Record statistics
        self.episode_count += 1
        self.episode_returns.append(episode_return)
        self.episode_lengths.append(episode_length)
        self.greedy_steps.append(greedy_count)
        self.trace_truncations.append(truncation_count)
        
        return episode_return, episode_length
    
    def learn(self,
             n_episodes: int = 1000,
             verbose: bool = True) -> ActionValueFunction:
        """
        学习Q函数
        Learn Q function
        
        Args:
            n_episodes: 回合数
                       Number of episodes
            verbose: 是否输出进度
                    Whether to output progress
        
        Returns:
            学习的Q函数
            Learned Q function
        """
        if verbose:
            print(f"\n开始Watkins's Q(λ)学习: {n_episodes}回合")
            print(f"Starting Watkins's Q(λ) learning: {n_episodes} episodes")
            print(f"  参数: γ={self.gamma}, λ={self.lambda_}")
            print(f"  初始ε: {self.behavior_policy.epsilon:.3f}")
        
        for episode in range(n_episodes):
            episode_return, episode_length = self.learn_episode()
            
            if verbose and (episode + 1) % max(1, n_episodes // 10) == 0:
                avg_return = np.mean(self.episode_returns[-100:]) \
                           if len(self.episode_returns) >= 100 \
                           else np.mean(self.episode_returns)
                avg_length = np.mean(self.episode_lengths[-100:]) \
                           if len(self.episode_lengths) >= 100 \
                           else np.mean(self.episode_lengths)
                avg_greedy = np.mean(self.greedy_steps[-100:]) \
                           if len(self.greedy_steps) >= 100 \
                           else np.mean(self.greedy_steps)
                avg_trunc = np.mean(self.trace_truncations[-100:]) \
                          if len(self.trace_truncations) >= 100 \
                          else np.mean(self.trace_truncations)
                
                print(f"Episode {episode + 1}/{n_episodes}: "
                      f"Return={episode_return:.2f}, "
                      f"Avg Return={avg_return:.2f}, "
                      f"Greedy={avg_greedy:.1f}/{avg_length:.1f}, "
                      f"Truncations={avg_trunc:.1f}, "
                      f"ε={self.behavior_policy.epsilon:.3f}")
        
        if verbose:
            print(f"\nWatkins's Q(λ)学习完成!")
            print(f"  最终ε: {self.behavior_policy.epsilon:.3f}")
            print(f"  总步数: {self.step_count}")
            
            # 分析贪婪比例
            # Analyze greedy ratio
            total_steps = sum(self.episode_lengths)
            total_greedy = sum(self.greedy_steps)
            greedy_ratio = total_greedy / total_steps if total_steps > 0 else 0
            print(f"  贪婪步比例: {greedy_ratio:.2%}")
        
        return self.Q


# ================================================================================
# 第6.3.3节：算法比较器
# Section 6.3.3: Algorithm Comparator
# ================================================================================

class TDLambdaControlComparator:
    """
    TD(λ)控制算法比较器
    TD(λ) Control Algorithm Comparator
    
    系统比较SARSA(λ)、Watkins's Q(λ)等
    Systematically compare SARSA(λ), Watkins's Q(λ), etc.
    """
    
    def __init__(self, env: MDPEnvironment):
        """
        初始化比较器
        Initialize comparator
        """
        self.env = env
        self.results = {}
        
        logger.info("初始化TD(λ)控制算法比较器")
    
    def run_comparison(self,
                       algorithms: Dict[str, Any] = None,
                       n_episodes: int = 1000,
                       n_runs: int = 5,
                       verbose: bool = True) -> Dict[str, Any]:
        """
        运行比较实验
        Run comparison experiment
        
        Args:
            algorithms: 算法配置字典
                       Algorithm configuration dict
            n_episodes: 每次运行的回合数
                       Episodes per run
            n_runs: 运行次数
                   Number of runs
            verbose: 是否输出进度
                    Whether to output progress
        
        Returns:
            比较结果
            Comparison results
        """
        if algorithms is None:
            # 默认配置
            # Default configuration
            algorithms = {
                'SARSA': {
                    'class': 'SARSA',
                    'params': {'gamma': 0.99, 'alpha': 0.1, 'epsilon': 0.1}
                },
                'SARSA(λ=0.8)': {
                    'class': SARSALambda,
                    'params': {'gamma': 0.99, 'lambda_': 0.8, 'alpha': 0.1, 'epsilon': 0.1}
                },
                'Q-Learning': {
                    'class': 'QLearning',
                    'params': {'gamma': 0.99, 'alpha': 0.1, 'epsilon': 0.1}
                },
                'Watkins Q(λ=0.8)': {
                    'class': WatkinsQLambda,
                    'params': {'gamma': 0.99, 'lambda_': 0.8, 'alpha': 0.1, 'epsilon': 0.1}
                }
            }
        
        if verbose:
            print("\n" + "="*80)
            print("TD(λ)控制算法比较实验")
            print("TD(λ) Control Algorithm Comparison")
            print("="*80)
            print(f"算法: {list(algorithms.keys())}")
            print(f"实验: {n_episodes}回合 × {n_runs}次运行")
        
        # 导入需要的算法
        # Import needed algorithms
        from ch05_temporal_difference.td_control import SARSA, QLearning
        
        results = {name: {
            'returns': [],
            'lengths': [],
            'final_returns': [],
            'convergence_episodes': []
        } for name in algorithms}
        
        for run in range(n_runs):
            if verbose:
                print(f"\n运行 {run + 1}/{n_runs}:")
            
            for name, config in algorithms.items():
                # 创建算法实例
                # Create algorithm instance
                algo_class = config['class']
                params = config['params']
                
                # 处理字符串类名
                # Handle string class names
                if algo_class == 'SARSA':
                    algo = SARSA(self.env, **params)
                elif algo_class == 'QLearning':
                    algo = QLearning(self.env, **params)
                else:
                    algo = algo_class(self.env, **params)
                
                # 设置固定的epsilon衰减
                # Set fixed epsilon decay
                if hasattr(algo, 'policy'):
                    algo.policy.epsilon_decay = 1.0  # 不衰减，公平比较
                elif hasattr(algo, 'behavior_policy'):
                    algo.behavior_policy.epsilon_decay = 1.0
                
                # 学习
                # Learn
                algo.learn(n_episodes, verbose=False)
                
                # 记录结果
                # Record results
                results[name]['returns'].append(algo.episode_returns)
                results[name]['lengths'].append(algo.episode_lengths)
                results[name]['final_returns'].append(algo.episode_returns[-1])
                
                # 计算收敛
                # Compute convergence
                convergence_ep = self._find_convergence_episode(algo.episode_returns)
                results[name]['convergence_episodes'].append(convergence_ep)
                
                if verbose:
                    print(f"  {name}: 最终回报={algo.episode_returns[-1]:.2f}, "
                          f"收敛={convergence_ep}")
        
        # 分析结果
        # Analyze results
        self.results = self._analyze_results(results)
        
        if verbose:
            self._print_comparison_summary()
        
        return self.results
    
    def _find_convergence_episode(self, returns: List[float],
                                  window: int = 100,
                                  threshold: float = 0.1) -> int:
        """
        找到收敛回合
        Find convergence episode
        """
        if len(returns) < window:
            return len(returns)
        
        for i in range(window, len(returns)):
            recent = returns[i-window:i]
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
        
        for name, data in results.items():
            returns_array = np.array(data['returns'])
            lengths_array = np.array(data['lengths'])
            
            analyzed[name] = {
                'mean_returns': np.mean(returns_array, axis=0),
                'std_returns': np.std(returns_array, axis=0),
                'mean_lengths': np.mean(lengths_array, axis=0),
                'final_return_mean': np.mean(data['final_returns']),
                'final_return_std': np.std(data['final_returns']),
                'convergence_mean': np.mean(data['convergence_episodes']),
                'convergence_std': np.std(data['convergence_episodes']),
                'stability': np.mean([np.var(run[-100:]) for run in returns_array])
            }
        
        return analyzed
    
    def _print_comparison_summary(self):
        """
        打印比较摘要
        Print comparison summary
        """
        print("\n" + "="*80)
        print("比较结果摘要")
        print("Comparison Results Summary")
        print("="*80)
        
        print(f"\n{'算法':<20} {'最终回报':<20} {'收敛回合':<20} {'稳定性':<15}")
        print("-" * 75)
        
        for name, data in self.results.items():
            final_return = f"{data['final_return_mean']:.2f} ± {data['final_return_std']:.2f}"
            convergence = f"{data['convergence_mean']:.0f} ± {data['convergence_std']:.0f}"
            stability = f"{data['stability']:.3f}"
            
            print(f"{name:<20} {final_return:<20} {convergence:<20} {stability:<15}")
        
        # 找最佳
        # Find best
        best_return = max(self.results.items(),
                         key=lambda x: x[1]['final_return_mean'])
        fastest = min(self.results.items(),
                     key=lambda x: x[1]['convergence_mean'])
        most_stable = min(self.results.items(),
                         key=lambda x: x[1]['stability'])
        
        print(f"\n最高回报: {best_return[0]}")
        print(f"最快收敛: {fastest[0]}")
        print(f"最稳定: {most_stable[0]}")
        
        print("""
        关键洞察 Key Insights:
        - TD(λ)通常比TD(0)更快
          TD(λ) usually faster than TD(0)
        - SARSA(λ)适合on-policy学习
          SARSA(λ) good for on-policy learning
        - Watkins Q(λ)保持off-policy但可能慢
          Watkins Q(λ) maintains off-policy but may be slow
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
        
        # 颜色映射
        # Color mapping
        colors = ['blue', 'lightblue', 'red', 'lightcoral']
        
        # 图1：学习曲线
        # Plot 1: Learning curves
        ax1 = axes[0, 0]
        for idx, (name, data) in enumerate(self.results.items()):
            mean_returns = data['mean_returns']
            std_returns = data['std_returns']
            episodes = range(len(mean_returns))
            
            ax1.plot(episodes, mean_returns, color=colors[idx % len(colors)],
                    label=name, linewidth=2)
            ax1.fill_between(episodes,
                            mean_returns - std_returns,
                            mean_returns + std_returns,
                            color=colors[idx % len(colors)], alpha=0.1)
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Return')
        ax1.set_title('Learning Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 图2：回合长度
        # Plot 2: Episode lengths
        ax2 = axes[0, 1]
        for idx, (name, data) in enumerate(self.results.items()):
            mean_lengths = data['mean_lengths']
            
            # 平滑
            # Smooth
            if len(mean_lengths) > 20:
                window = 20
                smoothed = np.convolve(mean_lengths,
                                      np.ones(window)/window,
                                      mode='valid')
                ax2.plot(range(len(smoothed)), smoothed,
                        color=colors[idx % len(colors)],
                        label=name, linewidth=2)
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Episode Length')
        ax2.set_title('Episode Lengths (Smoothed)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 图3：最终性能
        # Plot 3: Final performance
        ax3 = axes[1, 0]
        
        names_list = list(self.results.keys())
        positions = range(len(names_list))
        
        final_means = [self.results[name]['final_return_mean'] for name in names_list]
        final_stds = [self.results[name]['final_return_std'] for name in names_list]
        
        bars = ax3.bar(positions, final_means, yerr=final_stds,
                      capsize=5, color=colors[:len(names_list)])
        ax3.set_xticks(positions)
        ax3.set_xticklabels(names_list, rotation=45, ha='right')
        ax3.set_ylabel('Final Return')
        ax3.set_title('Final Performance')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 图4：收敛速度
        # Plot 4: Convergence speed
        ax4 = axes[1, 1]
        
        conv_means = [self.results[name]['convergence_mean'] for name in names_list]
        conv_stds = [self.results[name]['convergence_std'] for name in names_list]
        
        bars = ax4.bar(positions, conv_means, yerr=conv_stds,
                      capsize=5, color=colors[:len(names_list)])
        ax4.set_xticks(positions)
        ax4.set_xticklabels(names_list, rotation=45, ha='right')
        ax4.set_ylabel('Convergence Episode')
        ax4.set_title('Convergence Speed')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('TD(λ) Control Algorithms Comparison',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig


# ================================================================================
# 主函数：演示TD(λ)控制
# Main Function: Demonstrate TD(λ) Control
# ================================================================================

def demonstrate_td_lambda_control():
    """
    演示TD(λ)控制算法
    Demonstrate TD(λ) control algorithms
    """
    print("\n" + "="*80)
    print("第6.3节：TD(λ)控制算法")
    print("Section 6.3: TD(λ) Control Algorithms")
    print("="*80)
    
    from ch02_mdp.gridworld import GridWorld
    
    # 创建环境（带障碍）
    # Create environment (with obstacles)
    env = GridWorld(rows=5, cols=5,
                   start_pos=(0,0),
                   goal_pos=(4,4),
                   obstacles=[(2,2), (3,2)])
    
    print(f"\n创建5×5 GridWorld（含障碍）")
    print(f"  起点: (0,0)")
    print(f"  终点: (4,4)")
    print(f"  障碍: (2,2), (3,2)")
    
    # 1. SARSA(λ)测试
    # 1. SARSA(λ) test
    print("\n" + "="*60)
    print("1. SARSA(λ)测试")
    print("1. SARSA(λ) Test")
    print("="*60)
    
    # 测试不同迹类型
    # Test different trace types
    trace_types = ["replacing", "accumulating"]
    
    for trace_type in trace_types:
        print(f"\n测试{trace_type}迹的SARSA(λ):")
        sarsa_lambda = SARSALambda(
            env, gamma=0.99, lambda_=0.9, alpha=0.1,
            epsilon=0.1, trace_type=trace_type
        )
        
        Q = sarsa_lambda.learn(n_episodes=200, verbose=False)
        
        avg_return = np.mean(sarsa_lambda.episode_returns[-50:])
        avg_traces = np.mean(sarsa_lambda.max_traces_per_episode)
        
        print(f"  最终平均回报: {avg_return:.2f}")
        print(f"  平均活跃迹: {avg_traces:.1f}")
        
        # 显示一些Q值
        # Show some Q values
        sample_state = env.state_space[0]
        if not sample_state.is_terminal:
            print(f"  Q值示例:")
            for action in env.action_space[:2]:
                q_value = Q.get_value(sample_state, action)
                print(f"    Q(s0, {action.id}) = {q_value:.3f}")
    
    # 2. Watkins's Q(λ)测试
    # 2. Watkins's Q(λ) test
    print("\n" + "="*60)
    print("2. Watkins's Q(λ)测试")
    print("2. Watkins's Q(λ) Test")
    print("="*60)
    
    watkins_q = WatkinsQLambda(
        env, gamma=0.99, lambda_=0.9, alpha=0.1, epsilon=0.1
    )
    
    Q_watkins = watkins_q.learn(n_episodes=200, verbose=True)
    
    # 3. 算法比较
    # 3. Algorithm comparison
    print("\n" + "="*60)
    print("3. TD(λ)控制算法系统比较")
    print("3. Systematic TD(λ) Control Comparison")
    print("="*60)
    
    comparator = TDLambdaControlComparator(env)
    results = comparator.run_comparison(
        n_episodes=500,
        n_runs=3,
        verbose=True
    )
    
    # 绘图
    # Plot
    fig = comparator.plot_comparison()
    
    # 总结
    # Summary
    print("\n" + "="*80)
    print("TD(λ)控制算法总结")
    print("TD(λ) Control Algorithm Summary")
    print("="*80)
    
    print("""
    关键要点 Key Takeaways:
    =======================
    
    1. SARSA(λ)是自然的on-policy扩展
       SARSA(λ) is natural on-policy extension
       
    2. Watkins Q(λ)通过截断保持off-policy
       Watkins Q(λ) maintains off-policy via truncation
       
    3. 迹类型影响性能
       Trace type affects performance
       - 替换迹在控制中常用
         Replacing traces common in control
       - 累积迹理论性质好
         Accumulating traces have good theory
       
    4. λ≈0.8-0.9通常效果好
       λ≈0.8-0.9 usually works well
       
    5. 迹的内存管理很重要
       Trace memory management important
       - 使用稀疏表示
         Use sparse representation
       - 及时截断小迹
         Truncate small traces timely
    """)
    
    plt.show()


if __name__ == "__main__":
    demonstrate_td_lambda_control()