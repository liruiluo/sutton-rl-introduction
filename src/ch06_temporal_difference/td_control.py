"""
================================================================================
第5.4节：TD控制 - SARSA和Q-learning
Section 5.4: TD Control - SARSA and Q-learning
================================================================================

TD控制是现代强化学习的基石！
TD control is the cornerstone of modern RL!

两大经典算法：
Two classic algorithms:

1. SARSA (State-Action-Reward-State-Action)
   - On-policy TD控制
     On-policy TD control
   - 名字来源于使用的五元组(S,A,R,S',A')
     Name from the quintuple used
   - 保守、安全
     Conservative, safe

2. Q-learning (Watkins, 1989)
   - Off-policy TD控制
     Off-policy TD control
   - 最重要的强化学习算法！
     Most important RL algorithm!
   - 激进、最优
     Aggressive, optimal

核心区别 Core Difference:
SARSA: Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
       使用实际采取的A'
       Uses actually taken A'

Q-learning: Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]
            使用最优动作
            Uses optimal action

这个微小差异导致完全不同的行为！
This small difference leads to completely different behaviors!

Expected SARSA: 介于两者之间
                Between the two
Q(S,A) ← Q(S,A) + α[R + γ Σ_a π(a|S')Q(S',a) - Q(S,A)]

深度强化学习的基础：
Foundation of Deep RL:
- DQN = Q-learning + Deep Neural Network
- A3C = Actor-Critic + Asynchronous
- PPO = Policy Gradient + Clipping
都建立在这些基本算法之上！
All built on these basic algorithms!
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import seaborn as sns
from scipy import stats
import time

# 导入基础组件
# Import base components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.ch03_finite_mdp.mdp_framework import State, Action, MDPEnvironment, MDPAgent
from src.ch03_finite_mdp.policies_and_values import (
    Policy, StateValueFunction, ActionValueFunction,
    StochasticPolicy, DeterministicPolicy
)
from src.ch05_monte_carlo.mc_control import EpsilonGreedyPolicy
from .td_foundations import TDError, TDErrorAnalyzer

# 设置日志
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第5.4.1节：SARSA算法
# Section 5.4.1: SARSA Algorithm
# ================================================================================

class SARSA:
    """
    SARSA算法 - On-Policy TD控制
    SARSA Algorithm - On-Policy TD Control
    
    全称：State-Action-Reward-State-Action
    Full name: State-Action-Reward-State-Action
    
    更新规则 Update rule:
    Q(S_t, A_t) ← Q(S_t, A_t) + α[R_{t+1} + γQ(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]
    
    注意A_{t+1}是按照当前策略选择的！
    Note A_{t+1} is chosen according to current policy!
    
    算法步骤 Algorithm steps:
    1. 初始化Q(s,a)任意，Q(terminal,·)=0
       Initialize Q(s,a) arbitrarily, Q(terminal,·)=0
    2. 重复每个回合：
       Repeat for each episode:
       a. 初始化S
          Initialize S
       b. 从S选择A（使用ε-greedy）
          Choose A from S (using ε-greedy)
       c. 重复每步：
          Repeat for each step:
          - 执行A，观察R,S'
            Take A, observe R,S'
          - 从S'选择A'（使用ε-greedy）
            Choose A' from S' (using ε-greedy)
          - Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
          - S ← S'; A ← A'
       直到S是终止状态
       Until S is terminal
    
    特性 Properties:
    1. On-policy: 评估和改进同一策略
                  Evaluate and improve same policy
    2. 收敛到Q^π（ε-greedy策略的Q）
       Converges to Q^π (Q for ε-greedy policy)
    3. 保守：考虑探索动作的后果
       Conservative: considers consequences of exploratory actions
    4. 适合需要安全探索的场景
       Suitable for scenarios needing safe exploration
    
    与Q-learning的关键区别：
    Key difference from Q-learning:
    - SARSA学习实际执行的策略（包括探索）
      SARSA learns the policy being followed (including exploration)
    - Q-learning学习最优策略（忽略探索）
      Q-learning learns optimal policy (ignoring exploration)
    
    例子：悬崖行走（Cliff Walking）
    Example: Cliff Walking
    SARSA会学习远离悬崖的安全路径
    SARSA learns safe path away from cliff
    Q-learning会学习靠近悬崖的最优路径
    Q-learning learns optimal path near cliff
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 gamma: float = 0.99,
                 alpha: Union[float, Callable] = 0.1,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        初始化SARSA
        Initialize SARSA
        
        Args:
            env: 环境
                Environment
            gamma: 折扣因子
                  Discount factor
            alpha: 学习率
                  Learning rate
            epsilon: 探索率
                    Exploration rate
            epsilon_decay: ε衰减率
                         ε decay rate
            epsilon_min: 最小ε
                        Minimum ε
        """
        self.env = env
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
        
        logger.info(f"初始化SARSA: γ={gamma}, α={alpha}, ε={epsilon}")
    
    def learn_episode(self) -> Tuple[float, int]:
        """
        学习一个回合
        Learn one episode
        
        Returns:
            (回合回报, 回合长度)
            (episode return, episode length)
        """
        # 初始化S
        # Initialize S
        state = self.env.reset()
        
        # 选择A（从S使用策略）
        # Choose A (from S using policy)
        action = self.policy.select_action(state)
        
        episode_return = 0.0
        episode_length = 0
        
        while True:
            # 执行动作A，观察R和S'
            # Take action A, observe R and S'
            next_state, reward, done, info = self.env.step(action)
            
            # 选择A'（从S'使用策略）
            # Choose A' (from S' using policy)
            if not done:
                next_action = self.policy.select_action(next_state)
            else:
                next_action = None
            
            # SARSA更新
            # SARSA update
            if not state.is_terminal:
                alpha = self.alpha_func(self.step_count)
                
                # 计算TD目标
                # Compute TD target
                q_current = self.Q.get_value(state, action)
                if done:
                    td_target = reward  # 终止状态Q=0
                else:
                    q_next = self.Q.get_value(next_state, next_action)
                    td_target = reward + self.gamma * q_next
                
                # TD误差
                # TD error
                td_error = td_target - q_current
                
                # 更新Q
                # Update Q
                new_q = q_current + alpha * td_error
                self.Q.set_value(state, action, new_q)
                
                # 记录TD误差
                # Record TD error
                td_err_obj = TDError(
                    value=td_error,
                    timestep=self.step_count,
                    state=state,
                    next_state=next_state,
                    reward=reward,
                    state_value=q_current,
                    next_state_value=q_next if not done else 0.0
                )
                self.td_analyzer.add_error(td_err_obj)
            
            # 累积回报
            # Accumulate return
            episode_return += reward * (self.gamma ** episode_length)
            episode_length += 1
            self.step_count += 1
            
            # S ← S'; A ← A'
            state = next_state
            action = next_action
            
            if done:
                break
        
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
            print(f"\n开始SARSA学习: {n_episodes}回合")
            print(f"Starting SARSA learning: {n_episodes} episodes")
            print(f"  初始ε: {self.policy.epsilon:.3f}")
        
        for episode in range(n_episodes):
            episode_return, episode_length = self.learn_episode()
            
            if verbose and (episode + 1) % max(1, n_episodes // 10) == 0:
                avg_return = np.mean(self.episode_returns[-100:]) if len(self.episode_returns) >= 100 else np.mean(self.episode_returns)
                avg_length = np.mean(self.episode_lengths[-100:]) if len(self.episode_lengths) >= 100 else np.mean(self.episode_lengths)
                
                stats = self.td_analyzer.get_statistics()
                
                print(f"Episode {episode + 1}/{n_episodes}: "
                      f"Return={episode_return:.2f}, "
                      f"Avg Return={avg_return:.2f}, "
                      f"Avg Length={avg_length:.1f}, "
                      f"ε={self.policy.epsilon:.3f}, "
                      f"|δ|={stats.get('recent_abs_mean', 0):.4f}")
        
        if verbose:
            print(f"\nSARSA学习完成!")
            print(f"  最终ε: {self.policy.epsilon:.3f}")
            print(f"  总步数: {self.step_count}")
            
            # 分析最优策略
            # Analyze optimal policy
            self.analyze_learned_policy()
        
        return self.Q


# ================================================================================
# 第5.4.2节：Q-Learning算法
# Section 5.4.2: Q-Learning Algorithm
# ================================================================================

class QLearning:
    """
    Q-Learning算法 - Off-Policy TD控制
    Q-Learning Algorithm - Off-Policy TD Control
    
    最重要的强化学习算法！
    The most important RL algorithm!
    
    更新规则 Update rule:
    Q(S_t, A_t) ← Q(S_t, A_t) + α[R_{t+1} + γ max_a Q(S_{t+1}, a) - Q(S_t, A_t)]
    
    关键：使用max而不是实际的A'！
    Key: Uses max instead of actual A'!
    
    算法步骤 Algorithm steps:
    1. 初始化Q(s,a)任意，Q(terminal,·)=0
       Initialize Q(s,a) arbitrarily, Q(terminal,·)=0
    2. 重复每个回合：
       Repeat for each episode:
       a. 初始化S
          Initialize S
       b. 重复每步：
          Repeat for each step:
          - 从S选择A（使用ε-greedy）
            Choose A from S (using ε-greedy)
          - 执行A，观察R,S'
            Take A, observe R,S'
          - Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]
          - S ← S'
       直到S是终止状态
       Until S is terminal
    
    特性 Properties:
    1. Off-policy: 学习最优策略，同时用ε-greedy探索
                   Learn optimal policy while following ε-greedy
    2. 收敛到Q*（最优Q函数）
       Converges to Q* (optimal Q function)
    3. 激进：直接学习贪婪策略
       Aggressive: directly learns greedy policy
    4. 可能过度估计Q值（maximization bias）
       May overestimate Q values (maximization bias)
    
    为什么Q-learning如此成功？
    Why is Q-learning so successful?
    1. 简单：一行更新规则
       Simple: one-line update rule
    2. 有效：直接学习最优策略
       Effective: directly learns optimal policy
    3. 通用：适用于各种问题
       General: applicable to various problems
    4. 理论保证：收敛到最优
       Theoretical guarantee: converges to optimal
    
    深度Q网络(DQN)的基础：
    Foundation of Deep Q-Network (DQN):
    - 用神经网络近似Q函数
      Approximate Q function with neural network
    - 经验回放(Experience Replay)
    - 目标网络(Target Network)
    - 开启深度强化学习革命！
      Started the Deep RL revolution!
    
    Double Q-Learning解决过度估计：
    Double Q-Learning solves overestimation:
    使用两个Q函数，一个选择动作，一个评估
    Use two Q functions, one selects action, one evaluates
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 gamma: float = 0.99,
                 alpha: Union[float, Callable] = 0.1,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        初始化Q-Learning
        Initialize Q-Learning
        
        Args:
            env: 环境
                Environment
            gamma: 折扣因子
                  Discount factor
            alpha: 学习率
                  Learning rate
            epsilon: 探索率
                    Exploration rate
            epsilon_decay: ε衰减率
                         ε decay rate
            epsilon_min: 最小ε
                        Minimum ε
        """
        self.env = env
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
        
        # ε-贪婪策略（行为策略）
        # ε-greedy policy (behavior policy)
        self.behavior_policy = EpsilonGreedyPolicy(
            self.Q,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            action_space=env.action_space
        )
        
        # 目标策略（贪婪）
        # Target policy (greedy)
        self.target_policy = self._create_greedy_policy()
        
        # TD误差分析
        # TD error analysis
        self.td_analyzer = TDErrorAnalyzer()
        
        # 统计
        # Statistics
        self.episode_count = 0
        self.step_count = 0
        self.episode_returns = []
        self.episode_lengths = []
        
        # Q-learning特有：记录最大Q值
        # Q-learning specific: record max Q values
        self.max_q_values = []
        
        logger.info(f"初始化Q-Learning: γ={gamma}, α={alpha}, ε={epsilon}")
    
    def _create_greedy_policy(self) -> Policy:
        """
        创建贪婪策略
        Create greedy policy
        
        总是选择Q值最大的动作
        Always choose action with max Q value
        """
        class GreedyPolicy(Policy):
            def __init__(self, Q: ActionValueFunction, action_space: List[Action]):
                super().__init__()
                self.Q = Q
                self.action_space = action_space
            
            def select_action(self, state: State) -> Action:
                """贪婪选择"""
                if state.is_terminal:
                    return self.action_space[0]
                
                # 找最大Q值的动作
                # Find action with max Q value
                q_values = [self.Q.get_value(state, a) for a in self.action_space]
                max_q = max(q_values)
                
                # 如果有多个最大值，随机选择
                # If multiple max values, choose randomly
                max_actions = [a for a, q in zip(self.action_space, q_values) if q == max_q]
                return np.random.choice(max_actions)
            
            def get_action_probabilities(self, state: State) -> Dict[Action, float]:
                """获取动作概率"""
                selected = self.select_action(state)
                return {a: 1.0 if a == selected else 0.0 for a in self.action_space}
        
        return GreedyPolicy(self.Q, self.env.action_space)
    
    def learn_episode(self) -> Tuple[float, int]:
        """
        学习一个回合
        Learn one episode
        
        Returns:
            (回合回报, 回合长度)
            (episode return, episode length)
        """
        # 初始化S
        # Initialize S
        state = self.env.reset()
        
        episode_return = 0.0
        episode_length = 0
        
        while True:
            # 选择A（使用行为策略ε-greedy）
            # Choose A (using behavior policy ε-greedy)
            action = self.behavior_policy.select_action(state)
            
            # 执行动作，观察R和S'
            # Take action, observe R and S'
            next_state, reward, done, info = self.env.step(action)
            
            # Q-Learning更新
            # Q-Learning update
            if not state.is_terminal:
                alpha = self.alpha_func(self.step_count)
                
                # 当前Q值
                # Current Q value
                q_current = self.Q.get_value(state, action)
                
                # 计算TD目标（关键：使用max）
                # Compute TD target (key: use max)
                if done:
                    td_target = reward
                    max_q_next = 0.0
                else:
                    # 找S'的最大Q值
                    # Find max Q value for S'
                    q_values_next = [self.Q.get_value(next_state, a) 
                                    for a in self.env.action_space]
                    max_q_next = max(q_values_next)
                    td_target = reward + self.gamma * max_q_next
                
                # TD误差
                # TD error
                td_error = td_target - q_current
                
                # 更新Q
                # Update Q
                new_q = q_current + alpha * td_error
                self.Q.set_value(state, action, new_q)
                
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
                
                # 记录最大Q值（用于诊断）
                # Record max Q value (for diagnosis)
                self.max_q_values.append(max_q_next)
            
            # 累积回报
            # Accumulate return
            episode_return += reward * (self.gamma ** episode_length)
            episode_length += 1
            self.step_count += 1
            
            # S ← S'
            state = next_state
            
            if done:
                break
        
        # 衰减ε
        # Decay ε
        self.behavior_policy.decay_epsilon()
        
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
            print(f"\n开始Q-Learning学习: {n_episodes}回合")
            print(f"Starting Q-Learning: {n_episodes} episodes")
            print(f"  初始ε: {self.behavior_policy.epsilon:.3f}")
        
        for episode in range(n_episodes):
            episode_return, episode_length = self.learn_episode()
            
            if verbose and (episode + 1) % max(1, n_episodes // 10) == 0:
                avg_return = np.mean(self.episode_returns[-100:]) if len(self.episode_returns) >= 100 else np.mean(self.episode_returns)
                avg_length = np.mean(self.episode_lengths[-100:]) if len(self.episode_lengths) >= 100 else np.mean(self.episode_lengths)
                
                stats = self.td_analyzer.get_statistics()
                avg_max_q = np.mean(self.max_q_values[-1000:]) if self.max_q_values else 0
                
                print(f"Episode {episode + 1}/{n_episodes}: "
                      f"Return={episode_return:.2f}, "
                      f"Avg Return={avg_return:.2f}, "
                      f"Avg Length={avg_length:.1f}, "
                      f"ε={self.behavior_policy.epsilon:.3f}, "
                      f"Max Q={avg_max_q:.3f}")
        
        if verbose:
            print(f"\nQ-Learning学习完成!")
            print(f"  最终ε: {self.behavior_policy.epsilon:.3f}")
            print(f"  总步数: {self.step_count}")
            
            # 分析最优策略
            # Analyze optimal policy
            self.analyze_learned_policy()
        
        return self.Q
    
    def analyze_learned_policy(self):
        """
        分析学习的策略
        Analyze learned policy
        
        展示Q-learning学到了什么
        Show what Q-learning learned
        """
        print("\n学习策略分析 Learned Policy Analysis:")
        print("-" * 60)
        
        # 选择一些代表性状态
        # Select some representative states
        sample_states = []
        for state in self.env.state_space[:10]:  # 前10个状态
            if not state.is_terminal:
                sample_states.append(state)
        
        print("部分状态的Q值 Q values for sample states:")
        for state in sample_states[:5]:
            print(f"\n状态 State {state.id}:")
            
            # 获取所有动作的Q值
            # Get Q values for all actions
            q_values = [(a, self.Q.get_value(state, a)) 
                       for a in self.env.action_space]
            q_values.sort(key=lambda x: x[1], reverse=True)
            
            for action, q_val in q_values:
                is_best = "* " if q_val == q_values[0][1] else "  "
                print(f"  {is_best}Q({action.id}) = {q_val:.3f}")
        
        # 检查是否有过度估计
        # Check for overestimation
        all_q_values = []
        for state in self.env.state_space:
            if not state.is_terminal:
                for action in self.env.action_space:
                    all_q_values.append(self.Q.get_value(state, action))
        
        if all_q_values:
            print(f"\nQ值统计 Q value statistics:")
            print(f"  最大 Max: {max(all_q_values):.3f}")
            print(f"  平均 Mean: {np.mean(all_q_values):.3f}")
            print(f"  标准差 Std: {np.std(all_q_values):.3f}")
            
            # 检测可能的过度估计
            # Detect possible overestimation
            theoretical_max = 1.0 / (1 - self.gamma) if self.gamma < 1 else 100
            if max(all_q_values) > theoretical_max:
                print(f"\n⚠️ 警告：Q值可能过度估计!")
                print(f"   Warning: Q values may be overestimated!")
                print(f"   理论最大值 Theoretical max: {theoretical_max:.3f}")


# ================================================================================
# 第5.4.3节：Expected SARSA
# Section 5.4.3: Expected SARSA  
# ================================================================================

class ExpectedSARSA:
    """
    Expected SARSA - SARSA和Q-learning的统一
    Expected SARSA - Unification of SARSA and Q-learning
    
    更新规则 Update rule:
    Q(S_t, A_t) ← Q(S_t, A_t) + α[R_{t+1} + γ E_π[Q(S_{t+1}, A)] - Q(S_t, A_t)]
    
    其中 where:
    E_π[Q(S_{t+1}, A)] = Σ_a π(a|S_{t+1}) Q(S_{t+1}, a)
    
    特殊情况 Special cases:
    1. π是ε-greedy → Expected SARSA
    2. π是贪婪(ε=0) → Q-learning
    3. π是随机选择一个动作 → SARSA
       π randomly selects one action → SARSA
    
    优势 Advantages:
    1. 比SARSA方差更小（使用期望而非采样）
       Lower variance than SARSA (uses expectation not sampling)
    2. 比Q-learning更稳定（考虑所有动作）
       More stable than Q-learning (considers all actions)
    3. 理论上更优雅（统一框架）
       Theoretically more elegant (unified framework)
    
    实践表现：
    Performance in practice:
    通常介于SARSA和Q-learning之间
    Usually between SARSA and Q-learning
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 gamma: float = 0.99,
                 alpha: Union[float, Callable] = 0.1,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        初始化Expected SARSA
        Initialize Expected SARSA
        """
        self.env = env
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
        
        logger.info(f"初始化Expected SARSA: γ={gamma}, α={alpha}, ε={epsilon}")
    
    def _compute_expected_q(self, state: State) -> float:
        """
        计算期望Q值
        Compute expected Q value
        
        E_π[Q(s, A)] = Σ_a π(a|s) Q(s, a)
        
        Args:
            state: 状态
                  State
        
        Returns:
            期望Q值
            Expected Q value
        """
        if state.is_terminal:
            return 0.0
        
        # 获取动作概率
        # Get action probabilities
        action_probs = self.policy.get_action_probabilities(state)
        
        # 计算期望
        # Compute expectation
        expected_q = 0.0
        for action in self.env.action_space:
            prob = action_probs.get(action, 0.0)
            q_value = self.Q.get_value(state, action)
            expected_q += prob * q_value
        
        return expected_q
    
    def learn_episode(self) -> Tuple[float, int]:
        """
        学习一个回合
        Learn one episode
        """
        state = self.env.reset()
        
        episode_return = 0.0
        episode_length = 0
        
        while True:
            # 选择动作
            # Select action
            action = self.policy.select_action(state)
            
            # 执行动作
            # Execute action
            next_state, reward, done, info = self.env.step(action)
            
            # Expected SARSA更新
            # Expected SARSA update
            if not state.is_terminal:
                alpha = self.alpha_func(self.step_count)
                
                # 当前Q值
                # Current Q value
                q_current = self.Q.get_value(state, action)
                
                # 计算TD目标（使用期望）
                # Compute TD target (using expectation)
                if done:
                    td_target = reward
                    expected_q_next = 0.0
                else:
                    expected_q_next = self._compute_expected_q(next_state)
                    td_target = reward + self.gamma * expected_q_next
                
                # TD误差
                # TD error
                td_error = td_target - q_current
                
                # 更新Q
                # Update Q
                new_q = q_current + alpha * td_error
                self.Q.set_value(state, action, new_q)
                
                # 记录TD误差
                # Record TD error
                td_err_obj = TDError(
                    value=td_error,
                    timestep=self.step_count,
                    state=state,
                    next_state=next_state,
                    reward=reward,
                    state_value=q_current,
                    next_state_value=expected_q_next
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
            print(f"\n开始Expected SARSA学习: {n_episodes}回合")
            print(f"Starting Expected SARSA: {n_episodes} episodes")
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
            print(f"\nExpected SARSA学习完成!")
            print(f"  最终ε: {self.policy.epsilon:.3f}")
            print(f"  总步数: {self.step_count}")
        
        return self.Q


# ================================================================================
# 第5.4.4节：算法比较器
# Section 5.4.4: Algorithm Comparator
# ================================================================================

class TDControlComparator:
    """
    TD控制算法比较器
    TD Control Algorithm Comparator
    
    系统比较SARSA、Q-learning和Expected SARSA
    Systematically compare SARSA, Q-learning and Expected SARSA
    
    比较维度：
    Comparison dimensions:
    1. 学习速度（收敛速度）
       Learning speed (convergence rate)
    2. 最终性能（回报）
       Final performance (return)
    3. 稳定性（方差）
       Stability (variance)
    4. 探索效率
       Exploration efficiency
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
        
        logger.info("初始化TD控制算法比较器")
    
    def run_comparison(self,
                       n_episodes: int = 1000,
                       n_runs: int = 10,
                       gamma: float = 0.99,
                       alpha: float = 0.1,
                       epsilon: float = 0.1,
                       verbose: bool = True) -> Dict[str, Any]:
        """
        运行比较实验
        Run comparison experiment
        
        Args:
            n_episodes: 每次运行的回合数
                       Episodes per run
            n_runs: 运行次数（用于统计）
                   Number of runs (for statistics)
            gamma: 折扣因子
                  Discount factor
            alpha: 学习率
                  Learning rate
            epsilon: 探索率
                    Exploration rate
            verbose: 是否输出进度
                    Whether to output progress
        
        Returns:
            比较结果
            Comparison results
        """
        if verbose:
            print("\n" + "="*80)
            print("TD控制算法比较实验")
            print("TD Control Algorithm Comparison")
            print("="*80)
            print(f"参数: γ={gamma}, α={alpha}, ε={epsilon}")
            print(f"实验: {n_episodes}回合 × {n_runs}次运行")
        
        algorithms = {
            'SARSA': SARSA,
            'Q-Learning': QLearning,
            'Expected SARSA': ExpectedSARSA
        }
        
        results = {name: {
            'returns': [],
            'lengths': [],
            'final_returns': [],
            'convergence_episodes': []
        } for name in algorithms}
        
        for run in range(n_runs):
            if verbose:
                print(f"\n运行 {run + 1}/{n_runs}:")
            
            for name, AlgoClass in algorithms.items():
                # 创建算法实例
                # Create algorithm instance
                algo = AlgoClass(
                    self.env,
                    gamma=gamma,
                    alpha=alpha,
                    epsilon=epsilon,
                    epsilon_decay=1.0,  # 不衰减，公平比较
                    epsilon_min=epsilon
                )
                
                # 学习
                # Learn
                algo.learn(n_episodes, verbose=False)
                
                # 记录结果
                # Record results
                results[name]['returns'].append(algo.episode_returns)
                results[name]['lengths'].append(algo.episode_lengths)
                results[name]['final_returns'].append(algo.episode_returns[-1])
                
                # 计算收敛回合（回报稳定的回合）
                # Compute convergence episode (when return stabilizes)
                convergence_ep = self._find_convergence_episode(algo.episode_returns)
                results[name]['convergence_episodes'].append(convergence_ep)
                
                if verbose:
                    print(f"  {name}: 最终回报={algo.episode_returns[-1]:.2f}, "
                          f"收敛回合={convergence_ep}")
        
        # 统计分析
        # Statistical analysis
        self.results = self._analyze_results(results, n_episodes)
        
        if verbose:
            self._print_comparison_summary()
        
        return self.results
    
    def _find_convergence_episode(self, returns: List[float],
                                  window: int = 100,
                                  threshold: float = 0.1) -> int:
        """
        找到收敛回合
        Find convergence episode
        
        当移动平均稳定时认为收敛
        Consider converged when moving average stabilizes
        """
        if len(returns) < window:
            return len(returns)
        
        for i in range(window, len(returns)):
            recent = returns[i-window:i]
            mean = np.mean(recent)
            std = np.std(recent)
            
            # 变异系数小于阈值
            # Coefficient of variation below threshold
            if std / (abs(mean) + 1e-10) < threshold:
                return i
        
        return len(returns)
    
    def _analyze_results(self, results: Dict, n_episodes: int) -> Dict:
        """
        分析结果
        Analyze results
        """
        analyzed = {}
        
        for name, data in results.items():
            # 转换为numpy数组
            # Convert to numpy arrays
            returns_array = np.array(data['returns'])  # shape: (n_runs, n_episodes)
            lengths_array = np.array(data['lengths'])
            
            analyzed[name] = {
                # 学习曲线（平均和标准差）
                # Learning curves (mean and std)
                'mean_returns': np.mean(returns_array, axis=0),
                'std_returns': np.std(returns_array, axis=0),
                
                # 最终性能
                # Final performance
                'final_return_mean': np.mean(data['final_returns']),
                'final_return_std': np.std(data['final_returns']),
                
                # 收敛速度
                # Convergence speed
                'convergence_mean': np.mean(data['convergence_episodes']),
                'convergence_std': np.std(data['convergence_episodes']),
                
                # 平均回合长度
                # Average episode length
                'mean_lengths': np.mean(lengths_array, axis=0),
                
                # 稳定性（最后100回合的方差）
                # Stability (variance of last 100 episodes)
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
        
        # 表格标题
        # Table header
        print(f"\n{'算法 Algorithm':<20} {'最终回报 Final Return':<20} "
              f"{'收敛回合 Convergence':<20} {'稳定性 Stability':<15}")
        print("-" * 75)
        
        for name, data in self.results.items():
            final_return = f"{data['final_return_mean']:.2f} ± {data['final_return_std']:.2f}"
            convergence = f"{data['convergence_mean']:.0f} ± {data['convergence_std']:.0f}"
            stability = f"{data['stability']:.3f}"
            
            print(f"{name:<20} {final_return:<20} {convergence:<20} {stability:<15}")
        
        # 关键洞察
        # Key insights
        print("\n关键洞察 Key Insights:")
        print("-" * 40)
        
        # 找最佳算法
        # Find best algorithm
        best_return = max(self.results.items(), 
                         key=lambda x: x[1]['final_return_mean'])
        fastest = min(self.results.items(),
                     key=lambda x: x[1]['convergence_mean'])
        most_stable = min(self.results.items(),
                         key=lambda x: x[1]['stability'])
        
        print(f"最高回报 Best Return: {best_return[0]}")
        print(f"最快收敛 Fastest Convergence: {fastest[0]}")
        print(f"最稳定 Most Stable: {most_stable[0]}")
        
        print("""
        典型模式 Typical Patterns:
        - Q-Learning: 学习最优但可能不稳定
                      Learns optimal but may be unstable
        - SARSA: 更保守但稳定
                Conservative but stable
        - Expected SARSA: 平衡两者优点
                        Balances both advantages
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
        
        colors = {'SARSA': 'blue', 'Q-Learning': 'red', 'Expected SARSA': 'green'}
        
        # 图1：学习曲线
        # Plot 1: Learning curves
        ax1 = axes[0, 0]
        for name, data in self.results.items():
            mean_returns = data['mean_returns']
            std_returns = data['std_returns']
            episodes = range(len(mean_returns))
            
            ax1.plot(episodes, mean_returns, color=colors[name], 
                    label=name, linewidth=2)
            ax1.fill_between(episodes,
                            mean_returns - std_returns,
                            mean_returns + std_returns,
                            color=colors[name], alpha=0.2)
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Return')
        ax1.set_title('Learning Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 图2：回合长度
        # Plot 2: Episode lengths
        ax2 = axes[0, 1]
        for name, data in self.results.items():
            mean_lengths = data['mean_lengths']
            # 平滑处理
            # Smooth
            if len(mean_lengths) > 20:
                window = 20
                smoothed = np.convolve(mean_lengths, 
                                      np.ones(window)/window,
                                      mode='valid')
                ax2.plot(range(len(smoothed)), smoothed,
                        color=colors[name], label=name, linewidth=2)
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Episode Length')
        ax2.set_title('Episode Lengths (Smoothed)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 图3：最终性能分布
        # Plot 3: Final performance distribution
        ax3 = axes[1, 0]
        
        positions = [1, 2, 3]
        names_list = ['SARSA', 'Q-Learning', 'Expected SARSA']
        
        for i, name in enumerate(names_list):
            if name in self.results:
                mean = self.results[name]['final_return_mean']
                std = self.results[name]['final_return_std']
                
                ax3.bar(positions[i], mean, color=colors[name],
                       alpha=0.7, label=name)
                ax3.errorbar(positions[i], mean, yerr=std,
                           color='black', capsize=5)
        
        ax3.set_xticks(positions)
        ax3.set_xticklabels(names_list, rotation=45)
        ax3.set_ylabel('Final Return')
        ax3.set_title('Final Performance Comparison')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 图4：收敛速度
        # Plot 4: Convergence speed
        ax4 = axes[1, 1]
        
        for i, name in enumerate(names_list):
            if name in self.results:
                mean = self.results[name]['convergence_mean']
                std = self.results[name]['convergence_std']
                
                ax4.bar(positions[i], mean, color=colors[name],
                       alpha=0.7, label=name)
                ax4.errorbar(positions[i], mean, yerr=std,
                           color='black', capsize=5)
        
        ax4.set_xticks(positions)
        ax4.set_xticklabels(names_list, rotation=45)
        ax4.set_ylabel('Convergence Episode')
        ax4.set_title('Convergence Speed')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('TD Control Algorithms Comparison', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig


# ================================================================================
# 主函数：演示TD控制
# Main Function: Demonstrate TD Control
# ================================================================================

def demonstrate_td_control():
    """
    演示TD控制算法
    Demonstrate TD control algorithms
    """
    print("\n" + "="*80)
    print("第5.4节：TD控制 - SARSA vs Q-Learning vs Expected SARSA")
    print("Section 5.4: TD Control - SARSA vs Q-Learning vs Expected SARSA")
    print("="*80)
    
    from src.ch03_finite_mdp.gridworld import GridWorld
    
    # 创建环境
    # Create environment
    env = GridWorld(rows=5, cols=5, 
                   start_pos=(0,0), 
                   goal_pos=(4,4),
                   obstacles=[(2,2), (3,2)])  # 添加障碍
    
    print(f"\n创建5×5 GridWorld（含障碍）")
    print(f"Created 5×5 GridWorld (with obstacles)")
    print(f"  起点 Start: (0,0)")
    print(f"  终点 Goal: (4,4)")
    print(f"  障碍 Obstacles: (2,2), (3,2)")
    
    # 运行比较
    # Run comparison
    comparator = TDControlComparator(env)
    results = comparator.run_comparison(
        n_episodes=500,
        n_runs=5,
        gamma=0.99,
        alpha=0.1,
        epsilon=0.1,
        verbose=True
    )
    
    # 绘制比较
    # Plot comparison
    fig = comparator.plot_comparison()
    
    # 单独测试每个算法
    # Test each algorithm individually
    print("\n" + "="*60)
    print("详细测试各算法")
    print("Detailed Testing of Each Algorithm")
    print("="*60)
    
    # 1. SARSA
    print("\n1. SARSA:")
    sarsa = SARSA(env, gamma=0.99, alpha=0.1, epsilon=0.1)
    sarsa.learn(n_episodes=100, verbose=True)
    
    # 2. Q-Learning
    print("\n2. Q-Learning:")
    qlearning = QLearning(env, gamma=0.99, alpha=0.1, epsilon=0.1)
    qlearning.learn(n_episodes=100, verbose=True)
    
    # 3. Expected SARSA
    print("\n3. Expected SARSA:")
    expected_sarsa = ExpectedSARSA(env, gamma=0.99, alpha=0.1, epsilon=0.1)
    expected_sarsa.learn(n_episodes=100, verbose=True)
    
    print("\n" + "="*80)
    print("TD控制演示完成！")
    print("TD Control Demo Complete!")
    print("="*80)
    
    plt.show()


if __name__ == "__main__":
    demonstrate_td_control()