"""
================================================================================
第7.2节：n步SARSA - On-Policy n步控制
Section 7.2: n-step SARSA - On-Policy n-step Control  
================================================================================

n步SARSA扩展了SARSA到n步情况！
n-step SARSA extends SARSA to n-step case!

核心思想 Core Idea:
使用n步回报而不是1步回报进行Q函数更新
Use n-step returns instead of 1-step returns for Q-function updates

n步回报（动作价值）n-step Return (Action Values):
G_t:t+n = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γ^n Q(S_{t+n}, A_{t+n})

更新规则 Update Rule:
Q(S_t, A_t) ← Q(S_t, A_t) + α[G_t:t+n - Q(S_t, A_t)]

算法家族 Algorithm Family:
1. n步SARSA: 使用采样动作
            Use sampled action
2. n步期望SARSA: 使用期望
                Use expectation  
3. n步Tree Backup: 完全off-policy
                  Fully off-policy
4. n步Q(σ): 统一算法
           Unified algorithm

优势 Advantages:
- 更快的学习
  Faster learning
- 更好的信用分配
  Better credit assignment
- 灵活的on/off-policy控制
  Flexible on/off-policy control
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
# 第7.2.1节：n步SARSA算法
# Section 7.2.1: n-step SARSA Algorithm
# ================================================================================

class NStepSARSA:
    """
    n步SARSA算法 - On-Policy n步TD控制
    n-step SARSA Algorithm - On-Policy n-step TD Control
    
    算法步骤 Algorithm Steps:
    1. 初始化Q(s,a)任意，Q(terminal,·)=0
       Initialize Q(s,a) arbitrarily, Q(terminal,·)=0
    2. 初始化策略（如ε-greedy）
       Initialize policy (e.g., ε-greedy)
    3. 对每个回合：
       For each episode:
       a. 初始化S_0，选择A_0 ~ π(·|S_0)
          Initialize S_0, select A_0 ~ π(·|S_0)
       b. T = ∞
       c. 对t = 0, 1, 2, ...：
          For t = 0, 1, 2, ...:
          - 如果t < T：
            If t < T:
            执行A_t，观察R_{t+1}, S_{t+1}
            Take A_t, observe R_{t+1}, S_{t+1}
            如果S_{t+1}终止：
            If S_{t+1} terminal:
              T = t + 1
            否则：
            Else:
              选择A_{t+1} ~ π(·|S_{t+1})
              Select A_{t+1} ~ π(·|S_{t+1})
          - τ = t - n + 1
          - 如果τ ≥ 0：
            If τ ≥ 0:
            G = Σ_{i=τ+1}^{min(τ+n,T)} γ^{i-τ-1} R_i
            如果τ + n < T：
            If τ + n < T:
              G = G + γ^n Q(S_{τ+n}, A_{τ+n})
            Q(S_τ, A_τ) ← Q(S_τ, A_τ) + α[G - Q(S_τ, A_τ)]
            如果π是ε-greedy，更新π基于Q
            If π is ε-greedy, update π based on Q
       直到τ = T - 1
       Until τ = T - 1
    
    关键点 Key Points:
    - On-policy: 学习和行为策略相同
                Learning and behavior policy are same
    - 需要存储n步的(S,A,R)序列
      Need to store n-step (S,A,R) sequence
    - 延迟n-1步更新
      n-1 step delay for updates
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 n: int = 4,
                 gamma: float = 0.99,
                 alpha: Union[float, Callable] = 0.1,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        初始化n步SARSA
        Initialize n-step SARSA
        
        Args:
            env: 环境
                Environment
            n: 步数
               Number of steps
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
        
        logger.info(f"初始化{n}步SARSA: γ={gamma}, ε={epsilon}")
    
    def learn_episode(self) -> Tuple[float, int]:
        """
        学习一个回合
        Learn one episode
        
        Returns:
            (回合回报, 回合长度)
            (episode return, episode length)
        """
        # 初始化轨迹存储
        # Initialize trajectory storage
        states = []
        actions = []
        rewards = []
        
        # 初始化S_0，选择A_0
        # Initialize S_0, select A_0
        state = self.env.reset()
        action = self.policy.select_action(state)
        
        states.append(state)
        actions.append(action)
        
        t = 0
        T = float('inf')
        episode_return = 0.0
        
        # 主循环
        # Main loop
        while True:
            # 时间步τ需要更新
            # Time step τ to update
            tau = t - self.n + 1
            
            if t < T:
                # 执行动作，观察结果
                # Execute action, observe result
                next_state, reward, done, _ = self.env.step(action)
                
                # 存储
                # Store
                states.append(next_state)
                rewards.append(reward)
                
                # 累积回报
                # Accumulate return
                episode_return += reward * (self.gamma ** t)
                
                if done:
                    T = t + 1
                else:
                    # 选择下一个动作
                    # Select next action
                    next_action = self.policy.select_action(next_state)
                    actions.append(next_action)
                    action = next_action
                
                state = next_state
            
            # 如果可以更新τ时刻
            # If can update time τ
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
                if tau + n_actual < T and tau + n_actual < len(states) - 1:
                    bootstrap_state = states[tau + n_actual]
                    bootstrap_action = actions[tau + n_actual]
                    bootstrap_q = self.Q.get_value(bootstrap_state, bootstrap_action)
                    g_n += (self.gamma ** n_actual) * bootstrap_q
                
                # 更新Q函数
                # Update Q function
                update_state = states[tau]
                update_action = actions[tau]
                old_q = self.Q.get_value(update_state, update_action)
                alpha = self.alpha_func(self.step_count)
                new_q = old_q + alpha * (g_n - old_q)
                self.Q.set_value(update_state, update_action, new_q)
                
                # 记录TD误差
                # Record TD error
                td_error = g_n - old_q
                td_err_obj = TDError(
                    value=td_error,
                    timestep=self.step_count,
                    state=update_state,
                    next_state=states[tau + 1] if tau + 1 < len(states) else None,
                    reward=rewards[tau] if tau < len(rewards) else 0,
                    state_value=old_q,
                    next_state_value=g_n
                )
                self.td_analyzer.add_error(td_err_obj)
                
                self.step_count += 1
            
            t += 1
            
            # 检查是否结束
            # Check if done
            if tau == T - 1:
                break
        
        # 衰减ε
        # Decay ε
        self.policy.decay_epsilon()
        
        # 记录统计
        # Record statistics  
        self.episode_count += 1
        self.episode_returns.append(episode_return)
        self.episode_lengths.append(t)
        
        return episode_return, t
    
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
            print(f"\n开始{self.n}步SARSA学习: {n_episodes}回合")
            print(f"Starting {self.n}-step SARSA learning: {n_episodes} episodes")
            print(f"  参数: γ={self.gamma}, n={self.n}")
            print(f"  初始ε: {self.policy.epsilon:.3f}")
        
        for episode in range(n_episodes):
            episode_return, episode_length = self.learn_episode()
            
            if verbose and (episode + 1) % max(1, n_episodes // 10) == 0:
                avg_return = np.mean(self.episode_returns[-100:]) \
                           if len(self.episode_returns) >= 100 \
                           else np.mean(self.episode_returns)
                avg_length = np.mean(self.episode_lengths[-100:]) \
                           if len(self.episode_lengths) >= 100 \
                           else np.mean(self.episode_lengths)
                
                print(f"Episode {episode + 1}/{n_episodes}: "
                      f"Return={episode_return:.2f}, "
                      f"Avg Return={avg_return:.2f}, "
                      f"Avg Length={avg_length:.1f}, "
                      f"ε={self.policy.epsilon:.3f}")
        
        if verbose:
            print(f"\n{self.n}步SARSA学习完成!")
            print(f"  最终ε: {self.policy.epsilon:.3f}")
            print(f"  总步数: {self.step_count}")
        
        return self.Q


# ================================================================================
# 第7.2.2节：n步期望SARSA
# Section 7.2.2: n-step Expected SARSA
# ================================================================================

class NStepExpectedSARSA:
    """
    n步期望SARSA算法
    n-step Expected SARSA Algorithm
    
    使用期望而不是采样的下一个动作
    Uses expectation instead of sampled next action
    
    n步回报 n-step Return:
    G_t:t+n = R_{t+1} + ... + γ^{n-1}R_{t+n} + γ^n Σ_a π(a|S_{t+n})Q(S_{t+n}, a)
    
    优势 Advantages:
    - 比n步SARSA方差更低
      Lower variance than n-step SARSA
    - 仍然是on-policy
      Still on-policy
    - 更稳定的学习
      More stable learning
    
    代价 Cost:
    - 计算期望需要遍历所有动作
      Computing expectation requires iterating all actions
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 n: int = 4,
                 gamma: float = 0.99,
                 alpha: Union[float, Callable] = 0.1,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        初始化n步期望SARSA
        Initialize n-step Expected SARSA
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
        
        # 统计
        # Statistics
        self.episode_count = 0
        self.step_count = 0
        self.episode_returns = []
        self.episode_lengths = []
        
        logger.info(f"初始化{n}步期望SARSA: γ={gamma}, ε={epsilon}")
    
    def compute_expected_value(self, state: State) -> float:
        """
        计算状态的期望价值
        Compute expected value of state
        
        V_π(s) = Σ_a π(a|s) Q(s, a)
        
        Args:
            state: 状态
                  State
        
        Returns:
            期望价值
            Expected value
        """
        if state.is_terminal:
            return 0.0
        
        expected_value = 0.0
        
        # 获取动作概率
        # Get action probabilities
        for action in self.env.action_space:
            q_value = self.Q.get_value(state, action)
            
            # ε-贪婪策略的概率
            # Probability under ε-greedy policy
            q_values = [self.Q.get_value(state, a) for a in self.env.action_space]
            best_actions = [a for a, q in zip(self.env.action_space, q_values)
                          if q == max(q_values)]
            
            if action in best_actions:
                prob = (1 - self.policy.epsilon) / len(best_actions) + \
                       self.policy.epsilon / len(self.env.action_space)
            else:
                prob = self.policy.epsilon / len(self.env.action_space)
            
            expected_value += prob * q_value
        
        return expected_value
    
    def learn_episode(self) -> Tuple[float, int]:
        """
        学习一个回合
        Learn one episode
        """
        # 初始化轨迹存储
        # Initialize trajectory storage
        states = []
        actions = []
        rewards = []
        
        # 初始化S_0，选择A_0
        # Initialize S_0, select A_0
        state = self.env.reset()
        action = self.policy.select_action(state)
        
        states.append(state)
        actions.append(action)
        
        t = 0
        T = float('inf')
        episode_return = 0.0
        
        # 主循环
        # Main loop
        while True:
            # 时间步τ需要更新
            # Time step τ to update
            tau = t - self.n + 1
            
            if t < T:
                # 执行动作
                # Execute action
                next_state, reward, done, _ = self.env.step(action)
                
                # 存储
                # Store
                states.append(next_state)
                rewards.append(reward)
                
                # 累积回报
                # Accumulate return
                episode_return += reward * (self.gamma ** t)
                
                if done:
                    T = t + 1
                else:
                    # 选择下一个动作（但只用于执行，不用于更新）
                    # Select next action (only for execution, not for update)
                    next_action = self.policy.select_action(next_state)
                    actions.append(next_action)
                    action = next_action
                
                state = next_state
            
            # 如果可以更新τ时刻
            # If can update time τ
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
                
                # 添加期望bootstrap（关键差异）
                # Add expected bootstrap (key difference)
                if tau + n_actual < T and tau + n_actual < len(states) - 1:
                    bootstrap_state = states[tau + n_actual]
                    expected_value = self.compute_expected_value(bootstrap_state)
                    g_n += (self.gamma ** n_actual) * expected_value
                
                # 更新Q函数
                # Update Q function
                update_state = states[tau]
                update_action = actions[tau]
                old_q = self.Q.get_value(update_state, update_action)
                alpha = self.alpha_func(self.step_count)
                new_q = old_q + alpha * (g_n - old_q)
                self.Q.set_value(update_state, update_action, new_q)
                
                self.step_count += 1
            
            t += 1
            
            # 检查是否结束
            # Check if done
            if tau == T - 1:
                break
        
        # 衰减ε
        # Decay ε
        self.policy.decay_epsilon()
        
        # 记录统计
        # Record statistics
        self.episode_count += 1
        self.episode_returns.append(episode_return)
        self.episode_lengths.append(t)
        
        return episode_return, t


# ================================================================================
# 第7.2.3节：n步Q(σ)算法
# Section 7.2.3: n-step Q(σ) Algorithm
# ================================================================================

class NStepQSigma:
    """
    n步Q(σ)算法 - 统一的n步算法
    n-step Q(σ) Algorithm - Unified n-step Algorithm
    
    σ控制采样和期望的混合程度
    σ controls the degree of sampling vs expectation
    
    σ = 1: 完全采样（SARSA）
          Full sampling (SARSA)
    σ = 0: 完全期望（Tree Backup）
          Full expectation (Tree Backup)
    0 < σ < 1: 混合
             Mixed
    
    n步回报 n-step Return:
    G_t:t+n = R_{t+1} + γ(σ_{t+1}ρ_{t+1} + (1-σ_{t+1})π(A_{t+1}|S_{t+1}))
             × (G_{t+1:t+n} - Q(S_{t+1}, A_{t+1})) + γV̄(S_{t+1})
    
    其中 where:
    - ρ: 重要性采样比率
         Importance sampling ratio
    - V̄: 期望价值
         Expected value
    - σ: 控制参数
         Control parameter
    
    这是最通用的n步算法！
    This is the most general n-step algorithm!
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 n: int = 4,
                 gamma: float = 0.99,
                 alpha: Union[float, Callable] = 0.1,
                 sigma: Union[float, Callable] = 0.5,
                 epsilon: float = 0.1):
        """
        初始化n步Q(σ)
        Initialize n-step Q(σ)
        
        Args:
            env: 环境
                Environment
            n: 步数
               Number of steps
            gamma: 折扣因子
                  Discount factor
            alpha: 学习率
                  Learning rate
            sigma: σ参数（可以是函数）
                  σ parameter (can be function)
            epsilon: 探索率
                    Exploration rate
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
        
        # σ参数
        # σ parameter
        if callable(sigma):
            self.sigma_func = sigma
        else:
            self.sigma_func = lambda t: sigma
        
        # Q函数
        # Q function
        self.Q = ActionValueFunction(
            env.state_space,
            env.action_space,
            initial_value=0.0
        )
        
        # 策略
        # Policy
        self.policy = EpsilonGreedyPolicy(
            self.Q,
            epsilon=epsilon,
            epsilon_decay=1.0,  # 固定ε
            epsilon_min=epsilon,
            action_space=env.action_space
        )
        
        # 统计
        # Statistics
        self.episode_count = 0
        self.step_count = 0
        self.episode_returns = []
        
        logger.info(f"初始化{n}步Q(σ): γ={gamma}, σ={sigma}")
    
    def compute_n_step_return(self,
                              states: List[State],
                              actions: List[Action],
                              rewards: List[float],
                              tau: int,
                              T: int) -> float:
        """
        计算n步Q(σ)回报
        Compute n-step Q(σ) return
        
        这是算法的核心
        This is the core of the algorithm
        """
        n_actual = min(self.n, T - tau)
        g = 0.0
        
        for k in range(n_actual):
            t = tau + k
            
            if t < len(rewards):
                # 累积奖励
                # Accumulate reward
                g += (self.gamma ** k) * rewards[t]
                
                if t + 1 < len(states) and not states[t + 1].is_terminal:
                    # 获取σ值
                    # Get σ value
                    sigma = self.sigma_func(t)
                    
                    # 计算期望价值
                    # Compute expected value
                    expected_value = 0.0
                    for a in self.env.action_space:
                        q = self.Q.get_value(states[t + 1], a)
                        # ε-贪婪策略概率
                        # ε-greedy policy probability
                        prob = self.policy.epsilon / len(self.env.action_space)
                        if self.Q.get_value(states[t + 1], a) == max(
                            self.Q.get_value(states[t + 1], a2) 
                            for a2 in self.env.action_space
                        ):
                            prob += (1 - self.policy.epsilon) / sum(
                                1 for a2 in self.env.action_space
                                if self.Q.get_value(states[t + 1], a2) == max(
                                    self.Q.get_value(states[t + 1], a3)
                                    for a3 in self.env.action_space
                                )
                            )
                        expected_value += prob * q
                    
                    # 混合采样和期望
                    # Mix sampling and expectation
                    if t + 1 < len(actions):
                        sample_value = self.Q.get_value(states[t + 1], actions[t + 1])
                        mixed_value = sigma * sample_value + (1 - sigma) * expected_value
                    else:
                        mixed_value = expected_value
                    
                    g += (self.gamma ** (k + 1)) * mixed_value
        
        return g


# ================================================================================
# 主函数：演示n步SARSA
# Main Function: Demonstrate n-step SARSA
# ================================================================================

def demonstrate_n_step_sarsa():
    """
    演示n步SARSA控制算法
    Demonstrate n-step SARSA control algorithms
    """
    print("\n" + "="*80)
    print("第7.2节：n步SARSA控制")
    print("Section 7.2: n-step SARSA Control")
    print("="*80)
    
    from ch02_mdp.gridworld import GridWorld
    
    # 创建环境
    # Create environment
    env = GridWorld(rows=4, cols=4,
                   start_pos=(0,0),
                   goal_pos=(3,3),
                   obstacles=[(1,1), (2,2)])
    
    print(f"\n创建4×4 GridWorld（含障碍）")
    print(f"  起点: (0,0)")
    print(f"  终点: (3,3)")
    print(f"  障碍: (1,1), (2,2)")
    
    # 1. 测试不同n值的SARSA
    # 1. Test SARSA with different n values
    print("\n" + "="*60)
    print("1. 不同n值的SARSA")
    print("1. SARSA with Different n Values")
    print("="*60)
    
    n_values = [1, 2, 4, 8]
    
    for n in n_values:
        print(f"\n测试{n}步SARSA:")
        n_step_sarsa = NStepSARSA(
            env, n=n, gamma=0.99, alpha=0.1, epsilon=0.1
        )
        
        Q = n_step_sarsa.learn(n_episodes=200, verbose=False)
        
        # 显示结果
        # Show results
        avg_return = np.mean(n_step_sarsa.episode_returns[-50:])
        avg_length = np.mean(n_step_sarsa.episode_lengths[-50:])
        
        print(f"  最终平均回报: {avg_return:.2f}")
        print(f"  最终平均长度: {avg_length:.1f}")
        
        # 显示一些Q值
        # Show some Q values
        sample_state = env.state_space[0]
        if not sample_state.is_terminal:
            print(f"  Q值示例:")
            for action in env.action_space[:2]:
                q_value = Q.get_value(sample_state, action)
                print(f"    Q(s0, {action.id}) = {q_value:.3f}")
    
    # 2. 测试n步期望SARSA
    # 2. Test n-step Expected SARSA
    print("\n" + "="*60)
    print("2. n步期望SARSA")
    print("2. n-step Expected SARSA")
    print("="*60)
    
    n_step_expected = NStepExpectedSARSA(
        env, n=4, gamma=0.99, alpha=0.1, epsilon=0.1
    )
    
    Q_expected = n_step_expected.learn(n_episodes=200, verbose=False)
    
    avg_return = np.mean(n_step_expected.episode_returns[-50:])
    print(f"\n4步期望SARSA最终平均回报: {avg_return:.2f}")
    
    # 3. 比较SARSA vs 期望SARSA
    # 3. Compare SARSA vs Expected SARSA
    print("\n" + "="*60)
    print("3. SARSA vs 期望SARSA比较")
    print("3. SARSA vs Expected SARSA Comparison")
    print("="*60)
    
    n = 4
    n_runs = 5
    n_episodes = 200
    
    sarsa_returns = []
    expected_returns = []
    
    print(f"\n运行{n_runs}次实验，每次{n_episodes}回合...")
    
    for run in range(n_runs):
        # SARSA
        sarsa = NStepSARSA(env, n=n, gamma=0.99, alpha=0.1, epsilon=0.1)
        sarsa.learn(n_episodes=n_episodes, verbose=False)
        sarsa_returns.append(np.mean(sarsa.episode_returns[-50:]))
        
        # Expected SARSA
        expected = NStepExpectedSARSA(env, n=n, gamma=0.99, alpha=0.1, epsilon=0.1)
        expected.learn(n_episodes=n_episodes, verbose=False)
        expected_returns.append(np.mean(expected.episode_returns[-50:]))
    
    print(f"\n{n}步SARSA平均回报: {np.mean(sarsa_returns):.2f} ± {np.std(sarsa_returns):.2f}")
    print(f"{n}步期望SARSA平均回报: {np.mean(expected_returns):.2f} ± {np.std(expected_returns):.2f}")
    
    if np.mean(expected_returns) > np.mean(sarsa_returns):
        print("\n✓ 期望SARSA表现更好（方差更低）")
        print("✓ Expected SARSA performs better (lower variance)")
    else:
        print("\n✓ SARSA在此问题上表现更好")
        print("✓ SARSA performs better on this problem")
    
    # 总结
    # Summary
    print("\n" + "="*80)
    print("n步SARSA控制总结")
    print("n-step SARSA Control Summary")
    print("="*80)
    
    print("""
    关键要点 Key Takeaways:
    =======================
    
    1. n步SARSA扩展了SARSA到n步
       n-step SARSA extends SARSA to n steps
       
    2. 期望SARSA降低方差
       Expected SARSA reduces variance
       
    3. Q(σ)统一所有n步算法
       Q(σ) unifies all n-step algorithms
       
    4. n=4-8通常效果好
       n=4-8 usually works well
       
    5. 权衡：计算vs性能
       Tradeoff: computation vs performance
    """)


if __name__ == "__main__":
    demonstrate_n_step_sarsa()