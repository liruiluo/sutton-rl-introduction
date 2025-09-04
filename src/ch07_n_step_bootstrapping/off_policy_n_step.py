"""
================================================================================
第7.3节：Off-Policy n步方法 - 重要性采样修正
Section 7.3: Off-Policy n-step Methods - Importance Sampling Corrections
================================================================================

Off-policy学习的n步扩展！
n-step extension of off-policy learning!

核心挑战 Core Challenge:
行为策略b与目标策略π不同，需要修正
Behavior policy b differs from target policy π, needs correction

重要性采样比率 Importance Sampling Ratio:
ρ_t:t+n-1 = ∏_{k=t}^{min(t+n-1,T-1)} π(A_k|S_k) / b(A_k|S_k)

修正的n步回报 Corrected n-step Return:
G_t:t+n = ρ_t:t+n-1 [R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γ^n V(S_{t+n})]

关键技术 Key Techniques:
1. 重要性采样
   Importance sampling
2. 控制变量方法
   Control variates
3. Per-decision重要性采样
   Per-decision importance sampling
4. Tree Backup（无需重要性采样）
   Tree Backup (no importance sampling needed)

优势 Advantages:
- 可以从任意策略学习
  Can learn from any policy
- 数据效率高
  Data efficient
- 探索更安全
  Safer exploration

挑战 Challenges:
- 高方差
  High variance
- 重要性权重可能爆炸
  Importance weights may explode
- 需要已知行为策略概率
  Need known behavior policy probabilities
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
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

from src.ch03_finite_mdp.mdp_framework import State, Action, MDPEnvironment, MDPAgent
from src.ch03_finite_mdp.policies_and_values import (
    Policy, StateValueFunction, ActionValueFunction,
    StochasticPolicy, DeterministicPolicy
)
from src.ch05_monte_carlo.mc_control import EpsilonGreedyPolicy
from src.ch06_temporal_difference.td_foundations import TDError, TDErrorAnalyzer

# 设置日志
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第7.3.1节：重要性采样修正
# Section 7.3.1: Importance Sampling Correction
# ================================================================================

class ImportanceSamplingCorrection:
    """
    重要性采样修正计算
    Importance Sampling Correction Computation
    
    计算各种重要性采样比率
    Computes various importance sampling ratios
    
    关键公式 Key Formulas:
    1. 普通重要性采样 Ordinary IS:
       ρ = π(a|s) / b(a|s)
    
    2. 累积重要性采样 Cumulative IS:
       ρ_t:t+n = ∏_{k=t}^{t+n} ρ_k
    
    3. Per-decision IS:
       每步单独修正
       Correct each step separately
    
    实现技巧 Implementation Tricks:
    - 截断大的比率避免爆炸
      Truncate large ratios to avoid explosion
    - 使用对数空间避免数值问题
      Use log space to avoid numerical issues
    - 增量计算提高效率
      Incremental computation for efficiency
    """
    
    @staticmethod
    def compute_importance_ratio(
        target_prob: float,
        behavior_prob: float,
        truncate: Optional[float] = None
    ) -> float:
        """
        计算单步重要性采样比率
        Compute single-step importance sampling ratio
        
        Args:
            target_prob: 目标策略概率π(a|s)
                        Target policy probability
            behavior_prob: 行为策略概率b(a|s)
                          Behavior policy probability
            truncate: 截断阈值（可选）
                     Truncation threshold (optional)
        
        Returns:
            重要性采样比率
            Importance sampling ratio
        """
        # 避免除零
        # Avoid division by zero
        if behavior_prob < 1e-10:
            return 0.0
        
        ratio = target_prob / behavior_prob
        
        # 截断（如果需要）
        # Truncate (if needed)
        if truncate is not None:
            ratio = min(ratio, truncate)
        
        return ratio
    
    @staticmethod
    def compute_cumulative_ratio(
        target_probs: List[float],
        behavior_probs: List[float],
        truncate: Optional[float] = None
    ) -> float:
        """
        计算累积重要性采样比率
        Compute cumulative importance sampling ratio
        
        Args:
            target_probs: 目标策略概率序列
                         Target policy probability sequence
            behavior_probs: 行为策略概率序列
                           Behavior policy probability sequence
            truncate: 截断阈值
                     Truncation threshold
        
        Returns:
            累积比率ρ_t:t+n
            Cumulative ratio
        """
        ratio = 1.0
        
        for target_p, behavior_p in zip(target_probs, behavior_probs):
            step_ratio = ImportanceSamplingCorrection.compute_importance_ratio(
                target_p, behavior_p, truncate
            )
            ratio *= step_ratio
            
            # 早停如果比率太小
            # Early stop if ratio too small
            if ratio < 1e-10:
                return 0.0
        
        return ratio
    
    @staticmethod
    def compute_per_decision_ratios(
        target_probs: List[float],
        behavior_probs: List[float],
        gamma: float,
        truncate: Optional[float] = None
    ) -> List[float]:
        """
        计算per-decision重要性采样比率
        Compute per-decision importance sampling ratios
        
        每个奖励单独修正
        Correct each reward separately
        
        Args:
            target_probs: 目标策略概率
                         Target policy probabilities
            behavior_probs: 行为策略概率
                           Behavior policy probabilities
            gamma: 折扣因子
                  Discount factor
            truncate: 截断阈值
                     Truncation threshold
        
        Returns:
            Per-decision比率列表
            List of per-decision ratios
        """
        ratios = []
        cumulative = 1.0
        
        for i, (target_p, behavior_p) in enumerate(zip(target_probs, behavior_probs)):
            # 只需要到这一步的累积比率
            # Only need cumulative ratio up to this step
            step_ratio = ImportanceSamplingCorrection.compute_importance_ratio(
                target_p, behavior_p, truncate
            )
            cumulative *= step_ratio
            ratios.append(cumulative * (gamma ** i))
        
        return ratios


# ================================================================================
# 第7.3.2节：Off-Policy n步TD
# Section 7.3.2: Off-Policy n-step TD
# ================================================================================

class OffPolicyNStepTD:
    """
    Off-Policy n步TD预测
    Off-Policy n-step TD Prediction
    
    使用重要性采样从行为策略学习目标策略的价值
    Learn target policy value from behavior policy using importance sampling
    
    算法步骤 Algorithm Steps:
    1. 使用行为策略b收集数据
       Collect data using behavior policy b
    2. 计算重要性采样比率ρ
       Compute importance sampling ratio ρ
    3. 使用修正的回报更新价值函数
       Update value function with corrected return
    
    更新规则 Update Rule:
    V(S_t) ← V(S_t) + α ρ_t:t+n-1 [G_t:t+n - V(S_t)]
    
    关键点 Key Points:
    - 高方差是主要问题
      High variance is main issue
    - 需要知道b(a|s)
      Need to know b(a|s)
    - 可以使用各种方差减少技术
      Can use various variance reduction techniques
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 n: int = 4,
                 gamma: float = 0.99,
                 alpha: Union[float, Callable] = 0.1,
                 truncate_ratio: Optional[float] = 5.0):
        """
        初始化Off-Policy n步TD
        Initialize Off-Policy n-step TD
        
        Args:
            env: 环境
                Environment
            n: 步数
               Number of steps
            gamma: 折扣因子
                  Discount factor
            alpha: 学习率
                  Learning rate
            truncate_ratio: 重要性采样比率截断值
                           IS ratio truncation value
        """
        self.env = env
        self.n = n
        self.gamma = gamma
        self.truncate_ratio = truncate_ratio
        
        # 学习率
        # Learning rate
        if callable(alpha):
            self.alpha_func = alpha
        else:
            self.alpha_func = lambda t: alpha
        
        # 价值函数（目标策略）
        # Value function (target policy)
        self.V = StateValueFunction(env.state_space, initial_value=0.0)
        
        # TD误差分析
        # TD error analysis
        self.td_analyzer = TDErrorAnalyzer()
        
        # 统计
        # Statistics
        self.episode_count = 0
        self.step_count = 0
        self.episode_returns = []
        self.importance_ratios_history = []
        
        logger.info(f"初始化Off-Policy {n}步TD: γ={gamma}")
    
    def learn_episode(self,
                     behavior_policy: Policy,
                     target_policy: Policy) -> float:
        """
        学习一个回合
        Learn one episode
        
        Args:
            behavior_policy: 行为策略b
                           Behavior policy b
            target_policy: 目标策略π
                         Target policy π
        
        Returns:
            回合回报（在目标策略下的期望）
            Episode return (expected under target policy)
        """
        # 初始化轨迹存储
        # Initialize trajectory storage
        states = []
        actions = []
        rewards = []
        target_probs = []
        behavior_probs = []
        
        # 初始化
        # Initialize
        state = self.env.reset()
        states.append(state)
        
        t = 0
        T = float('inf')
        episode_return = 0.0
        
        # 收集轨迹
        # Collect trajectory
        while True:
            if t < T:
                # 使用行为策略选择动作
                # Select action using behavior policy
                action = behavior_policy.select_action(state)
                actions.append(action)
                
                # 获取概率（用于重要性采样）
                # Get probabilities (for importance sampling)
                if hasattr(behavior_policy, 'get_probability'):
                    b_prob = behavior_policy.get_probability(state, action)
                else:
                    # 假设ε-贪婪
                    # Assume ε-greedy
                    if hasattr(behavior_policy, 'epsilon'):
                        eps = behavior_policy.epsilon
                        n_actions = len(self.env.action_space)
                        if hasattr(behavior_policy, 'Q'):
                            q_values = [behavior_policy.Q.get_value(state, a) 
                                      for a in self.env.action_space]
                            is_greedy = action in [a for a, q in zip(self.env.action_space, q_values)
                                                  if q == max(q_values)]
                            b_prob = (1 - eps) / sum(1 for q in q_values if q == max(q_values)) + eps / n_actions if is_greedy else eps / n_actions
                        else:
                            b_prob = 1.0 / n_actions
                    else:
                        b_prob = 1.0 / len(self.env.action_space)
                
                if hasattr(target_policy, 'get_probability'):
                    pi_prob = target_policy.get_probability(state, action)
                else:
                    # 假设贪婪或ε-贪婪
                    # Assume greedy or ε-greedy
                    if hasattr(target_policy, 'epsilon'):
                        eps = target_policy.epsilon
                        n_actions = len(self.env.action_space)
                        if hasattr(target_policy, 'Q'):
                            q_values = [target_policy.Q.get_value(state, a)
                                      for a in self.env.action_space]
                            is_greedy = action in [a for a, q in zip(self.env.action_space, q_values)
                                                  if q == max(q_values)]
                            pi_prob = (1 - eps) / sum(1 for q in q_values if q == max(q_values)) + eps / n_actions if is_greedy else eps / n_actions
                        else:
                            pi_prob = 1.0 / n_actions
                    else:
                        pi_prob = 1.0 / len(self.env.action_space)
                
                target_probs.append(pi_prob)
                behavior_probs.append(b_prob)
                
                # 执行动作
                # Execute action
                next_state, reward, done, _ = self.env.step(action)
                rewards.append(reward)
                states.append(next_state)
                
                episode_return += reward * (self.gamma ** t)
                
                if done:
                    T = t + 1
                
                state = next_state
            
            # 时间步τ需要更新
            # Time step τ to update
            tau = t - self.n + 1
            
            if tau >= 0:
                # 计算重要性采样比率
                # Compute importance sampling ratio
                n_actual = min(self.n, T - tau)
                
                # 获取相关概率
                # Get relevant probabilities
                target_p = target_probs[tau:tau+n_actual] if tau+n_actual <= len(target_probs) else target_probs[tau:]
                behavior_p = behavior_probs[tau:tau+n_actual] if tau+n_actual <= len(behavior_probs) else behavior_probs[tau:]
                
                # 计算累积比率
                # Compute cumulative ratio
                rho = ImportanceSamplingCorrection.compute_cumulative_ratio(
                    target_p, behavior_p, self.truncate_ratio
                )
                
                self.importance_ratios_history.append(rho)
                
                # 计算n步回报
                # Compute n-step return
                g_n = 0.0
                for i in range(n_actual):
                    if tau + i < len(rewards):
                        g_n += (self.gamma ** i) * rewards[tau + i]
                
                # Bootstrap
                if tau + n_actual < T and tau + n_actual < len(states):
                    g_n += (self.gamma ** n_actual) * self.V.get_value(states[tau + n_actual])
                
                # 更新价值函数（使用重要性采样修正）
                # Update value function (with IS correction)
                update_state = states[tau]
                old_v = self.V.get_value(update_state)
                alpha = self.alpha_func(self.step_count)
                
                # 重要性采样修正的更新
                # Importance sampling corrected update
                new_v = old_v + alpha * rho * (g_n - old_v)
                self.V.set_value(update_state, new_v)
                
                # 记录TD误差
                # Record TD error
                td_error = rho * (g_n - old_v)
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
            
            t += 1
            
            if tau == T - 1:
                break
        
        # 记录统计
        # Record statistics
        self.episode_count += 1
        self.episode_returns.append(episode_return)
        
        return episode_return


# ================================================================================
# 第7.3.3节：Off-Policy n步SARSA
# Section 7.3.3: Off-Policy n-step SARSA
# ================================================================================

class OffPolicyNStepSARSA:
    """
    Off-Policy n步SARSA
    Off-Policy n-step SARSA
    
    n步SARSA的off-policy版本
    Off-policy version of n-step SARSA
    
    更新规则 Update Rule:
    Q(S_τ, A_τ) ← Q(S_τ, A_τ) + α ρ_{τ+1:τ+n-1} [G_{τ:τ+n} - Q(S_τ, A_τ)]
    
    其中 where:
    ρ_{τ+1:τ+n-1} = ∏_{k=τ+1}^{min(τ+n-1,T-1)} π(A_k|S_k) / b(A_k|S_k)
    
    注意 Note:
    - 第一个动作A_τ不包含在比率中
      First action A_τ not included in ratio
    - 这允许更稳定的学习
      This allows more stable learning
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 n: int = 4,
                 gamma: float = 0.99,
                 alpha: Union[float, Callable] = 0.1,
                 epsilon_behavior: float = 0.3,
                 epsilon_target: float = 0.1,
                 truncate_ratio: Optional[float] = 5.0):
        """
        初始化Off-Policy n步SARSA
        Initialize Off-Policy n-step SARSA
        """
        self.env = env
        self.n = n
        self.gamma = gamma
        self.truncate_ratio = truncate_ratio
        
        # 学习率
        # Learning rate
        if callable(alpha):
            self.alpha_func = alpha
        else:
            self.alpha_func = lambda t: alpha
        
        # Q函数（共享）
        # Q function (shared)
        self.Q = ActionValueFunction(
            env.state_space,
            env.action_space,
            initial_value=0.0
        )
        
        # 行为策略（更多探索）
        # Behavior policy (more exploration)
        self.behavior_policy = EpsilonGreedyPolicy(
            self.Q,
            epsilon=epsilon_behavior,
            epsilon_decay=1.0,
            epsilon_min=epsilon_behavior,
            action_space=env.action_space
        )
        
        # 目标策略（更贪婪）
        # Target policy (more greedy)
        self.target_policy = EpsilonGreedyPolicy(
            self.Q,
            epsilon=epsilon_target,
            epsilon_decay=1.0,
            epsilon_min=epsilon_target,
            action_space=env.action_space
        )
        
        # 统计
        # Statistics
        self.episode_count = 0
        self.step_count = 0
        self.episode_returns = []
        self.episode_lengths = []
        self.importance_ratios_history = []
        
        logger.info(f"初始化Off-Policy {n}步SARSA: "
                   f"ε_b={epsilon_behavior}, ε_π={epsilon_target}")
    
    def get_action_probability(self, policy: EpsilonGreedyPolicy, 
                              state: State, action: Action) -> float:
        """
        获取动作概率
        Get action probability
        """
        q_values = [self.Q.get_value(state, a) for a in self.env.action_space]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(self.env.action_space, q_values) if q == max_q]
        
        if action in best_actions:
            return (1 - policy.epsilon) / len(best_actions) + policy.epsilon / len(self.env.action_space)
        else:
            return policy.epsilon / len(self.env.action_space)
    
    def learn_episode(self) -> Tuple[float, int]:
        """
        学习一个回合
        Learn one episode
        """
        # 初始化
        # Initialize
        states = []
        actions = []
        rewards = []
        
        state = self.env.reset()
        action = self.behavior_policy.select_action(state)
        
        states.append(state)
        actions.append(action)
        
        t = 0
        T = float('inf')
        episode_return = 0.0
        
        # 主循环
        # Main loop
        while True:
            # τ时刻需要更新
            # Time τ to update
            tau = t - self.n + 1
            
            if t < T:
                # 执行动作
                # Execute action
                next_state, reward, done, _ = self.env.step(action)
                
                states.append(next_state)
                rewards.append(reward)
                episode_return += reward * (self.gamma ** t)
                
                if done:
                    T = t + 1
                else:
                    # 选择下一个动作（使用行为策略）
                    # Select next action (using behavior policy)
                    next_action = self.behavior_policy.select_action(next_state)
                    actions.append(next_action)
                    action = next_action
                
                state = next_state
            
            # 更新
            # Update
            if tau >= 0:
                # 计算重要性采样比率（从τ+1到τ+n-1）
                # Compute IS ratio (from τ+1 to τ+n-1)
                n_actual = min(self.n - 1, T - tau - 1)
                rho = 1.0
                
                for k in range(n_actual):
                    idx = tau + 1 + k
                    if idx < len(actions) and idx < len(states):
                        pi_prob = self.get_action_probability(
                            self.target_policy, states[idx], actions[idx]
                        )
                        b_prob = self.get_action_probability(
                            self.behavior_policy, states[idx], actions[idx]
                        )
                        
                        step_ratio = ImportanceSamplingCorrection.compute_importance_ratio(
                            pi_prob, b_prob, self.truncate_ratio
                        )
                        rho *= step_ratio
                
                self.importance_ratios_history.append(rho)
                
                # 计算n步回报
                # Compute n-step return
                n_steps = min(self.n, T - tau)
                g_n = 0.0
                
                for i in range(n_steps):
                    if tau + i < len(rewards):
                        g_n += (self.gamma ** i) * rewards[tau + i]
                
                # Bootstrap
                if tau + n_steps < T and tau + n_steps < len(actions):
                    bootstrap_q = self.Q.get_value(states[tau + n_steps], actions[tau + n_steps])
                    g_n += (self.gamma ** n_steps) * bootstrap_q
                
                # 更新Q函数
                # Update Q function
                update_state = states[tau]
                update_action = actions[tau]
                old_q = self.Q.get_value(update_state, update_action)
                alpha = self.alpha_func(self.step_count)
                
                # Off-policy更新
                # Off-policy update
                new_q = old_q + alpha * rho * (g_n - old_q)
                self.Q.set_value(update_state, update_action, new_q)
                
                self.step_count += 1
            
            t += 1
            
            if tau == T - 1:
                break
        
        # 记录统计
        # Record statistics
        self.episode_count += 1
        self.episode_returns.append(episode_return)
        self.episode_lengths.append(t)
        
        return episode_return, t


# ================================================================================
# 主函数：演示Off-Policy n步方法
# Main Function: Demonstrate Off-Policy n-step Methods
# ================================================================================

def demonstrate_off_policy_n_step():
    """
    演示Off-Policy n步方法
    Demonstrate Off-Policy n-step methods
    """
    print("\n" + "="*80)
    print("第7.3节：Off-Policy n步方法")
    print("Section 7.3: Off-Policy n-step Methods")
    print("="*80)
    
    from src.ch03_finite_mdp.gridworld import GridWorld
    from src.ch03_finite_mdp.policies_and_values import UniformRandomPolicy
    
    # 创建环境
    # Create environment
    env = GridWorld(rows=4, cols=4,
                   start_pos=(0,0),
                   goal_pos=(3,3))
    
    print(f"\n创建4×4 GridWorld")
    print(f"  起点: (0,0)")
    print(f"  终点: (3,3)")
    
    # 1. 演示重要性采样修正
    # 1. Demonstrate importance sampling correction
    print("\n" + "="*60)
    print("1. 重要性采样修正演示")
    print("1. Importance Sampling Correction Demo")
    print("="*60)
    
    # 示例概率
    # Example probabilities
    target_probs = [0.9, 0.8, 0.7]  # 目标策略更确定
    behavior_probs = [0.4, 0.3, 0.3]  # 行为策略更随机
    
    print(f"\n目标策略概率: {target_probs}")
    print(f"行为策略概率: {behavior_probs}")
    
    # 计算各种比率
    # Compute various ratios
    single_ratios = [ImportanceSamplingCorrection.compute_importance_ratio(tp, bp)
                     for tp, bp in zip(target_probs, behavior_probs)]
    
    cumulative_ratio = ImportanceSamplingCorrection.compute_cumulative_ratio(
        target_probs, behavior_probs
    )
    
    per_decision_ratios = ImportanceSamplingCorrection.compute_per_decision_ratios(
        target_probs, behavior_probs, gamma=0.9
    )
    
    print(f"\n单步比率: {[f'{r:.2f}' for r in single_ratios]}")
    print(f"累积比率: {cumulative_ratio:.2f}")
    print(f"Per-decision比率: {[f'{r:.2f}' for r in per_decision_ratios]}")
    
    # 2. 测试Off-Policy n步TD
    # 2. Test Off-Policy n-step TD
    print("\n" + "="*60)
    print("2. Off-Policy n步TD预测")
    print("2. Off-Policy n-step TD Prediction")
    print("="*60)
    
    # 创建行为策略（随机）和目标策略（更贪婪）
    # Create behavior policy (random) and target policy (more greedy)
    behavior_policy = UniformRandomPolicy(env.action_space)
    
    # 创建一个简单的贪婪目标策略
    # Create a simple greedy target policy
    from src.ch03_finite_mdp.policies_and_values import ActionValueFunction
    Q_target = ActionValueFunction(env.state_space, env.action_space, initial_value=0.0)
    # 手动设置一些Q值使策略有倾向性
    # Manually set some Q values to make policy biased
    for state in env.state_space[:5]:
        if not state.is_terminal:
            # 偏好向右和向下
            # Prefer right and down
            Q_target.set_value(state, env.action_space[1], 1.0)  # right
            Q_target.set_value(state, env.action_space[3], 1.0)  # down
    
    target_policy = EpsilonGreedyPolicy(
        Q_target, epsilon=0.1, epsilon_decay=1.0, 
        epsilon_min=0.1, action_space=env.action_space
    )
    
    off_policy_td = OffPolicyNStepTD(env, n=4, gamma=0.9, alpha=0.1)
    
    print("\n学习100回合...")
    for episode in range(100):
        episode_return = off_policy_td.learn_episode(behavior_policy, target_policy)
        
        if (episode + 1) % 20 == 0:
            avg_return = np.mean(off_policy_td.episode_returns[-20:])
            avg_ratio = np.mean(off_policy_td.importance_ratios_history[-100:]) \
                       if off_policy_td.importance_ratios_history else 0
            print(f"  Episode {episode + 1}: "
                  f"Avg Return={avg_return:.2f}, "
                  f"Avg IS Ratio={avg_ratio:.2f}")
    
    # 3. 测试Off-Policy n步SARSA
    # 3. Test Off-Policy n-step SARSA
    print("\n" + "="*60)
    print("3. Off-Policy n步SARSA控制")
    print("3. Off-Policy n-step SARSA Control")
    print("="*60)
    
    off_policy_sarsa = OffPolicyNStepSARSA(
        env, n=4, gamma=0.99, alpha=0.1,
        epsilon_behavior=0.3,  # 行为策略更探索
        epsilon_target=0.1     # 目标策略更贪婪
    )
    
    print(f"\n行为策略ε=0.3（更多探索）")
    print(f"目标策略ε=0.1（更贪婪）")
    print("\n学习200回合...")
    
    for episode in range(200):
        episode_return, episode_length = off_policy_sarsa.learn_episode()
        
        if (episode + 1) % 50 == 0:
            avg_return = np.mean(off_policy_sarsa.episode_returns[-50:])
            avg_length = np.mean(off_policy_sarsa.episode_lengths[-50:])
            avg_ratio = np.mean(off_policy_sarsa.importance_ratios_history[-100:]) \
                       if off_policy_sarsa.importance_ratios_history else 0
            
            print(f"  Episode {episode + 1}: "
                  f"Avg Return={avg_return:.2f}, "
                  f"Avg Length={avg_length:.1f}, "
                  f"Avg IS Ratio={avg_ratio:.2f}")
    
    # 显示学习的Q值
    # Show learned Q values
    print("\n学习的Q值（部分）:")
    sample_state = env.state_space[0]
    if not sample_state.is_terminal:
        for action in env.action_space:
            q_value = off_policy_sarsa.Q.get_value(sample_state, action)
            print(f"  Q(s0, {action.id}) = {q_value:.3f}")
    
    # 总结
    # Summary
    print("\n" + "="*80)
    print("Off-Policy n步方法总结")
    print("Off-Policy n-step Methods Summary")
    print("="*80)
    
    print("""
    关键要点 Key Takeaways:
    =======================
    
    1. 重要性采样修正偏差
       Importance sampling corrects bias
       
    2. 高方差是主要挑战
       High variance is main challenge
       
    3. 截断比率有助于稳定
       Truncating ratios helps stability
       
    4. Per-decision IS更精确
       Per-decision IS is more precise
       
    5. Off-policy允许安全探索
       Off-policy allows safe exploration
    
    实践建议 Practical Tips:
    - 使用截断避免比率爆炸
      Use truncation to avoid ratio explosion
    - 考虑控制变量减少方差
      Consider control variates to reduce variance
    - Tree Backup避免重要性采样
      Tree Backup avoids importance sampling
    """)


if __name__ == "__main__":
    demonstrate_off_policy_n_step()