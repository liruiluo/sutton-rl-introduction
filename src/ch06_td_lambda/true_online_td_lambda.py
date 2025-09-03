"""
================================================================================
第6.4节：真在线TD(λ) - 最新理论进展
Section 6.4: True Online TD(λ) - Latest Theoretical Advances
================================================================================

真在线TD(λ)是资格迹理论的重要突破！
True Online TD(λ) is a major breakthrough in eligibility trace theory!

传统TD(λ)的问题 Problems with Traditional TD(λ):
- 离线λ-回报与在线资格迹在函数近似下不等价
  Offline λ-return and online eligibility traces not equivalent with function approximation
- 需要特征向量归一化
  Requires feature vector normalization

真在线TD(λ)的创新 Innovations of True Online TD(λ):
1. Dutch迹：e = γλe + α(1 - γλe^T x)x
            Dutch traces
2. 精确等价于离线λ-回报（即使有函数近似）
   Exactly equivalent to offline λ-return (even with function approximation)
3. 更快收敛
   Faster convergence
4. 更好的理论性质
   Better theoretical properties

van Seijen & Sutton (2014)的重要贡献！
Important contribution by van Seijen & Sutton (2014)!

实践价值 Practical Value:
- 现代深度RL中的标准方法
  Standard method in modern deep RL
- A3C等算法的基础
  Foundation of algorithms like A3C
- 理论和实践的完美结合
  Perfect combination of theory and practice
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

# 设置日志
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第6.4.1节：真在线TD(λ)预测
# Section 6.4.1: True Online TD(λ) Prediction
# ================================================================================

class TrueOnlineTDLambda:
    """
    真在线TD(λ)预测
    True Online TD(λ) Prediction
    
    van Seijen & Sutton (2014)的算法
    Algorithm by van Seijen & Sutton (2014)
    
    关键创新 Key Innovations:
    1. Dutch迹（考虑特征重叠）
       Dutch traces (considering feature overlap)
    2. 修正的更新规则
       Modified update rule
    3. 精确等价性
       Exact equivalence
    
    算法 Algorithm:
    初始化 Initialize: V = 0, e = 0, V_old = 0
    对每个时间步 For each timestep:
        δ = R + γV(S') - V_old
        e = γλe + α(1 - γλe^T x)x
        V = V + δe + α(V_old - V(S))x
        V_old = V(S')
    
    其中x是状态S的特征向量
    where x is feature vector of state S
    
    优势 Advantages:
    - 真正的在线学习
      True online learning
    - 理论等价性
      Theoretical equivalence
    - 更快收敛
      Faster convergence
    - 无需归一化
      No normalization needed
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 gamma: float = 0.99,
                 lambda_: float = 0.9,
                 alpha: Union[float, Callable] = 0.1,
                 feature_func: Optional[Callable] = None):
        """
        初始化真在线TD(λ)
        Initialize True Online TD(λ)
        
        Args:
            env: 环境
                Environment
            gamma: 折扣因子
                  Discount factor
            lambda_: λ参数
                    Lambda parameter
            alpha: 学习率
                  Learning rate
            feature_func: 特征函数（可选）
                         Feature function (optional)
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
        
        # 特征函数（默认使用表格特征）
        # Feature function (default to tabular features)
        if feature_func is None:
            # 表格特征：每个状态一个特征
            # Tabular features: one feature per state
            self.feature_func = self._tabular_features
            self.n_features = len(env.state_space)
        else:
            self.feature_func = feature_func
            # 需要探测特征维度
            # Need to probe feature dimension
            test_features = feature_func(env.state_space[0])
            self.n_features = len(test_features)
        
        # 价值函数参数（线性函数近似）
        # Value function parameters (linear function approximation)
        self.w = np.zeros(self.n_features)
        
        # 资格迹（Dutch迹）
        # Eligibility trace (Dutch trace)
        self.e = np.zeros(self.n_features)
        
        # 上一步的价值
        # Previous value
        self.V_old = 0.0
        
        # TD误差分析
        # TD error analysis
        self.td_analyzer = TDErrorAnalyzer()
        
        # 统计
        # Statistics
        self.episode_count = 0
        self.step_count = 0
        self.episode_returns = []
        self.weight_norms = []
        self.trace_norms = []
        
        logger.info(f"初始化真在线TD(λ): γ={gamma}, λ={lambda_}, "
                   f"特征维度={self.n_features}")
    
    def _tabular_features(self, state: State) -> np.ndarray:
        """
        表格特征（one-hot编码）
        Tabular features (one-hot encoding)
        
        Args:
            state: 状态
                  State
        
        Returns:
            特征向量
            Feature vector
        """
        features = np.zeros(self.n_features)
        
        # 找到状态索引
        # Find state index
        for i, s in enumerate(self.env.state_space):
            if s == state:
                features[i] = 1.0
                break
        
        return features
    
    def get_value(self, state: State) -> float:
        """
        获取状态价值
        Get state value
        
        V(s) = w^T x(s)
        
        Args:
            state: 状态
                  State
        
        Returns:
            状态价值
            State value
        """
        if state.is_terminal:
            return 0.0
        
        features = self.feature_func(state)
        return np.dot(self.w, features)
    
    def learn_episode(self, policy: Policy) -> float:
        """
        学习一个回合
        Learn one episode
        
        真在线更新
        True online updates
        
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
        self.e = np.zeros(self.n_features)  # 重置迹
        self.V_old = self.get_value(state)
        
        episode_return = 0.0
        episode_steps = 0
        
        while True:
            # 获取当前特征
            # Get current features
            x = self.feature_func(state)
            
            # 选择动作
            # Select action
            action = policy.select_action(state)
            
            # 执行动作
            # Execute action
            next_state, reward, done, _ = self.env.step(action)
            
            # 获取价值
            # Get values
            V_current = np.dot(self.w, x)  # 更新前的价值
            V_next = self.get_value(next_state) if not done else 0.0
            
            # 计算TD误差
            # Compute TD error
            td_error = reward + self.gamma * V_next - self.V_old
            
            # 获取学习率
            # Get learning rate
            alpha = self.alpha_func(self.step_count)
            
            # 更新Dutch迹（真在线的关键）
            # Update Dutch trace (key to true online)
            e_dot_x = np.dot(self.e, x)
            self.e = self.gamma * self.lambda_ * self.e + \
                    alpha * (1.0 - self.gamma * self.lambda_ * e_dot_x) * x
            
            # 更新权重（真在线更新规则）
            # Update weights (true online update rule)
            self.w = self.w + td_error * self.e + \
                    alpha * (self.V_old - V_current) * x
            
            # 记录TD误差
            # Record TD error
            td_err_obj = TDError(
                value=td_error,
                timestep=self.step_count,
                state=state,
                next_state=next_state,
                reward=reward,
                state_value=self.V_old,
                next_state_value=V_next
            )
            self.td_analyzer.add_error(td_err_obj)
            
            # 更新V_old
            # Update V_old
            self.V_old = V_next
            
            # 累积回报和统计
            # Accumulate return and statistics
            episode_return += reward * (self.gamma ** episode_steps)
            episode_steps += 1
            self.step_count += 1
            
            # 记录范数
            # Record norms
            self.weight_norms.append(np.linalg.norm(self.w))
            self.trace_norms.append(np.linalg.norm(self.e))
            
            # 转移状态
            # Transition state
            state = next_state
            
            if done:
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
            print(f"\n开始真在线TD(λ)学习: {n_episodes}回合")
            print(f"Starting True Online TD(λ) learning: {n_episodes} episodes")
            print(f"  参数: γ={self.gamma}, λ={self.lambda_}")
            print(f"  特征维度: {self.n_features}")
        
        for episode in range(n_episodes):
            episode_return = self.learn_episode(policy)
            
            if verbose and (episode + 1) % max(1, n_episodes // 10) == 0:
                avg_return = np.mean(self.episode_returns[-100:]) \
                           if len(self.episode_returns) >= 100 \
                           else np.mean(self.episode_returns)
                
                stats = self.td_analyzer.get_statistics()
                avg_weight_norm = np.mean(self.weight_norms[-1000:]) \
                                if len(self.weight_norms) >= 1000 \
                                else np.mean(self.weight_norms)
                
                print(f"Episode {episode + 1}/{n_episodes}: "
                      f"Return={episode_return:.2f}, "
                      f"Avg Return={avg_return:.2f}, "
                      f"TD Error={stats.get('recent_abs_mean', 0):.4f}, "
                      f"||w||={avg_weight_norm:.2f}")
        
        # 创建价值函数对象
        # Create value function object
        V = StateValueFunction(self.env.state_space, initial_value=0.0)
        for state in self.env.state_space:
            if not state.is_terminal:
                V.set_value(state, self.get_value(state))
        
        if verbose:
            print(f"\n学习完成!")
            print(f"  最终||w||: {np.linalg.norm(self.w):.2f}")
            print(f"  最终平均回报: {np.mean(self.episode_returns[-100:]):.2f}")
        
        return V


# ================================================================================
# 第6.4.2节：真在线SARSA(λ)
# Section 6.4.2: True Online SARSA(λ)
# ================================================================================

class TrueOnlineSARSALambda:
    """
    真在线SARSA(λ)
    True Online SARSA(λ)
    
    将真在线思想扩展到控制
    Extending true online idea to control
    
    算法 Algorithm:
    初始化 Initialize: Q = 0, e = 0, Q_old = 0
    对每个时间步 For each timestep:
        取动作A，观察R,S'，选择A'
        Take A, observe R,S', choose A'
        x = φ(S,A), x' = φ(S',A')
        δ = R + γQ_old - Q_old
        e = γλe + α(1 - γλe^T x)x
        Q = Q + δe + α(Q_old - Q(S,A))x
        Q_old = Q(S',A')
        S = S', A = A'
    
    特点 Characteristics:
    - 精确等价性
      Exact equivalence
    - 更快收敛
      Faster convergence
    - 适合函数近似
      Suitable for function approximation
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 gamma: float = 0.99,
                 lambda_: float = 0.9,
                 alpha: Union[float, Callable] = 0.1,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 feature_func: Optional[Callable] = None):
        """
        初始化真在线SARSA(λ)
        Initialize True Online SARSA(λ)
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
        
        # 特征函数
        # Feature function
        if feature_func is None:
            # 表格特征
            # Tabular features
            self.feature_func = self._tabular_features
            self.n_features = len(env.state_space) * len(env.action_space)
        else:
            self.feature_func = feature_func
            # 探测维度
            # Probe dimension
            test_features = feature_func(env.state_space[0], env.action_space[0])
            self.n_features = len(test_features)
        
        # Q函数参数
        # Q function parameters
        self.w = np.zeros(self.n_features)
        
        # 资格迹
        # Eligibility trace
        self.e = np.zeros(self.n_features)
        
        # 上一步的Q值
        # Previous Q value
        self.Q_old = 0.0
        
        # 创建Q函数对象（用于策略）
        # Create Q function object (for policy)
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
        self.weight_norms = []
        
        logger.info(f"初始化真在线SARSA(λ): γ={gamma}, λ={lambda_}, "
                   f"ε={epsilon}, 特征维度={self.n_features}")
    
    def _tabular_features(self, state: State, action: Action) -> np.ndarray:
        """
        表格特征（状态-动作对的one-hot编码）
        Tabular features (one-hot encoding of state-action pair)
        """
        features = np.zeros(self.n_features)
        
        # 计算索引
        # Compute index
        state_idx = None
        action_idx = None
        
        for i, s in enumerate(self.env.state_space):
            if s == state:
                state_idx = i
                break
        
        for i, a in enumerate(self.env.action_space):
            if a == action:
                action_idx = i
                break
        
        if state_idx is not None and action_idx is not None:
            idx = state_idx * len(self.env.action_space) + action_idx
            features[idx] = 1.0
        
        return features
    
    def get_q_value(self, state: State, action: Action) -> float:
        """
        获取Q值
        Get Q value
        
        Q(s,a) = w^T φ(s,a)
        """
        if state.is_terminal:
            return 0.0
        
        features = self.feature_func(state, action)
        return np.dot(self.w, features)
    
    def update_Q_object(self):
        """
        更新Q函数对象（供策略使用）
        Update Q function object (for policy use)
        """
        for state in self.env.state_space:
            if not state.is_terminal:
                for action in self.env.action_space:
                    q_value = self.get_q_value(state, action)
                    self.Q.set_value(state, action, q_value)
    
    def learn_episode(self) -> Tuple[float, int]:
        """
        学习一个回合
        Learn one episode
        """
        # 初始化
        # Initialize
        state = self.env.reset()
        self.e = np.zeros(self.n_features)
        
        # 更新Q对象并选择动作
        # Update Q object and choose action
        self.update_Q_object()
        action = self.policy.select_action(state)
        
        self.Q_old = self.get_q_value(state, action)
        
        episode_return = 0.0
        episode_length = 0
        
        while True:
            # 获取当前特征
            # Get current features
            x = self.feature_func(state, action)
            
            # 执行动作
            # Execute action
            next_state, reward, done, _ = self.env.step(action)
            
            # 选择下一个动作
            # Choose next action
            if not done:
                self.update_Q_object()
                next_action = self.policy.select_action(next_state)
                x_next = self.feature_func(next_state, next_action)
                Q_next = np.dot(self.w, x_next)
            else:
                next_action = None
                Q_next = 0.0
            
            # 当前Q值（更新前）
            # Current Q value (before update)
            Q_current = np.dot(self.w, x)
            
            # TD误差
            # TD error
            td_error = reward + self.gamma * Q_next - self.Q_old
            
            # 学习率
            # Learning rate
            alpha = self.alpha_func(self.step_count)
            
            # 更新Dutch迹
            # Update Dutch trace
            e_dot_x = np.dot(self.e, x)
            self.e = self.gamma * self.lambda_ * self.e + \
                    alpha * (1.0 - self.gamma * self.lambda_ * e_dot_x) * x
            
            # 更新权重（真在线规则）
            # Update weights (true online rule)
            self.w = self.w + td_error * self.e + \
                    alpha * (self.Q_old - Q_current) * x
            
            # 更新Q_old
            # Update Q_old
            self.Q_old = Q_next
            
            # 累积回报
            # Accumulate return
            episode_return += reward * (self.gamma ** episode_length)
            episode_length += 1
            self.step_count += 1
            
            # 记录权重范数
            # Record weight norm
            self.weight_norms.append(np.linalg.norm(self.w))
            
            # 转移
            # Transition
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
        
        return episode_return, episode_length
    
    def learn(self,
             n_episodes: int = 1000,
             verbose: bool = True) -> ActionValueFunction:
        """
        学习Q函数
        Learn Q function
        """
        if verbose:
            print(f"\n开始真在线SARSA(λ)学习: {n_episodes}回合")
            print(f"Starting True Online SARSA(λ) learning: {n_episodes} episodes")
            print(f"  参数: γ={self.gamma}, λ={self.lambda_}")
            print(f"  初始ε: {self.policy.epsilon:.3f}")
            print(f"  特征维度: {self.n_features}")
        
        for episode in range(n_episodes):
            episode_return, episode_length = self.learn_episode()
            
            if verbose and (episode + 1) % max(1, n_episodes // 10) == 0:
                avg_return = np.mean(self.episode_returns[-100:]) \
                           if len(self.episode_returns) >= 100 \
                           else np.mean(self.episode_returns)
                avg_length = np.mean(self.episode_lengths[-100:]) \
                           if len(self.episode_lengths) >= 100 \
                           else np.mean(self.episode_lengths)
                avg_weight_norm = np.mean(self.weight_norms[-1000:]) \
                                if len(self.weight_norms) >= 1000 \
                                else np.mean(self.weight_norms)
                
                print(f"Episode {episode + 1}/{n_episodes}: "
                      f"Return={episode_return:.2f}, "
                      f"Avg Return={avg_return:.2f}, "
                      f"Avg Length={avg_length:.1f}, "
                      f"ε={self.policy.epsilon:.3f}, "
                      f"||w||={avg_weight_norm:.2f}")
        
        # 最终更新Q对象
        # Final update Q object
        self.update_Q_object()
        
        if verbose:
            print(f"\n真在线SARSA(λ)学习完成!")
            print(f"  最终ε: {self.policy.epsilon:.3f}")
            print(f"  最终||w||: {np.linalg.norm(self.w):.2f}")
            print(f"  总步数: {self.step_count}")
        
        return self.Q


# ================================================================================
# 第6.4.3节：真在线与传统TD(λ)比较
# Section 6.4.3: Comparison of True Online and Traditional TD(λ)
# ================================================================================

class TrueOnlineComparator:
    """
    真在线与传统TD(λ)比较器
    True Online vs Traditional TD(λ) Comparator
    
    展示真在线方法的优势
    Demonstrate advantages of true online methods
    """
    
    def __init__(self, env: MDPEnvironment):
        """
        初始化比较器
        Initialize comparator
        """
        self.env = env
        self.results = {}
        
        logger.info("初始化真在线比较器")
    
    def compare_methods(self,
                       n_episodes: int = 500,
                       n_runs: int = 10,
                       gamma: float = 0.99,
                       lambda_: float = 0.9,
                       alpha: float = 0.1,
                       verbose: bool = True) -> Dict[str, Any]:
        """
        比较真在线与传统方法
        Compare true online with traditional methods
        """
        if verbose:
            print("\n" + "="*80)
            print("真在线vs传统TD(λ)比较实验")
            print("True Online vs Traditional TD(λ) Comparison")
            print("="*80)
            print(f"参数: γ={gamma}, λ={lambda_}, α={alpha}")
            print(f"实验: {n_episodes}回合 × {n_runs}次运行")
        
        from ch02_mdp.policies_and_values import UniformRandomPolicy
        from ch06_td_lambda.td_lambda_prediction import OnlineTDLambda
        
        policy = UniformRandomPolicy(self.env.action_space)
        
        # 方法列表
        # Method list
        methods = {
            'Traditional TD(λ)': OnlineTDLambda,
            'True Online TD(λ)': TrueOnlineTDLambda
        }
        
        results = {name: {
            'returns': [],
            'final_returns': [],
            'convergence_episodes': [],
            'final_values': []
        } for name in methods}
        
        for run in range(n_runs):
            if verbose:
                print(f"\n运行 {run + 1}/{n_runs}:")
            
            for name, AlgoClass in methods.items():
                # 创建算法实例
                # Create algorithm instance
                algo = AlgoClass(
                    self.env,
                    gamma=gamma,
                    lambda_=lambda_,
                    alpha=alpha
                )
                
                # 学习
                # Learn
                for episode in range(n_episodes):
                    episode_return = algo.learn_episode(policy)
                
                # 记录结果
                # Record results
                results[name]['returns'].append(algo.episode_returns)
                results[name]['final_returns'].append(algo.episode_returns[-1])
                
                # 计算收敛
                # Compute convergence
                convergence_ep = self._find_convergence_episode(algo.episode_returns)
                results[name]['convergence_episodes'].append(convergence_ep)
                
                # 记录最终价值
                # Record final values
                if hasattr(algo, 'w'):
                    final_norm = np.linalg.norm(algo.w)
                    results[name]['final_values'].append(final_norm)
                
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
            
            analyzed[name] = {
                'mean_returns': np.mean(returns_array, axis=0),
                'std_returns': np.std(returns_array, axis=0),
                'final_return_mean': np.mean(data['final_returns']),
                'final_return_std': np.std(data['final_returns']),
                'convergence_mean': np.mean(data['convergence_episodes']),
                'convergence_std': np.std(data['convergence_episodes']),
            }
            
            if data['final_values']:
                analyzed[name]['final_value_mean'] = np.mean(data['final_values'])
                analyzed[name]['final_value_std'] = np.std(data['final_values'])
        
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
        
        print(f"\n{'方法':<25} {'最终回报':<20} {'收敛回合':<20}")
        print("-" * 65)
        
        for name, data in self.results.items():
            final_return = f"{data['final_return_mean']:.2f} ± {data['final_return_std']:.2f}"
            convergence = f"{data['convergence_mean']:.0f} ± {data['convergence_std']:.0f}"
            
            print(f"{name:<25} {final_return:<20} {convergence:<20}")
        
        # 比较收敛速度
        # Compare convergence speed
        trad_conv = self.results.get('Traditional TD(λ)', {}).get('convergence_mean', float('inf'))
        true_conv = self.results.get('True Online TD(λ)', {}).get('convergence_mean', float('inf'))
        
        if trad_conv > 0 and true_conv > 0:
            speedup = (trad_conv - true_conv) / trad_conv * 100
            print(f"\n真在线相对传统方法:")
            print(f"  收敛速度提升: {speedup:.1f}%")
        
        print("""
        关键优势 Key Advantages:
        - 真在线TD(λ)通常收敛更快
          True Online TD(λ) usually converges faster
        - 理论上精确等价于离线λ-回报
          Theoretically exact equivalence to offline λ-return
        - 更适合函数近似
          Better suited for function approximation
        - 现代深度RL的标准方法
          Standard method in modern deep RL
        """)
    
    def plot_comparison(self):
        """
        绘制比较图
        Plot comparison
        """
        if not self.results:
            print("请先运行比较实验")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 图1：学习曲线
        # Plot 1: Learning curves
        ax1 = axes[0]
        
        colors = {'Traditional TD(λ)': 'blue', 'True Online TD(λ)': 'red'}
        
        for name, data in self.results.items():
            mean_returns = data['mean_returns']
            std_returns = data['std_returns']
            episodes = range(len(mean_returns))
            
            ax1.plot(episodes, mean_returns, color=colors.get(name, 'gray'),
                    label=name, linewidth=2)
            ax1.fill_between(episodes,
                            mean_returns - std_returns,
                            mean_returns + std_returns,
                            color=colors.get(name, 'gray'), alpha=0.1)
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Return')
        ax1.set_title('Learning Curves: True Online vs Traditional')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 图2：收敛速度比较
        # Plot 2: Convergence speed comparison
        ax2 = axes[1]
        
        names = list(self.results.keys())
        conv_means = [self.results[name]['convergence_mean'] for name in names]
        conv_stds = [self.results[name]['convergence_std'] for name in names]
        
        x_pos = range(len(names))
        colors_list = [colors.get(name, 'gray') for name in names]
        
        bars = ax2.bar(x_pos, conv_means, yerr=conv_stds,
                      capsize=5, color=colors_list, alpha=0.7)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.set_ylabel('Convergence Episode')
        ax2.set_title('Convergence Speed Comparison')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        # Add value labels
        for bar, mean in zip(bars, conv_means):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean:.0f}',
                    ha='center', va='bottom')
        
        plt.suptitle('True Online TD(λ) vs Traditional TD(λ)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig


# ================================================================================
# 主函数：演示真在线TD(λ)
# Main Function: Demonstrate True Online TD(λ)
# ================================================================================

def demonstrate_true_online_td_lambda():
    """
    演示真在线TD(λ)
    Demonstrate True Online TD(λ)
    """
    print("\n" + "="*80)
    print("第6.4节：真在线TD(λ)")
    print("Section 6.4: True Online TD(λ)")
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
    
    # 1. 真在线TD(λ)预测测试
    # 1. True Online TD(λ) prediction test
    print("\n" + "="*60)
    print("1. 真在线TD(λ)预测")
    print("1. True Online TD(λ) Prediction")
    print("="*60)
    
    policy = UniformRandomPolicy(env.action_space)
    
    true_online_td = TrueOnlineTDLambda(
        env, gamma=0.9, lambda_=0.9, alpha=0.1
    )
    
    V = true_online_td.learn(policy, n_episodes=200, verbose=True)
    
    # 显示学习的价值
    # Show learned values
    print("\n学习的价值（部分）:")
    for i in range(min(5, len(env.state_space))):
        state = env.state_space[i]
        if not state.is_terminal:
            value = V.get_value(state)
            print(f"  V({state.id}) = {value:.3f}")
    
    # 2. 真在线SARSA(λ)测试
    # 2. True Online SARSA(λ) test
    print("\n" + "="*60)
    print("2. 真在线SARSA(λ)控制")
    print("2. True Online SARSA(λ) Control")
    print("="*60)
    
    # 创建带障碍的环境
    # Create environment with obstacles
    env_control = GridWorld(rows=4, cols=4,
                           start_pos=(0,0),
                           goal_pos=(3,3),
                           obstacles=[(1,1), (2,2)])
    
    print(f"\n创建4×4 GridWorld（含障碍）")
    print(f"  障碍: (1,1), (2,2)")
    
    true_online_sarsa = TrueOnlineSARSALambda(
        env_control, gamma=0.99, lambda_=0.9, alpha=0.1, epsilon=0.1
    )
    
    Q = true_online_sarsa.learn(n_episodes=300, verbose=True)
    
    # 显示一些Q值
    # Show some Q values
    print("\nQ值示例:")
    sample_state = env_control.state_space[0]
    if not sample_state.is_terminal:
        for action in env_control.action_space[:2]:
            q_value = Q.get_value(sample_state, action)
            print(f"  Q(s0, {action.id}) = {q_value:.3f}")
    
    # 3. 真在线vs传统比较
    # 3. True online vs traditional comparison
    print("\n" + "="*60)
    print("3. 真在线vs传统TD(λ)比较")
    print("3. True Online vs Traditional TD(λ) Comparison")
    print("="*60)
    
    comparator = TrueOnlineComparator(env)
    results = comparator.compare_methods(
        n_episodes=300,
        n_runs=5,
        gamma=0.9,
        lambda_=0.9,
        alpha=0.1,
        verbose=True
    )
    
    # 绘图
    # Plot
    fig = comparator.plot_comparison()
    
    # 总结
    # Summary
    print("\n" + "="*80)
    print("真在线TD(λ)总结")
    print("True Online TD(λ) Summary")
    print("="*80)
    
    print("""
    关键要点 Key Takeaways:
    =======================
    
    1. 真在线TD(λ)的理论突破
       Theoretical breakthrough of True Online TD(λ)
       - 精确等价于离线λ-回报
         Exact equivalence to offline λ-return
       - 即使有函数近似也成立
         Holds even with function approximation
    
    2. Dutch迹的创新
       Innovation of Dutch traces
       - e = γλe + α(1 - γλe^T x)x
       - 考虑特征重叠
         Considers feature overlap
    
    3. 实践优势
       Practical advantages
       - 更快收敛
         Faster convergence
       - 更稳定
         More stable
       - 适合深度学习
         Suitable for deep learning
    
    4. 现代应用
       Modern applications
       - A3C等算法的基础
         Foundation of algorithms like A3C
       - 深度RL的标准方法
         Standard method in deep RL
    
    5. van Seijen & Sutton (2014)的重要贡献
       Important contribution by van Seijen & Sutton (2014)
       - 解决了长期存在的理论问题
         Solved long-standing theoretical problem
       - 统一了理论和实践
         Unified theory and practice
    """)
    
    plt.show()


if __name__ == "__main__":
    demonstrate_true_online_td_lambda()