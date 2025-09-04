"""
================================================================================
第9.3-9.4节：线性方法 - 最简单但强大的函数近似
Section 9.3-9.4: Linear Methods - Simplest but Powerful Function Approximation
================================================================================

线性函数近似的优雅！
The elegance of linear function approximation!

值函数的线性形式 Linear Form of Value Function:
v̂(s,w) = w^T x(s) = Σᵢ wᵢ xᵢ(s)

其中 where:
- x(s) ∈ ℝᵈ: 特征向量
             Feature vector
- w ∈ ℝᵈ: 权重向量
          Weight vector

梯度特别简单 Gradient is particularly simple:
∇v̂(s,w) = x(s)

核心算法 Core Algorithms:
1. 梯度蒙特卡洛 Gradient Monte Carlo:
   w ← w + α[Gₜ - v̂(Sₜ,w)]x(Sₜ)
   
2. 半梯度TD(0) Semi-gradient TD(0):
   w ← w + α[Rₜ₊₁ + γv̂(Sₜ₊₁,w) - v̂(Sₜ,w)]x(Sₜ)
   
3. 半梯度TD(λ) Semi-gradient TD(λ):
   带资格迹的TD
   TD with eligibility traces

关键性质 Key Properties:
- 收敛性保证（在策略情况）
  Convergence guarantee (on-policy case)
- 计算效率高
  Computationally efficient
- 特征工程很重要
  Feature engineering is crucial
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import logging

# 导入基础组件
# Import base components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.ch03_finite_mdp.mdp_framework import State, Action, MDPEnvironment
from src.ch03_finite_mdp.policies_and_values import Policy

# 设置日志
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第9.3.1节：线性特征
# Section 9.3.1: Linear Features
# ================================================================================

class LinearFeatures:
    """
    线性特征提取器
    Linear Feature Extractor
    
    将状态转换为特征向量
    Convert states to feature vectors
    
    常见特征类型 Common Feature Types:
    1. 独热编码 One-hot encoding
    2. 径向基函数 Radial basis functions
    3. 多项式特征 Polynomial features
    4. 瓦片编码 Tile coding
    """
    
    def __init__(self, n_features: int):
        """
        初始化特征提取器
        Initialize feature extractor
        
        Args:
            n_features: 特征维度
                       Feature dimension
        """
        self.n_features = n_features
        
        # 特征统计
        # Feature statistics
        self.feature_counts = defaultdict(int)
        self.feature_means = np.zeros(n_features)
        self.feature_stds = np.ones(n_features)
        
        logger.info(f"初始化线性特征: d={n_features}")
    
    def extract(self, state: Any) -> np.ndarray:
        """
        提取特征
        Extract features
        
        Args:
            state: 状态
                  State
        
        Returns:
            特征向量 x(s)
            Feature vector
        """
        # 基础实现：根据状态类型选择
        # Basic implementation: choose based on state type
        
        if isinstance(state, State):
            # 从State对象提取特征
            # Extract features from State object
            if hasattr(state, 'features'):
                # 处理不同类型的features
                # Handle different types of features
                if isinstance(state.features, dict):
                    features = np.array(list(state.features.values()))
                elif isinstance(state.features, (list, tuple)):
                    features = np.array(state.features)
                elif isinstance(state.features, np.ndarray):
                    features = state.features
                else:
                    features = np.array([state.features])
                    
                # 填充或截断到正确维度
                # Pad or truncate to correct dimension
                if len(features) < self.n_features:
                    features = np.pad(features, (0, self.n_features - len(features)))
                else:
                    features = features[:self.n_features]
                return features
            else:
                # 使用状态ID的独热编码
                # Use one-hot encoding of state ID
                return self.one_hot(hash(state.id) % self.n_features)
        
        elif isinstance(state, (int, np.integer)):
            # 独热编码
            # One-hot encoding
            return self.one_hot(state)
        
        elif isinstance(state, (list, tuple, np.ndarray)):
            # 直接使用作为特征
            # Use directly as features
            features = np.array(state).flatten()
            if len(features) < self.n_features:
                features = np.pad(features, (0, self.n_features - len(features)))
            else:
                features = features[:self.n_features]
            return features
        
        else:
            # 默认：零向量
            # Default: zero vector
            return np.zeros(self.n_features)
    
    def one_hot(self, index: int) -> np.ndarray:
        """
        独热编码
        One-hot encoding
        
        Args:
            index: 索引
                  Index
        
        Returns:
            独热向量
            One-hot vector
        """
        features = np.zeros(self.n_features)
        if 0 <= index < self.n_features:
            features[index] = 1.0
        return features
    
    def normalize(self, features: np.ndarray) -> np.ndarray:
        """
        标准化特征
        Normalize features
        
        Args:
            features: 原始特征
                     Raw features
        
        Returns:
            标准化特征
            Normalized features
        """
        # 避免除零
        # Avoid division by zero
        safe_stds = np.where(self.feature_stds > 0, self.feature_stds, 1.0)
        return (features - self.feature_means) / safe_stds
    
    def update_statistics(self, features: np.ndarray):
        """
        更新特征统计
        Update feature statistics
        
        增量更新均值和标准差
        Incremental update of mean and std
        
        Args:
            features: 特征向量
                     Feature vector
        """
        # 简化实现：使用指数移动平均
        # Simplified: use exponential moving average
        alpha = 0.01
        self.feature_means = (1 - alpha) * self.feature_means + alpha * features
        variance = (features - self.feature_means) ** 2
        self.feature_stds = np.sqrt((1 - alpha) * self.feature_stds**2 + alpha * variance)


# ================================================================================
# 第9.3.2节：线性值函数
# Section 9.3.2: Linear Value Function
# ================================================================================

class LinearValueFunction:
    """
    线性值函数近似
    Linear Value Function Approximation
    
    v̂(s,w) = w^T x(s)
    
    关键性质 Key Properties:
    - 线性参数化
      Linear parameterization
    - 凸优化问题
      Convex optimization problem
    - 唯一最优解
      Unique optimal solution
    """
    
    def __init__(self, feature_extractor: LinearFeatures):
        """
        初始化线性值函数
        Initialize linear value function
        
        Args:
            feature_extractor: 特征提取器
                             Feature extractor
        """
        self.feature_extractor = feature_extractor
        self.n_features = feature_extractor.n_features
        
        # 权重向量
        # Weight vector
        self.weights = np.zeros(self.n_features)
        
        # 统计
        # Statistics
        self.update_count = 0
        self.td_errors = []
        
        logger.info(f"初始化线性值函数: d={self.n_features}")
    
    def predict(self, state: Any) -> float:
        """
        预测状态价值
        Predict state value
        
        Args:
            state: 状态
                  State
        
        Returns:
            v̂(s,w)
        """
        features = self.feature_extractor.extract(state)
        return np.dot(self.weights, features)
    
    def gradient(self, state: Any) -> np.ndarray:
        """
        计算梯度
        Compute gradient
        
        对线性函数：∇v̂(s,w) = x(s)
        For linear function: ∇v̂(s,w) = x(s)
        
        Args:
            state: 状态
                  State
        
        Returns:
            梯度向量
            Gradient vector
        """
        return self.feature_extractor.extract(state)
    
    def update(self, state: Any, target: float, alpha: float) -> float:
        """
        更新权重
        Update weights
        
        w ← w + α[target - v̂(s,w)]x(s)
        
        Args:
            state: 状态
                  State
            target: 目标值
                   Target value
            alpha: 学习率
                  Learning rate
        
        Returns:
            TD误差
            TD error
        """
        features = self.feature_extractor.extract(state)
        prediction = np.dot(self.weights, features)
        td_error = target - prediction
        
        # 权重更新
        # Weight update
        self.weights += alpha * td_error * features
        
        # 记录统计
        # Record statistics
        self.td_errors.append(td_error)
        self.update_count += 1
        
        return td_error
    
    def get_feature_importance(self) -> np.ndarray:
        """
        获取特征重要性
        Get feature importance
        
        基于权重绝对值
        Based on weight magnitude
        
        Returns:
            特征重要性
            Feature importance
        """
        return np.abs(self.weights)


# ================================================================================
# 第9.4.1节：梯度蒙特卡洛
# Section 9.4.1: Gradient Monte Carlo
# ================================================================================

class GradientMonteCarlo:
    """
    梯度蒙特卡洛预测
    Gradient Monte Carlo Prediction
    
    使用完整回报更新
    Update using complete returns
    
    更新规则 Update Rule:
    w ← w + α[Gₜ - v̂(Sₜ,w)]∇v̂(Sₜ,w)
    
    特点 Characteristics:
    - 无偏估计
      Unbiased estimate
    - 高方差
      High variance
    - 需要完整轨迹
      Requires complete episodes
    """
    
    def __init__(self,
                env: MDPEnvironment,
                feature_extractor: LinearFeatures,
                gamma: float = 0.99,
                alpha: float = 0.01):
        """
        初始化梯度MC
        Initialize Gradient MC
        
        Args:
            env: 环境
                Environment
            feature_extractor: 特征提取器
                             Feature extractor
            gamma: 折扣因子
                  Discount factor
            alpha: 学习率
                  Learning rate
        """
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        
        # 值函数
        # Value function
        self.value_function = LinearValueFunction(feature_extractor)
        
        # 轨迹缓冲
        # Episode buffer
        self.episode_states = []
        self.episode_rewards = []
        
        # 统计
        # Statistics
        self.episode_count = 0
        self.episode_returns = []
        
        logger.info(f"初始化梯度MC: γ={gamma}, α={alpha}")
    
    def generate_episode(self, policy: Policy) -> Tuple[List[State], List[float]]:
        """
        生成一个回合
        Generate an episode
        
        Args:
            policy: 策略
                   Policy
        
        Returns:
            (状态序列, 奖励序列)
            (state sequence, reward sequence)
        """
        states = []
        rewards = []
        
        state = self.env.reset()
        
        while not state.is_terminal:
            states.append(state)
            
            action = policy.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            
            rewards.append(reward)
            state = next_state
            
            if done:
                break
        
        return states, rewards
    
    def compute_returns(self, rewards: List[float]) -> List[float]:
        """
        计算回报
        Compute returns
        
        Gₜ = Rₜ₊₁ + γRₜ₊₂ + γ²Rₜ₊₃ + ...
        
        Args:
            rewards: 奖励序列
                    Reward sequence
        
        Returns:
            回报序列
            Return sequence
        """
        returns = []
        G = 0
        
        # 从后向前计算
        # Compute backward
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.append(G)
        
        returns.reverse()
        return returns
    
    def update_episode(self, states: List[State], returns: List[float]):
        """
        更新一个回合
        Update one episode
        
        Args:
            states: 状态序列
                   State sequence
            returns: 回报序列
                    Return sequence
        """
        for state, G in zip(states, returns):
            # 梯度MC更新
            # Gradient MC update
            self.value_function.update(state, G, self.alpha)
    
    def learn(self,
             policy: Policy,
             n_episodes: int = 100,
             verbose: bool = True) -> LinearValueFunction:
        """
        学习值函数
        Learn value function
        
        Args:
            policy: 要评估的策略
                   Policy to evaluate
            n_episodes: 回合数
                       Number of episodes
            verbose: 是否输出进度
                    Whether to output progress
        
        Returns:
            学习的值函数
            Learned value function
        """
        if verbose:
            print(f"\n开始梯度MC学习: {n_episodes}回合")
            print(f"Starting Gradient MC learning: {n_episodes} episodes")
        
        for episode in range(n_episodes):
            # 生成回合
            # Generate episode
            states, rewards = self.generate_episode(policy)
            
            # 计算回报
            # Compute returns
            returns = self.compute_returns(rewards)
            
            # 更新值函数
            # Update value function
            self.update_episode(states, returns)
            
            # 记录统计
            # Record statistics
            self.episode_count += 1
            if returns:
                self.episode_returns.append(returns[0])
            
            if verbose and (episode + 1) % max(1, n_episodes // 10) == 0:
                avg_return = np.mean(self.episode_returns[-10:]) \
                           if len(self.episode_returns) >= 10 \
                           else np.mean(self.episode_returns)
                
                avg_td_error = np.mean(np.abs(self.value_function.td_errors[-100:])) \
                             if len(self.value_function.td_errors) >= 100 \
                             else np.mean(np.abs(self.value_function.td_errors))
                
                print(f"Episode {episode + 1}/{n_episodes}: "
                      f"Avg Return={avg_return:.2f}, "
                      f"Avg |TD Error|={avg_td_error:.4f}")
        
        if verbose:
            print(f"\n梯度MC学习完成!")
            print(f"  总更新次数: {self.value_function.update_count}")
        
        return self.value_function


# ================================================================================
# 第9.4.2节：半梯度TD(0)
# Section 9.4.2: Semi-gradient TD(0)
# ================================================================================

class SemiGradientTD:
    """
    半梯度TD(0)
    Semi-gradient TD(0)
    
    最重要的函数近似算法之一！
    One of the most important function approximation algorithms!
    
    更新规则 Update Rule:
    w ← w + α[Rₜ₊₁ + γv̂(Sₜ₊₁,w) - v̂(Sₜ,w)]∇v̂(Sₜ,w)
    
    "半梯度"因为 "Semi-gradient" because:
    - 只对v̂(Sₜ,w)求梯度
      Only take gradient w.r.t. v̂(Sₜ,w)
    - 不对目标v̂(Sₜ₊₁,w)求梯度
      Not w.r.t. target v̂(Sₜ₊₁,w)
    
    特点 Characteristics:
    - 有偏但方差小
      Biased but low variance
    - 在线学习
      Online learning
    - 计算效率高
      Computationally efficient
    """
    
    def __init__(self,
                env: MDPEnvironment,
                feature_extractor: LinearFeatures,
                gamma: float = 0.99,
                alpha: float = 0.01):
        """
        初始化半梯度TD
        Initialize Semi-gradient TD
        
        Args:
            env: 环境
                Environment
            feature_extractor: 特征提取器
                             Feature extractor
            gamma: 折扣因子
                  Discount factor
            alpha: 学习率
                  Learning rate
        """
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        
        # 值函数
        # Value function
        self.value_function = LinearValueFunction(feature_extractor)
        
        # 统计
        # Statistics
        self.step_count = 0
        self.episode_count = 0
        self.td_errors = []
        
        logger.info(f"初始化半梯度TD(0): γ={gamma}, α={alpha}")
    
    def update_step(self, state: State, reward: float, next_state: State):
        """
        执行一步TD更新
        Execute one TD update step
        
        Args:
            state: 当前状态
                  Current state
            reward: 奖励
                   Reward
            next_state: 下一状态
                       Next state
        """
        # 计算TD目标
        # Compute TD target
        if next_state.is_terminal:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.value_function.predict(next_state)
        
        # 半梯度更新
        # Semi-gradient update
        td_error = self.value_function.update(state, td_target, self.alpha)
        
        # 记录统计
        # Record statistics
        self.td_errors.append(td_error)
        self.step_count += 1
    
    def learn_episode(self, policy: Policy) -> float:
        """
        学习一个回合
        Learn one episode
        
        Args:
            policy: 策略
                   Policy
        
        Returns:
            回合回报
            Episode return
        """
        state = self.env.reset()
        episode_return = 0.0
        
        while not state.is_terminal:
            # 选择动作
            # Select action
            action = policy.select_action(state)
            
            # 执行动作
            # Execute action
            next_state, reward, done, _ = self.env.step(action)
            
            # TD更新
            # TD update
            self.update_step(state, reward, next_state)
            
            episode_return += reward
            state = next_state
            
            if done:
                break
        
        self.episode_count += 1
        return episode_return
    
    def learn(self,
             policy: Policy,
             n_episodes: int = 100,
             verbose: bool = True) -> LinearValueFunction:
        """
        学习值函数
        Learn value function
        
        Args:
            policy: 要评估的策略
                   Policy to evaluate
            n_episodes: 回合数
                       Number of episodes
            verbose: 是否输出进度
                    Whether to output progress
        
        Returns:
            学习的值函数
            Learned value function
        """
        if verbose:
            print(f"\n开始半梯度TD(0)学习: {n_episodes}回合")
            print(f"Starting Semi-gradient TD(0) learning: {n_episodes} episodes")
        
        episode_returns = []
        
        for episode in range(n_episodes):
            episode_return = self.learn_episode(policy)
            episode_returns.append(episode_return)
            
            if verbose and (episode + 1) % max(1, n_episodes // 10) == 0:
                avg_return = np.mean(episode_returns[-10:]) \
                           if len(episode_returns) >= 10 \
                           else np.mean(episode_returns)
                
                avg_td_error = np.mean(np.abs(self.td_errors[-100:])) \
                             if len(self.td_errors) >= 100 \
                             else np.mean(np.abs(self.td_errors))
                
                print(f"Episode {episode + 1}/{n_episodes}: "
                      f"Avg Return={avg_return:.2f}, "
                      f"Avg |TD Error|={avg_td_error:.4f}")
        
        if verbose:
            print(f"\n半梯度TD(0)学习完成!")
            print(f"  总步数: {self.step_count}")
            print(f"  总更新次数: {self.value_function.update_count}")
        
        return self.value_function


# ================================================================================
# 第9.4.3节：半梯度TD(λ)
# Section 9.4.3: Semi-gradient TD(λ)
# ================================================================================

class SemiGradientTDLambda:
    """
    半梯度TD(λ)与资格迹
    Semi-gradient TD(λ) with Eligibility Traces
    
    结合TD和MC的优势！
    Combining advantages of TD and MC!
    
    更新规则 Update Rule:
    δₜ = Rₜ₊₁ + γv̂(Sₜ₊₁,w) - v̂(Sₜ,w)
    zₜ = γλzₜ₋₁ + ∇v̂(Sₜ,w)
    w ← w + αδₜzₜ
    
    其中 where:
    - z: 资格迹向量
        Eligibility trace vector
    - λ: 迹衰减参数
        Trace decay parameter
    
    λ的影响 Effect of λ:
    - λ=0: TD(0)
    - λ=1: MC
    - 0<λ<1: 介于两者之间
            Between TD and MC
    """
    
    def __init__(self,
                env: MDPEnvironment,
                feature_extractor: LinearFeatures,
                gamma: float = 0.99,
                alpha: float = 0.01,
                lambda_: float = 0.9):
        """
        初始化TD(λ)
        Initialize TD(λ)
        
        Args:
            env: 环境
                Environment
            feature_extractor: 特征提取器
                             Feature extractor
            gamma: 折扣因子
                  Discount factor
            alpha: 学习率
                  Learning rate
            lambda_: 迹衰减参数
                    Trace decay parameter
        """
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.lambda_ = lambda_
        
        # 值函数
        # Value function
        self.value_function = LinearValueFunction(feature_extractor)
        
        # 资格迹
        # Eligibility traces
        self.eligibility_traces = np.zeros(feature_extractor.n_features)
        
        # 统计
        # Statistics
        self.step_count = 0
        self.episode_count = 0
        self.td_errors = []
        
        logger.info(f"初始化半梯度TD(λ): γ={gamma}, α={alpha}, λ={lambda_}")
    
    def update_step(self, state: State, reward: float, next_state: State):
        """
        执行一步TD(λ)更新
        Execute one TD(λ) update step
        
        Args:
            state: 当前状态
                  Current state
            reward: 奖励
                   Reward
            next_state: 下一状态
                       Next state
        """
        # 获取特征和预测
        # Get features and predictions
        features = self.value_function.feature_extractor.extract(state)
        value = self.value_function.predict(state)
        
        # 计算TD误差
        # Compute TD error
        if next_state.is_terminal:
            td_error = reward - value
        else:
            next_value = self.value_function.predict(next_state)
            td_error = reward + self.gamma * next_value - value
        
        # 更新资格迹
        # Update eligibility traces
        self.eligibility_traces = (self.gamma * self.lambda_ * 
                                  self.eligibility_traces + features)
        
        # 更新权重
        # Update weights
        self.value_function.weights += self.alpha * td_error * self.eligibility_traces
        
        # 记录统计
        # Record statistics
        self.td_errors.append(td_error)
        self.value_function.td_errors.append(td_error)
        self.value_function.update_count += 1
        self.step_count += 1
    
    def learn_episode(self, policy: Policy) -> float:
        """
        学习一个回合
        Learn one episode
        
        Args:
            policy: 策略
                   Policy
        
        Returns:
            回合回报
            Episode return
        """
        # 重置资格迹
        # Reset eligibility traces
        self.eligibility_traces.fill(0)
        
        state = self.env.reset()
        episode_return = 0.0
        
        while not state.is_terminal:
            # 选择动作
            # Select action
            action = policy.select_action(state)
            
            # 执行动作
            # Execute action
            next_state, reward, done, _ = self.env.step(action)
            
            # TD(λ)更新
            # TD(λ) update
            self.update_step(state, reward, next_state)
            
            episode_return += reward
            state = next_state
            
            if done:
                break
        
        self.episode_count += 1
        return episode_return
    
    def learn(self,
             policy: Policy,
             n_episodes: int = 100,
             verbose: bool = True) -> LinearValueFunction:
        """
        学习值函数
        Learn value function
        
        Args:
            policy: 要评估的策略
                   Policy to evaluate
            n_episodes: 回合数
                       Number of episodes
            verbose: 是否输出进度
                    Whether to output progress
        
        Returns:
            学习的值函数
            Learned value function
        """
        if verbose:
            print(f"\n开始半梯度TD(λ)学习: {n_episodes}回合, λ={self.lambda_}")
            print(f"Starting Semi-gradient TD(λ) learning: {n_episodes} episodes, λ={self.lambda_}")
        
        episode_returns = []
        
        for episode in range(n_episodes):
            episode_return = self.learn_episode(policy)
            episode_returns.append(episode_return)
            
            if verbose and (episode + 1) % max(1, n_episodes // 10) == 0:
                avg_return = np.mean(episode_returns[-10:]) \
                           if len(episode_returns) >= 10 \
                           else np.mean(episode_returns)
                
                avg_td_error = np.mean(np.abs(self.td_errors[-100:])) \
                             if len(self.td_errors) >= 100 \
                             else np.mean(np.abs(self.td_errors))
                
                avg_trace = np.mean(np.abs(self.eligibility_traces))
                
                print(f"Episode {episode + 1}/{n_episodes}: "
                      f"Avg Return={avg_return:.2f}, "
                      f"Avg |TD Error|={avg_td_error:.4f}, "
                      f"Avg |Trace|={avg_trace:.4f}")
        
        if verbose:
            print(f"\n半梯度TD(λ)学习完成!")
            print(f"  总步数: {self.step_count}")
            print(f"  总更新次数: {self.value_function.update_count}")
        
        return self.value_function


# ================================================================================
# 主函数：演示线性函数近似
# Main Function: Demonstrate Linear Function Approximation
# ================================================================================

def demonstrate_linear_approximation():
    """
    演示线性函数近似
    Demonstrate linear function approximation
    """
    print("\n" + "="*80)
    print("第9.3-9.4节：线性函数近似")
    print("Section 9.3-9.4: Linear Function Approximation")
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
    
    # 创建特征提取器
    # Create feature extractor
    n_features = 16  # 简单独热编码
    feature_extractor = LinearFeatures(n_features)
    
    # 创建随机策略
    # Create random policy
    policy = UniformRandomPolicy(env.action_space)
    
    print(f"\n使用{n_features}维特征")
    print(f"策略: 均匀随机")
    
    # 1. 测试梯度蒙特卡洛
    # 1. Test Gradient Monte Carlo
    print("\n" + "="*60)
    print("1. 梯度蒙特卡洛")
    print("1. Gradient Monte Carlo")
    print("="*60)
    
    gmc = GradientMonteCarlo(env, feature_extractor, gamma=0.9, alpha=0.01)
    v_gmc = gmc.learn(policy, n_episodes=100, verbose=True)
    
    # 显示一些状态的值
    # Show values for some states
    print("\n学习的状态价值 (前5个状态):")
    for i in range(min(5, len(env.state_space))):
        state = env.state_space[i]
        value = v_gmc.predict(state)
        print(f"  V({state.id}) = {value:.3f}")
    
    # 2. 测试半梯度TD(0)
    # 2. Test Semi-gradient TD(0)
    print("\n" + "="*60)
    print("2. 半梯度TD(0)")
    print("2. Semi-gradient TD(0)")
    print("="*60)
    
    # 重新创建特征提取器（重置状态）
    # Recreate feature extractor (reset state)
    feature_extractor2 = LinearFeatures(n_features)
    sgtd = SemiGradientTD(env, feature_extractor2, gamma=0.9, alpha=0.01)
    v_sgtd = sgtd.learn(policy, n_episodes=100, verbose=True)
    
    print("\n学习的状态价值 (前5个状态):")
    for i in range(min(5, len(env.state_space))):
        state = env.state_space[i]
        value = v_sgtd.predict(state)
        print(f"  V({state.id}) = {value:.3f}")
    
    # 3. 测试半梯度TD(λ)
    # 3. Test Semi-gradient TD(λ)
    print("\n" + "="*60)
    print("3. 半梯度TD(λ)")
    print("3. Semi-gradient TD(λ)")
    print("="*60)
    
    # 测试不同的λ值
    # Test different λ values
    lambda_values = [0.0, 0.5, 0.9]
    
    for lambda_ in lambda_values:
        print(f"\nλ = {lambda_}:")
        
        feature_extractor3 = LinearFeatures(n_features)
        sgtd_lambda = SemiGradientTDLambda(
            env, feature_extractor3,
            gamma=0.9, alpha=0.01, lambda_=lambda_
        )
        
        v_lambda = sgtd_lambda.learn(policy, n_episodes=50, verbose=False)
        
        # 计算平均TD误差
        # Compute average TD error
        avg_td_error = np.mean(np.abs(sgtd_lambda.td_errors[-100:])) \
                      if len(sgtd_lambda.td_errors) >= 100 \
                      else np.mean(np.abs(sgtd_lambda.td_errors))
        
        print(f"  平均|TD误差|: {avg_td_error:.4f}")
        print(f"  总更新次数: {v_lambda.update_count}")
    
    # 4. 比较三种方法
    # 4. Compare three methods
    print("\n" + "="*60)
    print("4. 方法比较")
    print("4. Method Comparison")
    print("="*60)
    
    print(f"\n{'方法':<20} {'更新次数':<15} {'平均|TD误差|':<15}")
    print("-" * 50)
    
    gmc_td_error = np.mean(np.abs(v_gmc.td_errors[-100:])) \
                  if len(v_gmc.td_errors) >= 100 \
                  else np.mean(np.abs(v_gmc.td_errors))
    
    sgtd_td_error = np.mean(np.abs(v_sgtd.td_errors[-100:])) \
                   if len(v_sgtd.td_errors) >= 100 \
                   else np.mean(np.abs(v_sgtd.td_errors))
    
    print(f"{'梯度MC':<20} {v_gmc.update_count:<15} {gmc_td_error:<15.4f}")
    print(f"{'半梯度TD(0)':<20} {v_sgtd.update_count:<15} {sgtd_td_error:<15.4f}")
    
    # 总结
    # Summary
    print("\n" + "="*80)
    print("线性函数近似总结")
    print("Linear Function Approximation Summary")
    print("="*80)
    
    print("""
    关键要点 Key Takeaways:
    =======================
    
    1. 线性函数近似简单高效
       Linear approximation is simple and efficient
       
    2. 梯度MC：无偏但方差大
       Gradient MC: Unbiased but high variance
       
    3. 半梯度TD：有偏但方差小
       Semi-gradient TD: Biased but low variance
       
    4. TD(λ)：平衡偏差和方差
       TD(λ): Balance bias and variance
       
    5. 特征工程很关键
       Feature engineering is crucial
    
    收敛保证 Convergence Guarantee:
    - 线性函数 + 同策略 = 收敛
      Linear function + on-policy = convergence
    - 非线性或异策略可能发散
      Nonlinear or off-policy may diverge
    """)


if __name__ == "__main__":
    demonstrate_linear_approximation()