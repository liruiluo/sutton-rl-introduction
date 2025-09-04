"""
================================================================================
第10.5节：控制与函数近似的一般框架
Section 10.5: General Framework for Control with Function Approximation
================================================================================

统一的控制框架！
Unified control framework!

核心组件 Core Components:
1. 动作价值近似器
   Action-value approximator
2. 策略改进
   Policy improvement
3. 探索机制
   Exploration mechanism

关键挑战 Key Challenges:
- 致命三要素 (Deadly Triad)
  * 函数近似
    Function approximation
  * 自举
    Bootstrapping
  * 离策略
    Off-policy
- 不稳定性
  Instability
- 发散风险
  Risk of divergence

Actor-Critic架构:
- Actor: 策略π(a|s,θ)
- Critic: 价值函数v(s,w)或q(s,a,w)
- 协同学习
  Cooperative learning
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

# 导入基础组件
# Import base components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# 设置日志
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第10.5.1节：动作价值近似器
# Section 10.5.1: Action-Value Approximator
# ================================================================================

class ActionValueApproximator(ABC):
    """
    动作价值近似器基类
    Action-Value Approximator Base Class
    
    定义q(s,a,w)的接口
    Defines interface for q(s,a,w)
    
    关键方法 Key Methods:
    - predict: 预测q(s,a)
              Predict q(s,a)
    - gradient: 计算∇q(s,a,w)
               Compute gradient
    - update: 更新参数w
             Update parameters
    """
    
    @abstractmethod
    def predict(self, state: Any, action: int) -> float:
        """
        预测动作价值
        Predict action value
        
        Args:
            state: 状态
                  State
            action: 动作
                   Action
        
        Returns:
            q̂(s,a,w)
        """
        pass
    
    @abstractmethod
    def gradient(self, state: Any, action: int) -> np.ndarray:
        """
        计算梯度
        Compute gradient
        
        Args:
            state: 状态
                  State
            action: 动作
                   Action
        
        Returns:
            ∇q̂(s,a,w)
        """
        pass
    
    @abstractmethod
    def update(self, state: Any, action: int, target: float, alpha: float):
        """
        更新参数
        Update parameters
        
        Args:
            state: 状态
                  State
            action: 动作
                   Action
            target: 目标值
                   Target value
            alpha: 学习率
                  Learning rate
        """
        pass
    
    def get_best_action(self, state: Any, n_actions: int) -> int:
        """
        获取最佳动作
        Get best action
        
        Args:
            state: 状态
                  State
            n_actions: 动作数
                      Number of actions
        
        Returns:
            argmax_a q̂(s,a,w)
        """
        q_values = [self.predict(state, a) for a in range(n_actions)]
        # 随机打破平局
        # Random tie-breaking
        max_q = max(q_values)
        max_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return np.random.choice(max_actions)


# ================================================================================
# 第10.5.2节：线性动作价值函数
# Section 10.5.2: Linear Action-Value Function
# ================================================================================

class LinearActionValueFunction(ActionValueApproximator):
    """
    线性动作价值函数
    Linear Action-Value Function
    
    q̂(s,a,w) = w^T x(s,a)
    
    特征可以是:
    Features can be:
    - 状态-动作对特征
      State-action pair features
    - 状态特征×动作指示器
      State features × action indicators
    """
    
    def __init__(self,
                feature_extractor: Callable,
                n_features: int,
                n_actions: int):
        """
        初始化线性动作价值函数
        Initialize linear action-value function
        
        Args:
            feature_extractor: 特征提取函数
                             Feature extraction function
            n_features: 特征数
                       Number of features
            n_actions: 动作数
                      Number of actions
        """
        self.feature_extractor = feature_extractor
        self.n_features = n_features
        self.n_actions = n_actions
        
        # 权重矩阵 (每个动作一组权重)
        # Weight matrix (weights per action)
        self.weights = np.zeros((n_actions, n_features))
        
        # 统计
        # Statistics
        self.update_count = 0
        
        logger.info(f"初始化线性动作价值函数: {n_features}特征×{n_actions}动作")
    
    def get_features(self, state: Any, action: int) -> np.ndarray:
        """
        获取状态-动作特征
        Get state-action features
        """
        # 使用特征提取器
        # Use feature extractor
        return self.feature_extractor(state, action)
    
    def predict(self, state: Any, action: int) -> float:
        """
        预测动作价值
        Predict action value
        """
        features = self.get_features(state, action)
        return np.dot(self.weights[action], features)
    
    def gradient(self, state: Any, action: int) -> np.ndarray:
        """
        计算梯度
        Compute gradient
        
        对线性函数: ∇q̂(s,a,w) = x(s,a)
        For linear: gradient is features
        """
        return self.get_features(state, action)
    
    def update(self, state: Any, action: int, target: float, alpha: float):
        """
        更新权重
        Update weights
        
        w ← w + α[target - q̂(s,a,w)]∇q̂(s,a,w)
        """
        features = self.get_features(state, action)
        prediction = self.predict(state, action)
        td_error = target - prediction
        
        # 梯度更新
        # Gradient update
        self.weights[action] += alpha * td_error * features
        
        self.update_count += 1
        
        return td_error


# ================================================================================
# 第10.5.3节：通用控制框架
# Section 10.5.3: General Control Framework
# ================================================================================

class ControlWithFA:
    """
    函数近似的通用控制框架
    General Control Framework with Function Approximation
    
    支持各种算法:
    Supports various algorithms:
    - Sarsa
    - Q-learning
    - Expected Sarsa
    - n-step methods
    
    关键组件 Key Components:
    1. 近似器
       Approximator
    2. 探索策略
       Exploration policy
    3. 更新规则
       Update rule
    """
    
    def __init__(self,
                approximator: ActionValueApproximator,
                n_actions: int,
                alpha: float = 0.1,
                gamma: float = 0.99,
                epsilon: float = 0.1,
                method: str = 'sarsa'):
        """
        初始化控制框架
        Initialize control framework
        
        Args:
            approximator: 动作价值近似器
                        Action-value approximator
            n_actions: 动作数
                      Number of actions
            alpha: 学习率
                  Learning rate
            gamma: 折扣因子
                  Discount factor
            epsilon: 探索率
                    Exploration rate
            method: 方法 ('sarsa', 'q_learning', 'expected_sarsa')
                   Method
        """
        self.approximator = approximator
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.method = method
        
        # 统计
        # Statistics
        self.episode_count = 0
        self.step_count = 0
        self.episode_returns = []
        self.td_errors = []
        
        logger.info(f"初始化控制框架: method={method}, ε={epsilon}")
    
    def select_action(self, state: Any, greedy: bool = False) -> int:
        """
        选择动作
        Select action
        
        Args:
            state: 状态
                  State
            greedy: 是否贪婪选择
                   Whether greedy selection
        
        Returns:
            选择的动作
            Selected action
        """
        if not greedy and np.random.random() < self.epsilon:
            # 探索：随机动作
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # 利用：贪婪动作
            # Exploit: greedy action
            return self.approximator.get_best_action(state, self.n_actions)
    
    def compute_target(self, reward: float, next_state: Any,
                      next_action: Optional[int] = None,
                      done: bool = False) -> float:
        """
        计算TD目标
        Compute TD target
        
        Args:
            reward: 奖励
                   Reward
            next_state: 下一状态
                       Next state
            next_action: 下一动作(Sarsa需要)
                        Next action (for Sarsa)
            done: 是否终止
                 Whether terminal
        
        Returns:
            TD目标
            TD target
        """
        if done:
            return reward
        
        if self.method == 'sarsa':
            # Sarsa: 使用实际的下一动作
            # Sarsa: use actual next action
            if next_action is None:
                raise ValueError("Sarsa需要next_action")
            next_q = self.approximator.predict(next_state, next_action)
            
        elif self.method == 'q_learning':
            # Q-learning: 使用最大Q值
            # Q-learning: use max Q value
            next_q = max(
                self.approximator.predict(next_state, a)
                for a in range(self.n_actions)
            )
            
        elif self.method == 'expected_sarsa':
            # Expected Sarsa: 使用期望Q值
            # Expected Sarsa: use expected Q value
            q_values = [
                self.approximator.predict(next_state, a)
                for a in range(self.n_actions)
            ]
            max_q = max(q_values)
            
            # ε-贪婪策略的期望
            # Expected value for ε-greedy
            expected_q = 0.0
            for a, q in enumerate(q_values):
                if q == max_q:
                    prob = (1 - self.epsilon) + self.epsilon / self.n_actions
                else:
                    prob = self.epsilon / self.n_actions
                expected_q += prob * q
            
            next_q = expected_q
        
        else:
            raise ValueError(f"未知方法: {self.method}")
        
        return reward + self.gamma * next_q
    
    def learn_step(self, state: Any, action: int, reward: float,
                  next_state: Any, done: bool) -> float:
        """
        学习一步
        Learn one step
        
        Args:
            state: 当前状态
                  Current state
            action: 当前动作
                   Current action
            reward: 奖励
                   Reward
            next_state: 下一状态
                       Next state
            done: 是否终止
                 Whether done
        
        Returns:
            TD误差
            TD error
        """
        # 为Sarsa选择下一动作
        # Select next action for Sarsa
        next_action = None
        if self.method == 'sarsa' and not done:
            next_action = self.select_action(next_state)
        
        # 计算TD目标
        # Compute TD target
        target = self.compute_target(reward, next_state, next_action, done)
        
        # 更新近似器
        # Update approximator
        td_error = self.approximator.update(state, action, target, self.alpha)
        
        self.td_errors.append(td_error)
        self.step_count += 1
        
        return td_error
    
    def learn_episode(self, env: Any, max_steps: int = 1000) -> Tuple[float, int]:
        """
        学习一个回合
        Learn one episode
        
        Args:
            env: 环境
                Environment
            max_steps: 最大步数
                      Maximum steps
        
        Returns:
            (回合回报, 步数)
            (episode return, steps)
        """
        state = env.reset()
        episode_return = 0.0
        
        for step in range(max_steps):
            # 选择动作
            # Select action
            action = self.select_action(state)
            
            # 执行动作
            # Execute action
            next_state, reward, done, _ = env.step(action)
            
            # 学习
            # Learn
            self.learn_step(state, action, reward, next_state, done)
            
            episode_return += reward
            state = next_state
            
            if done:
                break
        
        self.episode_count += 1
        self.episode_returns.append(episode_return)
        
        return episode_return, step + 1


# ================================================================================
# 第10.5.4节：Actor-Critic架构
# Section 10.5.4: Actor-Critic Architecture
# ================================================================================

class ActorCriticWithFA:
    """
    Actor-Critic与函数近似
    Actor-Critic with Function Approximation
    
    分离策略和价值函数！
    Separate policy and value function!
    
    架构 Architecture:
    - Actor: π(a|s,θ) - 策略网络
            Policy network
    - Critic: v(s,w) - 价值网络
             Value network
    
    优势 Advantages:
    - 直接学习策略
      Learn policy directly
    - 低方差
      Low variance
    - 在线学习
      Online learning
    """
    
    def __init__(self,
                state_dim: int,
                n_actions: int,
                actor_lr: float = 0.01,
                critic_lr: float = 0.1,
                gamma: float = 0.99):
        """
        初始化Actor-Critic
        Initialize Actor-Critic
        
        Args:
            state_dim: 状态维度
                      State dimension
            n_actions: 动作数
                      Number of actions
            actor_lr: Actor学习率
                     Actor learning rate
            critic_lr: Critic学习率
                      Critic learning rate
            gamma: 折扣因子
                  Discount factor
        """
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        
        # Actor参数 (softmax策略)
        # Actor parameters (softmax policy)
        self.actor_weights = np.zeros((n_actions, state_dim))
        
        # Critic参数 (线性价值函数)
        # Critic parameters (linear value function)
        self.critic_weights = np.zeros(state_dim)
        
        # 统计
        # Statistics
        self.episode_count = 0
        
        logger.info(f"初始化Actor-Critic: {state_dim}维状态, {n_actions}动作")
    
    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """
        获取动作概率(softmax)
        Get action probabilities (softmax)
        
        π(a|s,θ) = exp(θ_a^T s) / Σ_b exp(θ_b^T s)
        
        Args:
            state: 状态
                  State
        
        Returns:
            动作概率分布
            Action probability distribution
        """
        # 计算偏好
        # Compute preferences
        preferences = np.dot(self.actor_weights, state)
        
        # Softmax
        # 减去最大值提高数值稳定性
        # Subtract max for numerical stability
        max_pref = np.max(preferences)
        exp_prefs = np.exp(preferences - max_pref)
        
        return exp_prefs / np.sum(exp_prefs)
    
    def select_action(self, state: np.ndarray) -> int:
        """
        根据策略选择动作
        Select action according to policy
        
        Args:
            state: 状态
                  State
        
        Returns:
            选择的动作
            Selected action
        """
        probs = self.get_action_probabilities(state)
        return np.random.choice(self.n_actions, p=probs)
    
    def get_state_value(self, state: np.ndarray) -> float:
        """
        获取状态价值
        Get state value
        
        v(s,w) = w^T s
        
        Args:
            state: 状态
                  State
        
        Returns:
            状态价值
            State value
        """
        return np.dot(self.critic_weights, state)
    
    def update(self, state: np.ndarray, action: int,
              reward: float, next_state: np.ndarray,
              done: bool):
        """
        Actor-Critic更新
        Actor-Critic update
        
        Args:
            state: 当前状态
                  Current state
            action: 执行的动作
                   Action taken
            reward: 奖励
                   Reward
            next_state: 下一状态
                       Next state
            done: 是否终止
                 Whether done
        """
        # Critic更新
        # Critic update
        current_value = self.get_state_value(state)
        
        if done:
            td_target = reward
        else:
            next_value = self.get_state_value(next_state)
            td_target = reward + self.gamma * next_value
        
        td_error = td_target - current_value
        
        # 更新Critic (梯度上升)
        # Update Critic (gradient ascent)
        self.critic_weights += self.critic_lr * td_error * state
        
        # Actor更新
        # Actor update
        probs = self.get_action_probabilities(state)
        
        # 计算梯度 ∇ln π(a|s,θ)
        # Compute gradient
        grad = np.zeros_like(self.actor_weights)
        grad[action] = state
        
        # 减去基线(所有动作的期望)
        # Subtract baseline (expected over all actions)
        for a in range(self.n_actions):
            grad[a] -= probs[a] * state
        
        # 更新Actor (使用TD误差作为优势)
        # Update Actor (use TD error as advantage)
        self.actor_weights += self.actor_lr * td_error * grad


# ================================================================================
# 主函数：演示控制框架
# Main Function: Demonstrate Control Framework
# ================================================================================

def demonstrate_control_with_fa():
    """
    演示函数近似控制框架
    Demonstrate control framework with function approximation
    """
    print("\n" + "="*80)
    print("第10.5节：控制与函数近似框架")
    print("Section 10.5: Control with Function Approximation Framework")
    print("="*80)
    
    # 设置
    # Setup
    n_features = 8
    n_actions = 3
    state_dim = 4
    
    # 1. 测试线性动作价值函数
    # 1. Test linear action-value function
    print("\n" + "="*60)
    print("1. 线性动作价值函数")
    print("1. Linear Action-Value Function")
    print("="*60)
    
    # 简单特征提取器
    # Simple feature extractor
    def simple_features(state, action):
        # 状态特征与动作独热编码的组合
        # Combination of state features and action one-hot
        if isinstance(state, np.ndarray):
            state_features = state[:n_features]
        else:
            state_features = np.random.randn(n_features)
        return state_features
    
    linear_q = LinearActionValueFunction(
        feature_extractor=simple_features,
        n_features=n_features,
        n_actions=n_actions
    )
    
    # 测试预测和更新
    # Test prediction and update
    test_state = np.random.randn(n_features)
    
    print("\n初始Q值:")
    for a in range(n_actions):
        q_val = linear_q.predict(test_state, a)
        print(f"  Q(s,{a}) = {q_val:.3f}")
    
    # 更新
    # Update
    print("\n执行更新...")
    for a in range(n_actions):
        target = np.random.randn()
        td_error = linear_q.update(test_state, a, target, alpha=0.1)
        print(f"  动作{a}: target={target:.3f}, TD误差={td_error:.3f}")
    
    print("\n更新后Q值:")
    for a in range(n_actions):
        q_val = linear_q.predict(test_state, a)
        print(f"  Q(s,{a}) = {q_val:.3f}")
    
    # 2. 测试通用控制框架
    # 2. Test general control framework
    print("\n" + "="*60)
    print("2. 通用控制框架")
    print("2. General Control Framework")
    print("="*60)
    
    # 测试不同方法
    # Test different methods
    methods = ['sarsa', 'q_learning', 'expected_sarsa']
    
    for method in methods:
        print(f"\n测试{method}:")
        
        # 创建新的近似器
        # Create new approximator
        approximator = LinearActionValueFunction(
            feature_extractor=simple_features,
            n_features=n_features,
            n_actions=n_actions
        )
        
        controller = ControlWithFA(
            approximator=approximator,
            n_actions=n_actions,
            alpha=0.1,
            gamma=0.9,
            epsilon=0.1,
            method=method
        )
        
        # 模拟学习
        # Simulate learning
        state = np.random.randn(n_features)
        
        for step in range(5):
            action = controller.select_action(state)
            reward = np.random.randn()
            next_state = np.random.randn(n_features)
            done = step == 4
            
            td_error = controller.learn_step(state, action, reward, next_state, done)
            
            print(f"    步{step+1}: a={action}, r={reward:.2f}, TD误差={td_error:.3f}")
            
            state = next_state
    
    # 3. 测试Actor-Critic
    # 3. Test Actor-Critic
    print("\n" + "="*60)
    print("3. Actor-Critic架构")
    print("3. Actor-Critic Architecture")
    print("="*60)
    
    ac = ActorCriticWithFA(
        state_dim=state_dim,
        n_actions=n_actions,
        actor_lr=0.01,
        critic_lr=0.1,
        gamma=0.9
    )
    
    print(f"\n架构配置:")
    print(f"  状态维度: {state_dim}")
    print(f"  动作数: {n_actions}")
    print(f"  Actor学习率: {ac.actor_lr}")
    print(f"  Critic学习率: {ac.critic_lr}")
    
    # 测试策略
    # Test policy
    test_state = np.random.randn(state_dim)
    probs = ac.get_action_probabilities(test_state)
    value = ac.get_state_value(test_state)
    
    print(f"\n初始策略:")
    for a in range(n_actions):
        print(f"  π({a}|s) = {probs[a]:.3f}")
    print(f"  V(s) = {value:.3f}")
    
    # 模拟更新
    # Simulate updates
    print("\n模拟Actor-Critic更新...")
    
    for step in range(5):
        state = np.random.randn(state_dim)
        action = ac.select_action(state)
        reward = np.random.randn() + 1.0
        next_state = np.random.randn(state_dim)
        done = step == 4
        
        ac.update(state, action, reward, next_state, done)
        
        value = ac.get_state_value(state)
        print(f"  步{step+1}: a={action}, r={reward:.2f}, V(s)={value:.3f}")
    
    # 显示更新后的策略
    # Show updated policy
    probs_after = ac.get_action_probabilities(test_state)
    value_after = ac.get_state_value(test_state)
    
    print(f"\n更新后策略:")
    for a in range(n_actions):
        print(f"  π({a}|s) = {probs_after[a]:.3f} (变化: {probs_after[a]-probs[a]:+.3f})")
    print(f"  V(s) = {value_after:.3f} (变化: {value_after-value:+.3f})")
    
    # 总结
    # Summary
    print("\n" + "="*80)
    print("控制框架总结")
    print("Control Framework Summary")
    print("="*80)
    
    print("""
    关键要点 Key Takeaways:
    =======================
    
    1. 统一框架支持多种算法
       Unified framework supports various algorithms
       
    2. 线性近似保证收敛(同策略)
       Linear approximation guarantees convergence (on-policy)
       
    3. Actor-Critic分离策略和价值
       Actor-Critic separates policy and value
       
    4. 探索-利用平衡关键
       Exploration-exploitation balance crucial
       
    5. 小心致命三要素
       Beware of deadly triad
    
    算法选择 Algorithm Selection:
    - Sarsa: 保守，同策略
            Conservative, on-policy
    - Q-learning: 激进，离策略
                 Aggressive, off-policy
    - Expected Sarsa: 平衡
                     Balanced
    - Actor-Critic: 直接策略学习
                   Direct policy learning
    """)


if __name__ == "__main__":
    demonstrate_control_with_fa()