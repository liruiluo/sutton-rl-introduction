"""
================================================================================
第13.4-13.5节：Actor-Critic方法
Section 13.4-13.5: Actor-Critic Methods
================================================================================

结合策略梯度和时序差分！
Combining policy gradient with temporal difference!

核心思想 Core Idea:
Actor: 策略，负责选择动作
       Policy, responsible for action selection
Critic: 价值函数，负责评估动作
        Value function, responsible for action evaluation

Actor-Critic更新 Actor-Critic Update:
Actor:  θ_{t+1} = θ_t + α_θ δ_t ∇ln π(A_t|S_t,θ_t)
Critic: w_{t+1} = w_t + α_w δ_t ∇v̂(S_t,w_t)

其中 Where:
δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)  (TD误差)
                                       (TD error)

优势 Advantages:
- 在线更新
  Online updates
- 低方差
  Low variance
- 持续任务适用
  Suitable for continuing tasks
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import logging

# 设置日志
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第13.4节：One-step Actor-Critic
# Section 13.4: One-step Actor-Critic
# ================================================================================

class OneStepActorCritic:
    """
    单步Actor-Critic算法
    One-step Actor-Critic Algorithm
    
    最简单的Actor-Critic形式
    Simplest form of Actor-Critic
    
    使用TD误差作为优势估计
    Use TD error as advantage estimate
    
    特点 Features:
    - 完全在线
      Fully online
    - 低计算成本
      Low computational cost
    - 适合连续任务
      Suitable for continuing tasks
    """
    
    def __init__(self,
                 actor: Any,  # 策略（Actor）
                 critic: Any,  # 价值函数（Critic）
                 alpha_theta: float = 0.01,
                 alpha_w: float = 0.05,
                 gamma: float = 0.99):
        """
        初始化单步Actor-Critic
        Initialize one-step Actor-Critic
        
        Args:
            actor: Actor（策略）
                  Actor (policy)
            critic: Critic（价值函数）
                   Critic (value function)
            alpha_theta: Actor学习率
                        Actor learning rate
            alpha_w: Critic学习率
                    Critic learning rate
            gamma: 折扣因子
                  Discount factor
        """
        self.actor = actor
        self.critic = critic
        self.alpha_theta = alpha_theta
        self.alpha_w = alpha_w
        self.gamma = gamma
        
        # 统计
        # Statistics
        self.step_count = 0
        self.episode_count = 0
        self.td_errors = []
        self.episode_returns = []
        
        logger.info(f"初始化One-step Actor-Critic: α_θ={alpha_theta}, α_w={alpha_w}")
    
    def compute_td_error(self, state: Any, reward: float, 
                         next_state: Any, done: bool) -> float:
        """
        计算TD误差
        Compute TD error
        
        δ = r + γV(s') - V(s)
        
        Args:
            state: 当前状态
                  Current state
            reward: 奖励
                   Reward
            next_state: 下一状态
                       Next state
            done: 是否终止
                 Whether terminal
        
        Returns:
            TD误差
            TD error
        """
        current_value = self.critic.get_value(state)
        
        if done:
            next_value = 0.0
        else:
            next_value = self.critic.get_value(next_state)
        
        td_error = reward + self.gamma * next_value - current_value
        
        return td_error
    
    def update(self, state: Any, action: int, reward: float,
               next_state: Any, done: bool):
        """
        Actor-Critic更新
        Actor-Critic update
        
        Args:
            state: 当前状态
                  Current state
            action: 执行的动作
                   Action taken
            reward: 获得的奖励
                   Reward received
            next_state: 下一状态
                       Next state
            done: 是否终止
                 Whether terminal
        """
        # 计算TD误差
        # Compute TD error
        td_error = self.compute_td_error(state, reward, next_state, done)
        self.td_errors.append(td_error)
        
        # 更新Critic（价值函数）
        # Update Critic (value function)
        # w ← w + α_w δ ∇V(s)
        self.critic.update(state, td_error, self.alpha_w)
        
        # 更新Actor（策略）
        # Update Actor (policy)
        # θ ← θ + α_θ δ ∇ln π(a|s)
        log_gradient = self.actor.compute_log_gradient(state, action)
        policy_gradient = self.alpha_theta * td_error * log_gradient
        self.actor.update_parameters(policy_gradient, step_size=1.0)
        
        self.step_count += 1
    
    def learn_episode(self, env: Any, max_steps: int = 1000) -> float:
        """
        学习一个回合
        Learn one episode
        
        Args:
            env: 环境
                Environment
            max_steps: 最大步数
                      Maximum steps
        
        Returns:
            回合总回报
            Episode total return
        """
        state = env.reset()
        episode_return = 0.0
        
        for step in range(max_steps):
            # 选择动作
            # Select action
            action = self.actor.select_action(state)
            
            # 执行动作
            # Execute action
            next_state, reward, done, _ = env.step(action)
            episode_return += reward
            
            # Actor-Critic更新
            # Actor-Critic update
            self.update(state, action, reward, next_state, done)
            
            if done:
                break
            
            state = next_state
        
        self.episode_count += 1
        self.episode_returns.append(episode_return)
        
        return episode_return
    
    def get_statistics(self) -> Dict[str, float]:
        """
        获取统计信息
        Get statistics
        """
        if len(self.td_errors) == 0:
            return {}
        
        recent_errors = self.td_errors[-1000:]
        recent_returns = self.episode_returns[-100:]
        
        return {
            'mean_td_error': np.mean(np.abs(recent_errors)),
            'std_td_error': np.std(recent_errors),
            'mean_return': np.mean(recent_returns) if recent_returns else 0,
            'std_return': np.std(recent_returns) if recent_returns else 0,
            'total_steps': self.step_count,
            'total_episodes': self.episode_count
        }


# ================================================================================
# 第13.5节：带资格迹的Actor-Critic
# Section 13.5: Actor-Critic with Eligibility Traces
# ================================================================================

class ActorCriticWithTraces:
    """
    带资格迹的Actor-Critic
    Actor-Critic with Eligibility Traces
    
    结合TD(λ)的Actor-Critic
    Actor-Critic combined with TD(λ)
    
    特点 Features:
    - 更快的信用分配
      Faster credit assignment
    - 更好的样本效率
      Better sample efficiency
    - 统一短期和长期学习
      Unifies short-term and long-term learning
    """
    
    def __init__(self,
                 actor: Any,
                 critic: Any,
                 lambda_theta: float = 0.9,  # Actor资格迹衰减
                 lambda_w: float = 0.9,      # Critic资格迹衰减
                 alpha_theta: float = 0.01,
                 alpha_w: float = 0.05,
                 gamma: float = 0.99):
        """
        初始化带资格迹的Actor-Critic
        Initialize Actor-Critic with eligibility traces
        
        Args:
            lambda_theta: Actor资格迹衰减率
                         Actor trace decay rate
            lambda_w: Critic资格迹衰减率
                     Critic trace decay rate
        """
        self.actor = actor
        self.critic = critic
        self.lambda_theta = lambda_theta
        self.lambda_w = lambda_w
        self.alpha_theta = alpha_theta
        self.alpha_w = alpha_w
        self.gamma = gamma
        
        # 资格迹
        # Eligibility traces
        self.trace_theta = None  # Actor迹
        self.trace_w = None      # Critic迹
        
        # 统计
        # Statistics
        self.step_count = 0
        self.episode_count = 0
        self.td_errors = []
        
        logger.info(f"初始化Actor-Critic with Traces: λ_θ={lambda_theta}, λ_w={lambda_w}")
    
    def initialize_traces(self):
        """
        初始化资格迹
        Initialize eligibility traces
        """
        # 根据参数形状初始化
        # Initialize based on parameter shape
        if hasattr(self.actor, 'theta'):
            self.trace_theta = np.zeros_like(self.actor.theta)
        
        if hasattr(self.critic, 'weights'):
            self.trace_w = np.zeros_like(self.critic.weights)
        elif hasattr(self.critic, 'w'):
            self.trace_w = np.zeros_like(self.critic.w)
    
    def update(self, state: Any, action: int, reward: float,
               next_state: Any, done: bool):
        """
        带资格迹的Actor-Critic更新
        Actor-Critic update with eligibility traces
        
        Args:
            state: 当前状态
                  Current state
            action: 执行的动作
                   Action taken
            reward: 获得的奖励
                   Reward received
            next_state: 下一状态
                       Next state
            done: 是否终止
                 Whether terminal
        """
        # 初始化迹（如果需要）
        # Initialize traces (if needed)
        if self.trace_theta is None:
            self.initialize_traces()
        
        # 计算TD误差
        # Compute TD error
        current_value = self.critic.get_value(state)
        next_value = 0.0 if done else self.critic.get_value(next_state)
        td_error = reward + self.gamma * next_value - current_value
        self.td_errors.append(td_error)
        
        # 更新Critic资格迹
        # Update Critic eligibility trace
        critic_features = self.critic.feature_extractor(state)
        self.trace_w = self.gamma * self.lambda_w * self.trace_w + critic_features
        
        # 更新Critic
        # Update Critic
        self.critic.weights += self.alpha_w * td_error * self.trace_w
        
        # 更新Actor资格迹
        # Update Actor eligibility trace
        log_gradient = self.actor.compute_log_gradient(state, action)
        if len(log_gradient.shape) == 1:
            # 处理一维梯度
            # Handle 1D gradient
            if len(self.trace_theta.shape) == 2:
                self.trace_theta[action] = (self.gamma * self.lambda_theta * 
                                           self.trace_theta[action] + log_gradient)
            else:
                self.trace_theta = (self.gamma * self.lambda_theta * 
                                  self.trace_theta + log_gradient)
        else:
            self.trace_theta = self.gamma * self.lambda_theta * self.trace_theta + log_gradient
        
        # 更新Actor
        # Update Actor
        if hasattr(self.actor, 'theta'):
            self.actor.theta += self.alpha_theta * td_error * self.trace_theta
        
        # 如果终止，重置迹
        # Reset traces if terminal
        if done:
            self.trace_theta = np.zeros_like(self.trace_theta) if self.trace_theta is not None else None
            self.trace_w = np.zeros_like(self.trace_w) if self.trace_w is not None else None
        
        self.step_count += 1
    
    def learn_episode(self, env: Any, max_steps: int = 1000) -> float:
        """
        学习一个回合
        Learn one episode
        """
        state = env.reset()
        episode_return = 0.0
        
        # 重置资格迹
        # Reset eligibility traces
        self.trace_theta = None
        self.trace_w = None
        
        for step in range(max_steps):
            action = self.actor.select_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_return += reward
            
            self.update(state, action, reward, next_state, done)
            
            if done:
                break
            
            state = next_state
        
        self.episode_count += 1
        
        return episode_return


# ================================================================================
# 第13.5.1节：A2C (Advantage Actor-Critic)
# Section 13.5.1: A2C (Advantage Actor-Critic)
# ================================================================================

class A2C:
    """
    优势Actor-Critic (A2C)
    Advantage Actor-Critic (A2C)
    
    使用优势函数代替TD误差
    Use advantage function instead of TD error
    
    优势 Advantage:
    A(s,a) = Q(s,a) - V(s)
    
    特点 Features:
    - 更稳定的学习
      More stable learning
    - 更好的方差-偏差权衡
      Better variance-bias tradeoff
    - 可以批量更新
      Can do batch updates
    """
    
    def __init__(self,
                 actor: Any,
                 critic: Any,
                 n_steps: int = 5,  # n-step回报
                 alpha_theta: float = 0.001,
                 alpha_w: float = 0.005,
                 gamma: float = 0.99,
                 use_gae: bool = True,  # 是否使用GAE
                 gae_lambda: float = 0.95):
        """
        初始化A2C
        Initialize A2C
        
        Args:
            n_steps: n-step回报的步数
                    Number of steps for n-step returns
            use_gae: 是否使用广义优势估计
                    Whether to use Generalized Advantage Estimation
            gae_lambda: GAE的λ参数
                       GAE lambda parameter
        """
        self.actor = actor
        self.critic = critic
        self.n_steps = n_steps
        self.alpha_theta = alpha_theta
        self.alpha_w = alpha_w
        self.gamma = gamma
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        
        # 经验缓冲
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        
        # 统计
        # Statistics
        self.step_count = 0
        self.episode_count = 0
        self.advantages_history = []
        
        logger.info(f"初始化A2C: n_steps={n_steps}, use_GAE={use_gae}")
    
    def compute_advantages(self) -> np.ndarray:
        """
        计算优势
        Compute advantages
        
        Returns:
            优势值数组
            Array of advantage values
        """
        if self.use_gae:
            return self.compute_gae()
        else:
            return self.compute_n_step_advantages()
    
    def compute_n_step_advantages(self) -> np.ndarray:
        """
        计算n-step优势
        Compute n-step advantages
        
        A_t = R_t + γR_{t+1} + ... + γ^{n-1}R_{t+n-1} + γ^n V(S_{t+n}) - V(S_t)
        """
        n = len(self.rewards)
        advantages = np.zeros(n)
        
        for t in range(n):
            # 计算n-step回报
            # Compute n-step return
            G = 0
            for k in range(min(self.n_steps, n - t)):
                G += (self.gamma ** k) * self.rewards[t + k]
            
            # 加上bootstrap值
            # Add bootstrap value
            if t + self.n_steps < n:
                G += (self.gamma ** self.n_steps) * self.values[t + self.n_steps]
            
            # 优势 = n-step回报 - 基线
            # Advantage = n-step return - baseline
            advantages[t] = G - self.values[t]
        
        return advantages
    
    def compute_gae(self) -> np.ndarray:
        """
        计算广义优势估计(GAE)
        Compute Generalized Advantage Estimation (GAE)
        
        A^GAE_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
        """
        n = len(self.rewards)
        advantages = np.zeros(n)
        gae = 0
        
        # 反向计算GAE
        # Backward computation of GAE
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = 0  # 终止状态
            else:
                next_value = self.values[t + 1]
            
            # TD误差
            # TD error
            delta = self.rewards[t] + self.gamma * next_value - self.values[t]
            
            # GAE
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae
        
        return advantages
    
    def update_batch(self):
        """
        批量更新Actor和Critic
        Batch update Actor and Critic
        """
        # 计算优势
        # Compute advantages
        advantages = self.compute_advantages()
        self.advantages_history.extend(advantages.tolist())
        
        # 归一化优势（可选，提高稳定性）
        # Normalize advantages (optional, improves stability)
        if len(advantages) > 1:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # 更新Critic
        # Update Critic
        for t in range(len(self.states)):
            # 计算目标价值
            # Compute target value
            target_value = self.values[t] + advantages[t]
            
            # 更新价值函数
            # Update value function
            self.critic.update(self.states[t], target_value, self.alpha_w)
        
        # 更新Actor
        # Update Actor
        for t in range(len(self.states)):
            # 计算策略梯度
            # Compute policy gradient
            log_gradient = self.actor.compute_log_gradient(self.states[t], self.actions[t])
            policy_gradient = self.alpha_theta * advantages[t] * log_gradient
            
            # 更新策略参数
            # Update policy parameters
            self.actor.update_parameters(policy_gradient, step_size=1.0)
        
        # 清空缓冲
        # Clear buffer
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
    
    def learn_steps(self, env: Any, n_steps: int) -> List[float]:
        """
        学习n步
        Learn for n steps
        
        Args:
            env: 环境
                Environment
            n_steps: 步数
                    Number of steps
        
        Returns:
            每步的奖励
            Rewards at each step
        """
        rewards = []
        
        if not hasattr(self, 'current_state'):
            self.current_state = env.reset()
        
        for _ in range(n_steps):
            # 选择动作
            # Select action
            action = self.actor.select_action(self.current_state)
            
            # 获取价值估计
            # Get value estimate
            value = self.critic.get_value(self.current_state)
            
            # 执行动作
            # Execute action
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            
            # 存储经验
            # Store experience
            self.states.append(self.current_state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.values.append(value)
            
            self.step_count += 1
            
            if done:
                self.current_state = env.reset()
                self.episode_count += 1
                # 批量更新
                # Batch update
                self.update_batch()
            else:
                self.current_state = next_state
                
                # 如果缓冲满了，执行更新
                # If buffer is full, perform update
                if len(self.states) >= self.n_steps:
                    self.update_batch()
        
        return rewards


# ================================================================================
# 简单的Actor和Critic实现
# Simple Actor and Critic Implementation
# ================================================================================

class SimpleActor:
    """
    简单的Actor（策略）
    Simple Actor (Policy)
    """
    
    def __init__(self, n_features: int, n_actions: int, feature_extractor: Callable):
        self.n_features = n_features
        self.n_actions = n_actions
        self.feature_extractor = feature_extractor
        self.theta = np.zeros((n_actions, n_features))
    
    def compute_action_probabilities(self, state: Any) -> np.ndarray:
        """计算动作概率"""
        preferences = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            features = self.feature_extractor(state, a)
            preferences[a] = np.dot(self.theta[a], features)
        
        # Softmax
        max_pref = np.max(preferences)
        exp_prefs = np.exp(preferences - max_pref)
        return exp_prefs / np.sum(exp_prefs)
    
    def select_action(self, state: Any) -> int:
        """选择动作"""
        probs = self.compute_action_probabilities(state)
        return np.random.choice(self.n_actions, p=probs)
    
    def compute_log_gradient(self, state: Any, action: int) -> np.ndarray:
        """计算对数策略梯度"""
        features = self.feature_extractor(state, action)
        probs = self.compute_action_probabilities(state)
        
        # 期望特征
        expected_features = np.zeros(self.n_features)
        for a in range(self.n_actions):
            f = self.feature_extractor(state, a)
            expected_features += probs[a] * f
        
        return features - expected_features
    
    def update_parameters(self, gradient: np.ndarray, step_size: float):
        """更新参数"""
        if len(gradient.shape) == 1:
            # 需要重塑梯度
            for a in range(self.n_actions):
                self.theta[a] += step_size * gradient
        else:
            self.theta += step_size * gradient


class SimpleCritic:
    """
    简单的Critic（价值函数）
    Simple Critic (Value Function)
    """
    
    def __init__(self, n_features: int, feature_extractor: Callable):
        self.n_features = n_features
        self.feature_extractor = feature_extractor
        self.weights = np.zeros(n_features)
    
    def get_value(self, state: Any) -> float:
        """获取状态价值"""
        features = self.feature_extractor(state)
        return np.dot(self.weights, features)
    
    def update(self, state: Any, td_error_or_target: float, alpha: float):
        """更新价值函数"""
        features = self.feature_extractor(state)
        
        # 检查输入是TD误差还是目标值
        if abs(td_error_or_target) < 100:  # 假设是TD误差
            # TD误差更新
            self.weights += alpha * td_error_or_target * features
        else:  # 假设是目标值
            # 监督学习更新
            current_value = self.get_value(state)
            error = td_error_or_target - current_value
            self.weights += alpha * error * features


# ================================================================================
# 主函数：演示Actor-Critic方法
# Main Function: Demonstrate Actor-Critic Methods
# ================================================================================

def demonstrate_actor_critic():
    """
    演示Actor-Critic方法
    Demonstrate Actor-Critic methods
    """
    print("\n" + "="*80)
    print("第13.4-13.5节：Actor-Critic方法")
    print("Section 13.4-13.5: Actor-Critic Methods")
    print("="*80)
    
    # 设置
    # Setup
    n_features = 8
    n_actions = 2
    n_states = 4
    
    # 特征提取器
    # Feature extractors
    def state_features(state):
        features = np.zeros(n_features)
        if isinstance(state, int):
            features[state % n_features] = 1.0
            features[(state * 2) % n_features] = 0.5
        return features
    
    def state_action_features(state, action):
        features = np.zeros(n_features)
        if isinstance(state, int):
            base_idx = (state * n_actions + action) % n_features
            features[base_idx] = 1.0
        return features
    
    # 简单环境
    # Simple environment
    class SimpleEnv:
        def __init__(self):
            self.state = 0
            self.step_count = 0
        
        def reset(self):
            self.state = 0
            self.step_count = 0
            return self.state
        
        def step(self, action):
            if action == 0:
                self.state = max(0, self.state - 1)
            else:
                self.state = min(n_states - 1, self.state + 1)
            
            if self.state == n_states - 1:
                reward = 10.0
                done = True
            else:
                reward = -1.0
                done = False
            
            self.step_count += 1
            if self.step_count >= 20:
                done = True
            
            return self.state, reward, done, {}
    
    # 1. 测试One-step Actor-Critic
    # 1. Test One-step Actor-Critic
    print("\n" + "="*60)
    print("1. One-step Actor-Critic")
    print("="*60)
    
    actor_1step = SimpleActor(n_features, n_actions, state_action_features)
    critic_1step = SimpleCritic(n_features, state_features)
    
    ac_1step = OneStepActorCritic(
        actor=actor_1step,
        critic=critic_1step,
        alpha_theta=0.05,
        alpha_w=0.1,
        gamma=0.9
    )
    
    print("\n训练One-step Actor-Critic...")
    env = SimpleEnv()
    
    for episode in range(50):
        episode_return = ac_1step.learn_episode(env, max_steps=20)
        
        if (episode + 1) % 10 == 0:
            stats = ac_1step.get_statistics()
            print(f"  回合{episode + 1}: "
                  f"回报={episode_return:.1f}, "
                  f"平均|TD误差|={stats['mean_td_error']:.3f}")
    
    # 2. 测试Actor-Critic with Traces
    # 2. Test Actor-Critic with Traces
    print("\n" + "="*60)
    print("2. Actor-Critic with Eligibility Traces")
    print("="*60)
    
    actor_traces = SimpleActor(n_features, n_actions, state_action_features)
    critic_traces = SimpleCritic(n_features, state_features)
    
    ac_traces = ActorCriticWithTraces(
        actor=actor_traces,
        critic=critic_traces,
        lambda_theta=0.9,
        lambda_w=0.9,
        alpha_theta=0.02,
        alpha_w=0.05,
        gamma=0.9
    )
    
    print("\n训练Actor-Critic with Traces...")
    
    for episode in range(50):
        episode_return = ac_traces.learn_episode(env, max_steps=20)
        
        if (episode + 1) % 10 == 0:
            mean_td_error = np.mean(np.abs(ac_traces.td_errors[-100:]))
            print(f"  回合{episode + 1}: "
                  f"回报={episode_return:.1f}, "
                  f"平均|TD误差|={mean_td_error:.3f}")
    
    # 3. 测试A2C
    # 3. Test A2C
    print("\n" + "="*60)
    print("3. A2C (Advantage Actor-Critic)")
    print("="*60)
    
    actor_a2c = SimpleActor(n_features, n_actions, state_action_features)
    critic_a2c = SimpleCritic(n_features, state_features)
    
    a2c = A2C(
        actor=actor_a2c,
        critic=critic_a2c,
        n_steps=5,
        alpha_theta=0.01,
        alpha_w=0.05,
        gamma=0.9,
        use_gae=True,
        gae_lambda=0.95
    )
    
    print("\n训练A2C...")
    print("(使用GAE优势估计)")
    
    total_rewards = []
    for i in range(200):
        rewards = a2c.learn_steps(env, n_steps=5)
        total_rewards.extend(rewards)
        
        if (i + 1) % 40 == 0:
            recent_return = sum(total_rewards[-100:])
            mean_advantage = np.mean(a2c.advantages_history[-50:]) if a2c.advantages_history else 0
            print(f"  步{(i+1)*5}: "
                  f"最近100步回报={recent_return:.1f}, "
                  f"平均优势={mean_advantage:.3f}")
    
    # 4. 比较不同Actor-Critic变体
    # 4. Compare different Actor-Critic variants
    print("\n" + "="*60)
    print("4. Actor-Critic变体比较")
    print("4. Actor-Critic Variants Comparison")
    print("="*60)
    
    print("\n算法性能总结:")
    print("-" * 40)
    
    # One-step性能
    if ac_1step.episode_returns:
        one_step_mean = np.mean(ac_1step.episode_returns[-10:])
        one_step_std = np.std(ac_1step.episode_returns[-10:])
        print(f"One-step AC:     {one_step_mean:.1f} ± {one_step_std:.1f}")
    
    # With traces性能
    print(f"AC with Traces:  总步数={ac_traces.step_count}")
    
    # A2C性能
    print(f"A2C:             总步数={a2c.step_count}, 回合数={a2c.episode_count}")
    
    # 总结
    # Summary
    print("\n" + "="*80)
    print("Actor-Critic方法总结")
    print("Actor-Critic Methods Summary")
    print("="*80)
    
    print("""
    关键要点 Key Takeaways:
    =======================
    
    1. Actor-Critic结合策略梯度和值函数
       Actor-Critic combines policy gradient and value function
       
    2. Critic减少Actor的方差
       Critic reduces Actor's variance
       
    3. 资格迹加速信用分配
       Eligibility traces speed up credit assignment
       
    4. A2C使用优势函数和批量更新
       A2C uses advantage function and batch updates
       
    5. GAE提供更好的偏差-方差权衡
       GAE provides better bias-variance tradeoff
    
    算法选择 Algorithm Selection:
    - 简单任务: One-step Actor-Critic
               One-step Actor-Critic for simple tasks
    - 需要快速学习: Actor-Critic with Traces
                  Actor-Critic with Traces for fast learning
    - 复杂环境: A2C with GAE
              A2C with GAE for complex environments
    """)


if __name__ == "__main__":
    demonstrate_actor_critic()