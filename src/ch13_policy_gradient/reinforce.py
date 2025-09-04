"""
================================================================================
第13.3节：REINFORCE算法
Section 13.3: REINFORCE Algorithm
================================================================================

蒙特卡洛策略梯度！
Monte Carlo Policy Gradient!

核心思想 Core Idea:
使用完整回合的回报作为强化信号
Use complete episode return as reinforcement signal

REINFORCE更新 REINFORCE Update:
θ_{t+1} = θ_t + α∇ln π(A_t|S_t,θ_t) G_t

其中 Where:
- G_t: 从时刻t开始的回报
       Return from time t
- ∇ln π: 对数策略梯度
        Log policy gradient

优点 Advantages:
- 无偏估计
  Unbiased estimate
- 简单直观
  Simple and intuitive
- 不需要值函数
  No value function needed

缺点 Disadvantages:
- 高方差
  High variance
- 需要完整回合
  Requires complete episodes
- 样本效率低
  Low sample efficiency
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
# 第13.3节：基础REINFORCE
# Section 13.3: Basic REINFORCE
# ================================================================================

class REINFORCE:
    """
    基础REINFORCE算法
    Basic REINFORCE Algorithm
    
    蒙特卡洛策略梯度
    Monte Carlo Policy Gradient
    
    算法流程 Algorithm Flow:
    1. 生成完整回合
       Generate complete episode
    2. 计算回报
       Compute returns
    3. 更新策略参数
       Update policy parameters
    
    关键特性 Key Features:
    - 只在回合结束后更新
      Updates only at episode end
    - 使用实际回报
      Uses actual returns
    - 无偏但高方差
      Unbiased but high variance
    """
    
    def __init__(self,
                 policy: Any,
                 alpha: float = 0.01,
                 gamma: float = 0.99):
        """
        初始化REINFORCE
        Initialize REINFORCE
        
        Args:
            policy: 策略对象（必须有compute_log_gradient方法）
                   Policy object (must have compute_log_gradient method)
            alpha: 学习率
                  Learning rate
            gamma: 折扣因子
                  Discount factor
        """
        self.policy = policy
        self.alpha = alpha
        self.gamma = gamma
        
        # 统计
        # Statistics
        self.episode_count = 0
        self.episode_returns = []
        self.gradient_norms = []
        
        logger.info(f"初始化REINFORCE: α={alpha}, γ={gamma}")
    
    def compute_returns(self, rewards: List[float]) -> List[float]:
        """
        计算折扣回报
        Compute discounted returns
        
        G_t = Σ_{k=0}^{T-t-1} γ^k R_{t+k+1}
        
        Args:
            rewards: 奖励序列
                    Reward sequence
        
        Returns:
            各时刻的回报
            Returns at each time step
        """
        T = len(rewards)
        returns = [0.0] * T
        G = 0.0
        
        # 反向计算回报
        # Backward computation of returns
        for t in reversed(range(T)):
            G = rewards[t] + self.gamma * G
            returns[t] = G
        
        return returns
    
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
        # 生成回合
        # Generate episode
        states = []
        actions = []
        rewards = []
        
        state = env.reset()
        
        for step in range(max_steps):
            # 选择动作
            # Select action
            action = self.policy.select_action(state)
            
            # 执行动作
            # Execute action
            next_state, reward, done, _ = env.step(action)
            
            # 记录轨迹
            # Record trajectory
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            if done:
                break
            
            state = next_state
        
        # 计算回报
        # Compute returns
        returns = self.compute_returns(rewards)
        
        # REINFORCE更新
        # REINFORCE update
        for t in range(len(states)):
            state = states[t]
            action = actions[t]
            G = returns[t]
            
            # 计算对数策略梯度
            # Compute log policy gradient
            log_grad = self.policy.compute_log_gradient(state, action)
            
            # 更新策略参数
            # Update policy parameters
            # θ ← θ + α∇ln π(a|s,θ) G
            gradient = self.alpha * G * log_grad
            self.policy.update_parameters(gradient, step_size=1.0)  # 步长已包含在gradient中
            
            # 记录梯度范数
            # Record gradient norm
            self.gradient_norms.append(np.linalg.norm(gradient))
        
        # 统计
        # Statistics
        episode_return = sum(rewards)
        self.episode_returns.append(episode_return)
        self.episode_count += 1
        
        return episode_return
    
    def get_statistics(self) -> Dict[str, float]:
        """
        获取统计信息
        Get statistics
        """
        if len(self.episode_returns) == 0:
            return {}
        
        recent_returns = self.episode_returns[-100:]
        recent_grad_norms = self.gradient_norms[-1000:]
        
        return {
            'mean_return': np.mean(recent_returns),
            'std_return': np.std(recent_returns),
            'max_return': np.max(recent_returns),
            'min_return': np.min(recent_returns),
            'mean_gradient_norm': np.mean(recent_grad_norms) if recent_grad_norms else 0,
            'total_episodes': self.episode_count
        }


# ================================================================================
# 第13.3.1节：带基线的REINFORCE
# Section 13.3.1: REINFORCE with Baseline
# ================================================================================

class REINFORCEWithBaseline:
    """
    带基线的REINFORCE
    REINFORCE with Baseline
    
    使用基线减少方差
    Use baseline to reduce variance
    
    更新规则 Update Rule:
    θ_{t+1} = θ_t + α∇ln π(A_t|S_t,θ_t) (G_t - b(S_t))
    
    其中 Where:
    b(S_t): 基线（通常是V(S_t)）
           Baseline (usually V(S_t))
    
    关键洞察 Key Insight:
    基线不改变梯度的期望但减少方差
    Baseline doesn't change expected gradient but reduces variance
    """
    
    def __init__(self,
                 policy: Any,
                 value_function: Any,
                 alpha_theta: float = 0.01,
                 alpha_w: float = 0.05,
                 gamma: float = 0.99):
        """
        初始化带基线的REINFORCE
        Initialize REINFORCE with baseline
        
        Args:
            policy: 策略
                   Policy
            value_function: 价值函数（基线）
                          Value function (baseline)
            alpha_theta: 策略学习率
                        Policy learning rate
            alpha_w: 价值函数学习率
                    Value function learning rate
            gamma: 折扣因子
                  Discount factor
        """
        self.policy = policy
        self.value_function = value_function
        self.alpha_theta = alpha_theta
        self.alpha_w = alpha_w
        self.gamma = gamma
        
        # 统计
        # Statistics
        self.episode_count = 0
        self.episode_returns = []
        self.advantages = []
        self.value_errors = []
        
        logger.info(f"初始化REINFORCE with Baseline: α_θ={alpha_theta}, α_w={alpha_w}")
    
    def compute_returns(self, rewards: List[float]) -> List[float]:
        """
        计算折扣回报
        Compute discounted returns
        """
        T = len(rewards)
        returns = [0.0] * T
        G = 0.0
        
        for t in reversed(range(T)):
            G = rewards[t] + self.gamma * G
            returns[t] = G
        
        return returns
    
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
        # 生成回合
        # Generate episode
        states = []
        actions = []
        rewards = []
        
        state = env.reset()
        
        for step in range(max_steps):
            action = self.policy.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            if done:
                break
            
            state = next_state
        
        # 计算回报
        # Compute returns
        returns = self.compute_returns(rewards)
        
        # REINFORCE with baseline更新
        # REINFORCE with baseline update
        for t in range(len(states)):
            state = states[t]
            action = actions[t]
            G = returns[t]
            
            # 计算基线（价值估计）
            # Compute baseline (value estimate)
            baseline = self.value_function.get_value(state)
            
            # 优势 = 回报 - 基线
            # Advantage = return - baseline
            advantage = G - baseline
            self.advantages.append(advantage)
            
            # 更新策略
            # Update policy
            # θ ← θ + α_θ ∇ln π(a|s,θ) (G - b(s))
            log_grad = self.policy.compute_log_gradient(state, action)
            policy_gradient = self.alpha_theta * advantage * log_grad
            self.policy.update_parameters(policy_gradient, step_size=1.0)
            
            # 更新价值函数（基线）
            # Update value function (baseline)
            # w ← w + α_w (G - V(s)) ∇V(s,w)
            value_error = G - baseline
            self.value_errors.append(value_error)
            self.value_function.update(state, G, self.alpha_w)
        
        # 统计
        # Statistics
        episode_return = sum(rewards)
        self.episode_returns.append(episode_return)
        self.episode_count += 1
        
        return episode_return
    
    def get_statistics(self) -> Dict[str, float]:
        """
        获取统计信息
        Get statistics
        """
        if len(self.episode_returns) == 0:
            return {}
        
        recent_returns = self.episode_returns[-100:]
        recent_advantages = self.advantages[-1000:]
        recent_value_errors = self.value_errors[-1000:]
        
        return {
            'mean_return': np.mean(recent_returns),
            'std_return': np.std(recent_returns),
            'mean_advantage': np.mean(recent_advantages) if recent_advantages else 0,
            'std_advantage': np.std(recent_advantages) if recent_advantages else 0,
            'mean_value_error': np.mean(np.abs(recent_value_errors)) if recent_value_errors else 0,
            'total_episodes': self.episode_count
        }


# ================================================================================
# 第13.3.2节：All-actions REINFORCE
# Section 13.3.2: All-actions REINFORCE
# ================================================================================

class AllActionsREINFORCE:
    """
    All-actions REINFORCE
    
    更新所有动作的概率
    Update probabilities for all actions
    
    梯度估计 Gradient Estimate:
    ∇J(θ) = E[Σ_a π(a|s,θ) q(s,a) ∇ln π(a|s,θ)]
    
    优势 Advantages:
    - 更低方差
      Lower variance
    - 更好的探索
      Better exploration
    - 需要动作价值函数
      Requires action-value function
    """
    
    def __init__(self,
                 policy: Any,
                 q_function: Any,
                 alpha: float = 0.01,
                 gamma: float = 0.99):
        """
        初始化All-actions REINFORCE
        Initialize All-actions REINFORCE
        
        Args:
            policy: 策略
                   Policy
            q_function: 动作价值函数
                       Action-value function
            alpha: 学习率
                  Learning rate
            gamma: 折扣因子
                  Discount factor
        """
        self.policy = policy
        self.q_function = q_function
        self.alpha = alpha
        self.gamma = gamma
        
        # 统计
        # Statistics
        self.episode_count = 0
        self.episode_returns = []
        self.gradient_updates = 0
        
        logger.info(f"初始化All-actions REINFORCE")
    
    def update_policy(self, state: Any):
        """
        更新策略（考虑所有动作）
        Update policy (considering all actions)
        
        Args:
            state: 状态
                  State
        """
        # 获取动作概率分布
        # Get action probability distribution
        action_probs = self.policy.compute_action_probabilities(state)
        
        # 计算期望梯度
        # Compute expected gradient
        expected_gradient = np.zeros_like(self.policy.theta)
        
        for action in range(self.policy.n_actions):
            # 获取Q值
            # Get Q value
            q_value = self.q_function.get_value(state, action)
            
            # 计算该动作的对数梯度
            # Compute log gradient for this action
            log_grad = self.policy.compute_log_gradient(state, action)
            
            # 加权累积
            # Weighted accumulation
            # ∇J ≈ Σ_a π(a|s) Q(s,a) ∇ln π(a|s)
            if hasattr(log_grad, 'shape') and len(log_grad.shape) == 1:
                expected_gradient[action] += action_probs[action] * q_value * log_grad
            else:
                expected_gradient += action_probs[action] * q_value * log_grad
        
        # 更新策略参数
        # Update policy parameters
        self.policy.update_parameters(self.alpha * expected_gradient, step_size=1.0)
        self.gradient_updates += 1
    
    def learn_episode(self, env: Any, max_steps: int = 1000) -> float:
        """
        学习一个回合
        Learn one episode
        """
        # 生成回合
        # Generate episode
        states = []
        actions = []
        rewards = []
        
        state = env.reset()
        
        for step in range(max_steps):
            # 记录状态
            # Record state
            states.append(state)
            
            # 选择动作
            # Select action
            action = self.policy.select_action(state)
            actions.append(action)
            
            # 执行动作
            # Execute action
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            
            if done:
                break
            
            state = next_state
        
        # 计算回报并更新Q函数
        # Compute returns and update Q function
        G = 0
        for t in reversed(range(len(states))):
            G = rewards[t] + self.gamma * G
            
            # 更新Q函数
            # Update Q function
            self.q_function.update(states[t], actions[t], G, self.alpha)
            
            # 使用所有动作更新策略
            # Update policy using all actions
            self.update_policy(states[t])
        
        # 统计
        # Statistics
        episode_return = sum(rewards)
        self.episode_returns.append(episode_return)
        self.episode_count += 1
        
        return episode_return


# ================================================================================
# 简单的价值函数和Q函数实现
# Simple Value Function and Q Function Implementation
# ================================================================================

class SimpleValueFunction:
    """
    简单的价值函数（用于基线）
    Simple value function (for baseline)
    """
    
    def __init__(self, n_features: int, feature_extractor: Callable):
        """
        初始化价值函数
        Initialize value function
        """
        self.n_features = n_features
        self.feature_extractor = feature_extractor
        self.weights = np.zeros(n_features)
    
    def get_value(self, state: Any) -> float:
        """
        获取状态价值
        Get state value
        """
        features = self.feature_extractor(state)
        return np.dot(self.weights, features)
    
    def update(self, state: Any, target: float, alpha: float):
        """
        更新价值函数
        Update value function
        """
        features = self.feature_extractor(state)
        current_value = self.get_value(state)
        error = target - current_value
        self.weights += alpha * error * features


class SimpleQFunction:
    """
    简单的Q函数
    Simple Q function
    """
    
    def __init__(self, n_features: int, n_actions: int, feature_extractor: Callable):
        """
        初始化Q函数
        Initialize Q function
        """
        self.n_features = n_features
        self.n_actions = n_actions
        self.feature_extractor = feature_extractor
        self.weights = np.zeros((n_actions, n_features))
    
    def get_value(self, state: Any, action: int) -> float:
        """
        获取动作价值
        Get action value
        """
        features = self.feature_extractor(state, action)
        return np.dot(self.weights[action], features)
    
    def update(self, state: Any, action: int, target: float, alpha: float):
        """
        更新Q函数
        Update Q function
        """
        features = self.feature_extractor(state, action)
        current_value = self.get_value(state, action)
        error = target - current_value
        self.weights[action] += alpha * error * features


# ================================================================================
# 主函数：演示REINFORCE算法
# Main Function: Demonstrate REINFORCE Algorithms
# ================================================================================

def demonstrate_reinforce():
    """
    演示REINFORCE算法
    Demonstrate REINFORCE algorithms
    """
    print("\n" + "="*80)
    print("第13.3节：REINFORCE算法")
    print("Section 13.3: REINFORCE Algorithms")
    print("="*80)
    
    # 导入策略类
    # Import policy classes
    from policy_gradient_theorem import SoftmaxPolicy
    
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
        """简单的网格环境"""
        def __init__(self):
            self.state = 0
            self.step_count = 0
        
        def reset(self):
            self.state = 0
            self.step_count = 0
            return self.state
        
        def step(self, action):
            # 动作0: 左/停留, 动作1: 右
            # Action 0: left/stay, Action 1: right
            if action == 0:
                self.state = max(0, self.state - 1)
            else:
                self.state = min(n_states - 1, self.state + 1)
            
            # 奖励：到达最右端获得大奖励
            # Reward: big reward for reaching rightmost state
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
    
    # 1. 测试基础REINFORCE
    # 1. Test basic REINFORCE
    print("\n" + "="*60)
    print("1. 基础REINFORCE")
    print("1. Basic REINFORCE")
    print("="*60)
    
    # 创建策略
    # Create policy
    basic_policy = SoftmaxPolicy(
        n_features=n_features,
        n_actions=n_actions,
        feature_extractor=state_action_features,
        temperature=1.0
    )
    
    # 创建REINFORCE算法
    # Create REINFORCE algorithm
    basic_reinforce = REINFORCE(
        policy=basic_policy,
        alpha=0.1,
        gamma=0.9
    )
    
    print("\n训练基础REINFORCE...")
    env = SimpleEnv()
    
    for episode in range(50):
        episode_return = basic_reinforce.learn_episode(env, max_steps=20)
        
        if (episode + 1) % 10 == 0:
            stats = basic_reinforce.get_statistics()
            print(f"  回合{episode + 1}: "
                  f"平均回报={stats['mean_return']:.1f} ± {stats['std_return']:.1f}, "
                  f"梯度范数={stats['mean_gradient_norm']:.4f}")
    
    # 2. 测试带基线的REINFORCE
    # 2. Test REINFORCE with baseline
    print("\n" + "="*60)
    print("2. 带基线的REINFORCE")
    print("2. REINFORCE with Baseline")
    print("="*60)
    
    # 创建新策略和价值函数
    # Create new policy and value function
    baseline_policy = SoftmaxPolicy(
        n_features=n_features,
        n_actions=n_actions,
        feature_extractor=state_action_features,
        temperature=1.0
    )
    
    value_function = SimpleValueFunction(
        n_features=n_features,
        feature_extractor=state_features
    )
    
    # 创建带基线的REINFORCE
    # Create REINFORCE with baseline
    reinforce_baseline = REINFORCEWithBaseline(
        policy=baseline_policy,
        value_function=value_function,
        alpha_theta=0.05,
        alpha_w=0.1,
        gamma=0.9
    )
    
    print("\n训练REINFORCE with Baseline...")
    
    for episode in range(50):
        episode_return = reinforce_baseline.learn_episode(env, max_steps=20)
        
        if (episode + 1) % 10 == 0:
            stats = reinforce_baseline.get_statistics()
            print(f"  回合{episode + 1}: "
                  f"平均回报={stats['mean_return']:.1f} ± {stats['std_return']:.1f}, "
                  f"优势={stats['mean_advantage']:.2f} ± {stats['std_advantage']:.2f}")
    
    # 3. 测试All-actions REINFORCE
    # 3. Test All-actions REINFORCE
    print("\n" + "="*60)
    print("3. All-actions REINFORCE")
    print("="*60)
    
    # 创建新策略和Q函数
    # Create new policy and Q function
    all_actions_policy = SoftmaxPolicy(
        n_features=n_features,
        n_actions=n_actions,
        feature_extractor=state_action_features,
        temperature=1.0
    )
    
    q_function = SimpleQFunction(
        n_features=n_features,
        n_actions=n_actions,
        feature_extractor=state_action_features
    )
    
    # 创建All-actions REINFORCE
    # Create All-actions REINFORCE
    all_actions_reinforce = AllActionsREINFORCE(
        policy=all_actions_policy,
        q_function=q_function,
        alpha=0.05,
        gamma=0.9
    )
    
    print("\n训练All-actions REINFORCE...")
    
    for episode in range(50):
        episode_return = all_actions_reinforce.learn_episode(env, max_steps=20)
        
        if (episode + 1) % 10 == 0:
            mean_return = np.mean(all_actions_reinforce.episode_returns[-10:])
            print(f"  回合{episode + 1}: "
                  f"平均回报={mean_return:.1f}, "
                  f"梯度更新数={all_actions_reinforce.gradient_updates}")
    
    # 4. 比较不同REINFORCE变体
    # 4. Compare different REINFORCE variants
    print("\n" + "="*60)
    print("4. REINFORCE变体比较")
    print("4. REINFORCE Variants Comparison")
    print("="*60)
    
    print("\n最终性能比较:")
    print("算法                  最终10回合平均回报")
    print("-" * 40)
    
    basic_final = np.mean(basic_reinforce.episode_returns[-10:]) if len(basic_reinforce.episode_returns) >= 10 else 0
    baseline_final = np.mean(reinforce_baseline.episode_returns[-10:]) if len(reinforce_baseline.episode_returns) >= 10 else 0
    all_actions_final = np.mean(all_actions_reinforce.episode_returns[-10:]) if len(all_actions_reinforce.episode_returns) >= 10 else 0
    
    print(f"基础REINFORCE:        {basic_final:8.2f}")
    print(f"带基线REINFORCE:      {baseline_final:8.2f}")
    print(f"All-actions:          {all_actions_final:8.2f}")
    
    print("\n方差比较:")
    basic_std = np.std(basic_reinforce.episode_returns[-10:]) if len(basic_reinforce.episode_returns) >= 10 else 0
    baseline_std = np.std(reinforce_baseline.episode_returns[-10:]) if len(reinforce_baseline.episode_returns) >= 10 else 0
    all_actions_std = np.std(all_actions_reinforce.episode_returns[-10:]) if len(all_actions_reinforce.episode_returns) >= 10 else 0
    
    print(f"基础REINFORCE标准差:   {basic_std:.2f}")
    print(f"带基线标准差:          {baseline_std:.2f}")
    print(f"All-actions标准差:     {all_actions_std:.2f}")
    
    # 总结
    # Summary
    print("\n" + "="*80)
    print("REINFORCE算法总结")
    print("REINFORCE Algorithms Summary")
    print("="*80)
    
    print("""
    关键要点 Key Takeaways:
    =======================
    
    1. REINFORCE是蒙特卡洛策略梯度
       REINFORCE is Monte Carlo policy gradient
       
    2. 基线减少方差但不改变期望
       Baseline reduces variance without changing expectation
       
    3. All-actions方法需要Q函数但方差更低
       All-actions method needs Q function but has lower variance
       
    4. 权衡：偏差vs方差
       Tradeoff: bias vs variance
       
    5. REINFORCE适合情节性任务
       REINFORCE suitable for episodic tasks
    
    实践建议 Practical Advice:
    - 总是使用基线
      Always use baseline
    - 仔细调整学习率
      Carefully tune learning rate
    - 考虑使用Actor-Critic
      Consider Actor-Critic
    - 归一化回报可能有帮助
      Normalizing returns may help
    """)


if __name__ == "__main__":
    demonstrate_reinforce()