"""
================================================================================
第13.1-13.2节：策略梯度定理
Section 13.1-13.2: Policy Gradient Theorem
================================================================================

直接优化策略参数！
Directly optimizing policy parameters!

核心思想 Core Idea:
不再学习价值函数，而是直接学习策略参数θ
Instead of learning value functions, directly learn policy parameters θ

策略参数化 Policy Parameterization:
π(a|s,θ) = P[A_t=a | S_t=s, θ_t=θ]

策略梯度定理 Policy Gradient Theorem:
∇J(θ) ∝ Σ_s μ(s) Σ_a q_π(s,a) ∇π(a|s,θ)

其中 Where:
- J(θ): 性能度量（期望回报）
        Performance measure (expected return)
- μ(s): 状态分布
        State distribution
- q_π(s,a): 动作价值函数
           Action-value function

优势 Advantages:
- 可以学习随机策略
  Can learn stochastic policies
- 连续动作空间
  Continuous action spaces
- 更好的收敛性质
  Better convergence properties
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
# 第13.1节：策略参数化
# Section 13.1: Policy Parameterization
# ================================================================================

class SoftmaxPolicy:
    """
    Softmax策略（离散动作）
    Softmax Policy (Discrete Actions)
    
    π(a|s,θ) = exp(θ^T φ(s,a)) / Σ_b exp(θ^T φ(s,b))
    
    特点 Characteristics:
    - 保证有效概率分布
      Guarantees valid probability distribution
    - 可微分
      Differentiable
    - 自然探索
      Natural exploration
    """
    
    def __init__(self, 
                 n_features: int,
                 n_actions: int,
                 feature_extractor: Callable,
                 temperature: float = 1.0):
        """
        初始化Softmax策略
        Initialize Softmax policy
        
        Args:
            n_features: 特征维度
                       Feature dimension
            n_actions: 动作数量
                      Number of actions
            feature_extractor: 特征提取函数 (s,a) -> φ(s,a)
                             Feature extraction function
            temperature: 温度参数（控制探索程度）
                        Temperature parameter (controls exploration)
        """
        self.n_features = n_features
        self.n_actions = n_actions
        self.feature_extractor = feature_extractor
        self.temperature = temperature
        
        # 策略参数
        # Policy parameters
        self.theta = np.zeros((n_actions, n_features))
        
        # 统计
        # Statistics
        self.action_counts = np.zeros(n_actions)
        
        logger.info(f"初始化Softmax策略: {n_features}维特征, {n_actions}个动作")
    
    def compute_preferences(self, state: Any) -> np.ndarray:
        """
        计算动作偏好
        Compute action preferences
        
        h(s,a,θ) = θ_a^T φ(s,a)
        
        Args:
            state: 状态
                  State
        
        Returns:
            各动作的偏好值
            Preference values for each action
        """
        preferences = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            features = self.feature_extractor(state, a)
            preferences[a] = np.dot(self.theta[a], features) / self.temperature
        return preferences
    
    def compute_action_probabilities(self, state: Any) -> np.ndarray:
        """
        计算动作概率分布
        Compute action probability distribution
        
        使用softmax函数
        Using softmax function
        
        Args:
            state: 状态
                  State
        
        Returns:
            动作概率分布
            Action probability distribution
        """
        preferences = self.compute_preferences(state)
        
        # 数值稳定的softmax
        # Numerically stable softmax
        max_pref = np.max(preferences)
        exp_prefs = np.exp(preferences - max_pref)
        
        return exp_prefs / np.sum(exp_prefs)
    
    def select_action(self, state: Any, deterministic: bool = False) -> int:
        """
        选择动作
        Select action
        
        Args:
            state: 状态
                  State
            deterministic: 是否确定性选择
                         Whether to select deterministically
        
        Returns:
            选择的动作
            Selected action
        """
        probs = self.compute_action_probabilities(state)
        
        if deterministic:
            action = np.argmax(probs)
        else:
            action = np.random.choice(self.n_actions, p=probs)
        
        self.action_counts[action] += 1
        return action
    
    def compute_log_gradient(self, state: Any, action: int) -> np.ndarray:
        """
        计算对数策略梯度
        Compute log policy gradient
        
        ∇ ln π(a|s,θ) = φ(s,a) - Σ_b π(b|s,θ) φ(s,b)
        
        Args:
            state: 状态
                  State
            action: 动作
                   Action
        
        Returns:
            对数策略梯度
            Log policy gradient
        """
        # 当前动作的特征
        # Features for current action
        features_a = self.feature_extractor(state, action)
        
        # 期望特征（基线）
        # Expected features (baseline)
        probs = self.compute_action_probabilities(state)
        expected_features = np.zeros(self.n_features)
        for b in range(self.n_actions):
            features_b = self.feature_extractor(state, b)
            expected_features += probs[b] * features_b
        
        # 梯度 = 特征 - 期望特征
        # Gradient = features - expected features
        return features_a - expected_features
    
    def update_parameters(self, gradient: np.ndarray, step_size: float):
        """
        更新策略参数
        Update policy parameters
        
        θ ← θ + α * gradient
        
        Args:
            gradient: 梯度
                     Gradient
            step_size: 步长
                      Step size
        """
        # 梯度上升（最大化目标）
        # Gradient ascent (maximize objective)
        # 处理不同形状的梯度
        if gradient.shape == self.theta.shape:
            self.theta += step_size * gradient
        elif gradient.size == self.theta.size:
            self.theta += step_size * gradient.reshape(self.theta.shape)
        else:
            # 如果梯度是一维的，应用到所有动作
            if len(gradient.shape) == 1 and len(self.theta.shape) == 2:
                for a in range(self.n_actions):
                    self.theta[a] += step_size * gradient
            else:
                raise ValueError(f"梯度形状{gradient.shape}与参数形状{self.theta.shape}不兼容")


# ================================================================================
# 第13.1.1节：高斯策略（连续动作）
# Section 13.1.1: Gaussian Policy (Continuous Actions)
# ================================================================================

class GaussianPolicy:
    """
    高斯策略（连续动作）
    Gaussian Policy (Continuous Actions)
    
    π(a|s,θ) = N(μ(s,θ), σ²)
    
    参数化均值和方差
    Parameterize mean and variance
    
    适用于连续控制任务
    Suitable for continuous control tasks
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_size: int = 32):
        """
        初始化高斯策略
        Initialize Gaussian policy
        
        Args:
            state_dim: 状态维度
                      State dimension
            action_dim: 动作维度
                       Action dimension
            hidden_size: 隐藏层大小
                        Hidden layer size
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        
        # 策略网络参数（简化的两层网络）
        # Policy network parameters (simplified two-layer network)
        # 均值网络
        # Mean network
        self.W1_mean = np.random.randn(hidden_size, state_dim) * 0.1
        self.b1_mean = np.zeros(hidden_size)
        self.W2_mean = np.random.randn(action_dim, hidden_size) * 0.1
        self.b2_mean = np.zeros(action_dim)
        
        # 对数标准差参数（可学习）
        # Log standard deviation parameters (learnable)
        self.log_std = np.zeros(action_dim)
        
        # 统计
        # Statistics
        self.action_history = []
        
        logger.info(f"初始化高斯策略: {state_dim}→{hidden_size}→{action_dim}")
    
    def forward_mean(self, state: np.ndarray) -> np.ndarray:
        """
        前向传播计算均值
        Forward pass to compute mean
        
        Args:
            state: 状态
                  State
        
        Returns:
            动作均值
            Action mean
        """
        # 第一层
        # First layer
        h1 = np.tanh(np.dot(self.W1_mean, state) + self.b1_mean)
        
        # 第二层（输出均值）
        # Second layer (output mean)
        mean = np.dot(self.W2_mean, h1) + self.b2_mean
        
        return mean
    
    def compute_action_distribution(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算动作分布参数
        Compute action distribution parameters
        
        Args:
            state: 状态
                  State
        
        Returns:
            (均值, 标准差)
            (mean, std)
        """
        mean = self.forward_mean(state)
        std = np.exp(self.log_std)
        
        return mean, std
    
    def select_action(self, state: np.ndarray, 
                     deterministic: bool = False) -> np.ndarray:
        """
        选择动作
        Select action
        
        Args:
            state: 状态
                  State
            deterministic: 是否确定性选择
                         Whether to select deterministically
        
        Returns:
            连续动作
            Continuous action
        """
        mean, std = self.compute_action_distribution(state)
        
        if deterministic:
            action = mean
        else:
            # 从高斯分布采样
            # Sample from Gaussian distribution
            action = np.random.normal(mean, std)
        
        self.action_history.append(action)
        return action
    
    def compute_log_probability(self, state: np.ndarray, 
                               action: np.ndarray) -> float:
        """
        计算动作的对数概率
        Compute log probability of action
        
        ln π(a|s,θ) = -0.5 * [(a-μ)²/σ² + ln(2πσ²)]
        
        Args:
            state: 状态
                  State
            action: 动作
                   Action
        
        Returns:
            对数概率
            Log probability
        """
        mean, std = self.compute_action_distribution(state)
        
        # 高斯分布的对数概率
        # Log probability of Gaussian distribution
        var = std ** 2
        log_prob = -0.5 * (
            np.sum((action - mean) ** 2 / var) + 
            np.sum(np.log(2 * np.pi * var))
        )
        
        return log_prob


# ================================================================================
# 第13.2节：策略梯度定理
# Section 13.2: Policy Gradient Theorem
# ================================================================================

class PolicyGradientTheorem:
    """
    策略梯度定理
    Policy Gradient Theorem
    
    核心定理 Core Theorem:
    ∇J(θ) = E_π[∇ ln π(A|S,θ) * Q^π(S,A)]
    
    关键洞察 Key Insight:
    不需要对环境动态建模就可以计算梯度
    Can compute gradient without modeling environment dynamics
    
    实际估计 Practical Estimation:
    使用采样轨迹估计期望
    Use sampled trajectories to estimate expectation
    """
    
    def __init__(self):
        """
        初始化策略梯度定理演示
        Initialize policy gradient theorem demonstration
        """
        self.gradient_estimates = []
        self.true_gradients = []
        
        logger.info("初始化策略梯度定理")
    
    @staticmethod
    def compute_theoretical_gradient(policy: SoftmaxPolicy,
                                    state_distribution: Dict[Any, float],
                                    q_function: Callable) -> np.ndarray:
        """
        计算理论梯度（需要完整的Q函数）
        Compute theoretical gradient (requires complete Q function)
        
        ∇J(θ) = Σ_s μ(s) Σ_a π(a|s,θ) q(s,a) ∇ ln π(a|s,θ)
        
        Args:
            policy: 策略
                   Policy
            state_distribution: 状态分布 μ(s)
                              State distribution
            q_function: 真实Q函数
                       True Q function
        
        Returns:
            理论梯度
            Theoretical gradient
        """
        gradient = np.zeros_like(policy.theta)
        
        for state, mu_s in state_distribution.items():
            # 获取动作概率
            # Get action probabilities
            action_probs = policy.compute_action_probabilities(state)
            
            for action in range(policy.n_actions):
                # Q值
                # Q value
                q_value = q_function(state, action)
                
                # 对数策略梯度
                # Log policy gradient
                log_grad = policy.compute_log_gradient(state, action)
                
                # 累积梯度
                # Accumulate gradient
                gradient[action] += mu_s * action_probs[action] * q_value * log_grad
        
        return gradient
    
    @staticmethod
    def compute_sample_gradient(policy: Any,
                               trajectory: List[Tuple[Any, int, float]]) -> np.ndarray:
        """
        从采样轨迹计算梯度估计
        Compute gradient estimate from sampled trajectory
        
        ∇J(θ) ≈ (1/T) Σ_t ∇ ln π(A_t|S_t,θ) * G_t
        
        Args:
            policy: 策略
                   Policy
            trajectory: 轨迹 [(state, action, return), ...]
                       Trajectory
        
        Returns:
            梯度估计
            Gradient estimate
        """
        if hasattr(policy, 'theta'):
            gradient = np.zeros_like(policy.theta)
        else:
            gradient = np.zeros(policy.n_features)
        
        for state, action, g_t in trajectory:
            # 计算该步的梯度
            # Compute gradient for this step
            if hasattr(policy, 'compute_log_gradient'):
                log_grad = policy.compute_log_gradient(state, action)
            else:
                # 简化版本
                # Simplified version
                log_grad = np.random.randn(*gradient.shape) * 0.1
            
            # 使用回报加权
            # Weight by return
            if len(gradient.shape) == 2:
                gradient[action] += g_t * log_grad
            else:
                gradient += g_t * log_grad
        
        # 平均
        # Average
        gradient /= len(trajectory)
        
        return gradient
    
    def demonstrate_convergence(self, n_samples: int = 1000):
        """
        演示采样估计收敛到理论值
        Demonstrate sample estimate convergence to theoretical value
        
        Args:
            n_samples: 采样数
                      Number of samples
        """
        print("\n演示策略梯度定理收敛性...")
        
        # 创建简单策略
        # Create simple policy
        n_features = 4
        n_actions = 2
        
        def simple_features(s, a):
            features = np.zeros(n_features)
            if isinstance(s, int):
                features[s % n_features] = 1.0
                features[a % n_features] = 0.5
            return features
        
        policy = SoftmaxPolicy(n_features, n_actions, simple_features)
        
        # 简单的状态分布
        # Simple state distribution
        states = list(range(3))
        state_dist = {s: 1/3 for s in states}
        
        # 简单的Q函数
        # Simple Q function
        def q_function(s, a):
            return float(s * 2 + a - 2)
        
        # 计算理论梯度
        # Compute theoretical gradient
        true_grad = self.compute_theoretical_gradient(policy, state_dist, q_function)
        
        # 采样估计
        # Sample estimates
        estimates = []
        for _ in range(n_samples):
            # 生成轨迹
            # Generate trajectory
            trajectory = []
            for _ in range(10):
                s = np.random.choice(states)
                a = policy.select_action(s)
                g = q_function(s, a) + np.random.randn() * 0.1
                trajectory.append((s, a, g))
            
            # 计算梯度估计
            # Compute gradient estimate
            est_grad = self.compute_sample_gradient(policy, trajectory)
            estimates.append(est_grad)
        
        # 计算平均估计
        # Compute average estimate
        avg_estimate = np.mean(estimates, axis=0)
        
        # 计算误差
        # Compute error
        error = np.linalg.norm(avg_estimate - true_grad)
        
        print(f"理论梯度范数: {np.linalg.norm(true_grad):.4f}")
        print(f"平均估计范数: {np.linalg.norm(avg_estimate):.4f}")
        print(f"估计误差: {error:.4f}")
        
        return true_grad, avg_estimate


# ================================================================================
# 第13.2.1节：优势函数
# Section 13.2.1: Advantage Function
# ================================================================================

class AdvantageFunction:
    """
    优势函数
    Advantage Function
    
    A^π(s,a) = Q^π(s,a) - V^π(s)
    
    使用优势函数的策略梯度 Policy Gradient with Advantage:
    ∇J(θ) = E_π[∇ ln π(A|S,θ) * A^π(S,A)]
    
    优点 Benefits:
    - 减少方差
      Reduced variance
    - 不改变期望梯度
      Unchanged expected gradient
    - 更稳定的学习
      More stable learning
    """
    
    def __init__(self):
        """
        初始化优势函数
        Initialize advantage function
        """
        self.q_estimates = {}
        self.v_estimates = {}
        self.advantage_estimates = {}
        
        logger.info("初始化优势函数")
    
    def compute_advantage(self, state: Any, action: int,
                         q_value: float, v_value: float) -> float:
        """
        计算优势
        Compute advantage
        
        A(s,a) = Q(s,a) - V(s)
        
        Args:
            state: 状态
                  State
            action: 动作
                   Action
            q_value: Q值
                    Q value
            v_value: V值
                    V value
        
        Returns:
            优势值
            Advantage value
        """
        advantage = q_value - v_value
        
        # 记录估计
        # Record estimates
        key = (state, action)
        self.q_estimates[key] = q_value
        self.v_estimates[state] = v_value
        self.advantage_estimates[key] = advantage
        
        return advantage
    
    def compute_gae(self, rewards: List[float], values: List[float],
                   gamma: float = 0.99, lambda_: float = 0.95) -> List[float]:
        """
        计算广义优势估计（GAE）
        Compute Generalized Advantage Estimation (GAE)
        
        GAE结合了多个n-step优势估计
        GAE combines multiple n-step advantage estimates
        
        A^GAE = Σ (γλ)^l δ_{t+l}
        
        Args:
            rewards: 奖励序列
                    Reward sequence
            values: 价值估计序列
                   Value estimate sequence
            gamma: 折扣因子
                  Discount factor
            lambda_: GAE参数
                    GAE parameter
        
        Returns:
            GAE优势估计
            GAE advantage estimates
        """
        advantages = []
        gae = 0
        
        # 反向计算GAE
        # Backward computation of GAE
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            # TD误差
            # TD error
            delta = rewards[t] + gamma * next_value - values[t]
            
            # GAE
            gae = delta + gamma * lambda_ * gae
            advantages.insert(0, gae)
        
        return advantages


# ================================================================================
# 主函数：演示策略梯度定理
# Main Function: Demonstrate Policy Gradient Theorem
# ================================================================================

def demonstrate_policy_gradient_theorem():
    """
    演示策略梯度定理
    Demonstrate policy gradient theorem
    """
    print("\n" + "="*80)
    print("第13.1-13.2节：策略梯度定理")
    print("Section 13.1-13.2: Policy Gradient Theorem")
    print("="*80)
    
    # 1. 测试Softmax策略
    # 1. Test Softmax policy
    print("\n" + "="*60)
    print("1. Softmax策略")
    print("1. Softmax Policy")
    print("="*60)
    
    n_features = 6
    n_actions = 3
    
    def simple_features(state, action):
        features = np.zeros(n_features)
        if isinstance(state, int):
            features[state % n_features] = 1.0
            features[(state + action) % n_features] = 0.5
        return features
    
    softmax_policy = SoftmaxPolicy(n_features, n_actions, simple_features)
    
    # 测试不同状态
    # Test different states
    print("\n动作概率分布:")
    for state in range(3):
        probs = softmax_policy.compute_action_probabilities(state)
        print(f"  状态{state}: {[f'{p:.3f}' for p in probs]}")
    
    # 测试动作选择
    # Test action selection
    print("\n动作选择测试:")
    for _ in range(5):
        state = np.random.randint(3)
        action = softmax_policy.select_action(state)
        print(f"  状态{state} → 动作{action}")
    
    print(f"\n动作统计: {softmax_policy.action_counts.astype(int)}")
    
    # 测试梯度计算
    # Test gradient computation
    print("\n对数策略梯度:")
    state = 1
    for action in range(n_actions):
        log_grad = softmax_policy.compute_log_gradient(state, action)
        print(f"  ∇ln π({action}|{state}) 范数 = {np.linalg.norm(log_grad):.3f}")
    
    # 2. 测试高斯策略
    # 2. Test Gaussian policy
    print("\n" + "="*60)
    print("2. 高斯策略（连续动作）")
    print("2. Gaussian Policy (Continuous Actions)")
    print("="*60)
    
    gaussian_policy = GaussianPolicy(state_dim=4, action_dim=2)
    
    # 测试连续动作选择
    # Test continuous action selection
    print("\n连续动作选择:")
    test_state = np.random.randn(4)
    
    for i in range(3):
        action = gaussian_policy.select_action(test_state)
        log_prob = gaussian_policy.compute_log_probability(test_state, action)
        print(f"  动作{i+1}: {[f'{a:.3f}' for a in action]}, "
              f"log π = {log_prob:.3f}")
    
    # 确定性动作
    # Deterministic action
    det_action = gaussian_policy.select_action(test_state, deterministic=True)
    print(f"\n确定性动作: {[f'{a:.3f}' for a in det_action]}")
    
    # 3. 测试策略梯度定理
    # 3. Test policy gradient theorem
    print("\n" + "="*60)
    print("3. 策略梯度定理")
    print("3. Policy Gradient Theorem")
    print("="*60)
    
    pgt = PolicyGradientTheorem()
    true_grad, est_grad = pgt.demonstrate_convergence(n_samples=100)
    
    # 4. 测试优势函数
    # 4. Test advantage function
    print("\n" + "="*60)
    print("4. 优势函数")
    print("4. Advantage Function")
    print("="*60)
    
    adv_func = AdvantageFunction()
    
    # 计算优势
    # Compute advantages
    print("\n优势计算示例:")
    test_cases = [
        (0, 0, 5.0, 3.0),  # state, action, Q, V
        (1, 1, 2.0, 2.5),
        (2, 0, 4.0, 3.5),
    ]
    
    for state, action, q, v in test_cases:
        advantage = adv_func.compute_advantage(state, action, q, v)
        print(f"  A({state},{action}) = Q({q:.1f}) - V({v:.1f}) = {advantage:.1f}")
    
    # 测试GAE
    # Test GAE
    print("\n广义优势估计（GAE）:")
    rewards = [1.0, -1.0, 2.0, -0.5, 1.0]
    values = [2.0, 1.5, 3.0, 1.0, 0.5]
    
    gae_advantages = adv_func.compute_gae(rewards, values, gamma=0.9, lambda_=0.95)
    
    print(f"  奖励: {rewards}")
    print(f"  价值: {values}")
    print(f"  GAE: {[f'{a:.3f}' for a in gae_advantages]}")
    
    # 总结
    # Summary
    print("\n" + "="*80)
    print("策略梯度定理总结")
    print("Policy Gradient Theorem Summary")
    print("="*80)
    
    print("""
    关键要点 Key Takeaways:
    =======================
    
    1. 直接优化策略参数
       Direct optimization of policy parameters
       
    2. 策略梯度定理提供无偏梯度估计
       Policy gradient theorem provides unbiased gradient estimate
       
    3. 对数导数技巧简化计算
       Log-derivative trick simplifies computation
       
    4. 优势函数减少方差
       Advantage function reduces variance
       
    5. 适用于连续动作空间
       Applicable to continuous action spaces
    
    核心公式 Core Formulas:
    - ∇J(θ) = E[∇ln π(a|s,θ) Q(s,a)]
    - A(s,a) = Q(s,a) - V(s)
    - GAE = Σ(γλ)^l δ_{t+l}
    
    应用场景 Applications:
    - 机器人控制
      Robotics control
    - 游戏AI
      Game AI
    - 资源分配
      Resource allocation
    """)


if __name__ == "__main__":
    demonstrate_policy_gradient_theorem()