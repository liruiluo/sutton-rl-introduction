"""
================================================================================
第11.1-11.2节：重要性采样与函数近似
Section 11.1-11.2: Importance Sampling with Function Approximation
================================================================================

离策略学习的核心技术！
Core technique for off-policy learning!

重要性采样 Importance Sampling:
- 修正行为策略与目标策略的差异
  Correcting differences between behavior and target policies
- 重要性比率 ρ = π(A|S)/b(A|S)
  Importance ratio

加权重要性采样 Weighted Importance Sampling:
- 减少方差但增加偏差
  Reduces variance but adds bias
- 更实用的选择
  More practical choice

Per-decision重要性采样 Per-decision Importance Sampling:
- 只对更新的部分应用重要性比率
  Apply importance ratio only to updated part
- 更低的方差
  Lower variance
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass
from collections import deque
import logging

# 设置日志
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第11.1节：重要性采样基础
# Section 11.1: Importance Sampling Fundamentals
# ================================================================================

@dataclass
class Trajectory:
    """
    轨迹数据
    Trajectory data
    
    存储一条完整的经验轨迹
    Stores a complete experience trajectory
    """
    states: List[Any]        # 状态序列
    actions: List[int]       # 动作序列
    rewards: List[float]     # 奖励序列
    probs_b: List[float]     # 行为策略概率
    probs_pi: List[float]    # 目标策略概率
    
    @property
    def length(self) -> int:
        """轨迹长度"""
        return len(self.rewards)
    
    def compute_importance_ratio(self, start: int = 0, end: Optional[int] = None) -> float:
        """
        计算重要性比率
        Compute importance ratio
        
        ρ_{t:T-1} = Π_{k=t}^{T-1} π(A_k|S_k)/b(A_k|S_k)
        
        Args:
            start: 起始时间步
                  Start time step
            end: 结束时间步
                End time step
        
        Returns:
            重要性比率
            Importance ratio
        """
        if end is None:
            end = self.length
        
        ratio = 1.0
        for t in range(start, min(end, self.length)):
            if self.probs_b[t] == 0:
                return 0.0
            ratio *= self.probs_pi[t] / self.probs_b[t]
        
        return ratio
    
    def compute_return(self, gamma: float, start: int = 0) -> float:
        """
        计算回报
        Compute return
        
        G_t = Σ_{k=0}^{T-t-1} γ^k R_{t+k+1}
        
        Args:
            gamma: 折扣因子
                  Discount factor
            start: 起始时间步
                  Start time step
        
        Returns:
            折扣回报
            Discounted return
        """
        g = 0.0
        for k in range(start, self.length):
            g += (gamma ** (k - start)) * self.rewards[k]
        return g


class ImportanceSampling:
    """
    重要性采样的离策略评估
    Off-policy Evaluation with Importance Sampling
    
    从行为策略b学习目标策略π的价值
    Learn value of target policy π from behavior policy b
    
    两种变体 Two Variants:
    1. 普通重要性采样 (高方差，无偏)
       Ordinary IS (high variance, unbiased)
    2. 加权重要性采样 (低方差，有偏)
       Weighted IS (low variance, biased)
    """
    
    def __init__(self,
                 n_states: int,
                 gamma: float = 0.99,
                 weighted: bool = True):
        """
        初始化重要性采样
        Initialize importance sampling
        
        Args:
            n_states: 状态数
                     Number of states
            gamma: 折扣因子
                  Discount factor
            weighted: 是否使用加权IS
                     Whether to use weighted IS
        """
        self.n_states = n_states
        self.gamma = gamma
        self.weighted = weighted
        
        # 价值估计
        # Value estimates
        self.v = np.zeros(n_states)
        
        # 加权IS的累积权重
        # Cumulative weights for weighted IS
        self.c = np.zeros(n_states)
        
        # 访问计数
        # Visit counts
        self.visit_counts = np.zeros(n_states)
        
        # 统计
        # Statistics
        self.update_count = 0
        self.trajectories = []
        
        logger.info(f"初始化重要性采样: weighted={weighted}")
    
    def update_from_trajectory(self, trajectory: Trajectory):
        """
        从轨迹更新价值估计
        Update value estimates from trajectory
        
        Args:
            trajectory: 经验轨迹
                       Experience trajectory
        """
        # 对每个访问的状态更新
        # Update for each visited state
        for t in range(trajectory.length):
            state = trajectory.states[t]
            
            # 计算重要性比率
            # Compute importance ratio
            rho = trajectory.compute_importance_ratio(t)
            
            if rho == 0:
                continue
            
            # 计算回报
            # Compute return
            g = trajectory.compute_return(self.gamma, t)
            
            # 更新价值估计
            # Update value estimate
            if self.weighted:
                # 加权重要性采样
                # Weighted importance sampling
                self.c[state] += rho
                if self.c[state] > 0:
                    self.v[state] += (rho / self.c[state]) * (g - self.v[state])
            else:
                # 普通重要性采样
                # Ordinary importance sampling
                self.visit_counts[state] += 1
                self.v[state] += (1.0 / self.visit_counts[state]) * (rho * g - self.v[state])
            
            self.update_count += 1
        
        self.trajectories.append(trajectory)
    
    def get_value(self, state: int) -> float:
        """
        获取状态价值
        Get state value
        
        Args:
            state: 状态
                  State
        
        Returns:
            价值估计
            Value estimate
        """
        return self.v[state]
    
    def compute_mse(self, true_values: np.ndarray) -> float:
        """
        计算均方误差
        Compute mean squared error
        
        Args:
            true_values: 真实价值
                        True values
        
        Returns:
            MSE
        """
        return np.mean((self.v - true_values) ** 2)


# ================================================================================
# 第11.2节：半梯度离策略TD
# Section 11.2: Semi-gradient Off-policy TD
# ================================================================================

class SemiGradientOffPolicyTD:
    """
    半梯度离策略TD
    Semi-gradient Off-policy TD
    
    使用重要性采样的离策略TD学习
    Off-policy TD learning with importance sampling
    
    更新规则 Update Rule:
    δ = R + γV(S') - V(S)
    w ← w + αρδ∇v̂(S,w)
    
    其中ρ是重要性比率
    Where ρ is importance ratio
    """
    
    def __init__(self,
                 feature_extractor: Callable,
                 n_features: int,
                 alpha: float = 0.01,
                 gamma: float = 0.99):
        """
        初始化离策略TD
        Initialize off-policy TD
        
        Args:
            feature_extractor: 特征提取函数
                             Feature extraction function
            n_features: 特征维度
                       Feature dimension
            alpha: 学习率
                  Learning rate
            gamma: 折扣因子
                  Discount factor
        """
        self.feature_extractor = feature_extractor
        self.n_features = n_features
        self.alpha = alpha
        self.gamma = gamma
        
        # 权重向量
        # Weight vector
        self.weights = np.zeros(n_features)
        
        # 统计
        # Statistics
        self.update_count = 0
        self.td_errors = []
        
        logger.info(f"初始化半梯度离策略TD: α={alpha}")
    
    def get_value(self, state: Any) -> float:
        """
        获取状态价值
        Get state value
        
        v̂(s,w) = w^T x(s)
        
        Args:
            state: 状态
                  State
        
        Returns:
            价值估计
            Value estimate
        """
        features = self.feature_extractor(state)
        return np.dot(self.weights, features)
    
    def update(self, state: Any, reward: float, next_state: Any,
               done: bool, importance_ratio: float = 1.0):
        """
        离策略TD更新
        Off-policy TD update
        
        Args:
            state: 当前状态
                  Current state
            reward: 奖励
                   Reward
            next_state: 下一状态
                       Next state
            done: 是否终止
                 Whether done
            importance_ratio: 重要性比率ρ
                            Importance ratio
        """
        # 提取特征
        # Extract features
        features = self.feature_extractor(state)
        
        # 计算TD目标
        # Compute TD target
        current_value = self.get_value(state)
        
        if done:
            td_target = reward
        else:
            next_value = self.get_value(next_state)
            td_target = reward + self.gamma * next_value
        
        # TD误差
        # TD error
        td_error = td_target - current_value
        self.td_errors.append(td_error)
        
        # 离策略更新 (使用重要性比率)
        # Off-policy update (with importance ratio)
        self.weights += self.alpha * importance_ratio * td_error * features
        
        self.update_count += 1
        
        return td_error


# ================================================================================
# 第11.2.1节：Per-decision重要性采样
# Section 11.2.1: Per-decision Importance Sampling
# ================================================================================

class PerDecisionImportanceSampling:
    """
    Per-decision重要性采样
    Per-decision Importance Sampling
    
    更精细的重要性采样方法
    More refined importance sampling method
    
    优势 Advantages:
    - 更低的方差
      Lower variance
    - 更快的收敛
      Faster convergence
    
    关键思想 Key Idea:
    只对当前决策应用重要性比率，而不是整个轨迹
    Apply importance ratio only to current decision, not entire trajectory
    """
    
    def __init__(self,
                 n_features: int,
                 feature_extractor: Callable,
                 alpha: float = 0.01,
                 gamma: float = 0.99,
                 lambda_: float = 0.9):
        """
        初始化Per-decision IS
        Initialize per-decision IS
        
        Args:
            n_features: 特征数
                       Number of features
            feature_extractor: 特征提取器
                             Feature extractor
            alpha: 学习率
                  Learning rate
            gamma: 折扣因子
                  Discount factor
            lambda_: 资格迹衰减率
                    Eligibility trace decay
        """
        self.n_features = n_features
        self.feature_extractor = feature_extractor
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        
        # 权重和资格迹
        # Weights and eligibility traces
        self.weights = np.zeros(n_features)
        self.traces = np.zeros(n_features)
        
        # 统计
        # Statistics
        self.update_count = 0
        
        logger.info(f"初始化Per-decision IS: λ={lambda_}")
    
    def get_value(self, state: Any) -> float:
        """获取状态价值"""
        features = self.feature_extractor(state)
        return np.dot(self.weights, features)
    
    def update(self, state: Any, action: int, reward: float,
               next_state: Any, done: bool,
               prob_b: float, prob_pi: float):
        """
        Per-decision更新
        Per-decision update
        
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
            prob_b: 行为策略概率b(a|s)
                   Behavior policy probability
            prob_pi: 目标策略概率π(a|s)
                    Target policy probability
        """
        # 计算重要性比率
        # Compute importance ratio
        rho = prob_pi / prob_b if prob_b > 0 else 0
        
        # 提取特征
        # Extract features
        features = self.feature_extractor(state)
        
        # 计算TD误差
        # Compute TD error
        current_value = self.get_value(state)
        if done:
            td_target = reward
        else:
            next_value = self.get_value(next_state)
            td_target = reward + self.gamma * next_value
        
        td_error = td_target - current_value
        
        # 更新资格迹
        # Update eligibility traces
        self.traces = self.gamma * self.lambda_ * rho * self.traces + features
        
        # 更新权重
        # Update weights
        self.weights += self.alpha * td_error * self.traces
        
        # 如果终止，重置资格迹
        # Reset traces if terminal
        if done:
            self.traces = np.zeros(self.n_features)
        
        self.update_count += 1
        
        return td_error


# ================================================================================
# 第11.2.2节：n-step离策略学习
# Section 11.2.2: n-step Off-policy Learning
# ================================================================================

class NStepOffPolicyTD:
    """
    n-step离策略TD
    n-step Off-policy TD
    
    结合n-step方法和重要性采样
    Combining n-step methods with importance sampling
    
    关键公式 Key Formula:
    G_{t:t+n} = ρ_t[R_{t+1} + γρ_{t+1}[R_{t+2} + ... + γV(S_{t+n})]]
    
    优势 Advantages:
    - 更快的信用分配
      Faster credit assignment
    - 可调节偏差-方差权衡
      Tunable bias-variance tradeoff
    """
    
    def __init__(self,
                 n_features: int,
                 feature_extractor: Callable,
                 n: int = 4,
                 alpha: float = 0.01,
                 gamma: float = 0.99):
        """
        初始化n-step离策略TD
        Initialize n-step off-policy TD
        
        Args:
            n_features: 特征数
                       Number of features
            feature_extractor: 特征提取器
                             Feature extractor
            n: n步数
              n-step parameter
            alpha: 学习率
                  Learning rate
            gamma: 折扣因子
                  Discount factor
        """
        self.n_features = n_features
        self.feature_extractor = feature_extractor
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        
        # 权重
        # Weights
        self.weights = np.zeros(n_features)
        
        # n-step缓冲
        # n-step buffers
        self.state_buffer = deque(maxlen=n+1)
        self.reward_buffer = deque(maxlen=n)
        self.rho_buffer = deque(maxlen=n)
        
        # 统计
        # Statistics
        self.update_count = 0
        
        logger.info(f"初始化{n}-step离策略TD")
    
    def get_value(self, state: Any) -> float:
        """获取状态价值"""
        features = self.feature_extractor(state)
        return np.dot(self.weights, features)
    
    def compute_n_step_return(self) -> Tuple[float, float]:
        """
        计算n-step回报和重要性比率
        Compute n-step return and importance ratio
        
        Returns:
            (n-step回报, 累积重要性比率)
            (n-step return, cumulative importance ratio)
        """
        n_steps = len(self.reward_buffer)
        if n_steps == 0:
            return 0.0, 1.0
        
        # 累积重要性比率
        # Cumulative importance ratio
        rho_prod = 1.0
        
        # 计算n-step回报
        # Compute n-step return
        g = 0.0
        for i in range(n_steps):
            rho_prod *= self.rho_buffer[i]
            g += (self.gamma ** i) * self.reward_buffer[i] * rho_prod
        
        # 添加bootstrap值
        # Add bootstrap value
        if len(self.state_buffer) > n_steps:
            terminal_state = self.state_buffer[n_steps]
            terminal_value = self.get_value(terminal_state)
            g += (self.gamma ** n_steps) * rho_prod * terminal_value
        
        return g, rho_prod
    
    def add_experience(self, state: Any, reward: float, importance_ratio: float):
        """
        添加经验到缓冲
        Add experience to buffer
        
        Args:
            state: 状态
                  State
            reward: 奖励
                   Reward
            importance_ratio: 重要性比率
                            Importance ratio
        """
        self.state_buffer.append(state)
        if len(self.state_buffer) > 1:  # 跳过第一个状态
            self.reward_buffer.append(reward)
            self.rho_buffer.append(importance_ratio)
        
        # 如果缓冲满了，执行更新
        # If buffer is full, perform update
        if len(self.reward_buffer) == self.n:
            self.update()
    
    def update(self):
        """
        执行n-step更新
        Perform n-step update
        """
        if len(self.state_buffer) == 0:
            return
        
        # 获取要更新的状态
        # Get state to update
        state = self.state_buffer[0]
        features = self.feature_extractor(state)
        
        # 计算n-step回报
        # Compute n-step return
        g, rho_prod = self.compute_n_step_return()
        
        # 当前价值
        # Current value
        current_value = self.get_value(state)
        
        # TD误差
        # TD error
        td_error = g - current_value
        
        # 更新权重 (不再乘以重要性比率，因为已经在回报中考虑)
        # Update weights (don't multiply by importance ratio as it's in the return)
        self.weights += self.alpha * td_error * features
        
        self.update_count += 1


# ================================================================================
# 主函数：演示重要性采样
# Main Function: Demonstrate Importance Sampling
# ================================================================================

def demonstrate_importance_sampling():
    """
    演示重要性采样方法
    Demonstrate importance sampling methods
    """
    print("\n" + "="*80)
    print("第11.1-11.2节：重要性采样与函数近似")
    print("Section 11.1-11.2: Importance Sampling with Function Approximation")
    print("="*80)
    
    # 1. 测试轨迹和重要性比率
    # 1. Test trajectory and importance ratio
    print("\n" + "="*60)
    print("1. 轨迹和重要性比率")
    print("1. Trajectory and Importance Ratio")
    print("="*60)
    
    # 创建示例轨迹
    # Create example trajectory
    trajectory = Trajectory(
        states=[0, 1, 2, 3, 4],
        actions=[0, 1, 0, 1, 0],
        rewards=[1.0, -1.0, 2.0, 0.0, 1.0],
        probs_b=[0.5, 0.5, 0.6, 0.4, 0.5],
        probs_pi=[0.8, 0.2, 0.7, 0.3, 0.9]
    )
    
    print(f"\n轨迹长度: {trajectory.length}")
    
    # 计算不同片段的重要性比率
    # Compute importance ratios for different segments
    print("\n重要性比率:")
    for t in range(3):
        rho = trajectory.compute_importance_ratio(t, t+3)
        print(f"  ρ_{{{t}:{t+2}}} = {rho:.3f}")
    
    # 计算回报
    # Compute returns
    gamma = 0.9
    print(f"\n回报 (γ={gamma}):")
    for t in range(3):
        g = trajectory.compute_return(gamma, t)
        print(f"  G_{t} = {g:.3f}")
    
    # 2. 测试重要性采样评估
    # 2. Test importance sampling evaluation
    print("\n" + "="*60)
    print("2. 重要性采样评估")
    print("2. Importance Sampling Evaluation")
    print("="*60)
    
    n_states = 5
    
    # 普通IS
    # Ordinary IS
    ordinary_is = ImportanceSampling(n_states, gamma=0.9, weighted=False)
    ordinary_is.update_from_trajectory(trajectory)
    
    print("\n普通重要性采样:")
    for s in range(n_states):
        if ordinary_is.visit_counts[s] > 0:
            print(f"  V({s}) = {ordinary_is.get_value(s):.3f}")
    
    # 加权IS
    # Weighted IS
    weighted_is = ImportanceSampling(n_states, gamma=0.9, weighted=True)
    weighted_is.update_from_trajectory(trajectory)
    
    print("\n加权重要性采样:")
    for s in range(n_states):
        if weighted_is.c[s] > 0:
            print(f"  V({s}) = {weighted_is.get_value(s):.3f}, C({s}) = {weighted_is.c[s]:.3f}")
    
    # 3. 测试半梯度离策略TD
    # 3. Test semi-gradient off-policy TD
    print("\n" + "="*60)
    print("3. 半梯度离策略TD")
    print("3. Semi-gradient Off-policy TD")
    print("="*60)
    
    n_features = 8
    
    # 简单特征提取器
    # Simple feature extractor
    def simple_features(state):
        features = np.zeros(n_features)
        if isinstance(state, int):
            features[state % n_features] = 1.0
        return features
    
    off_td = SemiGradientOffPolicyTD(
        feature_extractor=simple_features,
        n_features=n_features,
        alpha=0.1,
        gamma=0.9
    )
    
    print("\n模拟离策略学习...")
    for step in range(10):
        state = step % n_states
        next_state = (step + 1) % n_states
        reward = np.random.randn()
        
        # 模拟重要性比率
        # Simulate importance ratio
        rho = np.random.uniform(0.5, 2.0)
        
        td_error = off_td.update(state, reward, next_state, False, rho)
        
        if step % 3 == 0:
            value = off_td.get_value(state)
            print(f"  步{step+1}: V({state})={value:.3f}, δ={td_error:.3f}, ρ={rho:.2f}")
    
    # 4. 测试Per-decision IS
    # 4. Test per-decision IS
    print("\n" + "="*60)
    print("4. Per-decision重要性采样")
    print("4. Per-decision Importance Sampling")
    print("="*60)
    
    pd_is = PerDecisionImportanceSampling(
        n_features=n_features,
        feature_extractor=simple_features,
        alpha=0.1,
        gamma=0.9,
        lambda_=0.5
    )
    
    print("\n模拟Per-decision更新...")
    for step in range(10):
        state = step % n_states
        action = step % 2
        reward = -1.0 + 0.5 * np.random.randn()
        next_state = (step + 1) % n_states
        
        # 模拟策略概率
        # Simulate policy probabilities
        prob_b = np.random.uniform(0.3, 0.7)
        prob_pi = np.random.uniform(0.2, 0.8)
        
        td_error = pd_is.update(
            state, action, reward, next_state, 
            done=(step == 9),
            prob_b=prob_b, prob_pi=prob_pi
        )
        
        if step % 3 == 0:
            value = pd_is.get_value(state)
            rho = prob_pi / prob_b
            print(f"  步{step+1}: V({state})={value:.3f}, ρ={rho:.2f}")
    
    # 5. 测试n-step离策略TD
    # 5. Test n-step off-policy TD
    print("\n" + "="*60)
    print("5. n-step离策略TD")
    print("5. n-step Off-policy TD")
    print("="*60)
    
    n_step_td = NStepOffPolicyTD(
        n_features=n_features,
        feature_extractor=simple_features,
        n=4,
        alpha=0.1,
        gamma=0.9
    )
    
    print(f"\n使用{n_step_td.n}步更新")
    
    # 模拟经验流
    # Simulate experience stream
    for step in range(15):
        state = step % n_states
        reward = -1.0 + np.random.randn() * 0.5
        rho = np.random.uniform(0.5, 1.5)
        
        n_step_td.add_experience(state, reward, rho)
        
        if n_step_td.update_count > 0 and step % 5 == 0:
            value = n_step_td.get_value(0)
            print(f"  步{step+1}: 更新数={n_step_td.update_count}, V(0)={value:.3f}")
    
    # 总结
    # Summary
    print("\n" + "="*80)
    print("重要性采样总结")
    print("Importance Sampling Summary")
    print("="*80)
    
    print("""
    关键要点 Key Takeaways:
    =======================
    
    1. 重要性采样纠正策略差异
       IS corrects for policy mismatch
       
    2. 加权IS降低方差
       Weighted IS reduces variance
       
    3. Per-decision IS更高效
       Per-decision IS more efficient
       
    4. n-step方法加速学习
       n-step methods speed up learning
       
    5. 离策略学习更灵活但更困难
       Off-policy more flexible but harder
    
    实践建议 Practical Advice:
    - 优先使用加权IS
      Prefer weighted IS
    - 注意重要性比率的稳定性
      Watch importance ratio stability
    - 考虑截断重要性比率
      Consider truncating importance ratios
    - 结合经验回放
      Combine with experience replay
    """)


if __name__ == "__main__":
    demonstrate_importance_sampling()