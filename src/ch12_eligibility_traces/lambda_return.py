"""
================================================================================
第12.1-12.2节：λ-return
Section 12.1-12.2: The λ-return
================================================================================

统一TD和MC的优雅方式！
Elegant unification of TD and MC!

λ-return的核心思想 Core Idea of λ-return:
结合所有n-step returns的加权平均
Weighted average of all n-step returns

G_t^λ = (1-λ) Σ_{n=1}^∞ λ^{n-1} G_t:t+n

特殊情况 Special Cases:
- λ=0: TD(0)
- λ=1: Monte Carlo

前向视角 Forward View:
- 需要未来数据
  Requires future data
- 离线更新
  Offline updates

后向视角 Backward View:
- 使用资格迹
  Uses eligibility traces
- 在线更新
  Online updates
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass
import logging

# 设置日志
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第12.1节：λ-return定义
# Section 12.1: λ-return Definition
# ================================================================================

@dataclass
class Episode:
    """
    回合数据
    Episode data
    
    存储完整回合的所有信息
    Stores all information for a complete episode
    """
    states: List[Any]      # 状态序列
    actions: List[int]     # 动作序列
    rewards: List[float]   # 奖励序列
    
    @property
    def length(self) -> int:
        """回合长度"""
        return len(self.rewards)
    
    def compute_return(self, t: int, gamma: float) -> float:
        """
        计算从时间t开始的回报
        Compute return starting from time t
        
        G_t = Σ_{k=0}^{T-t-1} γ^k R_{t+k+1}
        """
        G = 0.0
        for k in range(t, self.length):
            G += (gamma ** (k - t)) * self.rewards[k]
        return G
    
    def compute_n_step_return(self, t: int, n: int, gamma: float, 
                             V: Optional[Callable] = None) -> float:
        """
        计算n步回报
        Compute n-step return
        
        G_{t:t+n} = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γ^n V(S_{t+n})
        """
        G = 0.0
        
        # 累积n步奖励
        # Accumulate n-step rewards
        for k in range(min(n, self.length - t)):
            G += (gamma ** k) * self.rewards[t + k]
        
        # 添加bootstrap值
        # Add bootstrap value
        if V is not None and t + n < len(self.states):
            terminal_state = self.states[t + n]
            G += (gamma ** n) * V(terminal_state)
        
        return G


class LambdaReturn:
    """
    λ-return计算
    λ-return Computation
    
    复合回报的加权平均
    Weighted average of compound returns
    
    关键公式 Key Formula:
    G_t^λ = (1-λ) Σ_{n=1}^{T-t-1} λ^{n-1} G_{t:t+n} + λ^{T-t-1} G_t
    
    性质 Properties:
    - λ=0时退化为1步TD
      Degenerates to 1-step TD when λ=0
    - λ=1时退化为MC
      Degenerates to MC when λ=1
    - 0<λ<1时在两者之间权衡
      Trades off between them when 0<λ<1
    """
    
    def __init__(self, lambda_: float = 0.9, gamma: float = 0.99):
        """
        初始化λ-return计算器
        Initialize λ-return calculator
        
        Args:
            lambda_: λ参数，控制衰减率
                    λ parameter, controls decay rate
            gamma: 折扣因子
                  Discount factor
        """
        self.lambda_ = lambda_
        self.gamma = gamma
        
        # 缓存计算结果
        # Cache computation results
        self.return_cache = {}
        
        logger.info(f"初始化λ-return: λ={lambda_}, γ={gamma}")
    
    def compute_lambda_return(self, episode: Episode, t: int, 
                             V: Optional[Callable] = None) -> float:
        """
        计算λ-return
        Compute λ-return
        
        Args:
            episode: 回合数据
                    Episode data
            t: 时间步
              Time step
            V: 价值函数(用于bootstrap)
              Value function (for bootstrap)
        
        Returns:
            G_t^λ
        """
        T = episode.length
        
        # 如果已经计算过，直接返回
        # Return cached result if already computed
        cache_key = (id(episode), t)
        if cache_key in self.return_cache:
            return self.return_cache[cache_key]
        
        G_lambda = 0.0
        
        # 累积所有n-step returns
        # Accumulate all n-step returns
        for n in range(1, T - t):
            # n步回报
            # n-step return
            G_n = episode.compute_n_step_return(t, n, self.gamma, V)
            
            # 加权系数
            # Weight coefficient
            weight = (1 - self.lambda_) * (self.lambda_ ** (n - 1))
            
            G_lambda += weight * G_n
        
        # 添加完整回报（最后一项）
        # Add complete return (last term)
        if T > t:
            G_T = episode.compute_return(t, self.gamma)
            weight_T = self.lambda_ ** (T - t - 1)
            G_lambda += weight_T * G_T
        
        # 缓存结果
        # Cache result
        self.return_cache[cache_key] = G_lambda
        
        return G_lambda
    
    def compute_truncated_lambda_return(self, rewards: List[float], 
                                       values: List[float], 
                                       horizon: int) -> float:
        """
        计算截断的λ-return（用于在线算法）
        Compute truncated λ-return (for online algorithms)
        
        Args:
            rewards: 奖励序列
                    Reward sequence
            values: 价值估计序列
                   Value estimate sequence
            horizon: 截断长度
                    Truncation horizon
        
        Returns:
            截断的λ-return
            Truncated λ-return
        """
        G_lambda = 0.0
        
        for n in range(1, min(horizon, len(rewards)) + 1):
            # n步回报
            # n-step return
            G_n = 0.0
            for k in range(n):
                if k < len(rewards):
                    G_n += (self.gamma ** k) * rewards[k]
            
            # 添加bootstrap值
            # Add bootstrap value
            if n < len(values):
                G_n += (self.gamma ** n) * values[n]
            
            # 权重
            # Weight
            if n < horizon:
                weight = (1 - self.lambda_) * (self.lambda_ ** (n - 1))
            else:
                weight = self.lambda_ ** (n - 1)
            
            G_lambda += weight * G_n
        
        return G_lambda


# ================================================================================
# 第12.2节：离线λ-return算法
# Section 12.2: Offline λ-return Algorithm
# ================================================================================

class OfflineLambdaReturn:
    """
    离线λ-return算法
    Offline λ-return Algorithm
    
    前向视角的实现
    Forward view implementation
    
    特点 Characteristics:
    - 需要完整回合
      Requires complete episode
    - 精确计算λ-return
      Exact λ-return computation
    - 批量更新
      Batch updates
    """
    
    def __init__(self,
                 n_features: int,
                 feature_extractor: Callable,
                 lambda_: float = 0.9,
                 alpha: float = 0.01,
                 gamma: float = 0.99):
        """
        初始化离线λ-return
        Initialize offline λ-return
        
        Args:
            n_features: 特征维度
                       Feature dimension
            feature_extractor: 特征提取器
                             Feature extractor
            lambda_: λ参数
                    λ parameter
            alpha: 学习率
                  Learning rate
            gamma: 折扣因子
                  Discount factor
        """
        self.n_features = n_features
        self.feature_extractor = feature_extractor
        self.lambda_ = lambda_
        self.alpha = alpha
        self.gamma = gamma
        
        # 权重向量
        # Weight vector
        self.weights = np.zeros(n_features)
        
        # λ-return计算器
        # λ-return calculator
        self.lambda_return_calc = LambdaReturn(lambda_, gamma)
        
        # 统计
        # Statistics
        self.episode_count = 0
        self.total_updates = 0
        
        logger.info(f"初始化离线λ-return: λ={lambda_}, α={alpha}")
    
    def get_value(self, state: Any) -> float:
        """
        获取状态价值
        Get state value
        """
        features = self.feature_extractor(state)
        return np.dot(self.weights, features)
    
    def learn_episode(self, episode: Episode):
        """
        从完整回合学习
        Learn from complete episode
        
        Args:
            episode: 回合数据
                    Episode data
        """
        # 计算每个时间步的λ-return
        # Compute λ-return for each time step
        lambda_returns = []
        for t in range(episode.length):
            G_lambda = self.lambda_return_calc.compute_lambda_return(
                episode, t, self.get_value
            )
            lambda_returns.append(G_lambda)
        
        # 更新权重
        # Update weights
        for t in range(episode.length):
            state = episode.states[t]
            features = self.feature_extractor(state)
            
            # 预测值
            # Prediction
            prediction = np.dot(self.weights, features)
            
            # TD误差
            # TD error
            td_error = lambda_returns[t] - prediction
            
            # 梯度更新
            # Gradient update
            self.weights += self.alpha * td_error * features
            
            self.total_updates += 1
        
        self.episode_count += 1
    
    def batch_learn(self, episodes: List[Episode]):
        """
        批量学习多个回合
        Batch learning from multiple episodes
        
        Args:
            episodes: 回合列表
                     List of episodes
        """
        for episode in episodes:
            self.learn_episode(episode)
        
        logger.info(f"批量学习完成: {len(episodes)}个回合, "
                   f"总更新{self.total_updates}次")


# ================================================================================
# 第12.2.1节：半梯度λ-return
# Section 12.2.1: Semi-gradient λ-return
# ================================================================================

class SemiGradientLambdaReturn:
    """
    半梯度λ-return
    Semi-gradient λ-return
    
    使用函数近似的λ-return算法
    λ-return algorithm with function approximation
    
    更新规则 Update Rule:
    w ← w + α[G_t^λ - v̂(S_t,w)]∇v̂(S_t,w)
    """
    
    def __init__(self,
                 n_features: int,
                 feature_extractor: Callable,
                 lambda_: float = 0.9,
                 alpha: float = 0.01,
                 gamma: float = 0.99):
        """
        初始化半梯度λ-return
        Initialize semi-gradient λ-return
        """
        self.n_features = n_features
        self.feature_extractor = feature_extractor
        self.lambda_ = lambda_
        self.alpha = alpha
        self.gamma = gamma
        
        # 权重
        # Weights
        self.weights = np.zeros(n_features)
        
        # 回合缓冲（用于计算λ-return）
        # Episode buffer (for computing λ-return)
        self.state_buffer = []
        self.reward_buffer = []
        
        # 统计
        # Statistics
        self.update_count = 0
        self.episode_returns = []
        
        logger.info(f"初始化半梯度λ-return: λ={lambda_}")
    
    def get_value(self, state: Any) -> float:
        """获取状态价值"""
        features = self.feature_extractor(state)
        return np.dot(self.weights, features)
    
    def start_episode(self):
        """
        开始新回合
        Start new episode
        """
        self.state_buffer = []
        self.reward_buffer = []
    
    def step(self, state: Any, reward: float, next_state: Any, done: bool):
        """
        执行一步（在线收集数据）
        Execute one step (online data collection)
        
        Args:
            state: 当前状态
                  Current state
            reward: 奖励
                   Reward
            next_state: 下一状态
                       Next state
            done: 是否终止
                 Whether done
        """
        # 收集数据
        # Collect data
        if len(self.state_buffer) == 0:
            self.state_buffer.append(state)
        
        self.state_buffer.append(next_state)
        self.reward_buffer.append(reward)
        
        # 如果回合结束，执行更新
        # If episode ends, perform updates
        if done:
            self.update_episode()
            self.episode_returns.append(sum(self.reward_buffer))
    
    def update_episode(self):
        """
        回合结束后更新
        Update after episode ends
        """
        T = len(self.reward_buffer)
        
        # 计算每个时间步的λ-return
        # Compute λ-return for each time step
        for t in range(T):
            # 计算G_t^λ
            G_lambda = 0.0
            
            # 累积n-step returns
            for n in range(1, T - t + 1):
                # n步回报
                # n-step return
                G_n = 0.0
                for k in range(min(n, T - t)):
                    G_n += (self.gamma ** k) * self.reward_buffer[t + k]
                
                # 添加bootstrap值（如果不是终止）
                # Add bootstrap value (if not terminal)
                if t + n < T:
                    terminal_state = self.state_buffer[t + n]
                    G_n += (self.gamma ** n) * self.get_value(terminal_state)
                
                # 权重
                # Weight
                if n < T - t:
                    weight = (1 - self.lambda_) * (self.lambda_ ** (n - 1))
                else:
                    weight = self.lambda_ ** (n - 1)
                
                G_lambda += weight * G_n
            
            # 更新权重
            # Update weights
            state = self.state_buffer[t]
            features = self.feature_extractor(state)
            prediction = self.get_value(state)
            
            td_error = G_lambda - prediction
            self.weights += self.alpha * td_error * features
            
            self.update_count += 1


# ================================================================================
# 第12.2.2节：在线λ-return（TTD算法）
# Section 12.2.2: Online λ-return (TTD Algorithm)
# ================================================================================

class TTD:
    """
    TTD (Truncated TD)
    截断时序差分
    
    在线计算λ-return的近似方法
    Online approximation of λ-return
    
    关键思想 Key Idea:
    使用固定长度的截断窗口
    Use fixed-length truncation window
    
    优势 Advantages:
    - 不需要等待回合结束
      No need to wait for episode end
    - 内存有限
      Bounded memory
    - 持续学习
      Continual learning
    """
    
    def __init__(self,
                 n_features: int,
                 feature_extractor: Callable,
                 lambda_: float = 0.9,
                 alpha: float = 0.01,
                 gamma: float = 0.99,
                 horizon: int = 10):
        """
        初始化TTD
        Initialize TTD
        
        Args:
            n_features: 特征维度
                       Feature dimension
            feature_extractor: 特征提取器
                             Feature extractor
            lambda_: λ参数
                    λ parameter
            alpha: 学习率
                  Learning rate
            gamma: 折扣因子
                  Discount factor
            horizon: 截断长度
                    Truncation horizon
        """
        self.n_features = n_features
        self.feature_extractor = feature_extractor
        self.lambda_ = lambda_
        self.alpha = alpha
        self.gamma = gamma
        self.horizon = horizon
        
        # 权重
        # Weights
        self.weights = np.zeros(n_features)
        
        # 滑动窗口缓冲
        # Sliding window buffer
        self.state_buffer = []
        self.reward_buffer = []
        
        # λ-return计算器
        # λ-return calculator
        self.lambda_calc = LambdaReturn(lambda_, gamma)
        
        # 统计
        # Statistics
        self.update_count = 0
        self.step_count = 0
        
        logger.info(f"初始化TTD: λ={lambda_}, horizon={horizon}")
    
    def get_value(self, state: Any) -> float:
        """获取状态价值"""
        features = self.feature_extractor(state)
        return np.dot(self.weights, features)
    
    def step(self, state: Any, reward: float, next_state: Any, done: bool):
        """
        执行一步
        Execute one step
        
        在线更新，使用截断的λ-return
        Online update with truncated λ-return
        """
        # 添加到缓冲
        # Add to buffer
        if len(self.state_buffer) == 0:
            self.state_buffer.append(state)
        
        self.state_buffer.append(next_state)
        self.reward_buffer.append(reward)
        
        # 保持窗口大小
        # Maintain window size
        if len(self.state_buffer) > self.horizon + 1:
            self.state_buffer.pop(0)
        if len(self.reward_buffer) > self.horizon:
            self.reward_buffer.pop(0)
        
        # 如果有足够数据或回合结束，执行更新
        # If enough data or episode ends, perform update
        if len(self.reward_buffer) >= self.horizon or done:
            self.update()
        
        # 如果回合结束，清空缓冲
        # If episode ends, clear buffer
        if done:
            self.state_buffer = []
            self.reward_buffer = []
        
        self.step_count += 1
    
    def update(self):
        """
        执行TTD更新
        Perform TTD update
        """
        if len(self.state_buffer) == 0:
            return
        
        # 获取最早的状态
        # Get earliest state
        state = self.state_buffer[0]
        features = self.feature_extractor(state)
        
        # 计算截断的λ-return
        # Compute truncated λ-return
        values = [self.get_value(s) for s in self.state_buffer]
        G_lambda = self.lambda_calc.compute_truncated_lambda_return(
            self.reward_buffer, values, self.horizon
        )
        
        # 当前预测
        # Current prediction
        prediction = self.get_value(state)
        
        # TD误差
        # TD error
        td_error = G_lambda - prediction
        
        # 更新权重
        # Update weights
        self.weights += self.alpha * td_error * features
        
        self.update_count += 1


# ================================================================================
# 主函数：演示λ-return
# Main Function: Demonstrate λ-return
# ================================================================================

def demonstrate_lambda_return():
    """
    演示λ-return方法
    Demonstrate λ-return methods
    """
    print("\n" + "="*80)
    print("第12.1-12.2节：λ-return")
    print("Section 12.1-12.2: The λ-return")
    print("="*80)
    
    # 设置
    # Setup
    n_features = 8
    n_states = 5
    
    # 简单特征提取器
    # Simple feature extractor
    def simple_features(state):
        if isinstance(state, int):
            features = np.zeros(n_features)
            features[state % n_features] = 1.0
            features[(state + 1) % n_features] = 0.3
            return features / np.linalg.norm(features)
        return np.random.randn(n_features) * 0.1
    
    # 1. 测试λ-return计算
    # 1. Test λ-return computation
    print("\n" + "="*60)
    print("1. λ-return计算")
    print("1. λ-return Computation")
    print("="*60)
    
    # 创建示例回合
    # Create example episode
    episode = Episode(
        states=[0, 1, 2, 3, 4],
        actions=[0, 1, 0, 1, 0],
        rewards=[1.0, -2.0, 3.0, -1.0, 5.0]
    )
    
    print(f"\n回合长度: {episode.length}")
    print(f"奖励序列: {episode.rewards}")
    
    # 测试不同λ值
    # Test different λ values
    lambda_calc = LambdaReturn(lambda_=0.5, gamma=0.9)
    
    print("\n不同时间步的λ-return (λ=0.5):")
    for t in range(3):
        G_lambda = lambda_calc.compute_lambda_return(episode, t)
        G_mc = episode.compute_return(t, 0.9)
        print(f"  t={t}: G^λ={G_lambda:.3f}, G^MC={G_mc:.3f}")
    
    # 测试不同λ值的效果
    # Test effect of different λ values
    print("\n不同λ值的比较 (t=0):")
    for lambda_val in [0.0, 0.5, 0.9, 1.0]:
        calc = LambdaReturn(lambda_=lambda_val, gamma=0.9)
        G = calc.compute_lambda_return(episode, 0)
        print(f"  λ={lambda_val}: G^λ={G:.3f}")
    
    # 2. 测试离线λ-return算法
    # 2. Test offline λ-return algorithm
    print("\n" + "="*60)
    print("2. 离线λ-return算法")
    print("2. Offline λ-return Algorithm")
    print("="*60)
    
    offline_lambda = OfflineLambdaReturn(
        n_features=n_features,
        feature_extractor=simple_features,
        lambda_=0.8,
        alpha=0.1,
        gamma=0.9
    )
    
    # 创建多个回合
    # Create multiple episodes
    episodes = []
    for i in range(5):
        states = [(i + j) % n_states for j in range(5)]
        actions = [j % 2 for j in range(5)]
        rewards = [-1.0 if s != 2 else 3.0 for s in states]
        episodes.append(Episode(states, actions, rewards))
    
    print(f"\n训练{len(episodes)}个回合...")
    offline_lambda.batch_learn(episodes)
    
    print("\n学习后的价值估计:")
    for state in range(n_states):
        value = offline_lambda.get_value(state)
        print(f"  V({state}) = {value:.3f}")
    
    print(f"\n总更新次数: {offline_lambda.total_updates}")
    
    # 3. 测试半梯度λ-return
    # 3. Test semi-gradient λ-return
    print("\n" + "="*60)
    print("3. 半梯度λ-return")
    print("3. Semi-gradient λ-return")
    print("="*60)
    
    sg_lambda = SemiGradientLambdaReturn(
        n_features=n_features,
        feature_extractor=simple_features,
        lambda_=0.7,
        alpha=0.05,
        gamma=0.9
    )
    
    print("\n模拟在线回合...")
    
    # 模拟多个回合
    # Simulate multiple episodes
    for ep in range(3):
        print(f"\n回合 {ep + 1}:")
        sg_lambda.start_episode()
        
        episode_reward = 0.0
        for step in range(5):
            state = step % n_states
            reward = -1.0 if state != 2 else 5.0
            next_state = (step + 1) % n_states
            done = step == 4
            
            sg_lambda.step(state, reward, next_state, done)
            episode_reward += reward
        
        print(f"  回合总奖励: {episode_reward:.1f}")
        print(f"  更新次数: {sg_lambda.update_count}")
    
    print("\n最终价值估计:")
    for state in range(n_states):
        value = sg_lambda.get_value(state)
        print(f"  V({state}) = {value:.3f}")
    
    # 4. 测试TTD算法
    # 4. Test TTD algorithm
    print("\n" + "="*60)
    print("4. TTD (截断TD)")
    print("4. TTD (Truncated TD)")
    print("="*60)
    
    ttd = TTD(
        n_features=n_features,
        feature_extractor=simple_features,
        lambda_=0.9,
        alpha=0.05,
        gamma=0.9,
        horizon=3
    )
    
    print(f"\n使用截断窗口: {ttd.horizon}")
    print("模拟连续学习...")
    
    # 模拟连续步骤
    # Simulate continuous steps
    for step in range(20):
        state = step % n_states
        reward = -1.0 if state != 2 else 3.0
        next_state = (step + 1) % n_states
        done = (step + 1) % 7 == 0  # 每7步结束
        
        ttd.step(state, reward, next_state, done)
        
        if (step + 1) % 5 == 0:
            print(f"  步{step + 1}: 更新{ttd.update_count}次")
    
    print("\nTTD价值估计:")
    for state in range(n_states):
        value = ttd.get_value(state)
        print(f"  V({state}) = {value:.3f}")
    
    # 5. 比较不同λ值
    # 5. Compare different λ values
    print("\n" + "="*60)
    print("5. λ值比较")
    print("5. λ Value Comparison")
    print("="*60)
    
    lambda_values = [0.0, 0.5, 0.9, 1.0]
    results = {}
    
    for lambda_val in lambda_values:
        learner = OfflineLambdaReturn(
            n_features=n_features,
            feature_extractor=simple_features,
            lambda_=lambda_val,
            alpha=0.1,
            gamma=0.9
        )
        
        # 训练
        # Train
        learner.batch_learn(episodes)
        
        # 记录结果
        # Record results
        values = [learner.get_value(s) for s in range(n_states)]
        results[lambda_val] = values
    
    print("\n不同λ值的价值估计:")
    print("状态   λ=0.0   λ=0.5   λ=0.9   λ=1.0")
    print("-" * 40)
    for state in range(n_states):
        print(f"{state:3d}", end="")
        for lambda_val in lambda_values:
            print(f"  {results[lambda_val][state]:7.3f}", end="")
        print()
    
    # 总结
    # Summary
    print("\n" + "="*80)
    print("λ-return总结")
    print("λ-return Summary")
    print("="*80)
    
    print("""
    关键要点 Key Takeaways:
    =======================
    
    1. λ-return统一TD和MC
       λ-return unifies TD and MC
       
    2. λ控制偏差-方差权衡
       λ controls bias-variance tradeoff
       
    3. 前向视角需要未来数据
       Forward view needs future data
       
    4. TTD实现在线近似
       TTD enables online approximation
       
    5. λ=0最快但偏差大，λ=1无偏但方差大
       λ=0 fastest but biased, λ=1 unbiased but high variance
    
    实践建议 Practical Advice:
    - 典型λ值: 0.8-0.95
      Typical λ values: 0.8-0.95
    - 问题相关的调优
      Problem-specific tuning
    - 考虑计算-性能权衡
      Consider computation-performance tradeoff
    """)


if __name__ == "__main__":
    demonstrate_lambda_return()