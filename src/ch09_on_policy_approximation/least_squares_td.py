"""
================================================================================
第9.7-9.8节：最小二乘TD - 数据效率的极致
Section 9.7-9.8: Least-Squares TD - Ultimate Data Efficiency  
================================================================================

一步到位的解决方案！
One-shot solution!

LSTD核心思想 LSTD Core Idea:
不是增量更新，而是直接求解线性系统
Instead of incremental updates, directly solve linear system

Aw = b

其中 where:
A = Σₜ xₜ(xₜ - γxₜ₊₁)ᵀ
b = Σₜ xₜrₜ₊₁

解 Solution:
w = A⁻¹b

优势 Advantages:
- 数据效率最高
  Maximum data efficiency
- 无需调参(无学习率)
  No hyperparameter tuning
- 一次求解
  One-shot solution

劣势 Disadvantages:
- O(d³)计算复杂度
  O(d³) computational complexity
- 需要存储矩阵
  Need to store matrices
- 仅适用于线性函数
  Only for linear functions
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging

# 导入基础组件
# Import base components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.ch03_finite_mdp.mdp_framework import State, Action, MDPEnvironment
from src.ch03_finite_mdp.policies_and_values import Policy

from .linear_approximation import LinearFeatures

# 设置日志
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第9.7节：最小二乘TD(0)
# Section 9.7: Least-Squares TD(0)
# ================================================================================

class LeastSquaresTD:
    """
    最小二乘TD(0) - LSTD(0)
    Least-Squares TD(0)
    
    批量求解TD固定点
    Batch solution to TD fixed point
    
    TD固定点 TD Fixed Point:
    w* = argmin E[(δₜ)²]
    
    其中TD误差 where TD error:
    δₜ = Rₜ₊₁ + γv̂(Sₜ₊₁,w) - v̂(Sₜ,w)
    
    对线性函数 For linear function:
    直接求解 Aw = b
    Direct solution
    
    关键创新 Key Innovation:
    使用Sherman-Morrison公式增量更新A⁻¹
    Use Sherman-Morrison formula for incremental A⁻¹ update
    """
    
    def __init__(self,
                feature_extractor: LinearFeatures,
                gamma: float = 0.99,
                epsilon: float = 0.01):
        """
        初始化LSTD
        Initialize LSTD
        
        Args:
            feature_extractor: 特征提取器
                             Feature extractor
            gamma: 折扣因子
                  Discount factor
            epsilon: 正则化参数
                    Regularization parameter
        """
        self.feature_extractor = feature_extractor
        self.n_features = feature_extractor.n_features
        self.gamma = gamma
        self.epsilon = epsilon
        
        # 初始化矩阵
        # Initialize matrices
        self.reset()
        
        logger.info(f"初始化LSTD: d={self.n_features}, γ={gamma}, ε={epsilon}")
    
    def reset(self):
        """
        重置算法状态
        Reset algorithm state
        """
        # A矩阵和b向量
        # A matrix and b vector
        self.A = self.epsilon * np.eye(self.n_features)
        self.b = np.zeros(self.n_features)
        
        # 权重
        # Weights
        self.weights = np.zeros(self.n_features)
        
        # 统计
        # Statistics
        self.n_samples = 0
        self.td_errors = []
    
    def add_sample(self, state: State, reward: float, next_state: State):
        """
        添加一个样本
        Add one sample
        
        更新A和b矩阵
        Update A and b matrices
        
        Args:
            state: 当前状态
                  Current state
            reward: 奖励
                   Reward
            next_state: 下一状态
                       Next state
        """
        # 提取特征
        # Extract features
        phi = self.feature_extractor.extract(state)
        
        if next_state.is_terminal:
            phi_next = np.zeros(self.n_features)
        else:
            phi_next = self.feature_extractor.extract(next_state)
        
        # 更新A矩阵: A += φ(φ - γφ')ᵀ
        # Update A matrix
        self.A += np.outer(phi, phi - self.gamma * phi_next)
        
        # 更新b向量: b += φr
        # Update b vector
        self.b += phi * reward
        
        self.n_samples += 1
    
    def solve(self) -> np.ndarray:
        """
        求解权重
        Solve for weights
        
        w = A⁻¹b
        
        Returns:
            权重向量
            Weight vector
        """
        try:
            # 直接求解
            # Direct solve
            self.weights = np.linalg.solve(self.A, self.b)
        except np.linalg.LinAlgError:
            # 矩阵奇异，使用伪逆
            # Matrix singular, use pseudoinverse
            logger.warning("A矩阵奇异，使用伪逆")
            self.weights = np.linalg.pinv(self.A) @ self.b
        
        return self.weights
    
    def predict(self, state: State) -> float:
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
    
    def compute_td_error(self, state: State, reward: float, next_state: State) -> float:
        """
        计算TD误差
        Compute TD error
        
        Args:
            state: 当前状态
                  Current state
            reward: 奖励
                   Reward
            next_state: 下一状态
                       Next state
        
        Returns:
            TD误差
            TD error
        """
        value = self.predict(state)
        
        if next_state.is_terminal:
            next_value = 0.0
        else:
            next_value = self.predict(next_state)
        
        td_error = reward + self.gamma * next_value - value
        self.td_errors.append(td_error)
        
        return td_error


# ================================================================================
# 第9.8节：最小二乘TD(λ)
# Section 9.8: Least-Squares TD(λ)
# ================================================================================

class LeastSquaresTDLambda:
    """
    最小二乘TD(λ) - LSTD(λ)
    Least-Squares TD(λ)
    
    带资格迹的LSTD
    LSTD with eligibility traces
    
    结合前向视角和数据效率！
    Combining forward view with data efficiency!
    
    更新规则 Update Rule:
    A = Σₜ zₜ(φₜ - γφₜ₊₁)ᵀ
    b = Σₜ zₜrₜ₊₁
    
    其中资格迹 where eligibility trace:
    zₜ = γλzₜ₋₁ + φₜ
    """
    
    def __init__(self,
                feature_extractor: LinearFeatures,
                gamma: float = 0.99,
                lambda_: float = 0.9,
                epsilon: float = 0.01):
        """
        初始化LSTD(λ)
        Initialize LSTD(λ)
        
        Args:
            feature_extractor: 特征提取器
                             Feature extractor
            gamma: 折扣因子
                  Discount factor
            lambda_: 迹衰减参数
                    Trace decay parameter
            epsilon: 正则化参数
                    Regularization parameter
        """
        self.feature_extractor = feature_extractor
        self.n_features = feature_extractor.n_features
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon = epsilon
        
        # 初始化
        # Initialize
        self.reset()
        
        logger.info(f"初始化LSTD(λ): d={self.n_features}, γ={gamma}, λ={lambda_}")
    
    def reset(self):
        """
        重置算法状态
        Reset algorithm state
        """
        # 矩阵
        # Matrices
        self.A = self.epsilon * np.eye(self.n_features)
        self.b = np.zeros(self.n_features)
        
        # 资格迹
        # Eligibility trace
        self.z = np.zeros(self.n_features)
        
        # 权重
        # Weights
        self.weights = np.zeros(self.n_features)
        
        # 统计
        # Statistics
        self.n_samples = 0
        self.td_errors = []
    
    def add_sample(self, state: State, reward: float, next_state: State):
        """
        添加一个样本
        Add one sample
        
        Args:
            state: 当前状态
                  Current state
            reward: 奖励
                   Reward
            next_state: 下一状态
                       Next state
        """
        # 提取特征
        # Extract features
        phi = self.feature_extractor.extract(state)
        
        # 更新资格迹: z = γλz + φ
        # Update eligibility trace
        self.z = self.gamma * self.lambda_ * self.z + phi
        
        if next_state.is_terminal:
            phi_next = np.zeros(self.n_features)
            # 终止时重置迹
            # Reset trace at termination
            self.z = np.zeros(self.n_features)
        else:
            phi_next = self.feature_extractor.extract(next_state)
        
        # 更新A矩阵: A += z(φ - γφ')ᵀ
        # Update A matrix
        self.A += np.outer(self.z, phi - self.gamma * phi_next)
        
        # 更新b向量: b += zr
        # Update b vector
        self.b += self.z * reward
        
        self.n_samples += 1
    
    def solve(self) -> np.ndarray:
        """
        求解权重
        Solve for weights
        
        Returns:
            权重向量
            Weight vector
        """
        try:
            self.weights = np.linalg.solve(self.A, self.b)
        except np.linalg.LinAlgError:
            logger.warning("A矩阵奇异，使用伪逆")
            self.weights = np.linalg.pinv(self.A) @ self.b
        
        return self.weights
    
    def predict(self, state: State) -> float:
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


# ================================================================================
# 第9.8.1节：递归最小二乘TD
# Section 9.8.1: Recursive Least-Squares TD
# ================================================================================

class RecursiveLeastSquaresTD:
    """
    递归最小二乘TD - RLSTD
    Recursive Least-Squares TD
    
    增量更新A⁻¹避免矩阵求逆！
    Incrementally update A⁻¹ to avoid matrix inversion!
    
    Sherman-Morrison公式：
    (A + uvᵀ)⁻¹ = A⁻¹ - (A⁻¹uvᵀA⁻¹)/(1 + vᵀA⁻¹u)
    
    优势 Advantages:
    - O(d²)每步复杂度
      O(d²) per-step complexity
    - 无需存储A
      No need to store A
    - 增量更新
      Incremental update
    """
    
    def __init__(self,
                feature_extractor: LinearFeatures,
                gamma: float = 0.99,
                lambda_: float = 0.0,
                epsilon: float = 0.01):
        """
        初始化RLSTD
        Initialize RLSTD
        
        Args:
            feature_extractor: 特征提取器
                             Feature extractor
            gamma: 折扣因子
                  Discount factor
            lambda_: 迹衰减参数(0表示无迹)
                    Trace decay (0 for no trace)
            epsilon: 正则化参数
                    Regularization parameter
        """
        self.feature_extractor = feature_extractor
        self.n_features = feature_extractor.n_features
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon = epsilon
        
        # 初始化
        # Initialize
        self.reset()
        
        logger.info(f"初始化递归LSTD: d={self.n_features}, γ={gamma}")
    
    def reset(self):
        """
        重置算法状态
        Reset algorithm state
        """
        # P = A⁻¹
        self.P = (1.0 / self.epsilon) * np.eye(self.n_features)
        
        # 权重
        # Weights
        self.weights = np.zeros(self.n_features)
        
        # 资格迹（如果使用）
        # Eligibility trace (if used)
        self.z = np.zeros(self.n_features)
        
        # 统计
        # Statistics
        self.n_samples = 0
        self.td_errors = []
    
    def update(self, state: State, reward: float, next_state: State):
        """
        递归更新
        Recursive update
        
        使用Sherman-Morrison公式
        Using Sherman-Morrison formula
        
        Args:
            state: 当前状态
                  Current state
            reward: 奖励
                   Reward
            next_state: 下一状态
                       Next state
        """
        # 提取特征
        # Extract features
        phi = self.feature_extractor.extract(state)
        
        if self.lambda_ > 0:
            # 更新资格迹
            # Update eligibility trace
            self.z = self.gamma * self.lambda_ * self.z + phi
            features = self.z
        else:
            features = phi
        
        if next_state.is_terminal:
            phi_next = np.zeros(self.n_features)
            if self.lambda_ > 0:
                self.z = np.zeros(self.n_features)
        else:
            phi_next = self.feature_extractor.extract(next_state)
        
        # v = φ - γφ'
        v = phi - self.gamma * phi_next
        
        # k = Pv / (1 + vᵀPv)
        Pv = self.P @ v
        denominator = 1 + np.dot(v, Pv)
        
        # 避免除零
        # Avoid division by zero
        if abs(denominator) < 1e-10:
            return
        
        # 更新P (Sherman-Morrison)
        # Update P
        self.P = self.P - np.outer(Pv, Pv) / denominator
        
        # 计算TD误差
        # Compute TD error
        td_error = reward + self.gamma * np.dot(self.weights, phi_next) - np.dot(self.weights, phi)
        
        # 更新权重
        # Update weights
        self.weights = self.weights + td_error * self.P @ features
        
        # 记录统计
        # Record statistics
        self.td_errors.append(td_error)
        self.n_samples += 1
    
    def predict(self, state: State) -> float:
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


# ================================================================================
# 主函数：演示LSTD
# Main Function: Demonstrate LSTD
# ================================================================================

def demonstrate_lstd():
    """
    演示最小二乘TD算法
    Demonstrate Least-Squares TD algorithms
    """
    print("\n" + "="*80)
    print("第9.7-9.8节：最小二乘TD")
    print("Section 9.7-9.8: Least-Squares TD")
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
    n_features = 16
    feature_extractor = LinearFeatures(n_features)
    
    # 创建随机策略
    # Create random policy
    policy = UniformRandomPolicy(env.action_space)
    
    print(f"\n使用{n_features}维特征")
    print(f"策略: 均匀随机")
    
    # 收集数据
    # Collect data
    print("\n收集训练数据...")
    print("Collecting training data...")
    
    samples = []
    n_episodes = 50
    
    for episode in range(n_episodes):
        state = env.reset()
        
        while not state.is_terminal:
            action = policy.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            samples.append((state, reward, next_state))
            
            state = next_state
            if done:
                break
    
    print(f"  收集了{len(samples)}个样本")
    
    # 1. 测试LSTD(0)
    # 1. Test LSTD(0)
    print("\n" + "="*60)
    print("1. LSTD(0) - 批量求解")
    print("1. LSTD(0) - Batch Solution")
    print("="*60)
    
    lstd = LeastSquaresTD(feature_extractor, gamma=0.9, epsilon=0.01)
    
    # 添加所有样本
    # Add all samples
    for state, reward, next_state in samples:
        lstd.add_sample(state, reward, next_state)
    
    # 求解
    # Solve
    weights = lstd.solve()
    
    print(f"\n求解完成:")
    print(f"  样本数: {lstd.n_samples}")
    print(f"  权重范数: {np.linalg.norm(weights):.3f}")
    print(f"  A矩阵条件数: {np.linalg.cond(lstd.A):.2e}")
    
    # 显示一些状态的值
    # Show values for some states
    print("\n学习的状态价值 (前5个状态):")
    for i in range(min(5, len(env.state_space))):
        state = env.state_space[i]
        value = lstd.predict(state)
        print(f"  V({state.id}) = {value:.3f}")
    
    # 2. 测试LSTD(λ)
    # 2. Test LSTD(λ)
    print("\n" + "="*60)
    print("2. LSTD(λ) - 带资格迹")
    print("2. LSTD(λ) - With Eligibility Traces")
    print("="*60)
    
    lambda_values = [0.0, 0.5, 0.9]
    
    for lambda_ in lambda_values:
        print(f"\nλ = {lambda_}:")
        
        lstd_lambda = LeastSquaresTDLambda(
            feature_extractor, gamma=0.9, lambda_=lambda_, epsilon=0.01
        )
        
        # 添加样本（需要按顺序）
        # Add samples (need sequential order)
        episode_starts = []
        current_episode = -1
        
        for i, (state, reward, next_state) in enumerate(samples):
            # 检测新回合开始
            # Detect new episode start
            if i == 0 or samples[i-1][2].is_terminal:
                lstd_lambda.z = np.zeros(n_features)  # 重置迹
                current_episode += 1
                episode_starts.append(i)
            
            lstd_lambda.add_sample(state, reward, next_state)
        
        # 求解
        # Solve
        weights_lambda = lstd_lambda.solve()
        
        print(f"  权重范数: {np.linalg.norm(weights_lambda):.3f}")
        print(f"  与LSTD(0)权重差异: {np.linalg.norm(weights_lambda - weights):.3f}")
    
    # 3. 测试递归LSTD
    # 3. Test Recursive LSTD
    print("\n" + "="*60)
    print("3. 递归LSTD - 增量更新")
    print("3. Recursive LSTD - Incremental Update")
    print("="*60)
    
    rlstd = RecursiveLeastSquaresTD(feature_extractor, gamma=0.9, epsilon=0.01)
    
    # 增量处理样本
    # Process samples incrementally
    print("\n增量处理样本...")
    
    for i, (state, reward, next_state) in enumerate(samples):
        # 检测新回合
        # Detect new episode
        if i > 0 and samples[i-1][2].is_terminal:
            rlstd.z = np.zeros(n_features)
        
        rlstd.update(state, reward, next_state)
        
        # 定期显示进度
        # Show progress periodically
        if (i + 1) % (len(samples) // 5) == 0:
            current_norm = np.linalg.norm(rlstd.weights)
            print(f"  样本 {i+1}/{len(samples)}: ||w|| = {current_norm:.3f}")
    
    print(f"\n最终权重范数: {np.linalg.norm(rlstd.weights):.3f}")
    print(f"与批量LSTD权重差异: {np.linalg.norm(rlstd.weights - weights):.3f}")
    
    # 4. 效率比较
    # 4. Efficiency Comparison
    print("\n" + "="*60)
    print("4. 效率比较")
    print("4. Efficiency Comparison")
    print("="*60)
    
    import time
    
    # LSTD批量求解时间
    # LSTD batch solve time
    lstd_batch = LeastSquaresTD(feature_extractor)
    start_time = time.time()
    for state, reward, next_state in samples:
        lstd_batch.add_sample(state, reward, next_state)
    lstd_batch.solve()
    batch_time = time.time() - start_time
    
    # 递归LSTD时间
    # Recursive LSTD time
    rlstd_test = RecursiveLeastSquaresTD(feature_extractor)
    start_time = time.time()
    for state, reward, next_state in samples:
        rlstd_test.update(state, reward, next_state)
    recursive_time = time.time() - start_time
    
    print(f"\n处理{len(samples)}个样本的时间:")
    print(f"  批量LSTD: {batch_time*1000:.2f}ms")
    print(f"  递归LSTD: {recursive_time*1000:.2f}ms")
    print(f"  加速比: {batch_time/recursive_time:.2f}x")
    
    # 总结
    # Summary
    print("\n" + "="*80)
    print("LSTD总结")
    print("LSTD Summary")
    print("="*80)
    
    print("""
    关键要点 Key Takeaways:
    =======================
    
    1. LSTD一次求解最优权重
       LSTD solves for optimal weights in one shot
       
    2. 数据效率最高
       Maximum data efficiency
       
    3. 无需学习率调参
       No learning rate tuning
       
    4. 递归版本避免矩阵求逆
       Recursive version avoids matrix inversion
       
    5. 仅适用于线性函数近似
       Only for linear function approximation
    
    复杂度 Complexity:
    - 批量LSTD: O(d³)求解 + O(d²)存储
      Batch LSTD: O(d³) solve + O(d²) storage
    - 递归LSTD: O(d²)每步 + O(d²)存储
      Recursive LSTD: O(d²) per step + O(d²) storage
    
    适用场景 Use Cases:
    - 小特征维度
      Small feature dimension
    - 批量数据可用
      Batch data available
    - 需要最优解
      Need optimal solution
    """)


if __name__ == "__main__":
    demonstrate_lstd()