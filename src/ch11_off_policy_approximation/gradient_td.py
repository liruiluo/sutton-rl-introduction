"""
================================================================================
第11.3-11.5节：梯度TD方法
Section 11.3-11.5: Gradient TD Methods
================================================================================

解决致命三要素问题！
Solving the deadly triad problem!

致命三要素 The Deadly Triad:
1. 函数近似
   Function approximation
2. 自举
   Bootstrapping
3. 离策略学习
   Off-policy learning

梯度TD方法 Gradient TD Methods:
- 保证收敛性
  Guaranteed convergence
- 最小化投影Bellman误差
  Minimize projected Bellman error
- 双时间尺度学习
  Two-timescale learning

主要算法 Main Algorithms:
- TDC (TD with Gradient Correction)
- GTD2 (Gradient TD version 2)
- HTD (Hybrid TD)
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
# 第11.3节：投影Bellman误差
# Section 11.3: Projected Bellman Error
# ================================================================================

class ProjectedBellmanError:
    """
    投影Bellman误差
    Projected Bellman Error
    
    目标函数 Objective Function:
    PBE = ||Πδ||²_μ
    
    其中 Where:
    - δ是TD误差
      δ is TD error
    - Π是投影算子
      Π is projection operator
    - μ是状态分布
      μ is state distribution
    
    关键思想 Key Idea:
    最小化投影到特征空间的Bellman误差
    Minimize Bellman error projected onto feature space
    """
    
    def __init__(self,
                 n_features: int,
                 feature_extractor: Callable,
                 gamma: float = 0.99):
        """
        初始化PBE
        Initialize PBE
        
        Args:
            n_features: 特征数
                       Number of features
            feature_extractor: 特征提取函数
                             Feature extraction function
            gamma: 折扣因子
                  Discount factor
        """
        self.n_features = n_features
        self.feature_extractor = feature_extractor
        self.gamma = gamma
        
        # 统计量
        # Statistics
        self.A = np.zeros((n_features, n_features))  # 特征协方差
        self.b = np.zeros(n_features)                # 特征-奖励相关
        self.C = np.zeros((n_features, n_features))  # 特征-TD误差协方差
        
        # 样本计数
        # Sample count
        self.n_samples = 0
        
        logger.info(f"初始化投影Bellman误差: {n_features}维特征")
    
    def update_statistics(self, state: Any, reward: float, 
                         next_state: Any, done: bool):
        """
        更新统计量
        Update statistics
        
        增量更新A, b, C矩阵
        Incrementally update A, b, C matrices
        
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
        # 提取特征
        # Extract features
        phi = self.feature_extractor(state)
        
        if done:
            phi_next = np.zeros(self.n_features)
        else:
            phi_next = self.feature_extractor(next_state)
        
        # 时序差分特征
        # Temporal difference features
        phi_diff = phi - self.gamma * phi_next
        
        # 增量更新
        # Incremental update
        self.n_samples += 1
        alpha = 1.0 / self.n_samples
        
        # A = E[φ(φ - γφ')^T]
        self.A = (1 - alpha) * self.A + alpha * np.outer(phi, phi_diff)
        
        # b = E[Rφ]
        self.b = (1 - alpha) * self.b + alpha * reward * phi
        
        # C = E[φφ^T]
        self.C = (1 - alpha) * self.C + alpha * np.outer(phi, phi)
    
    def compute_pbe(self, weights: np.ndarray) -> float:
        """
        计算投影Bellman误差
        Compute projected Bellman error
        
        PBE(w) = (Aw - b)^T C^{-1} (Aw - b)
        
        Args:
            weights: 权重向量
                    Weight vector
        
        Returns:
            PBE值
            PBE value
        """
        if self.n_samples == 0:
            return 0.0
        
        # Bellman误差向量
        # Bellman error vector
        bellman_error = self.A @ weights - self.b
        
        # 投影Bellman误差
        # Projected Bellman error
        try:
            C_inv = np.linalg.pinv(self.C + 0.001 * np.eye(self.n_features))
            pbe = bellman_error.T @ C_inv @ bellman_error
            return pbe
        except:
            return np.linalg.norm(bellman_error)
    
    def compute_gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        计算PBE梯度
        Compute PBE gradient
        
        ∇PBE(w) = 2A^T C^{-1} (Aw - b)
        
        Args:
            weights: 权重向量
                    Weight vector
        
        Returns:
            梯度向量
            Gradient vector
        """
        if self.n_samples == 0:
            return np.zeros(self.n_features)
        
        bellman_error = self.A @ weights - self.b
        
        try:
            C_inv = np.linalg.pinv(self.C + 0.001 * np.eye(self.n_features))
            gradient = 2 * self.A.T @ C_inv @ bellman_error
            return gradient
        except:
            return 2 * self.A.T @ bellman_error


# ================================================================================
# 第11.4节：GTD2和TDC算法
# Section 11.4: GTD2 and TDC Algorithms
# ================================================================================

class GTD2:
    """
    梯度TD版本2
    Gradient TD Version 2
    
    最小化投影Bellman误差的梯度算法
    Gradient algorithm for minimizing projected Bellman error
    
    双参数更新 Two-parameter Update:
    - w: 价值函数参数
         Value function parameters
    - v: 辅助参数(用于无偏梯度估计)
         Auxiliary parameters (for unbiased gradient estimation)
    
    收敛保证 Convergence Guarantee:
    即使离策略+函数近似+自举也收敛！
    Converges even with off-policy + function approximation + bootstrapping!
    """
    
    def __init__(self,
                 n_features: int,
                 feature_extractor: Callable,
                 alpha_w: float = 0.01,
                 alpha_v: float = 0.1,
                 gamma: float = 0.99):
        """
        初始化GTD2
        Initialize GTD2
        
        Args:
            n_features: 特征维度
                       Feature dimension
            feature_extractor: 特征提取器
                             Feature extractor
            alpha_w: 主参数学习率
                    Primary parameter learning rate
            alpha_v: 辅助参数学习率
                    Auxiliary parameter learning rate
            gamma: 折扣因子
                  Discount factor
        """
        self.n_features = n_features
        self.feature_extractor = feature_extractor
        self.alpha_w = alpha_w
        self.alpha_v = alpha_v
        self.gamma = gamma
        
        # 参数
        # Parameters
        self.w = np.zeros(n_features)  # 价值函数参数
        self.v = np.zeros(n_features)  # 辅助参数
        
        # 统计
        # Statistics
        self.update_count = 0
        self.td_errors = []
        
        logger.info(f"初始化GTD2: αw={alpha_w}, αv={alpha_v}")
    
    def get_value(self, state: Any) -> float:
        """
        获取状态价值
        Get state value
        
        v̂(s,w) = w^T φ(s)
        """
        phi = self.feature_extractor(state)
        return np.dot(self.w, phi)
    
    def update(self, state: Any, reward: float, next_state: Any,
               done: bool, importance_ratio: float = 1.0):
        """
        GTD2更新
        GTD2 update
        
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
        phi = self.feature_extractor(state)
        
        if done:
            phi_next = np.zeros(self.n_features)
        else:
            phi_next = self.feature_extractor(next_state)
        
        # TD误差
        # TD error
        td_error = reward + self.gamma * np.dot(self.w, phi_next) - np.dot(self.w, phi)
        self.td_errors.append(td_error)
        
        # 更新辅助参数v
        # Update auxiliary parameters v
        self.v += self.alpha_v * importance_ratio * (td_error * phi - self.v)
        
        # 更新主参数w
        # Update primary parameters w
        self.w += self.alpha_w * importance_ratio * (
            phi * np.dot(phi - self.gamma * phi_next, self.v)
        )
        
        self.update_count += 1
        
        return td_error


class TDC:
    """
    TD with Gradient Correction
    带梯度修正的TD
    
    GTD2的改进版本
    Improved version of GTD2
    
    关键区别 Key Difference:
    使用不同的参数更新顺序，通常性能更好
    Different parameter update order, usually better performance
    
    优势 Advantages:
    - 更稳定
      More stable
    - 收敛更快
      Faster convergence
    """
    
    def __init__(self,
                 n_features: int,
                 feature_extractor: Callable,
                 alpha_w: float = 0.01,
                 alpha_v: float = 0.1,
                 gamma: float = 0.99):
        """
        初始化TDC
        Initialize TDC
        """
        self.n_features = n_features
        self.feature_extractor = feature_extractor
        self.alpha_w = alpha_w
        self.alpha_v = alpha_v
        self.gamma = gamma
        
        # 参数
        # Parameters
        self.w = np.zeros(n_features)
        self.v = np.zeros(n_features)
        
        # 统计
        # Statistics
        self.update_count = 0
        self.td_errors = []
        
        logger.info(f"初始化TDC: αw={alpha_w}, αv={alpha_v}")
    
    def get_value(self, state: Any) -> float:
        """获取状态价值"""
        phi = self.feature_extractor(state)
        return np.dot(self.w, phi)
    
    def update(self, state: Any, reward: float, next_state: Any,
               done: bool, importance_ratio: float = 1.0):
        """
        TDC更新
        TDC update
        
        与GTD2的主要区别在更新顺序
        Main difference from GTD2 is update order
        """
        # 提取特征
        # Extract features
        phi = self.feature_extractor(state)
        
        if done:
            phi_next = np.zeros(self.n_features)
        else:
            phi_next = self.feature_extractor(next_state)
        
        # TD误差
        # TD error
        td_error = reward + self.gamma * np.dot(self.w, phi_next) - np.dot(self.w, phi)
        self.td_errors.append(td_error)
        
        # TDC的关键：使用修正的TD误差
        # TDC key: use corrected TD error
        corrected_td_error = td_error - np.dot(self.gamma * phi_next, self.v)
        
        # 更新主参数w
        # Update primary parameters w
        self.w += self.alpha_w * importance_ratio * corrected_td_error * phi
        
        # 更新辅助参数v
        # Update auxiliary parameters v
        self.v += self.alpha_v * importance_ratio * (td_error * phi - self.v)
        
        self.update_count += 1
        
        return td_error


# ================================================================================
# 第11.5节：HTD (混合TD)
# Section 11.5: HTD (Hybrid TD)
# ================================================================================

class HTD:
    """
    混合TD
    Hybrid TD
    
    结合GTD和传统TD的优点
    Combines advantages of GTD and conventional TD
    
    关键思想 Key Idea:
    - 在同策略时退化为普通TD
      Degenerates to regular TD when on-policy
    - 离策略时保持稳定性
      Maintains stability when off-policy
    
    自适应权衡 Adaptive Tradeoff:
    根据重要性比率自动调整
    Automatically adjusts based on importance ratio
    """
    
    def __init__(self,
                 n_features: int,
                 feature_extractor: Callable,
                 alpha: float = 0.01,
                 beta: float = 0.1,
                 gamma: float = 0.99,
                 lambda_: float = 0.9):
        """
        初始化HTD
        Initialize HTD
        
        Args:
            n_features: 特征维度
                       Feature dimension
            feature_extractor: 特征提取器
                             Feature extractor
            alpha: 主学习率
                  Primary learning rate
            beta: 辅助学习率
                 Auxiliary learning rate
            gamma: 折扣因子
                  Discount factor
            lambda_: 资格迹衰减率
                    Eligibility trace decay
        """
        self.n_features = n_features
        self.feature_extractor = feature_extractor
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_ = lambda_
        
        # 参数
        # Parameters
        self.w = np.zeros(n_features)      # 价值函数参数
        self.v = np.zeros(n_features)      # 辅助参数
        self.e = np.zeros(n_features)      # 资格迹
        self.f = np.zeros(n_features)      # 辅助资格迹
        
        # 统计
        # Statistics
        self.update_count = 0
        self.avg_importance_ratio = 1.0
        
        logger.info(f"初始化HTD: α={alpha}, β={beta}, λ={lambda_}")
    
    def get_value(self, state: Any) -> float:
        """获取状态价值"""
        phi = self.feature_extractor(state)
        return np.dot(self.w, phi)
    
    def update(self, state: Any, reward: float, next_state: Any,
               done: bool, importance_ratio: float = 1.0):
        """
        HTD更新
        HTD update
        
        混合TD更新规则
        Hybrid TD update rule
        """
        # 提取特征
        # Extract features
        phi = self.feature_extractor(state)
        
        if done:
            phi_next = np.zeros(self.n_features)
        else:
            phi_next = self.feature_extractor(next_state)
        
        # TD误差
        # TD error
        td_error = reward + self.gamma * np.dot(self.w, phi_next) - np.dot(self.w, phi)
        
        # 更新平均重要性比率
        # Update average importance ratio
        self.avg_importance_ratio = 0.99 * self.avg_importance_ratio + 0.01 * importance_ratio
        
        # 混合因子(根据重要性比率调整)
        # Mixing factor (adjusted by importance ratio)
        mix = min(1.0, importance_ratio / max(0.01, self.avg_importance_ratio))
        
        # 更新资格迹
        # Update eligibility traces
        self.e = self.gamma * self.lambda_ * importance_ratio * self.e + phi
        self.f = self.gamma * self.lambda_ * importance_ratio * self.f + phi
        
        # 混合更新：结合TD和GTD
        # Hybrid update: combine TD and GTD
        td_update = td_error * self.e
        gtd_correction = np.dot(phi - self.gamma * phi_next, self.v) * self.f
        
        # 根据混合因子加权
        # Weight by mixing factor
        self.w += self.alpha * (mix * td_update + (1 - mix) * importance_ratio * gtd_correction)
        
        # 更新辅助参数
        # Update auxiliary parameters
        self.v += self.beta * importance_ratio * (td_error * phi - self.v)
        
        # 如果终止，重置资格迹
        # Reset traces if terminal
        if done:
            self.e = np.zeros(self.n_features)
            self.f = np.zeros(self.n_features)
        
        self.update_count += 1
        
        return td_error


# ================================================================================
# 第11.5.1节：LSTD的梯度版本
# Section 11.5.1: Gradient Version of LSTD
# ================================================================================

class GradientLSTD:
    """
    梯度最小二乘TD
    Gradient Least-Squares TD
    
    使用梯度下降实现LSTD
    Implement LSTD using gradient descent
    
    优势 Advantages:
    - 不需要矩阵求逆
      No matrix inversion needed
    - 可以在线更新
      Can update online
    - 内存效率高
      Memory efficient
    """
    
    def __init__(self,
                 n_features: int,
                 feature_extractor: Callable,
                 alpha: float = 0.01,
                 gamma: float = 0.99,
                 reg: float = 0.001):
        """
        初始化梯度LSTD
        Initialize Gradient LSTD
        
        Args:
            n_features: 特征维度
                       Feature dimension
            feature_extractor: 特征提取器
                             Feature extractor
            alpha: 学习率
                  Learning rate
            gamma: 折扣因子
                  Discount factor
            reg: 正则化系数
                Regularization coefficient
        """
        self.n_features = n_features
        self.feature_extractor = feature_extractor
        self.alpha = alpha
        self.gamma = gamma
        self.reg = reg
        
        # 参数
        # Parameters
        self.w = np.zeros(n_features)
        
        # Sherman-Morrison增量逆矩阵
        # Sherman-Morrison incremental inverse
        self.P = np.eye(n_features) / reg
        
        # 统计
        # Statistics
        self.update_count = 0
        
        logger.info(f"初始化梯度LSTD: α={alpha}, reg={reg}")
    
    def get_value(self, state: Any) -> float:
        """获取状态价值"""
        phi = self.feature_extractor(state)
        return np.dot(self.w, phi)
    
    def update(self, state: Any, reward: float, next_state: Any,
               done: bool, importance_ratio: float = 1.0):
        """
        梯度LSTD更新
        Gradient LSTD update
        
        使用Sherman-Morrison公式增量更新
        Use Sherman-Morrison formula for incremental update
        """
        # 提取特征
        # Extract features
        phi = self.feature_extractor(state)
        
        if done:
            phi_next = np.zeros(self.n_features)
        else:
            phi_next = self.feature_extractor(next_state)
        
        # 时序差分特征
        # Temporal difference features
        phi_diff = phi - self.gamma * phi_next
        
        # Sherman-Morrison更新P矩阵
        # Sherman-Morrison update for P matrix
        k = (self.P @ phi_diff) / (1 + phi.T @ self.P @ phi_diff)
        self.P = self.P - np.outer(k, phi.T @ self.P)
        
        # TD误差
        # TD error
        td_error = reward + self.gamma * np.dot(self.w, phi_next) - np.dot(self.w, phi)
        
        # 梯度更新
        # Gradient update
        gradient = -2 * phi_diff * td_error
        self.w -= self.alpha * importance_ratio * self.P @ gradient
        
        self.update_count += 1
        
        return td_error


# ================================================================================
# 主函数：演示梯度TD方法
# Main Function: Demonstrate Gradient TD Methods
# ================================================================================

def demonstrate_gradient_td():
    """
    演示梯度TD方法
    Demonstrate gradient TD methods
    """
    print("\n" + "="*80)
    print("第11.3-11.5节：梯度TD方法")
    print("Section 11.3-11.5: Gradient TD Methods")
    print("="*80)
    
    # 设置
    # Setup
    n_features = 8
    gamma = 0.9
    
    # 简单特征提取器
    # Simple feature extractor
    def simple_features(state):
        if isinstance(state, int):
            features = np.zeros(n_features)
            features[state % n_features] = 1.0
            # 添加一些重叠特征
            # Add some overlapping features
            features[(state + 1) % n_features] = 0.5
            return features / np.linalg.norm(features)
        return np.random.randn(n_features) * 0.1
    
    # 1. 测试投影Bellman误差
    # 1. Test Projected Bellman Error
    print("\n" + "="*60)
    print("1. 投影Bellman误差")
    print("1. Projected Bellman Error")
    print("="*60)
    
    pbe = ProjectedBellmanError(n_features, simple_features, gamma)
    
    # 收集一些样本
    # Collect some samples
    print("\n收集样本更新统计...")
    for i in range(20):
        state = i % 5
        reward = np.random.randn()
        next_state = (i + 1) % 5
        done = i == 19
        
        pbe.update_statistics(state, reward, next_state, done)
    
    # 计算不同权重的PBE
    # Compute PBE for different weights
    print("\n不同权重的PBE值:")
    test_weights = [
        np.zeros(n_features),
        np.ones(n_features) * 0.1,
        np.random.randn(n_features) * 0.5
    ]
    
    for i, w in enumerate(test_weights):
        pbe_value = pbe.compute_pbe(w)
        gradient_norm = np.linalg.norm(pbe.compute_gradient(w))
        print(f"  权重{i+1}: PBE={pbe_value:.4f}, ||∇PBE||={gradient_norm:.4f}")
    
    # 2. 测试GTD2
    # 2. Test GTD2
    print("\n" + "="*60)
    print("2. GTD2算法")
    print("2. GTD2 Algorithm")
    print("="*60)
    
    gtd2 = GTD2(n_features, simple_features, alpha_w=0.01, alpha_v=0.1, gamma=gamma)
    
    print("\n训练GTD2...")
    for step in range(30):
        state = step % 5
        reward = -1.0 + np.random.randn() * 0.5
        next_state = (step + 1) % 5
        done = False
        
        # 模拟重要性比率
        # Simulate importance ratio
        rho = np.random.uniform(0.5, 1.5)
        
        td_error = gtd2.update(state, reward, next_state, done, rho)
        
        if step % 10 == 0:
            value = gtd2.get_value(state)
            print(f"  步{step}: V({state})={value:.3f}, δ={td_error:.3f}, ρ={rho:.2f}")
    
    print(f"\n最终参数范数: ||w||={np.linalg.norm(gtd2.w):.3f}, ||v||={np.linalg.norm(gtd2.v):.3f}")
    
    # 3. 测试TDC
    # 3. Test TDC
    print("\n" + "="*60)
    print("3. TDC算法")
    print("3. TDC Algorithm")
    print("="*60)
    
    tdc = TDC(n_features, simple_features, alpha_w=0.01, alpha_v=0.1, gamma=gamma)
    
    print("\n训练TDC...")
    for step in range(30):
        state = step % 5
        reward = -1.0 + np.random.randn() * 0.5
        next_state = (step + 1) % 5
        done = False
        
        rho = np.random.uniform(0.5, 1.5)
        
        td_error = tdc.update(state, reward, next_state, done, rho)
        
        if step % 10 == 0:
            value = tdc.get_value(state)
            print(f"  步{step}: V({state})={value:.3f}, δ={td_error:.3f}")
    
    # 比较GTD2和TDC
    # Compare GTD2 and TDC
    print("\n比较GTD2和TDC:")
    for state in range(5):
        gtd2_value = gtd2.get_value(state)
        tdc_value = tdc.get_value(state)
        print(f"  状态{state}: GTD2={gtd2_value:.3f}, TDC={tdc_value:.3f}")
    
    # 4. 测试HTD
    # 4. Test HTD
    print("\n" + "="*60)
    print("4. HTD (混合TD)")
    print("4. HTD (Hybrid TD)")
    print("="*60)
    
    htd = HTD(n_features, simple_features, alpha=0.01, beta=0.1, gamma=gamma, lambda_=0.5)
    
    print("\n训练HTD...")
    for step in range(30):
        state = step % 5
        reward = -1.0 + np.random.randn() * 0.5
        next_state = (step + 1) % 5
        done = (step + 1) % 10 == 0
        
        # 变化的重要性比率
        # Varying importance ratio
        if step < 10:
            rho = 1.0  # 同策略
        elif step < 20:
            rho = np.random.uniform(0.8, 1.2)  # 近同策略
        else:
            rho = np.random.uniform(0.5, 2.0)  # 离策略
        
        td_error = htd.update(state, reward, next_state, done, rho)
        
        if step % 10 == 0:
            value = htd.get_value(state)
            print(f"  步{step}: V({state})={value:.3f}, "
                  f"avg_ρ={htd.avg_importance_ratio:.2f}")
    
    # 5. 测试梯度LSTD
    # 5. Test Gradient LSTD
    print("\n" + "="*60)
    print("5. 梯度LSTD")
    print("5. Gradient LSTD")
    print("="*60)
    
    glstd = GradientLSTD(n_features, simple_features, alpha=0.1, gamma=gamma)
    
    print("\n训练梯度LSTD...")
    for step in range(30):
        state = step % 5
        reward = -1.0 + np.random.randn() * 0.5
        next_state = (step + 1) % 5
        done = False
        
        rho = np.random.uniform(0.8, 1.2)
        
        td_error = glstd.update(state, reward, next_state, done, rho)
        
        if step % 10 == 0:
            value = glstd.get_value(state)
            print(f"  步{step}: V({state})={value:.3f}, δ={td_error:.3f}")
    
    # 比较所有方法
    # Compare all methods
    print("\n" + "="*60)
    print("方法比较")
    print("Method Comparison")
    print("="*60)
    
    print("\n各方法的价值估计:")
    print("状态  GTD2    TDC     HTD     GLSTD")
    print("-" * 40)
    for state in range(5):
        values = [
            gtd2.get_value(state),
            tdc.get_value(state),
            htd.get_value(state),
            glstd.get_value(state)
        ]
        print(f"{state:3d}   {values[0]:6.3f}  {values[1]:6.3f}  "
              f"{values[2]:6.3f}  {values[3]:6.3f}")
    
    # 总结
    # Summary
    print("\n" + "="*80)
    print("梯度TD方法总结")
    print("Gradient TD Methods Summary")
    print("="*80)
    
    print("""
    关键要点 Key Takeaways:
    =======================
    
    1. 梯度TD解决致命三要素
       Gradient TD solves deadly triad
       
    2. 双时间尺度学习稳定收敛
       Two-timescale learning for stable convergence
       
    3. TDC通常优于GTD2
       TDC usually better than GTD2
       
    4. HTD自适应混合最实用
       HTD adaptive mixing most practical
       
    5. 投影Bellman误差是关键目标
       Projected Bellman error is key objective
    
    算法选择 Algorithm Selection:
    - 同策略: 普通TD即可
             Regular TD sufficient
    - 近同策略: HTD
              HTD for near on-policy
    - 强离策略: TDC或GTD2
              TDC or GTD2 for strong off-policy
    - 需要快速: 梯度LSTD
              Gradient LSTD for speed
    """)


if __name__ == "__main__":
    demonstrate_gradient_td()