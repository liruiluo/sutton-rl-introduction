"""
================================================================================
第11.6-11.8节：强调TD方法
Section 11.6-11.8: Emphatic TD Methods
================================================================================

解决分布偏移问题！
Solving the distribution shift problem!

核心思想 Core Idea:
强调那些在目标策略下重要但在行为策略下罕见的状态
Emphasize states that are important under target policy but rare under behavior policy

强调权重 Emphasis Weights:
M_t = λI_t + (1-λ)F_t
其中F_t是跟随度，I_t是兴趣度
Where F_t is followon trace, I_t is interest

主要算法 Main Algorithms:
- Emphatic TD(λ)
- Emphatic TDC
- ELSTD (Emphatic LSTD)
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
# 第11.6节：强调权重
# Section 11.6: Emphasis Weights
# ================================================================================

class EmphasisWeights:
    """
    强调权重计算
    Emphasis Weight Computation
    
    解决离策略学习的分布偏移
    Solve distribution shift in off-policy learning
    
    核心概念 Core Concepts:
    1. 兴趣度 I_t: 对状态的关注程度
       Interest: degree of attention to state
    2. 跟随度 F_t: 未来的累积重要性
       Followon: cumulative future importance
    3. 强调度 M_t: 最终的权重
       Emphasis: final weight
    
    递归关系 Recursive Relations:
    F_t = γρ_{t-1}F_{t-1} + I_t
    M_t = λI_t + (1-λ)F_t
    """
    
    def __init__(self,
                 gamma: float = 0.99,
                 lambda_: float = 0.9,
                 interest_fn: Optional[Callable] = None):
        """
        初始化强调权重
        Initialize emphasis weights
        
        Args:
            gamma: 折扣因子
                  Discount factor
            lambda_: 权重混合参数
                    Weight mixing parameter
            interest_fn: 兴趣函数 I(s)
                        Interest function
        """
        self.gamma = gamma
        self.lambda_ = lambda_
        self.interest_fn = interest_fn or (lambda s: 1.0)
        
        # 跟随度
        # Followon trace
        self.followon = 0.0
        
        # 统计
        # Statistics
        self.emphasis_history = []
        self.followon_history = []
        
        logger.info(f"初始化强调权重: γ={gamma}, λ={lambda_}")
    
    def compute_emphasis(self, state: Any, importance_ratio: float) -> float:
        """
        计算强调权重
        Compute emphasis weight
        
        Args:
            state: 当前状态
                  Current state
            importance_ratio: 重要性比率ρ
                            Importance ratio
        
        Returns:
            强调权重M
            Emphasis weight
        """
        # 获取兴趣度
        # Get interest
        interest = self.interest_fn(state)
        
        # 更新跟随度
        # Update followon
        self.followon = self.gamma * importance_ratio * self.followon + interest
        
        # 计算强调权重
        # Compute emphasis weight
        emphasis = self.lambda_ * interest + (1 - self.lambda_) * self.followon
        
        # 记录历史
        # Record history
        self.emphasis_history.append(emphasis)
        self.followon_history.append(self.followon)
        
        return emphasis
    
    def reset(self):
        """
        重置跟随度(新回合开始)
        Reset followon (new episode)
        """
        self.followon = 0.0
    
    def get_statistics(self) -> Dict[str, float]:
        """
        获取统计信息
        Get statistics
        """
        if len(self.emphasis_history) == 0:
            return {'mean_emphasis': 0.0, 'max_emphasis': 0.0}
        
        return {
            'mean_emphasis': np.mean(self.emphasis_history[-100:]),
            'max_emphasis': np.max(self.emphasis_history[-100:]),
            'mean_followon': np.mean(self.followon_history[-100:]),
            'current_followon': self.followon
        }


# ================================================================================
# 第11.7节：强调TD(λ)
# Section 11.7: Emphatic TD(λ)
# ================================================================================

class EmphaticTDLambda:
    """
    强调TD(λ)
    Emphatic TD(λ)
    
    使用强调权重的TD(λ)算法
    TD(λ) algorithm with emphasis weights
    
    更新规则 Update Rule:
    δ = R + γV(S') - V(S)
    e ← ρ(γλe + M∇v̂(S,w))
    w ← w + αδe
    
    关键创新 Key Innovation:
    资格迹包含强调权重M
    Eligibility trace includes emphasis weight M
    """
    
    def __init__(self,
                 n_features: int,
                 feature_extractor: Callable,
                 alpha: float = 0.01,
                 gamma: float = 0.99,
                 lambda_: float = 0.9,
                 interest_fn: Optional[Callable] = None):
        """
        初始化强调TD(λ)
        Initialize Emphatic TD(λ)
        
        Args:
            n_features: 特征维度
                       Feature dimension
            feature_extractor: 特征提取器
                             Feature extractor
            alpha: 学习率
                  Learning rate
            gamma: 折扣因子
                  Discount factor
            lambda_: 资格迹衰减率
                    Eligibility trace decay
            interest_fn: 兴趣函数
                        Interest function
        """
        self.n_features = n_features
        self.feature_extractor = feature_extractor
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        
        # 权重和资格迹
        # Weights and eligibility trace
        self.w = np.zeros(n_features)
        self.e = np.zeros(n_features)
        
        # 强调权重计算器
        # Emphasis weight computer
        self.emphasis_computer = EmphasisWeights(gamma, lambda_, interest_fn)
        
        # 统计
        # Statistics
        self.update_count = 0
        self.td_errors = []
        
        logger.info(f"初始化强调TD(λ): α={alpha}, λ={lambda_}")
    
    def get_value(self, state: Any) -> float:
        """
        获取状态价值
        Get state value
        """
        phi = self.feature_extractor(state)
        return np.dot(self.w, phi)
    
    def update(self, state: Any, reward: float, next_state: Any,
               done: bool, importance_ratio: float = 1.0):
        """
        强调TD(λ)更新
        Emphatic TD(λ) update
        
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
        
        # 计算强调权重
        # Compute emphasis weight
        emphasis = self.emphasis_computer.compute_emphasis(state, importance_ratio)
        
        # 计算TD误差
        # Compute TD error
        current_value = self.get_value(state)
        if done:
            td_target = reward
        else:
            next_value = self.get_value(next_state)
            td_target = reward + self.gamma * next_value
        
        td_error = td_target - current_value
        self.td_errors.append(td_error)
        
        # 更新资格迹（包含强调权重）
        # Update eligibility trace (with emphasis)
        self.e = importance_ratio * (self.gamma * self.lambda_ * self.e + emphasis * phi)
        
        # 更新权重
        # Update weights
        self.w += self.alpha * td_error * self.e
        
        # 如果终止，重置
        # Reset if terminal
        if done:
            self.e = np.zeros(self.n_features)
            self.emphasis_computer.reset()
        
        self.update_count += 1
        
        return td_error
    
    def get_emphasis_stats(self) -> Dict[str, float]:
        """获取强调统计"""
        return self.emphasis_computer.get_statistics()


# ================================================================================
# 第11.7.1节：强调TDC
# Section 11.7.1: Emphatic TDC
# ================================================================================

class EmphaticTDC:
    """
    强调TDC
    Emphatic TDC
    
    结合强调权重和梯度修正
    Combine emphasis weights with gradient correction
    
    优势 Advantages:
    - 解决分布偏移
      Solve distribution shift
    - 保持收敛性
      Maintain convergence
    - 更好的样本效率
      Better sample efficiency
    """
    
    def __init__(self,
                 n_features: int,
                 feature_extractor: Callable,
                 alpha_w: float = 0.01,
                 alpha_v: float = 0.1,
                 gamma: float = 0.99,
                 lambda_: float = 0.9,
                 interest_fn: Optional[Callable] = None):
        """
        初始化强调TDC
        Initialize Emphatic TDC
        """
        self.n_features = n_features
        self.feature_extractor = feature_extractor
        self.alpha_w = alpha_w
        self.alpha_v = alpha_v
        self.gamma = gamma
        self.lambda_ = lambda_
        
        # 参数
        # Parameters
        self.w = np.zeros(n_features)  # 价值函数参数
        self.v = np.zeros(n_features)  # 辅助参数
        self.e = np.zeros(n_features)  # 资格迹
        
        # 强调权重计算器
        # Emphasis weight computer
        self.emphasis_computer = EmphasisWeights(gamma, lambda_, interest_fn)
        
        # 统计
        # Statistics
        self.update_count = 0
        
        logger.info(f"初始化强调TDC: αw={alpha_w}, αv={alpha_v}")
    
    def get_value(self, state: Any) -> float:
        """获取状态价值"""
        phi = self.feature_extractor(state)
        return np.dot(self.w, phi)
    
    def update(self, state: Any, reward: float, next_state: Any,
               done: bool, importance_ratio: float = 1.0):
        """
        强调TDC更新
        Emphatic TDC update
        """
        # 提取特征
        # Extract features
        phi = self.feature_extractor(state)
        
        if done:
            phi_next = np.zeros(self.n_features)
        else:
            phi_next = self.feature_extractor(next_state)
        
        # 计算强调权重
        # Compute emphasis weight
        emphasis = self.emphasis_computer.compute_emphasis(state, importance_ratio)
        
        # TD误差
        # TD error
        td_error = reward + self.gamma * np.dot(self.w, phi_next) - np.dot(self.w, phi)
        
        # 修正的TD误差(TDC特性)
        # Corrected TD error (TDC feature)
        corrected_td_error = td_error - np.dot(self.gamma * phi_next, self.v)
        
        # 更新资格迹
        # Update eligibility trace
        self.e = importance_ratio * (self.gamma * self.lambda_ * self.e + emphasis * phi)
        
        # 更新主参数w
        # Update primary parameters w
        self.w += self.alpha_w * corrected_td_error * self.e
        
        # 更新辅助参数v
        # Update auxiliary parameters v
        self.v += self.alpha_v * importance_ratio * emphasis * (td_error * phi - self.v)
        
        # 如果终止，重置
        # Reset if terminal
        if done:
            self.e = np.zeros(self.n_features)
            self.emphasis_computer.reset()
        
        self.update_count += 1
        
        return td_error


# ================================================================================
# 第11.8节：ELSTD (强调LSTD)
# Section 11.8: ELSTD (Emphatic LSTD)
# ================================================================================

class ELSTD:
    """
    强调最小二乘TD
    Emphatic Least-Squares TD
    
    LSTD with emphasis weights
    
    关键修改 Key Modification:
    在LSTD的A和b矩阵中加入强调权重
    Include emphasis weights in LSTD's A and b matrices
    
    A = Σ M_t φ_t(φ_t - γφ_{t+1})^T
    b = Σ M_t R_{t+1} φ_t
    """
    
    def __init__(self,
                 n_features: int,
                 feature_extractor: Callable,
                 gamma: float = 0.99,
                 lambda_: float = 0.9,
                 epsilon: float = 0.001,
                 interest_fn: Optional[Callable] = None):
        """
        初始化ELSTD
        Initialize ELSTD
        
        Args:
            n_features: 特征维度
                       Feature dimension
            feature_extractor: 特征提取器
                             Feature extractor
            gamma: 折扣因子
                  Discount factor
            lambda_: 强调混合参数
                    Emphasis mixing parameter
            epsilon: 正则化参数
                    Regularization parameter
            interest_fn: 兴趣函数
                        Interest function
        """
        self.n_features = n_features
        self.feature_extractor = feature_extractor
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon = epsilon
        
        # LSTD矩阵
        # LSTD matrices
        self.A = np.zeros((n_features, n_features))
        self.b = np.zeros(n_features)
        
        # 资格迹
        # Eligibility trace
        self.z = np.zeros(n_features)
        
        # 强调权重计算器
        # Emphasis weight computer
        self.emphasis_computer = EmphasisWeights(gamma, lambda_, interest_fn)
        
        # 权重
        # Weights
        self.w = np.zeros(n_features)
        
        # 统计
        # Statistics
        self.n_samples = 0
        
        logger.info(f"初始化ELSTD: λ={lambda_}, ε={epsilon}")
    
    def add_sample(self, state: Any, reward: float, next_state: Any,
                   done: bool, importance_ratio: float = 1.0):
        """
        添加样本
        Add sample
        
        Args:
            state: 当前状态
                  Current state
            reward: 奖励
                   Reward
            next_state: 下一状态
                       Next state
            done: 是否终止
                 Whether done
            importance_ratio: 重要性比率
                            Importance ratio
        """
        # 提取特征
        # Extract features
        phi = self.feature_extractor(state)
        
        if done:
            phi_next = np.zeros(self.n_features)
        else:
            phi_next = self.feature_extractor(next_state)
        
        # 计算强调权重
        # Compute emphasis weight
        emphasis = self.emphasis_computer.compute_emphasis(state, importance_ratio)
        
        # 更新资格迹
        # Update eligibility trace
        self.z = self.gamma * self.lambda_ * importance_ratio * self.z + emphasis * phi
        
        # 更新A和b矩阵
        # Update A and b matrices
        self.A += np.outer(self.z, phi - self.gamma * phi_next)
        self.b += reward * self.z
        
        self.n_samples += 1
        
        # 如果终止，重置
        # Reset if terminal
        if done:
            self.z = np.zeros(self.n_features)
            self.emphasis_computer.reset()
    
    def solve(self) -> np.ndarray:
        """
        求解权重
        Solve for weights
        
        w = (A + εI)^{-1} b
        
        Returns:
            权重向量
            Weight vector
        """
        if self.n_samples == 0:
            return self.w
        
        # 添加正则化
        # Add regularization
        A_reg = self.A + self.epsilon * np.eye(self.n_features)
        
        try:
            # 求解线性系统
            # Solve linear system
            self.w = np.linalg.solve(A_reg, self.b)
        except np.linalg.LinAlgError:
            # 如果奇异，使用伪逆
            # Use pseudo-inverse if singular
            self.w = np.linalg.pinv(A_reg) @ self.b
        
        return self.w
    
    def get_value(self, state: Any) -> float:
        """获取状态价值"""
        phi = self.feature_extractor(state)
        return np.dot(self.w, phi)


# ================================================================================
# 第11.8.1节：真正的在线强调TD
# Section 11.8.1: True Online Emphatic TD
# ================================================================================

class TrueOnlineEmphaticTD:
    """
    真正的在线强调TD
    True Online Emphatic TD
    
    Dutch trace版本的强调TD
    Dutch trace version of Emphatic TD
    
    优势 Advantages:
    - 更高效的计算
      More efficient computation
    - 更好的收敛性
      Better convergence
    - 真正的在线更新
      True online updates
    """
    
    def __init__(self,
                 n_features: int,
                 feature_extractor: Callable,
                 alpha: float = 0.01,
                 gamma: float = 0.99,
                 lambda_: float = 0.9,
                 interest_fn: Optional[Callable] = None):
        """
        初始化真正的在线强调TD
        Initialize True Online Emphatic TD
        """
        self.n_features = n_features
        self.feature_extractor = feature_extractor
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        
        # 权重
        # Weights
        self.w = np.zeros(n_features)
        
        # Dutch traces
        self.e = np.zeros(n_features)  # 资格迹
        self.e_bar = np.zeros(n_features)  # 辅助资格迹
        
        # 强调权重计算器
        # Emphasis weight computer
        self.emphasis_computer = EmphasisWeights(gamma, lambda_, interest_fn)
        
        # 上一步的特征和价值
        # Previous features and value
        self.last_phi = None
        self.last_value = 0.0
        
        # 统计
        # Statistics
        self.update_count = 0
        
        logger.info(f"初始化真正的在线强调TD: α={alpha}, λ={lambda_}")
    
    def get_value(self, state: Any) -> float:
        """获取状态价值"""
        phi = self.feature_extractor(state)
        return np.dot(self.w, phi)
    
    def update(self, state: Any, reward: float, next_state: Any,
               done: bool, importance_ratio: float = 1.0):
        """
        真正的在线强调TD更新
        True Online Emphatic TD update
        """
        # 提取特征
        # Extract features
        phi = self.feature_extractor(state)
        
        # 计算强调权重
        # Compute emphasis weight
        emphasis = self.emphasis_computer.compute_emphasis(state, importance_ratio)
        
        # 当前价值
        # Current value
        current_value = self.get_value(state)
        
        # TD误差
        # TD error
        if done:
            td_target = reward
        else:
            next_value = self.get_value(next_state)
            td_target = reward + self.gamma * next_value
        
        td_error = td_target - current_value
        
        # Dutch trace更新
        # Dutch trace update
        if self.last_phi is not None:
            # 计算Dutch修正
            # Compute Dutch correction
            dutch_correction = self.alpha * (self.last_value - np.dot(self.w, self.last_phi)) * self.e
            
            # 更新权重（包含Dutch修正）
            # Update weights (with Dutch correction)
            self.w += self.alpha * td_error * self.e + dutch_correction
        
        # 更新资格迹
        # Update eligibility traces
        self.e = importance_ratio * (self.gamma * self.lambda_ * self.e + emphasis * phi)
        self.e_bar = importance_ratio * (self.gamma * self.lambda_ * self.e_bar + phi)
        
        # 保存当前特征和价值
        # Save current features and value
        self.last_phi = phi
        self.last_value = current_value
        
        # 如果终止，重置
        # Reset if terminal
        if done:
            self.e = np.zeros(self.n_features)
            self.e_bar = np.zeros(self.n_features)
            self.emphasis_computer.reset()
            self.last_phi = None
            self.last_value = 0.0
        
        self.update_count += 1
        
        return td_error


# ================================================================================
# 主函数：演示强调TD方法
# Main Function: Demonstrate Emphatic TD Methods
# ================================================================================

def demonstrate_emphatic_td():
    """
    演示强调TD方法
    Demonstrate Emphatic TD methods
    """
    print("\n" + "="*80)
    print("第11.6-11.8节：强调TD方法")
    print("Section 11.6-11.8: Emphatic TD Methods")
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
    
    # 兴趣函数（对某些状态更感兴趣）
    # Interest function (more interested in some states)
    def interest_function(state):
        if isinstance(state, int):
            # 对状态0和2更感兴趣
            # More interested in states 0 and 2
            if state % n_states in [0, 2]:
                return 2.0
            return 0.5
        return 1.0
    
    # 1. 测试强调权重计算
    # 1. Test emphasis weight computation
    print("\n" + "="*60)
    print("1. 强调权重计算")
    print("1. Emphasis Weight Computation")
    print("="*60)
    
    emphasis_computer = EmphasisWeights(
        gamma=0.9,
        lambda_=0.8,
        interest_fn=interest_function
    )
    
    print("\n模拟强调权重序列:")
    print("状态  ρ     I(s)   F_t    M_t")
    print("-" * 35)
    
    for step in range(10):
        state = step % n_states
        rho = np.random.uniform(0.5, 1.5)
        emphasis = emphasis_computer.compute_emphasis(state, rho)
        
        interest = interest_function(state)
        followon = emphasis_computer.followon
        
        print(f"{state:3d}  {rho:.2f}  {interest:.1f}   {followon:.3f}  {emphasis:.3f}")
    
    stats = emphasis_computer.get_statistics()
    print(f"\n统计: 平均强调={stats['mean_emphasis']:.3f}, "
          f"最大强调={stats['max_emphasis']:.3f}")
    
    # 2. 测试强调TD(λ)
    # 2. Test Emphatic TD(λ)
    print("\n" + "="*60)
    print("2. 强调TD(λ)")
    print("2. Emphatic TD(λ)")
    print("="*60)
    
    emphatic_td = EmphaticTDLambda(
        n_features=n_features,
        feature_extractor=simple_features,
        alpha=0.05,
        gamma=0.9,
        lambda_=0.8,
        interest_fn=interest_function
    )
    
    print("\n训练强调TD(λ)...")
    for step in range(30):
        state = step % n_states
        reward = -1.0 if state != 2 else 10.0  # 状态2有高奖励
        next_state = (step + 1) % n_states
        done = (step + 1) % 10 == 0
        
        # 模拟重要性比率
        # Simulate importance ratio
        rho = np.random.uniform(0.7, 1.3)
        
        td_error = emphatic_td.update(state, reward, next_state, done, rho)
        
        if step % 10 == 0:
            value = emphatic_td.get_value(state)
            emphasis_stats = emphatic_td.get_emphasis_stats()
            print(f"  步{step}: V({state})={value:.3f}, δ={td_error:.3f}, "
                  f"平均M={emphasis_stats['mean_emphasis']:.2f}")
    
    # 3. 测试强调TDC
    # 3. Test Emphatic TDC
    print("\n" + "="*60)
    print("3. 强调TDC")
    print("3. Emphatic TDC")
    print("="*60)
    
    emphatic_tdc = EmphaticTDC(
        n_features=n_features,
        feature_extractor=simple_features,
        alpha_w=0.01,
        alpha_v=0.1,
        gamma=0.9,
        lambda_=0.8,
        interest_fn=interest_function
    )
    
    print("\n训练强调TDC...")
    for step in range(30):
        state = step % n_states
        reward = -1.0 if state != 2 else 10.0
        next_state = (step + 1) % n_states
        done = (step + 1) % 10 == 0
        
        rho = np.random.uniform(0.7, 1.3)
        
        td_error = emphatic_tdc.update(state, reward, next_state, done, rho)
        
        if step % 10 == 0:
            value = emphatic_tdc.get_value(state)
            print(f"  步{step}: V({state})={value:.3f}, δ={td_error:.3f}")
    
    # 4. 测试ELSTD
    # 4. Test ELSTD
    print("\n" + "="*60)
    print("4. ELSTD (强调LSTD)")
    print("4. ELSTD (Emphatic LSTD)")
    print("="*60)
    
    elstd = ELSTD(
        n_features=n_features,
        feature_extractor=simple_features,
        gamma=0.9,
        lambda_=0.8,
        epsilon=0.01,
        interest_fn=interest_function
    )
    
    print("\n收集样本...")
    for step in range(50):
        state = step % n_states
        reward = -1.0 if state != 2 else 10.0
        next_state = (step + 1) % n_states
        done = (step + 1) % 10 == 0
        
        rho = np.random.uniform(0.8, 1.2)
        
        elstd.add_sample(state, reward, next_state, done, rho)
    
    # 求解
    # Solve
    weights = elstd.solve()
    print(f"\n求解完成，权重范数: ||w||={np.linalg.norm(weights):.3f}")
    
    print("\nELSTD价值估计:")
    for state in range(n_states):
        value = elstd.get_value(state)
        interest = interest_function(state)
        print(f"  V({state}) = {value:.3f}, I({state}) = {interest:.1f}")
    
    # 5. 测试真正的在线强调TD
    # 5. Test True Online Emphatic TD
    print("\n" + "="*60)
    print("5. 真正的在线强调TD")
    print("5. True Online Emphatic TD")
    print("="*60)
    
    true_online_etd = TrueOnlineEmphaticTD(
        n_features=n_features,
        feature_extractor=simple_features,
        alpha=0.05,
        gamma=0.9,
        lambda_=0.8,
        interest_fn=interest_function
    )
    
    print("\n训练真正的在线强调TD...")
    for step in range(30):
        state = step % n_states
        reward = -1.0 if state != 2 else 10.0
        next_state = (step + 1) % n_states
        done = (step + 1) % 10 == 0
        
        rho = np.random.uniform(0.7, 1.3)
        
        td_error = true_online_etd.update(state, reward, next_state, done, rho)
        
        if step % 10 == 0:
            value = true_online_etd.get_value(state)
            print(f"  步{step}: V({state})={value:.3f}, δ={td_error:.3f}")
    
    # 比较所有方法
    # Compare all methods
    print("\n" + "="*60)
    print("方法比较")
    print("Method Comparison")
    print("="*60)
    
    print("\n各方法的价值估计:")
    print("状态  ETD(λ)  ETDC    ELSTD   True-ETD  Interest")
    print("-" * 50)
    
    for state in range(n_states):
        values = [
            emphatic_td.get_value(state),
            emphatic_tdc.get_value(state),
            elstd.get_value(state),
            true_online_etd.get_value(state)
        ]
        interest = interest_function(state)
        
        print(f"{state:3d}   {values[0]:6.2f}  {values[1]:6.2f}  "
              f"{values[2]:6.2f}  {values[3]:7.2f}     {interest:.1f}")
    
    # 总结
    # Summary
    print("\n" + "="*80)
    print("强调TD方法总结")
    print("Emphatic TD Methods Summary")
    print("="*80)
    
    print("""
    关键要点 Key Takeaways:
    =======================
    
    1. 强调权重解决分布偏移
       Emphasis weights solve distribution shift
       
    2. 兴趣函数指导学习重点
       Interest function guides learning focus
       
    3. 跟随度追踪长期影响
       Followon trace tracks long-term impact
       
    4. ELSTD最准确但计算密集
       ELSTD most accurate but computationally intensive
       
    5. 真正的在线版本最实用
       True online version most practical
    
    应用场景 Application Scenarios:
    - 重要状态稀疏访问: 使用强调TD
                      Use Emphatic TD for rare important states
    - 需要稳定性: 使用强调TDC
                 Use Emphatic TDC for stability
    - 批量数据: 使用ELSTD
               Use ELSTD for batch data
    - 在线学习: 使用真正的在线强调TD
               Use True Online Emphatic TD for online learning
    
    离策略学习的未来！
    The future of off-policy learning!
    """)


if __name__ == "__main__":
    demonstrate_emphatic_td()