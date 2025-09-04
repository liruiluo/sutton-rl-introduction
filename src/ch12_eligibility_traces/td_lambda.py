"""
================================================================================
第12.3-12.5节：TD(λ)算法
Section 12.3-12.5: TD(λ) Algorithms
================================================================================

资格迹的后向视角！
Backward view with eligibility traces!

资格迹 Eligibility Traces:
记录状态的"资格"获得未来奖励
Record state's "eligibility" for future rewards

e_t = γλe_{t-1} + ∇v̂(S_t,w_t)

TD(λ)更新 TD(λ) Update:
w_{t+1} = w_t + αδ_t e_t

等价性 Equivalence:
前向视角(λ-return) ≡ 后向视角(资格迹)
Forward view (λ-return) ≡ Backward view (eligibility traces)

不同类型的迹 Different Types of Traces:
1. 累积迹 (Accumulating traces)
2. 替换迹 (Replacing traces)  
3. Dutch迹 (Dutch traces)
4. 真正的在线TD(λ) (True Online TD(λ))
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
# 第12.3节：基础TD(λ)
# Section 12.3: Basic TD(λ)
# ================================================================================

class TDLambda:
    """
    TD(λ)算法
    TD(λ) Algorithm
    
    使用资格迹的后向视角实现
    Backward view implementation with eligibility traces
    
    核心思想 Core Idea:
    资格迹记录每个状态对当前TD误差的贡献
    Eligibility trace records each state's contribution to current TD error
    
    更新规则 Update Rule:
    δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)
    e_t = γλe_{t-1} + ∇v̂(S_t,w_t)
    w_{t+1} = w_t + αδ_t e_t
    """
    
    def __init__(self,
                 n_features: int,
                 feature_extractor: Callable,
                 lambda_: float = 0.9,
                 alpha: float = 0.01,
                 gamma: float = 0.99,
                 trace_type: str = 'accumulating'):
        """
        初始化TD(λ)
        Initialize TD(λ)
        
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
            trace_type: 迹类型 ('accumulating', 'replacing')
                       Trace type
        """
        self.n_features = n_features
        self.feature_extractor = feature_extractor
        self.lambda_ = lambda_
        self.alpha = alpha
        self.gamma = gamma
        self.trace_type = trace_type
        
        # 权重和资格迹
        # Weights and eligibility traces
        self.weights = np.zeros(n_features)
        self.traces = np.zeros(n_features)
        
        # 统计
        # Statistics
        self.update_count = 0
        self.td_errors = []
        self.trace_magnitudes = []
        
        logger.info(f"初始化TD(λ): λ={lambda_}, trace_type={trace_type}")
    
    def get_value(self, state: Any) -> float:
        """
        获取状态价值
        Get state value
        
        v̂(s,w) = w^T φ(s)
        """
        features = self.feature_extractor(state)
        return np.dot(self.weights, features)
    
    def update_traces(self, features: np.ndarray):
        """
        更新资格迹
        Update eligibility traces
        
        Args:
            features: 当前状态特征
                     Current state features
        """
        if self.trace_type == 'accumulating':
            # 累积迹
            # Accumulating traces
            self.traces = self.gamma * self.lambda_ * self.traces + features
            
        elif self.trace_type == 'replacing':
            # 替换迹
            # Replacing traces
            self.traces *= self.gamma * self.lambda_
            # 对于激活的特征，设置为1而不是累积
            # For active features, set to 1 instead of accumulating
            active_indices = np.where(features > 0)[0]
            self.traces[active_indices] = features[active_indices]
        
        # 记录迹的大小
        # Record trace magnitude
        self.trace_magnitudes.append(np.linalg.norm(self.traces))
    
    def update(self, state: Any, reward: float, next_state: Any, done: bool):
        """
        TD(λ)更新
        TD(λ) update
        
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
        self.td_errors.append(td_error)
        
        # 更新资格迹
        # Update eligibility traces
        self.update_traces(features)
        
        # 更新权重
        # Update weights
        self.weights += self.alpha * td_error * self.traces
        
        # 如果终止，重置资格迹
        # Reset traces if terminal
        if done:
            self.traces = np.zeros(self.n_features)
        
        self.update_count += 1
        
        return td_error
    
    def get_statistics(self) -> Dict[str, float]:
        """
        获取统计信息
        Get statistics
        """
        if len(self.td_errors) == 0:
            return {}
        
        return {
            'mean_td_error': np.mean(np.abs(self.td_errors[-100:])),
            'mean_trace_magnitude': np.mean(self.trace_magnitudes[-100:]),
            'max_trace_magnitude': np.max(self.trace_magnitudes[-100:]) if self.trace_magnitudes else 0,
            'weight_norm': np.linalg.norm(self.weights)
        }


# ================================================================================
# 第12.4节：真正的在线TD(λ)
# Section 12.4: True Online TD(λ)
# ================================================================================

class TrueOnlineTDLambda:
    """
    真正的在线TD(λ)
    True Online TD(λ)
    
    使用Dutch traces的改进版本
    Improved version with Dutch traces
    
    关键创新 Key Innovation:
    修正了传统TD(λ)的近似误差
    Corrects approximation error of conventional TD(λ)
    
    Dutch trace更新 Dutch Trace Update:
    e_t = γλe_{t-1} + (1 - αγλe_{t-1}^T φ_t)φ_t
    
    性能优势 Performance Advantage:
    更接近理想的λ-return算法
    Closer to ideal λ-return algorithm
    """
    
    def __init__(self,
                 n_features: int,
                 feature_extractor: Callable,
                 lambda_: float = 0.9,
                 alpha: float = 0.01,
                 gamma: float = 0.99):
        """
        初始化真正的在线TD(λ)
        Initialize True Online TD(λ)
        """
        self.n_features = n_features
        self.feature_extractor = feature_extractor
        self.lambda_ = lambda_
        self.alpha = alpha
        self.gamma = gamma
        
        # 权重和资格迹
        # Weights and eligibility traces
        self.weights = np.zeros(n_features)
        self.traces = np.zeros(n_features)
        
        # 上一步的值和特征
        # Previous value and features
        self.old_value = 0.0
        self.old_features = np.zeros(n_features)
        
        # 统计
        # Statistics
        self.update_count = 0
        self.td_errors = []
        
        logger.info(f"初始化真正的在线TD(λ): λ={lambda_}")
    
    def get_value(self, state: Any) -> float:
        """获取状态价值"""
        features = self.feature_extractor(state)
        return np.dot(self.weights, features)
    
    def update(self, state: Any, reward: float, next_state: Any, done: bool):
        """
        真正的在线TD(λ)更新
        True Online TD(λ) update
        
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
        features = self.feature_extractor(state)
        
        # 当前价值
        # Current value
        current_value = np.dot(self.weights, features)
        
        # TD误差
        # TD error
        if done:
            td_target = reward
        else:
            next_value = self.get_value(next_state)
            td_target = reward + self.gamma * next_value
        
        td_error = td_target - current_value
        self.td_errors.append(td_error)
        
        # Dutch trace更新
        # Dutch trace update
        dutch_factor = 1 - self.alpha * self.gamma * self.lambda_ * np.dot(self.traces, features)
        self.traces = self.gamma * self.lambda_ * self.traces + dutch_factor * features
        
        # 权重更新（包含修正项）
        # Weight update (with correction term)
        self.weights += self.alpha * (td_error + current_value - self.old_value) * self.traces
        self.weights -= self.alpha * (current_value - self.old_value) * features
        
        # 保存当前值
        # Save current value
        self.old_value = next_value if not done else 0.0
        self.old_features = features.copy()
        
        # 如果终止，重置
        # Reset if terminal
        if done:
            self.traces = np.zeros(self.n_features)
            self.old_value = 0.0
            self.old_features = np.zeros(self.n_features)
        
        self.update_count += 1
        
        return td_error


# ================================================================================
# 第12.5节：截断TD(λ) - TTD(λ)
# Section 12.5: Truncated TD(λ) - TTD(λ)
# ================================================================================

class TruncatedTDLambda:
    """
    截断TD(λ)
    Truncated TD(λ)
    
    使用有限步骤的资格迹
    Eligibility traces with finite steps
    
    动机 Motivation:
    - 计算效率
      Computational efficiency
    - 内存限制
      Memory constraints
    - 在线学习
      Online learning
    
    截断策略 Truncation Strategy:
    当迹衰减到阈值以下时截断
    Truncate when trace decays below threshold
    """
    
    def __init__(self,
                 n_features: int,
                 feature_extractor: Callable,
                 lambda_: float = 0.9,
                 alpha: float = 0.01,
                 gamma: float = 0.99,
                 trace_threshold: float = 0.01):
        """
        初始化截断TD(λ)
        Initialize Truncated TD(λ)
        
        Args:
            trace_threshold: 迹截断阈值
                           Trace truncation threshold
        """
        self.n_features = n_features
        self.feature_extractor = feature_extractor
        self.lambda_ = lambda_
        self.alpha = alpha
        self.gamma = gamma
        self.trace_threshold = trace_threshold
        
        # 权重
        # Weights
        self.weights = np.zeros(n_features)
        
        # 稀疏资格迹（只存储非零元素）
        # Sparse eligibility traces (store only non-zero elements)
        self.active_traces = {}  # index -> trace value
        
        # 统计
        # Statistics
        self.update_count = 0
        self.active_trace_count = []
        
        logger.info(f"初始化截断TD(λ): threshold={trace_threshold}")
    
    def get_value(self, state: Any) -> float:
        """获取状态价值"""
        features = self.feature_extractor(state)
        return np.dot(self.weights, features)
    
    def update_sparse_traces(self, features: np.ndarray):
        """
        更新稀疏资格迹
        Update sparse eligibility traces
        
        Args:
            features: 当前特征
                     Current features
        """
        # 衰减现有迹
        # Decay existing traces
        decay_factor = self.gamma * self.lambda_
        traces_to_remove = []
        
        for idx in list(self.active_traces.keys()):
            self.active_traces[idx] *= decay_factor
            
            # 如果低于阈值，移除
            # Remove if below threshold
            if abs(self.active_traces[idx]) < self.trace_threshold:
                traces_to_remove.append(idx)
        
        for idx in traces_to_remove:
            del self.active_traces[idx]
        
        # 添加新特征
        # Add new features
        for idx, value in enumerate(features):
            if value != 0:
                if idx in self.active_traces:
                    self.active_traces[idx] += value
                else:
                    self.active_traces[idx] = value
        
        # 记录活跃迹数量
        # Record number of active traces
        self.active_trace_count.append(len(self.active_traces))
    
    def update(self, state: Any, reward: float, next_state: Any, done: bool):
        """
        截断TD(λ)更新
        Truncated TD(λ) update
        """
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
        
        # 更新稀疏资格迹
        # Update sparse eligibility traces
        self.update_sparse_traces(features)
        
        # 更新权重（只更新活跃迹对应的权重）
        # Update weights (only for active traces)
        for idx, trace_value in self.active_traces.items():
            self.weights[idx] += self.alpha * td_error * trace_value
        
        # 如果终止，清空迹
        # Clear traces if terminal
        if done:
            self.active_traces.clear()
        
        self.update_count += 1
        
        return td_error
    
    def get_statistics(self) -> Dict[str, float]:
        """获取统计信息"""
        if len(self.active_trace_count) == 0:
            return {'active_traces': 0}
        
        return {
            'mean_active_traces': np.mean(self.active_trace_count[-100:]),
            'max_active_traces': np.max(self.active_trace_count[-100:]),
            'current_active_traces': len(self.active_traces),
            'weight_norm': np.linalg.norm(self.weights)
        }


# ================================================================================
# 第12.5.1节：变λ的TD算法
# Section 12.5.1: TD with Variable λ
# ================================================================================

class VariableLambdaTD:
    """
    变λ的TD算法
    TD with Variable λ
    
    λ可以随状态变化
    λ can vary with state
    
    λ(s)的选择 Choice of λ(s):
    - 不确定性高的状态使用大λ
      High λ for uncertain states
    - 确定性高的状态使用小λ
      Low λ for certain states
    
    应用 Applications:
    - 自适应学习
      Adaptive learning
    - 状态相关的偏差-方差权衡
      State-dependent bias-variance tradeoff
    """
    
    def __init__(self,
                 n_features: int,
                 feature_extractor: Callable,
                 lambda_function: Callable,
                 alpha: float = 0.01,
                 gamma: float = 0.99):
        """
        初始化变λTD
        Initialize Variable λ TD
        
        Args:
            lambda_function: λ(s)函数
                           λ(s) function
        """
        self.n_features = n_features
        self.feature_extractor = feature_extractor
        self.lambda_function = lambda_function
        self.alpha = alpha
        self.gamma = gamma
        
        # 权重和资格迹
        # Weights and eligibility traces
        self.weights = np.zeros(n_features)
        self.traces = np.zeros(n_features)
        
        # 当前λ值
        # Current λ value
        self.current_lambda = 0.9
        
        # 统计
        # Statistics
        self.update_count = 0
        self.lambda_history = []
        
        logger.info("初始化变λTD")
    
    def get_value(self, state: Any) -> float:
        """获取状态价值"""
        features = self.feature_extractor(state)
        return np.dot(self.weights, features)
    
    def update(self, state: Any, reward: float, next_state: Any, done: bool):
        """
        变λTD更新
        Variable λ TD update
        """
        # 获取当前状态的λ值
        # Get λ value for current state
        self.current_lambda = self.lambda_function(state)
        self.lambda_history.append(self.current_lambda)
        
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
        
        # 使用状态相关的λ更新资格迹
        # Update eligibility traces with state-dependent λ
        self.traces = self.gamma * self.current_lambda * self.traces + features
        
        # 更新权重
        # Update weights
        self.weights += self.alpha * td_error * self.traces
        
        # 如果终止，重置迹
        # Reset traces if terminal
        if done:
            self.traces = np.zeros(self.n_features)
        
        self.update_count += 1
        
        return td_error


# ================================================================================
# 主函数：演示TD(λ)算法
# Main Function: Demonstrate TD(λ) Algorithms
# ================================================================================

def demonstrate_td_lambda():
    """
    演示TD(λ)算法
    Demonstrate TD(λ) algorithms
    """
    print("\n" + "="*80)
    print("第12.3-12.5节：TD(λ)算法")
    print("Section 12.3-12.5: TD(λ) Algorithms")
    print("="*80)
    
    # 设置
    # Setup
    n_features = 10
    n_states = 5
    
    # 瓦片编码特征（稀疏特征）
    # Tile coding features (sparse features)
    def tile_features(state):
        if isinstance(state, int):
            features = np.zeros(n_features)
            # 每个状态激活2个特征
            # Each state activates 2 features
            features[state % n_features] = 1.0
            features[(state * 2) % n_features] = 1.0
            return features
        return np.random.randn(n_features) * 0.1
    
    # 1. 测试基础TD(λ)
    # 1. Test basic TD(λ)
    print("\n" + "="*60)
    print("1. 基础TD(λ)")
    print("1. Basic TD(λ)")
    print("="*60)
    
    # 累积迹
    # Accumulating traces
    td_lambda_acc = TDLambda(
        n_features=n_features,
        feature_extractor=tile_features,
        lambda_=0.9,
        alpha=0.05,
        gamma=0.95,
        trace_type='accumulating'
    )
    
    # 替换迹
    # Replacing traces
    td_lambda_rep = TDLambda(
        n_features=n_features,
        feature_extractor=tile_features,
        lambda_=0.9,
        alpha=0.05,
        gamma=0.95,
        trace_type='replacing'
    )
    
    print("\n训练两种迹类型...")
    
    # 模拟回合
    # Simulate episode
    for step in range(20):
        state = step % n_states
        reward = -1.0 if state != 2 else 5.0
        next_state = (step + 1) % n_states
        done = (step + 1) % 10 == 0
        
        td_lambda_acc.update(state, reward, next_state, done)
        td_lambda_rep.update(state, reward, next_state, done)
        
        if (step + 1) % 10 == 0:
            stats_acc = td_lambda_acc.get_statistics()
            stats_rep = td_lambda_rep.get_statistics()
            print(f"\n步{step + 1}:")
            print(f"  累积迹 - 平均迹大小: {stats_acc['mean_trace_magnitude']:.3f}")
            print(f"  替换迹 - 平均迹大小: {stats_rep['mean_trace_magnitude']:.3f}")
    
    print("\n价值估计比较:")
    print("状态  累积迹    替换迹")
    print("-" * 25)
    for state in range(n_states):
        v_acc = td_lambda_acc.get_value(state)
        v_rep = td_lambda_rep.get_value(state)
        print(f"{state:3d}  {v_acc:8.3f}  {v_rep:8.3f}")
    
    # 2. 测试真正的在线TD(λ)
    # 2. Test True Online TD(λ)
    print("\n" + "="*60)
    print("2. 真正的在线TD(λ)")
    print("2. True Online TD(λ)")
    print("="*60)
    
    true_online_td = TrueOnlineTDLambda(
        n_features=n_features,
        feature_extractor=tile_features,
        lambda_=0.9,
        alpha=0.05,
        gamma=0.95
    )
    
    print("\n训练真正的在线TD(λ)...")
    
    for step in range(30):
        state = step % n_states
        reward = -1.0 if state != 2 else 5.0
        next_state = (step + 1) % n_states
        done = (step + 1) % 10 == 0
        
        td_error = true_online_td.update(state, reward, next_state, done)
        
        if (step + 1) % 10 == 0:
            mean_error = np.mean(np.abs(true_online_td.td_errors[-10:]))
            print(f"  步{step + 1}: 平均|TD误差|={mean_error:.3f}")
    
    print("\n真正的在线TD(λ)价值估计:")
    for state in range(n_states):
        value = true_online_td.get_value(state)
        print(f"  V({state}) = {value:.3f}")
    
    # 3. 测试截断TD(λ)
    # 3. Test Truncated TD(λ)
    print("\n" + "="*60)
    print("3. 截断TD(λ)")
    print("3. Truncated TD(λ)")
    print("="*60)
    
    truncated_td = TruncatedTDLambda(
        n_features=n_features,
        feature_extractor=tile_features,
        lambda_=0.9,
        alpha=0.05,
        gamma=0.95,
        trace_threshold=0.01
    )
    
    print(f"\n使用截断阈值: {truncated_td.trace_threshold}")
    print("训练截断TD(λ)...")
    
    for step in range(50):
        state = step % n_states
        reward = -1.0 if state != 2 else 5.0
        next_state = (step + 1) % n_states
        done = (step + 1) % 10 == 0
        
        truncated_td.update(state, reward, next_state, done)
        
        if (step + 1) % 10 == 0:
            stats = truncated_td.get_statistics()
            print(f"  步{step + 1}: 活跃迹数={stats['current_active_traces']}, "
                  f"平均活跃迹={stats['mean_active_traces']:.1f}")
    
    print("\n截断TD(λ)价值估计:")
    for state in range(n_states):
        value = truncated_td.get_value(state)
        print(f"  V({state}) = {value:.3f}")
    
    # 4. 测试变λTD
    # 4. Test Variable λ TD
    print("\n" + "="*60)
    print("4. 变λTD")
    print("4. Variable λ TD")
    print("="*60)
    
    # λ函数：对不同状态使用不同的λ
    # λ function: different λ for different states
    def state_dependent_lambda(state):
        if isinstance(state, int):
            # 状态2使用高λ（更不确定）
            # State 2 uses high λ (more uncertain)
            if state % n_states == 2:
                return 0.95
            # 其他状态使用低λ
            # Other states use low λ
            else:
                return 0.5
        return 0.7
    
    variable_lambda_td = VariableLambdaTD(
        n_features=n_features,
        feature_extractor=tile_features,
        lambda_function=state_dependent_lambda,
        alpha=0.05,
        gamma=0.95
    )
    
    print("\n训练变λTD...")
    print("λ(s)设置: s=2时λ=0.95, 其他状态λ=0.5")
    
    for step in range(30):
        state = step % n_states
        reward = -1.0 if state != 2 else 5.0
        next_state = (step + 1) % n_states
        done = (step + 1) % 10 == 0
        
        td_error = variable_lambda_td.update(state, reward, next_state, done)
    
    print("\n不同状态的λ值统计:")
    lambda_by_state = {}
    for i, state_idx in enumerate([i % n_states for i in range(30)]):
        if state_idx not in lambda_by_state:
            lambda_by_state[state_idx] = []
        if i < len(variable_lambda_td.lambda_history):
            lambda_by_state[state_idx].append(variable_lambda_td.lambda_history[i])
    
    for state in range(n_states):
        if state in lambda_by_state:
            avg_lambda = np.mean(lambda_by_state[state])
            print(f"  状态{state}: 平均λ={avg_lambda:.3f}")
    
    # 5. 比较不同算法
    # 5. Compare different algorithms
    print("\n" + "="*60)
    print("5. 算法比较")
    print("5. Algorithm Comparison")
    print("="*60)
    
    print("\n各算法的价值估计:")
    print("状态  TD(λ)-累积  TD(λ)-替换  真正在线   截断TD    变λTD")
    print("-" * 60)
    
    for state in range(n_states):
        values = [
            td_lambda_acc.get_value(state),
            td_lambda_rep.get_value(state),
            true_online_td.get_value(state),
            truncated_td.get_value(state),
            variable_lambda_td.get_value(state)
        ]
        
        print(f"{state:3d}", end="")
        for v in values:
            print(f"  {v:10.3f}", end="")
        print()
    
    # 总结
    # Summary
    print("\n" + "="*80)
    print("TD(λ)算法总结")
    print("TD(λ) Algorithms Summary")
    print("="*80)
    
    print("""
    关键要点 Key Takeaways:
    =======================
    
    1. 资格迹实现后向视角
       Eligibility traces implement backward view
       
    2. 累积迹vs替换迹的权衡
       Tradeoff between accumulating vs replacing traces
       
    3. 真正的在线TD(λ)更准确
       True Online TD(λ) more accurate
       
    4. 截断提高计算效率
       Truncation improves efficiency
       
    5. 变λ允许状态相关调整
       Variable λ allows state-dependent tuning
    
    实践建议 Practical Advice:
    - 稀疏特征用替换迹
      Use replacing traces for sparse features
    - 稠密特征用累积迹
      Use accumulating traces for dense features
    - 大规模问题用截断
      Use truncation for large-scale problems
    - 考虑真正的在线版本
      Consider True Online version
    """)


if __name__ == "__main__":
    demonstrate_td_lambda()