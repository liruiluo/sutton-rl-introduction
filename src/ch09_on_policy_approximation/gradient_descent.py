"""
================================================================================
第9.2节：梯度下降 - 函数近似的基础
Section 9.2: Gradient Descent - Foundation of Function Approximation
================================================================================

从表格到函数近似的跳跃！
The leap from tabular to function approximation!

核心思想 Core Idea:
不再存储每个状态的值，而是学习参数化函数
Instead of storing values for each state, learn parameterized function

v̂(s,w) ≈ vπ(s)

其中 where:
- w ∈ ℝᵈ: 参数向量 (权重)
         Parameter vector (weights)
- v̂: 近似值函数
    Approximate value function

梯度下降更新 Gradient Descent Update:
w_{t+1} = w_t + α[v_π(S_t) - v̂(S_t,w_t)]∇v̂(S_t,w_t)

关键概念 Key Concepts:
1. 目标函数 (均方误差)
   Objective function (MSE)
   J(w) = E_π[(v_π(s) - v̂(s,w))²]

2. 梯度
   Gradient
   ∇J(w) = -2E_π[(v_π(s) - v̂(s,w))∇v̂(s,w)]

3. 随机梯度下降 (SGD)
   Stochastic Gradient Descent
   使用样本而非期望
   Use samples instead of expectation

挑战 Challenges:
- 函数近似能力
  Function approximation capacity
- 收敛性
  Convergence
- 偏差-方差权衡
  Bias-variance tradeoff
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
# 第9.2.1节：值函数近似器基类
# Section 9.2.1: Value Function Approximator Base Class
# ================================================================================

class ValueFunctionApproximator(ABC):
    """
    值函数近似器基类
    Value Function Approximator Base Class
    
    定义函数近似的接口
    Defines interface for function approximation
    
    关键方法 Key Methods:
    - predict: 预测值 v̂(s,w)
              Predict value
    - gradient: 计算梯度 ∇v̂(s,w)
               Compute gradient
    - update: 更新参数 w
             Update parameters
    """
    
    def __init__(self, n_features: int):
        """
        初始化近似器
        Initialize approximator
        
        Args:
            n_features: 特征维度
                       Feature dimension
        """
        self.n_features = n_features
        self.weights = np.zeros(n_features)
        
        # 统计
        # Statistics
        self.update_count = 0
        self.loss_history = []
        
        logger.info(f"初始化值函数近似器: d={n_features}")
    
    @abstractmethod
    def extract_features(self, state: Any) -> np.ndarray:
        """
        提取状态特征
        Extract state features
        
        Args:
            state: 原始状态
                  Raw state
        
        Returns:
            特征向量 x(s)
            Feature vector
        """
        pass
    
    def predict(self, state: Any) -> float:
        """
        预测状态价值
        Predict state value
        
        v̂(s,w) = w^T x(s)
        
        Args:
            state: 状态
                  State
        
        Returns:
            预测值
            Predicted value
        """
        features = self.extract_features(state)
        return np.dot(self.weights, features)
    
    def gradient(self, state: Any) -> np.ndarray:
        """
        计算梯度
        Compute gradient
        
        ∇v̂(s,w) = x(s) (线性情况)
        
        Args:
            state: 状态
                  State
        
        Returns:
            梯度向量
            Gradient vector
        """
        return self.extract_features(state)
    
    def update(self, state: Any, target: float, alpha: float):
        """
        更新参数
        Update parameters
        
        w ← w + α[target - v̂(s,w)]∇v̂(s,w)
        
        Args:
            state: 状态
                  State
            target: 目标值
                   Target value
            alpha: 学习率
                  Learning rate
        """
        prediction = self.predict(state)
        error = target - prediction
        grad = self.gradient(state)
        
        # 梯度更新
        # Gradient update
        self.weights += alpha * error * grad
        
        # 记录损失
        # Record loss
        loss = 0.5 * error**2  # MSE
        self.loss_history.append(loss)
        self.update_count += 1
    
    def get_weights(self) -> np.ndarray:
        """
        获取权重
        Get weights
        """
        return self.weights.copy()
    
    def set_weights(self, weights: np.ndarray):
        """
        设置权重
        Set weights
        """
        assert len(weights) == self.n_features
        self.weights = weights.copy()


# ================================================================================
# 第9.2.2节：梯度下降优化器
# Section 9.2.2: Gradient Descent Optimizer
# ================================================================================

class GradientDescent:
    """
    批量梯度下降
    Batch Gradient Descent
    
    使用所有数据计算梯度
    Use all data to compute gradient
    
    更新规则 Update Rule:
    w ← w - α∇J(w)
    
    其中 where:
    ∇J(w) = -1/N Σᵢ (yᵢ - ŷᵢ)∇ŷᵢ
    
    特点 Characteristics:
    - 确定性
      Deterministic
    - 收敛到局部最小值
      Converges to local minimum
    - 计算成本高
      High computational cost
    """
    
    def __init__(self, learning_rate: float = 0.01):
        """
        初始化批量梯度下降
        Initialize batch gradient descent
        
        Args:
            learning_rate: 学习率
                         Learning rate
        """
        self.learning_rate = learning_rate
        self.iteration = 0
        
        logger.info(f"初始化批量梯度下降: α={learning_rate}")
    
    def compute_gradient(self, 
                        approximator: ValueFunctionApproximator,
                        states: List[Any],
                        targets: List[float]) -> np.ndarray:
        """
        计算批量梯度
        Compute batch gradient
        
        Args:
            approximator: 近似器
                        Approximator
            states: 状态批量
                   State batch
            targets: 目标值批量
                    Target batch
        
        Returns:
            平均梯度
            Average gradient
        """
        total_gradient = np.zeros(approximator.n_features)
        
        for state, target in zip(states, targets):
            prediction = approximator.predict(state)
            error = target - prediction
            gradient = approximator.gradient(state)
            
            # 累积梯度
            # Accumulate gradient
            total_gradient += error * gradient
        
        # 平均梯度
        # Average gradient
        return total_gradient / len(states)
    
    def step(self,
            approximator: ValueFunctionApproximator,
            states: List[Any],
            targets: List[float]):
        """
        执行一步批量梯度下降
        Execute one batch gradient descent step
        
        Args:
            approximator: 近似器
                        Approximator
            states: 状态批量
                   State batch
            targets: 目标值批量
                    Target batch
        """
        # 计算梯度
        # Compute gradient
        gradient = self.compute_gradient(approximator, states, targets)
        
        # 更新权重
        # Update weights
        approximator.weights += self.learning_rate * gradient
        
        self.iteration += 1
    
    def optimize(self,
                approximator: ValueFunctionApproximator,
                states: List[Any],
                targets: List[float],
                n_epochs: int = 100,
                tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        优化到收敛
        Optimize to convergence
        
        Args:
            approximator: 近似器
                        Approximator
            states: 状态批量
                   State batch
            targets: 目标值批量
                    Target batch
            n_epochs: 最大迭代次数
                     Maximum iterations
            tolerance: 收敛容差
                     Convergence tolerance
        
        Returns:
            优化结果
            Optimization results
        """
        losses = []
        weights_history = []
        
        for epoch in range(n_epochs):
            # 保存旧权重
            # Save old weights
            old_weights = approximator.get_weights()
            
            # 梯度下降步
            # Gradient descent step
            self.step(approximator, states, targets)
            
            # 计算损失
            # Compute loss
            loss = 0.0
            for state, target in zip(states, targets):
                prediction = approximator.predict(state)
                loss += 0.5 * (target - prediction)**2
            loss /= len(states)
            
            losses.append(loss)
            weights_history.append(approximator.get_weights())
            
            # 检查收敛
            # Check convergence
            weight_change = np.linalg.norm(
                approximator.get_weights() - old_weights
            )
            
            if weight_change < tolerance:
                logger.info(f"批量梯度下降收敛于epoch {epoch+1}")
                break
        
        return {
            'losses': losses,
            'weights_history': weights_history,
            'final_weights': approximator.get_weights(),
            'n_iterations': epoch + 1
        }


# ================================================================================
# 第9.2.3节：随机梯度下降
# Section 9.2.3: Stochastic Gradient Descent
# ================================================================================

class StochasticGradientDescent:
    """
    随机梯度下降 (SGD)
    Stochastic Gradient Descent
    
    使用单个样本更新
    Update using single sample
    
    更新规则 Update Rule:
    w ← w + α[y - ŷ]∇ŷ
    
    优势 Advantages:
    - 在线学习
      Online learning
    - 计算效率高
      Computationally efficient
    - 能逃离局部最小值
      Can escape local minima
    
    劣势 Disadvantages:
    - 噪声大
      Noisy
    - 需要调整学习率
      Need to tune learning rate
    """
    
    def __init__(self, 
                learning_rate: float = 0.01,
                decay_rate: float = 0.999):
        """
        初始化SGD
        Initialize SGD
        
        Args:
            learning_rate: 初始学习率
                         Initial learning rate
            decay_rate: 学习率衰减
                       Learning rate decay
        """
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.iteration = 0
        
        logger.info(f"初始化SGD: α₀={learning_rate}, decay={decay_rate}")
    
    def get_learning_rate(self) -> float:
        """
        获取当前学习率
        Get current learning rate
        
        使用衰减 Using decay:
        α_t = α₀ * decay^t
        """
        return self.initial_learning_rate * (self.decay_rate ** self.iteration)
    
    def step(self,
            approximator: ValueFunctionApproximator,
            state: Any,
            target: float):
        """
        执行一步SGD
        Execute one SGD step
        
        Args:
            approximator: 近似器
                        Approximator
            state: 状态
                  State
            target: 目标值
                   Target value
        """
        # 获取当前学习率
        # Get current learning rate
        alpha = self.get_learning_rate()
        
        # 更新
        # Update
        approximator.update(state, target, alpha)
        
        # 更新学习率
        # Update learning rate
        self.learning_rate = alpha
        self.iteration += 1
    
    def train(self,
             approximator: ValueFunctionApproximator,
             states: List[Any],
             targets: List[float],
             n_epochs: int = 10,
             shuffle: bool = True) -> Dict[str, Any]:
        """
        训练多个epoch
        Train multiple epochs
        
        Args:
            approximator: 近似器
                        Approximator
            states: 状态列表
                   State list
            targets: 目标值列表
                    Target list
            n_epochs: 训练轮数
                     Training epochs
            shuffle: 是否打乱数据
                    Whether to shuffle data
        
        Returns:
            训练结果
            Training results
        """
        losses = []
        
        for epoch in range(n_epochs):
            # 打乱数据
            # Shuffle data
            if shuffle:
                indices = np.random.permutation(len(states))
            else:
                indices = np.arange(len(states))
            
            epoch_loss = 0.0
            
            # 遍历所有样本
            # Iterate through all samples
            for idx in indices:
                state = states[idx]
                target = targets[idx]
                
                # 计算损失
                # Compute loss
                prediction = approximator.predict(state)
                loss = 0.5 * (target - prediction)**2
                epoch_loss += loss
                
                # SGD步骤
                # SGD step
                self.step(approximator, state, target)
            
            epoch_loss /= len(states)
            losses.append(epoch_loss)
            
            if (epoch + 1) % max(1, n_epochs // 10) == 0:
                logger.info(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.6f}, α: {self.learning_rate:.6f}")
        
        return {
            'losses': losses,
            'final_weights': approximator.get_weights(),
            'final_learning_rate': self.learning_rate,
            'n_iterations': self.iteration
        }


# ================================================================================
# 第9.2.4节：小批量梯度下降
# Section 9.2.4: Mini-batch Gradient Descent
# ================================================================================

class MiniBatchGradientDescent:
    """
    小批量梯度下降
    Mini-batch Gradient Descent
    
    平衡批量和SGD
    Balance between batch and SGD
    
    使用小批量数据更新
    Update using mini-batch of data
    
    优势 Advantages:
    - 减少方差
      Reduced variance
    - 利用向量化
      Leverage vectorization
    - 稳定性好
      Good stability
    """
    
    def __init__(self,
                learning_rate: float = 0.01,
                batch_size: int = 32,
                momentum: float = 0.9):
        """
        初始化小批量梯度下降
        Initialize mini-batch gradient descent
        
        Args:
            learning_rate: 学习率
                         Learning rate
            batch_size: 批量大小
                       Batch size
            momentum: 动量系数
                     Momentum coefficient
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.momentum = momentum
        
        # 动量向量
        # Momentum vector
        self.velocity = None
        
        self.iteration = 0
        
        logger.info(f"初始化小批量GD: α={learning_rate}, batch={batch_size}, β={momentum}")
    
    def step(self,
            approximator: ValueFunctionApproximator,
            states: List[Any],
            targets: List[float]):
        """
        执行一步小批量梯度下降
        Execute one mini-batch gradient descent step
        
        使用动量 Using momentum:
        v_t = β*v_{t-1} + α*∇J
        w_t = w_{t-1} - v_t
        
        Args:
            approximator: 近似器
                        Approximator
            states: 状态批量
                   State batch
            targets: 目标值批量
                    Target batch
        """
        # 初始化动量
        # Initialize momentum
        if self.velocity is None:
            self.velocity = np.zeros(approximator.n_features)
        
        # 计算批量梯度
        # Compute batch gradient
        total_gradient = np.zeros(approximator.n_features)
        
        for state, target in zip(states, targets):
            prediction = approximator.predict(state)
            error = target - prediction
            gradient = approximator.gradient(state)
            total_gradient += error * gradient
        
        # 平均梯度
        # Average gradient
        avg_gradient = total_gradient / len(states)
        
        # 更新动量
        # Update momentum
        self.velocity = (self.momentum * self.velocity + 
                        self.learning_rate * avg_gradient)
        
        # 更新权重
        # Update weights
        approximator.weights += self.velocity
        
        self.iteration += 1
    
    def train(self,
             approximator: ValueFunctionApproximator,
             states: List[Any],
             targets: List[float],
             n_epochs: int = 10) -> Dict[str, Any]:
        """
        训练多个epoch
        Train multiple epochs
        
        Args:
            approximator: 近似器
                        Approximator
            states: 状态列表
                   State list
            targets: 目标值列表
                    Target list
            n_epochs: 训练轮数
                     Training epochs
        
        Returns:
            训练结果
            Training results
        """
        n_samples = len(states)
        losses = []
        
        for epoch in range(n_epochs):
            # 打乱数据
            # Shuffle data
            indices = np.random.permutation(n_samples)
            
            epoch_loss = 0.0
            n_batches = 0
            
            # 分批处理
            # Process in batches
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = [states[i] for i in batch_indices]
                batch_targets = [targets[i] for i in batch_indices]
                
                # 计算批量损失
                # Compute batch loss
                batch_loss = 0.0
                for state, target in zip(batch_states, batch_targets):
                    prediction = approximator.predict(state)
                    batch_loss += 0.5 * (target - prediction)**2
                batch_loss /= len(batch_states)
                epoch_loss += batch_loss
                
                # 小批量梯度下降步
                # Mini-batch gradient descent step
                self.step(approximator, batch_states, batch_targets)
                
                n_batches += 1
            
            epoch_loss /= n_batches
            losses.append(epoch_loss)
            
            if (epoch + 1) % max(1, n_epochs // 10) == 0:
                logger.info(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.6f}")
        
        return {
            'losses': losses,
            'final_weights': approximator.get_weights(),
            'n_iterations': self.iteration
        }


# ================================================================================
# 第9.2.5节：示例近似器
# Section 9.2.5: Example Approximator
# ================================================================================

class SimpleLinearApproximator(ValueFunctionApproximator):
    """
    简单线性近似器
    Simple Linear Approximator
    
    用于演示的具体实现
    Concrete implementation for demonstration
    
    v̂(s,w) = w^T x(s)
    """
    
    def __init__(self, n_features: int, feature_fn: Optional[Callable] = None):
        """
        初始化线性近似器
        Initialize linear approximator
        
        Args:
            n_features: 特征维度
                       Feature dimension
            feature_fn: 特征函数
                       Feature function
        """
        super().__init__(n_features)
        self.feature_fn = feature_fn or self.default_features
    
    def default_features(self, state: Any) -> np.ndarray:
        """
        默认特征函数
        Default feature function
        
        简单的独热编码或标准化
        Simple one-hot encoding or normalization
        """
        if isinstance(state, (int, np.integer)):
            # 独热编码
            # One-hot encoding
            features = np.zeros(self.n_features)
            if 0 <= state < self.n_features:
                features[state] = 1.0
            return features
        elif isinstance(state, (float, np.floating)):
            # 多项式特征
            # Polynomial features
            features = np.array([state**i for i in range(self.n_features)])
            return features
        else:
            # 假设state已经是特征向量
            # Assume state is already feature vector
            return np.array(state)[:self.n_features]
    
    def extract_features(self, state: Any) -> np.ndarray:
        """
        提取特征
        Extract features
        """
        return self.feature_fn(state)


# ================================================================================
# 主函数：演示梯度下降
# Main Function: Demonstrate Gradient Descent
# ================================================================================

def demonstrate_gradient_descent():
    """
    演示梯度下降方法
    Demonstrate gradient descent methods
    """
    print("\n" + "="*80)
    print("第9.2节：梯度下降")
    print("Section 9.2: Gradient Descent")
    print("="*80)
    
    # 创建示例数据
    # Create example data
    np.random.seed(42)
    
    # 生成简单的1D函数近似问题
    # Generate simple 1D function approximation problem
    n_samples = 100
    x = np.linspace(-1, 1, n_samples)
    true_function = lambda s: np.sin(3 * s) + 0.5 * s  # 真实函数
    y_true = [true_function(xi) for xi in x]
    
    # 添加噪声
    # Add noise
    noise = np.random.normal(0, 0.1, n_samples)
    y_noisy = [y + n for y, n in zip(y_true, noise)]
    
    print(f"\n生成{n_samples}个训练样本")
    print(f"Generated {n_samples} training samples")
    print(f"真实函数: sin(3x) + 0.5x + noise")
    
    # 1. 批量梯度下降
    # 1. Batch Gradient Descent
    print("\n" + "="*60)
    print("1. 批量梯度下降")
    print("1. Batch Gradient Descent")
    print("="*60)
    
    # 创建近似器（使用多项式特征）
    # Create approximator (using polynomial features)
    n_features = 10
    def poly_features(s):
        return np.array([s**i for i in range(n_features)])
    
    bgd_approx = SimpleLinearApproximator(n_features, poly_features)
    bgd = GradientDescent(learning_rate=0.1)
    
    print(f"\n使用{n_features}维多项式特征")
    
    # 优化
    # Optimize
    bgd_results = bgd.optimize(
        bgd_approx, x, y_noisy,
        n_epochs=100, tolerance=1e-6
    )
    
    print(f"收敛于{bgd_results['n_iterations']}次迭代")
    print(f"最终损失: {bgd_results['losses'][-1]:.6f}")
    
    # 2. 随机梯度下降
    # 2. Stochastic Gradient Descent
    print("\n" + "="*60)
    print("2. 随机梯度下降 (SGD)")
    print("2. Stochastic Gradient Descent")
    print("="*60)
    
    sgd_approx = SimpleLinearApproximator(n_features, poly_features)
    sgd = StochasticGradientDescent(learning_rate=0.1, decay_rate=0.995)
    
    sgd_results = sgd.train(
        sgd_approx, list(x), y_noisy,
        n_epochs=20, shuffle=True
    )
    
    print(f"最终损失: {sgd_results['losses'][-1]:.6f}")
    print(f"最终学习率: {sgd_results['final_learning_rate']:.6f}")
    
    # 3. 小批量梯度下降
    # 3. Mini-batch Gradient Descent
    print("\n" + "="*60)
    print("3. 小批量梯度下降")
    print("3. Mini-batch Gradient Descent")
    print("="*60)
    
    mbgd_approx = SimpleLinearApproximator(n_features, poly_features)
    mbgd = MiniBatchGradientDescent(
        learning_rate=0.1, batch_size=10, momentum=0.9
    )
    
    mbgd_results = mbgd.train(
        mbgd_approx, list(x), y_noisy,
        n_epochs=20
    )
    
    print(f"最终损失: {mbgd_results['losses'][-1]:.6f}")
    print(f"总迭代次数: {mbgd_results['n_iterations']}")
    
    # 4. 比较三种方法
    # 4. Compare three methods
    print("\n" + "="*60)
    print("4. 方法比较")
    print("4. Method Comparison")
    print("="*60)
    
    # 计算预测误差
    # Compute prediction errors
    bgd_mse = np.mean([(bgd_approx.predict(xi) - yi)**2 
                       for xi, yi in zip(x, y_true)])
    sgd_mse = np.mean([(sgd_approx.predict(xi) - yi)**2 
                       for xi, yi in zip(x, y_true)])
    mbgd_mse = np.mean([(mbgd_approx.predict(xi) - yi)**2 
                        for xi, yi in zip(x, y_true)])
    
    print(f"\n在真实函数上的MSE:")
    print(f"MSE on true function:")
    print(f"  批量GD: {bgd_mse:.6f}")
    print(f"  SGD: {sgd_mse:.6f}")
    print(f"  小批量GD: {mbgd_mse:.6f}")
    
    # 总结
    # Summary
    print("\n" + "="*80)
    print("梯度下降总结")
    print("Gradient Descent Summary")
    print("="*80)
    
    print("""
    关键要点 Key Takeaways:
    =======================
    
    1. 批量梯度下降：准确但慢
       Batch GD: Accurate but slow
       
    2. SGD：快但噪声大
       SGD: Fast but noisy
       
    3. 小批量：平衡折中
       Mini-batch: Balanced compromise
       
    4. 学习率很关键
       Learning rate is crucial
       
    5. 动量加速收敛
       Momentum accelerates convergence
    
    函数近似优势 Function Approximation Advantages:
    - 泛化到未见状态
      Generalize to unseen states
    - 内存效率
      Memory efficient
    - 连续空间
      Continuous spaces
    """)


if __name__ == "__main__":
    demonstrate_gradient_descent()