"""
================================================================================
第9.6节：非线性函数近似 - 深度学习的力量
Section 9.6: Nonlinear Function Approximation - Power of Deep Learning
================================================================================

神经网络带来无限表达能力！
Neural networks bring unlimited expressiveness!

从线性到非线性 From Linear to Nonlinear:
v̂(s,w) = f(s; w)

其中f是神经网络 where f is neural network

关键挑战 Key Challenges:
1. 收敛性不再保证
   Convergence no longer guaranteed
2. 致命三要素 (Deadly Triad):
   - 函数近似
     Function approximation
   - 自举 (Bootstrapping)
   - 离策略
     Off-policy
3. 需要经验回放
   Need experience replay
4. 需要目标网络
   Need target network

深度Q网络 (DQN) 创新:
1. 经验回放打破相关性
   Experience replay breaks correlation
2. 目标网络稳定训练
   Target network stabilizes training
3. 梯度裁剪防止爆炸
   Gradient clipping prevents explosion
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import logging
import random

# 设置日志
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第9.6.1节：简单神经网络
# Section 9.6.1: Simple Neural Network
# ================================================================================

class NeuralNetwork:
    """
    简单前馈神经网络
    Simple Feedforward Neural Network
    
    两层网络用于值函数近似
    Two-layer network for value function approximation
    
    架构 Architecture:
    输入 -> 隐藏层(ReLU) -> 输出
    Input -> Hidden(ReLU) -> Output
    
    前向传播 Forward:
    h = ReLU(W₁x + b₁)
    y = W₂h + b₂
    
    反向传播 Backward:
    梯度下降更新权重
    Gradient descent to update weights
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1,
                learning_rate: float = 0.001):
        """
        初始化神经网络
        Initialize neural network
        
        Args:
            input_size: 输入维度
                       Input dimension
            hidden_size: 隐藏层大小
                        Hidden layer size
            output_size: 输出维度(值函数为1)
                        Output dimension (1 for value)
            learning_rate: 学习率
                         Learning rate
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # 初始化权重 (Xavier初始化)
        # Initialize weights (Xavier initialization)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)
        
        # 缓存前向传播值（用于反向传播）
        # Cache forward values (for backprop)
        self.cache = {}
        
        # 统计
        # Statistics
        self.update_count = 0
        self.loss_history = []
        
        logger.info(f"初始化神经网络: {input_size}-{hidden_size}-{output_size}")
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """ReLU导数"""
        return (x > 0).astype(float)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        Forward propagation
        
        Args:
            x: 输入 (batch_size, input_size)
              Input
        
        Returns:
            输出 (batch_size, output_size)
            Output
        """
        # 确保输入是2D
        # Ensure input is 2D
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # 第一层
        # First layer
        z1 = x @ self.W1 + self.b1
        h1 = self.relu(z1)
        
        # 第二层
        # Second layer
        z2 = h1 @ self.W2 + self.b2
        
        # 缓存中间值
        # Cache intermediate values
        self.cache = {
            'x': x,
            'z1': z1,
            'h1': h1,
            'z2': z2
        }
        
        return z2
    
    def backward(self, grad_output: np.ndarray) -> Dict[str, np.ndarray]:
        """
        反向传播
        Backward propagation
        
        Args:
            grad_output: 输出梯度 (batch_size, output_size)
                        Output gradient
        
        Returns:
            参数梯度
            Parameter gradients
        """
        batch_size = self.cache['x'].shape[0]
        
        # 输出层梯度
        # Output layer gradients
        grad_W2 = self.cache['h1'].T @ grad_output / batch_size
        grad_b2 = np.sum(grad_output, axis=0) / batch_size
        
        # 隐藏层梯度
        # Hidden layer gradients
        grad_h1 = grad_output @ self.W2.T
        grad_z1 = grad_h1 * self.relu_derivative(self.cache['z1'])
        
        grad_W1 = self.cache['x'].T @ grad_z1 / batch_size
        grad_b1 = np.sum(grad_z1, axis=0) / batch_size
        
        return {
            'W1': grad_W1,
            'b1': grad_b1,
            'W2': grad_W2,
            'b2': grad_b2
        }
    
    def update(self, gradients: Dict[str, np.ndarray], clip_norm: float = 5.0):
        """
        更新参数
        Update parameters
        
        Args:
            gradients: 参数梯度
                     Parameter gradients
            clip_norm: 梯度裁剪范数
                      Gradient clipping norm
        """
        # 梯度裁剪
        # Gradient clipping
        total_norm = 0
        for key in ['W1', 'b1', 'W2', 'b2']:
            total_norm += np.sum(gradients[key] ** 2)
        total_norm = np.sqrt(total_norm)
        
        if total_norm > clip_norm:
            clip_ratio = clip_norm / total_norm
            for key in gradients:
                gradients[key] *= clip_ratio
        
        # 梯度下降更新
        # Gradient descent update
        self.W1 -= self.learning_rate * gradients['W1']
        self.b1 -= self.learning_rate * gradients['b1']
        self.W2 -= self.learning_rate * gradients['W2']
        self.b2 -= self.learning_rate * gradients['b2']
        
        self.update_count += 1
    
    def predict(self, x: np.ndarray) -> float:
        """
        预测单个值
        Predict single value
        
        Args:
            x: 输入
              Input
        
        Returns:
            预测值
            Predicted value
        """
        output = self.forward(x)
        return output.item() if output.size == 1 else output.squeeze()
    
    def train_step(self, x: np.ndarray, target: float) -> float:
        """
        训练一步
        Train one step
        
        Args:
            x: 输入
              Input
            target: 目标值
                   Target value
        
        Returns:
            损失
            Loss
        """
        # 前向传播
        # Forward pass
        prediction = self.forward(x)
        
        # 计算损失 (MSE)
        # Compute loss
        loss = 0.5 * np.mean((prediction - target) ** 2)
        self.loss_history.append(loss)
        
        # 反向传播
        # Backward pass
        grad_output = prediction - target
        gradients = self.backward(grad_output)
        
        # 更新参数
        # Update parameters
        self.update(gradients)
        
        return loss
    
    def copy_weights_from(self, other: 'NeuralNetwork'):
        """
        复制另一个网络的权重
        Copy weights from another network
        
        Args:
            other: 源网络
                  Source network
        """
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()


# ================================================================================
# 第9.6.2节：经验回放缓冲
# Section 9.6.2: Experience Replay Buffer
# ================================================================================

@dataclass
class Experience:
    """
    经验样本
    Experience sample
    """
    state: Any
    action: Optional[Any]
    reward: float
    next_state: Any
    done: bool


class ReplayBuffer:
    """
    经验回放缓冲
    Experience Replay Buffer
    
    DQN的关键组件！
    Key component of DQN!
    
    作用 Purpose:
    1. 打破样本相关性
       Break sample correlation
    2. 提高样本效率
       Improve sample efficiency
    3. 稳定训练
       Stabilize training
    """
    
    def __init__(self, capacity: int = 10000):
        """
        初始化回放缓冲
        Initialize replay buffer
        
        Args:
            capacity: 缓冲容量
                     Buffer capacity
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
        logger.info(f"初始化回放缓冲: capacity={capacity}")
    
    def push(self, experience: Experience):
        """
        添加经验
        Add experience
        
        Args:
            experience: 经验样本
                       Experience sample
        """
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """
        随机采样
        Random sample
        
        Args:
            batch_size: 批量大小
                       Batch size
        
        Returns:
            经验批量
            Experience batch
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        """缓冲大小"""
        return len(self.buffer)
    
    def is_ready(self, min_size: int) -> bool:
        """
        检查是否准备好训练
        Check if ready for training
        
        Args:
            min_size: 最小缓冲大小
                     Minimum buffer size
        
        Returns:
            是否准备好
            Whether ready
        """
        return len(self.buffer) >= min_size


# ================================================================================
# 第9.6.3节：深度Q网络 (简化版)
# Section 9.6.3: Deep Q-Network (Simplified)
# ================================================================================

class DeepQNetwork:
    """
    深度Q网络 (DQN)
    Deep Q-Network
    
    Mnih et al. 2015的突破！
    Breakthrough of Mnih et al. 2015!
    
    关键创新 Key Innovations:
    1. 经验回放
       Experience replay
    2. 目标网络
       Target network
    3. 梯度裁剪
       Gradient clipping
    
    这里是简化版，用于教学
    Simplified version for teaching
    """
    
    def __init__(self,
                state_size: int,
                action_size: int,
                hidden_size: int = 128,
                learning_rate: float = 0.001,
                gamma: float = 0.99,
                buffer_size: int = 10000,
                batch_size: int = 32,
                target_update_freq: int = 100):
        """
        初始化DQN
        Initialize DQN
        
        Args:
            state_size: 状态维度
                       State dimension
            action_size: 动作数
                        Action count
            hidden_size: 隐藏层大小
                        Hidden layer size
            learning_rate: 学习率
                         Learning rate
            gamma: 折扣因子
                  Discount factor
            buffer_size: 缓冲大小
                        Buffer size
            batch_size: 批量大小
                       Batch size
            target_update_freq: 目标网络更新频率
                              Target network update frequency
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Q网络和目标网络
        # Q-network and target network
        self.q_network = NeuralNetwork(
            state_size, hidden_size, action_size, learning_rate
        )
        self.target_network = NeuralNetwork(
            state_size, hidden_size, action_size, learning_rate
        )
        
        # 初始同步
        # Initial sync
        self.target_network.copy_weights_from(self.q_network)
        
        # 经验回放缓冲
        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # 统计
        # Statistics
        self.update_count = 0
        self.loss_history = []
        
        logger.info(f"初始化DQN: state={state_size}, action={action_size}")
    
    def get_q_values(self, state: np.ndarray, use_target: bool = False) -> np.ndarray:
        """
        获取Q值
        Get Q-values
        
        Args:
            state: 状态
                  State
            use_target: 是否使用目标网络
                       Whether to use target network
        
        Returns:
            Q值向量
            Q-value vector
        """
        network = self.target_network if use_target else self.q_network
        return network.forward(state).squeeze()
    
    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """
        ε-贪婪选择动作
        ε-greedy action selection
        
        Args:
            state: 状态
                  State
            epsilon: 探索率
                    Exploration rate
        
        Returns:
            动作索引
            Action index
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.action_size)
        else:
            q_values = self.get_q_values(state)
            return np.argmax(q_values)
    
    def store_experience(self, state: Any, action: int, reward: float,
                        next_state: Any, done: bool):
        """
        存储经验
        Store experience
        
        Args:
            state: 当前状态
                  Current state
            action: 动作
                   Action
            reward: 奖励
                   Reward
            next_state: 下一状态
                       Next state
            done: 是否结束
                 Whether done
        """
        experience = Experience(state, action, reward, next_state, done)
        self.replay_buffer.push(experience)
    
    def train_batch(self) -> float:
        """
        训练一个批量
        Train one batch
        
        Returns:
            平均损失
            Average loss
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return 0.0
        
        # 采样批量
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        
        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch])
        
        # 计算当前Q值
        # Compute current Q-values
        current_q = self.q_network.forward(states)
        current_q_selected = current_q[np.arange(self.batch_size), actions]
        
        # 计算目标Q值 (使用目标网络)
        # Compute target Q-values (using target network)
        next_q = self.target_network.forward(next_states)
        max_next_q = np.max(next_q, axis=1)
        
        # TD目标
        # TD targets
        targets = rewards + self.gamma * max_next_q * (1 - dones)
        
        # 计算损失
        # Compute loss
        loss = 0.5 * np.mean((current_q_selected - targets) ** 2)
        self.loss_history.append(loss)
        
        # 构建梯度
        # Construct gradients
        grad_output = np.zeros_like(current_q)
        grad_output[np.arange(self.batch_size), actions] = current_q_selected - targets
        
        # 反向传播
        # Backpropagation
        gradients = self.q_network.backward(grad_output)
        
        # 更新Q网络
        # Update Q-network
        self.q_network.update(gradients)
        
        # 更新计数
        # Update count
        self.update_count += 1
        
        # 定期更新目标网络
        # Periodically update target network
        if self.update_count % self.target_update_freq == 0:
            self.target_network.copy_weights_from(self.q_network)
            logger.info(f"更新目标网络 (step {self.update_count})")
        
        return loss


# ================================================================================
# 第9.6.4节：梯度TD神经网络
# Section 9.6.4: Gradient TD Neural Network
# ================================================================================

class GradientTDNN:
    """
    使用神经网络的梯度TD
    Gradient TD with Neural Network
    
    非线性半梯度TD(0)
    Nonlinear Semi-gradient TD(0)
    
    警告 Warning:
    收敛性不保证！
    Convergence not guaranteed!
    
    需要小心调参
    Requires careful tuning
    """
    
    def __init__(self,
                state_size: int,
                hidden_size: int = 64,
                learning_rate: float = 0.001,
                gamma: float = 0.99):
        """
        初始化梯度TD神经网络
        Initialize Gradient TD Neural Network
        
        Args:
            state_size: 状态维度
                       State dimension
            hidden_size: 隐藏层大小
                        Hidden layer size
            learning_rate: 学习率
                         Learning rate
            gamma: 折扣因子
                  Discount factor
        """
        self.gamma = gamma
        
        # 值函数网络
        # Value function network
        self.value_network = NeuralNetwork(
            state_size, hidden_size, 1, learning_rate
        )
        
        # 统计
        # Statistics
        self.td_errors = []
        self.update_count = 0
        
        logger.info(f"初始化梯度TD神经网络: γ={gamma}")
    
    def predict(self, state: np.ndarray) -> float:
        """
        预测状态价值
        Predict state value
        
        Args:
            state: 状态
                  State
        
        Returns:
            v̂(s,w)
        """
        return self.value_network.predict(state)
    
    def update(self, state: np.ndarray, reward: float, 
              next_state: np.ndarray, done: bool) -> float:
        """
        TD更新
        TD update
        
        Args:
            state: 当前状态
                  Current state
            reward: 奖励
                   Reward
            next_state: 下一状态
                       Next state
            done: 是否结束
                 Whether done
        
        Returns:
            TD误差
            TD error
        """
        # 计算TD目标
        # Compute TD target
        if done:
            td_target = reward
        else:
            next_value = self.predict(next_state)
            td_target = reward + self.gamma * next_value
        
        # 训练网络
        # Train network
        loss = self.value_network.train_step(state, td_target)
        
        # 计算TD误差
        # Compute TD error
        current_value = self.predict(state)
        td_error = td_target - current_value
        
        self.td_errors.append(td_error)
        self.update_count += 1
        
        return td_error


# ================================================================================
# 主函数：演示神经网络近似
# Main Function: Demonstrate Neural Approximation
# ================================================================================

def demonstrate_neural_approximation():
    """
    演示神经网络函数近似
    Demonstrate neural network function approximation
    """
    print("\n" + "="*80)
    print("第9.6节：神经网络函数近似")
    print("Section 9.6: Neural Network Function Approximation")
    print("="*80)
    
    # 1. 测试简单神经网络
    # 1. Test simple neural network
    print("\n" + "="*60)
    print("1. 简单神经网络测试")
    print("1. Simple Neural Network Test")
    print("="*60)
    
    # 创建XOR问题
    # Create XOR problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    print("\nXOR问题:")
    for i in range(len(X)):
        print(f"  输入: {X[i]}, 目标: {y[i][0]}")
    
    # 创建网络
    # Create network
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1,
                      learning_rate=0.1)
    
    # 训练
    # Train
    print("\n训练神经网络...")
    n_epochs = 1000
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        for i in range(len(X)):
            loss = nn.train_step(X[i], y[i])
            epoch_loss += loss
        
        if (epoch + 1) % (n_epochs // 10) == 0:
            print(f"  Epoch {epoch+1}: Loss = {epoch_loss/len(X):.4f}")
    
    # 测试
    # Test
    print("\n测试结果:")
    for i in range(len(X)):
        prediction = nn.predict(X[i])
        print(f"  输入: {X[i]}, 预测: {prediction:.3f}, 目标: {y[i][0]}")
    
    # 2. 测试经验回放
    # 2. Test experience replay
    print("\n" + "="*60)
    print("2. 经验回放缓冲测试")
    print("2. Experience Replay Buffer Test")
    print("="*60)
    
    buffer = ReplayBuffer(capacity=100)
    
    # 添加一些经验
    # Add some experiences
    print("\n添加经验样本...")
    for i in range(50):
        exp = Experience(
            state=np.random.randn(4),
            action=np.random.randint(2),
            reward=np.random.randn(),
            next_state=np.random.randn(4),
            done=np.random.random() > 0.9
        )
        buffer.push(exp)
    
    print(f"  缓冲大小: {len(buffer)}")
    
    # 采样批量
    # Sample batch
    batch = buffer.sample(10)
    print(f"  采样10个样本")
    print(f"  第一个样本奖励: {batch[0].reward:.3f}")
    
    # 3. 测试简化DQN
    # 3. Test simplified DQN
    print("\n" + "="*60)
    print("3. 简化DQN测试")
    print("3. Simplified DQN Test")
    print("="*60)
    
    # 创建DQN
    # Create DQN
    dqn = DeepQNetwork(
        state_size=4,
        action_size=2,
        hidden_size=32,
        learning_rate=0.001,
        buffer_size=1000,
        batch_size=16
    )
    
    print("\n生成随机经验...")
    # 生成一些随机经验
    # Generate random experiences
    for _ in range(100):
        state = np.random.randn(4)
        action = dqn.select_action(state, epsilon=0.5)
        reward = np.random.randn()
        next_state = np.random.randn(4)
        done = np.random.random() > 0.9
        
        dqn.store_experience(state, action, reward, next_state, done)
    
    print(f"  缓冲大小: {len(dqn.replay_buffer)}")
    
    # 训练几个批次
    # Train a few batches
    print("\n训练DQN...")
    for i in range(10):
        loss = dqn.train_batch()
        if i % 2 == 0:
            print(f"  批次 {i+1}: Loss = {loss:.4f}")
    
    # 4. 测试梯度TD神经网络
    # 4. Test Gradient TD Neural Network
    print("\n" + "="*60)
    print("4. 梯度TD神经网络")
    print("4. Gradient TD Neural Network")
    print("="*60)
    
    gtd_nn = GradientTDNN(state_size=4, hidden_size=32)
    
    print("\n模拟TD学习...")
    for step in range(50):
        state = np.random.randn(4)
        reward = np.random.randn()
        next_state = np.random.randn(4)
        done = np.random.random() > 0.9
        
        td_error = gtd_nn.update(state, reward, next_state, done)
        
        if (step + 1) % 10 == 0:
            avg_td_error = np.mean(np.abs(gtd_nn.td_errors[-10:]))
            print(f"  步 {step+1}: 平均|TD误差| = {avg_td_error:.4f}")
    
    # 总结
    # Summary
    print("\n" + "="*80)
    print("神经网络近似总结")
    print("Neural Network Approximation Summary")
    print("="*80)
    
    print("""
    关键要点 Key Takeaways:
    =======================
    
    1. 神经网络提供强大表达能力
       Neural networks provide powerful expressiveness
       
    2. 收敛性不再保证
       Convergence no longer guaranteed
       
    3. DQN的关键创新：
       DQN's key innovations:
       - 经验回放 Experience replay
       - 目标网络 Target network
       - 梯度裁剪 Gradient clipping
       
    4. 需要仔细调参
       Requires careful tuning
       
    5. 致命三要素要小心
       Beware of deadly triad
    
    实践建议 Practical Tips:
    - 使用小学习率
      Use small learning rate
    - 监控TD误差
      Monitor TD errors
    - 使用梯度裁剪
      Use gradient clipping
    - 考虑双网络
      Consider double networks
    """)


if __name__ == "__main__":
    demonstrate_neural_approximation()