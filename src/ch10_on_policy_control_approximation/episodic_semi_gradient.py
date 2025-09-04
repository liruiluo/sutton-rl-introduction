"""
================================================================================
第10.1-10.2节：回合式半梯度控制
Section 10.1-10.2: Episodic Semi-gradient Control
================================================================================

将函数近似扩展到控制问题！
Extending function approximation to control!

核心思想 Core Idea:
近似动作价值函数而非状态价值函数
Approximate action-value function instead of state-value function

q̂(s,a,w) ≈ qπ(s,a)

半梯度Sarsa更新 Semi-gradient Sarsa Update:
w ← w + α[R + γq̂(S',A',w) - q̂(S,A,w)]∇q̂(S,A,w)

Mountain Car例子 Mountain Car Example:
- 连续状态空间
  Continuous state space
- 欠驱动控制
  Underactuated control
- 稀疏奖励
  Sparse rewards
- 需要动量积累
  Requires momentum buildup

瓦片编码的优势 Tile Coding Advantages:
- 线性复杂度
  Linear complexity
- 稀疏特征
  Sparse features
- 精确控制泛化
  Precise control of generalization
"""

import numpy as np
from typing import List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

# 导入基础组件
# Import base components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.ch03_finite_mdp.mdp_framework import State, Action, MDPEnvironment
from ch09_on_policy_approximation.feature_construction import TileCoding, Iht

# 设置日志
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第10.1.1节：半梯度Sarsa
# Section 10.1.1: Semi-gradient Sarsa
# ================================================================================

class SemiGradientSarsa:
    """
    半梯度Sarsa
    Semi-gradient Sarsa
    
    动作价值函数的TD(0)控制
    TD(0) control for action-value function
    
    关键点 Key Points:
    1. 近似q(s,a)而非v(s)
       Approximate q(s,a) not v(s)
    2. ε-贪婪探索
       ε-greedy exploration
    3. 线性函数近似
       Linear function approximation
    
    更新规则 Update Rule:
    δ = R + γq̂(S',A',w) - q̂(S,A,w)
    w ← w + αδ∇q̂(S,A,w)
    """
    
    def __init__(self,
                n_features: int,
                n_actions: int,
                alpha: float = 0.1,
                gamma: float = 1.0,
                epsilon: float = 0.1):
        """
        初始化半梯度Sarsa
        Initialize Semi-gradient Sarsa
        
        Args:
            n_features: 特征数
                       Number of features
            n_actions: 动作数
                      Number of actions
            alpha: 学习率
                  Learning rate
            gamma: 折扣因子
                  Discount factor
            epsilon: 探索率
                    Exploration rate
        """
        self.n_features = n_features
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # 权重向量 (每个动作一组权重)
        # Weight vectors (one for each action)
        self.weights = np.zeros((n_actions, n_features))
        
        # 统计
        # Statistics
        self.episode_count = 0
        self.step_count = 0
        self.td_errors = []
        
        logger.info(f"初始化半梯度Sarsa: features={n_features}, actions={n_actions}")
    
    def get_features(self, state: Any, action: int) -> np.ndarray:
        """
        获取状态-动作特征
        Get state-action features
        
        这里假设特征提取器已经处理了状态
        Assumes feature extractor already processed state
        
        Args:
            state: 状态特征
                  State features
            action: 动作索引
                   Action index
        
        Returns:
            特征向量
            Feature vector
        """
        # 简单实现：直接使用状态特征
        # Simple implementation: use state features directly
        if isinstance(state, np.ndarray):
            return state
        else:
            return np.array(state)
    
    def get_q_value(self, state: Any, action: int) -> float:
        """
        获取动作价值
        Get action value
        
        q̂(s,a,w) = w_a^T x(s)
        
        Args:
            state: 状态
                  State
            action: 动作
                   Action
        
        Returns:
            动作价值
            Action value
        """
        features = self.get_features(state, action)
        return np.dot(self.weights[action], features)
    
    def select_action(self, state: Any) -> int:
        """
        ε-贪婪动作选择
        ε-greedy action selection
        
        Args:
            state: 状态
                  State
        
        Returns:
            选择的动作
            Selected action
        """
        if np.random.random() < self.epsilon:
            # 探索：随机动作
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # 利用：贪婪动作
            # Exploit: greedy action
            q_values = [self.get_q_value(state, a) for a in range(self.n_actions)]
            
            # 打破平局
            # Break ties
            max_q = max(q_values)
            max_actions = [a for a, q in enumerate(q_values) if q == max_q]
            return np.random.choice(max_actions)
    
    def update(self, state: Any, action: int, reward: float,
              next_state: Any, next_action: int, done: bool):
        """
        Sarsa更新
        Sarsa update
        
        Args:
            state: 当前状态
                  Current state
            action: 当前动作
                   Current action
            reward: 奖励
                   Reward
            next_state: 下一状态
                       Next state
            next_action: 下一动作
                        Next action
            done: 是否结束
                 Whether done
        """
        # 获取特征
        # Get features
        features = self.get_features(state, action)
        
        # 计算TD目标
        # Compute TD target
        current_q = self.get_q_value(state, action)
        
        if done:
            td_target = reward
        else:
            next_q = self.get_q_value(next_state, next_action)
            td_target = reward + self.gamma * next_q
        
        # TD误差
        # TD error
        td_error = td_target - current_q
        self.td_errors.append(td_error)
        
        # 更新权重
        # Update weights
        self.weights[action] += self.alpha * td_error * features
        
        self.step_count += 1
    
    def learn_episode(self, env: Any, max_steps: int = 10000) -> Tuple[float, int]:
        """
        学习一个回合
        Learn one episode
        
        Args:
            env: 环境
                Environment
            max_steps: 最大步数
                      Maximum steps
        
        Returns:
            (回合回报, 步数)
            (episode return, steps)
        """
        state = env.reset()
        action = self.select_action(state)
        
        episode_return = 0.0
        steps = 0
        
        for _ in range(max_steps):
            # 执行动作
            # Execute action
            next_state, reward, done, _ = env.step(action)
            episode_return += reward
            
            if done:
                # 终止更新
                # Terminal update
                self.update(state, action, reward, next_state, 0, True)
                steps += 1
                break
            else:
                # 选择下一动作
                # Select next action
                next_action = self.select_action(next_state)
                
                # Sarsa更新
                # Sarsa update
                self.update(state, action, reward, next_state, next_action, False)
                
                state = next_state
                action = next_action
                steps += 1
        
        self.episode_count += 1
        return episode_return, steps


# ================================================================================
# 第10.1.2节：半梯度Expected Sarsa
# Section 10.1.2: Semi-gradient Expected Sarsa
# ================================================================================

class SemiGradientExpectedSarsa:
    """
    半梯度Expected Sarsa
    Semi-gradient Expected Sarsa
    
    使用期望而非采样的下一动作价值
    Use expected rather than sampled next action value
    
    更新规则 Update Rule:
    δ = R + γΣ_a π(a|S')q̂(S',a,w) - q̂(S,A,w)
    w ← w + αδ∇q̂(S,A,w)
    
    优势 Advantages:
    - 方差更小
      Lower variance
    - 通常学习更快
      Usually learns faster
    """
    
    def __init__(self,
                n_features: int,
                n_actions: int,
                alpha: float = 0.1,
                gamma: float = 1.0,
                epsilon: float = 0.1):
        """
        初始化Expected Sarsa
        Initialize Expected Sarsa
        """
        self.n_features = n_features
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # 权重向量
        # Weight vectors
        self.weights = np.zeros((n_actions, n_features))
        
        # 统计
        # Statistics
        self.episode_count = 0
        self.step_count = 0
        
        logger.info(f"初始化半梯度Expected Sarsa")
    
    def get_features(self, state: Any, action: int) -> np.ndarray:
        """获取特征"""
        if isinstance(state, np.ndarray):
            return state
        return np.array(state)
    
    def get_q_value(self, state: Any, action: int) -> float:
        """获取动作价值"""
        features = self.get_features(state, action)
        return np.dot(self.weights[action], features)
    
    def get_expected_value(self, state: Any) -> float:
        """
        计算期望价值
        Compute expected value
        
        E[Q(s,a)] = Σ_a π(a|s)q̂(s,a,w)
        
        对ε-贪婪策略:
        For ε-greedy policy:
        E[Q] = ε/|A| Σ_a Q(s,a) + (1-ε)max_a Q(s,a)
        """
        q_values = [self.get_q_value(state, a) for a in range(self.n_actions)]
        max_q = max(q_values)
        
        # ε-贪婪策略的期望
        # Expected value for ε-greedy policy
        expected = 0.0
        for q in q_values:
            if q == max_q:
                # 贪婪动作的概率
                # Probability of greedy action
                prob = (1 - self.epsilon) + self.epsilon / self.n_actions
            else:
                # 非贪婪动作的概率
                # Probability of non-greedy action
                prob = self.epsilon / self.n_actions
            expected += prob * q
        
        return expected
    
    def select_action(self, state: Any) -> int:
        """ε-贪婪动作选择"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            q_values = [self.get_q_value(state, a) for a in range(self.n_actions)]
            return np.argmax(q_values)
    
    def update(self, state: Any, action: int, reward: float,
              next_state: Any, done: bool):
        """
        Expected Sarsa更新
        Expected Sarsa update
        """
        features = self.get_features(state, action)
        current_q = self.get_q_value(state, action)
        
        if done:
            td_target = reward
        else:
            # 使用期望价值
            # Use expected value
            expected_next_q = self.get_expected_value(next_state)
            td_target = reward + self.gamma * expected_next_q
        
        td_error = td_target - current_q
        
        # 更新权重
        # Update weights
        self.weights[action] += self.alpha * td_error * features
        
        self.step_count += 1


# ================================================================================
# 第10.1.3节：半梯度n-step Sarsa
# Section 10.1.3: Semi-gradient n-step Sarsa
# ================================================================================

class SemiGradientNStepSarsa:
    """
    半梯度n-step Sarsa
    Semi-gradient n-step Sarsa
    
    n步回报的控制版本
    Control version of n-step returns
    
    G_t:t+n = R_t+1 + γR_t+2 + ... + γ^(n-1)R_t+n + γ^n q̂(S_t+n, A_t+n, w)
    
    优势 Advantages:
    - 更快传播信息
      Faster information propagation
    - 平衡偏差和方差
      Balance bias and variance
    """
    
    def __init__(self,
                n_features: int,
                n_actions: int,
                n: int = 4,
                alpha: float = 0.1,
                gamma: float = 1.0,
                epsilon: float = 0.1):
        """
        初始化n-step Sarsa
        Initialize n-step Sarsa
        
        Args:
            n: n步数
              n-step parameter
        """
        self.n_features = n_features
        self.n_actions = n_actions
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # 权重
        # Weights
        self.weights = np.zeros((n_actions, n_features))
        
        # n步缓冲
        # n-step buffer
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        
        logger.info(f"初始化{n}-step Sarsa")
    
    def get_features(self, state: Any, action: int) -> np.ndarray:
        """获取特征"""
        if isinstance(state, np.ndarray):
            return state
        return np.array(state)
    
    def get_q_value(self, state: Any, action: int) -> float:
        """获取动作价值"""
        features = self.get_features(state, action)
        return np.dot(self.weights[action], features)
    
    def select_action(self, state: Any) -> int:
        """ε-贪婪动作选择"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            q_values = [self.get_q_value(state, a) for a in range(self.n_actions)]
            return np.argmax(q_values)
    
    def compute_n_step_return(self, tau: int) -> float:
        """
        计算n步回报
        Compute n-step return
        
        Args:
            tau: 要更新的时间步
                Time step to update
        
        Returns:
            n步回报
            n-step return
        """
        n_steps = min(self.n, len(self.reward_buffer) - tau)
        G = 0.0
        
        # 累积奖励
        # Accumulate rewards
        for i in range(n_steps):
            G += (self.gamma ** i) * self.reward_buffer[tau + i]
        
        # 添加bootstrap值
        # Add bootstrap value
        if tau + n_steps < len(self.state_buffer):
            terminal_state = self.state_buffer[tau + n_steps]
            terminal_action = self.action_buffer[tau + n_steps]
            G += (self.gamma ** n_steps) * self.get_q_value(terminal_state, terminal_action)
        
        return G
    
    def update_at_tau(self, tau: int):
        """
        在时间步tau更新
        Update at time step tau
        
        Args:
            tau: 更新的时间步
                Time step to update
        """
        if tau < 0 or tau >= len(self.state_buffer) - self.n:
            return
        
        state = self.state_buffer[tau]
        action = self.action_buffer[tau]
        features = self.get_features(state, action)
        
        # n步回报
        # n-step return
        G = self.compute_n_step_return(tau)
        
        # 当前估计
        # Current estimate
        current_q = self.get_q_value(state, action)
        
        # TD误差
        # TD error
        td_error = G - current_q
        
        # 更新权重
        # Update weights
        self.weights[action] += self.alpha * td_error * features


# ================================================================================
# 第10.2节：Mountain Car瓦片编码
# Section 10.2: Mountain Car Tile Coding
# ================================================================================

@dataclass
class MountainCarState:
    """
    Mountain Car状态
    Mountain Car State
    
    位置和速度的连续状态
    Continuous state of position and velocity
    """
    position: float  # 范围 [-1.2, 0.6]
    velocity: float  # 范围 [-0.07, 0.07]
    
    @property
    def is_terminal(self) -> bool:
        """到达目标位置"""
        return self.position >= 0.5


class MountainCarTileCoding:
    """
    Mountain Car的瓦片编码
    Tile Coding for Mountain Car
    
    经典的连续控制问题
    Classic continuous control problem
    
    特点 Features:
    - 2D连续状态空间
      2D continuous state space
    - 稀疏奖励(-1每步)
      Sparse rewards (-1 per step)
    - 需要动量积累
      Requires momentum buildup
    
    瓦片编码设置 Tile Coding Setup:
    - 8个瓦片
      8 tilings
    - 每维8个tiles
      8 tiles per dimension
    - 2048大小的IHT
      IHT size of 2048
    """
    
    def __init__(self,
                n_tilings: int = 8,
                tiles_per_dim: int = 8,
                iht_size: int = 2048):
        """
        初始化Mountain Car瓦片编码
        Initialize Mountain Car tile coding
        
        Args:
            n_tilings: 瓦片数
                      Number of tilings
            tiles_per_dim: 每维瓦片数
                          Tiles per dimension
            iht_size: 索引哈希表大小
                     Index hash table size
        """
        self.n_tilings = n_tilings
        self.tiles_per_dim = tiles_per_dim
        
        # 状态空间边界
        # State space bounds
        self.position_bounds = (-1.2, 0.6)
        self.velocity_bounds = (-0.07, 0.07)
        bounds = [self.position_bounds, self.velocity_bounds]
        
        # 瓦片编码器
        # Tile coder
        self.tile_coder = TileCoding(
            n_tilings=n_tilings,
            bounds=bounds,
            n_tiles_per_dim=tiles_per_dim,
            iht_size=iht_size
        )
        
        # 动作空间: 后退(-1), 不动(0), 前进(1)
        # Action space: reverse(-1), neutral(0), forward(1)
        self.n_actions = 3
        
        logger.info(f"初始化Mountain Car瓦片编码: {n_tilings}x{tiles_per_dim}x{tiles_per_dim}")
    
    def get_features(self, state: MountainCarState) -> np.ndarray:
        """
        获取状态特征
        Get state features
        
        Args:
            state: Mountain Car状态
                  Mountain Car state
        
        Returns:
            瓦片编码特征
            Tile coding features
        """
        state_array = np.array([state.position, state.velocity])
        return self.tile_coder.transform(state_array)
    
    def get_active_tiles(self, state: MountainCarState) -> List[int]:
        """
        获取活跃瓦片索引
        Get active tile indices
        
        Args:
            state: Mountain Car状态
                  Mountain Car state
        
        Returns:
            活跃瓦片索引
            Active tile indices
        """
        state_array = np.array([state.position, state.velocity])
        return self.tile_coder.get_tiles(state_array)


# ================================================================================
# 主函数：演示回合式控制
# Main Function: Demonstrate Episodic Control
# ================================================================================

def demonstrate_episodic_control():
    """
    演示回合式半梯度控制
    Demonstrate episodic semi-gradient control
    """
    print("\n" + "="*80)
    print("第10.1-10.2节：回合式半梯度控制")
    print("Section 10.1-10.2: Episodic Semi-gradient Control")
    print("="*80)
    
    # 创建简单的测试环境
    # Create simple test environment
    n_features = 16
    n_actions = 4
    
    # 1. 测试半梯度Sarsa
    # 1. Test Semi-gradient Sarsa
    print("\n" + "="*60)
    print("1. 半梯度Sarsa")
    print("1. Semi-gradient Sarsa")
    print("="*60)
    
    sarsa = SemiGradientSarsa(
        n_features=n_features,
        n_actions=n_actions,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.1
    )
    
    # 模拟一些更新
    # Simulate some updates
    print("\n模拟学习步骤...")
    for step in range(10):
        state = np.random.randn(n_features)
        action = sarsa.select_action(state)
        reward = np.random.randn()
        next_state = np.random.randn(n_features)
        next_action = sarsa.select_action(next_state)
        done = step == 9
        
        sarsa.update(state, action, reward, next_state, next_action, done)
        
        if step % 3 == 0:
            q_value = sarsa.get_q_value(state, action)
            print(f"  步 {step+1}: Q值 = {q_value:.3f}")
    
    print(f"\n总更新步数: {sarsa.step_count}")
    
    # 2. 测试Expected Sarsa
    # 2. Test Expected Sarsa
    print("\n" + "="*60)
    print("2. 半梯度Expected Sarsa")
    print("2. Semi-gradient Expected Sarsa")
    print("="*60)
    
    expected_sarsa = SemiGradientExpectedSarsa(
        n_features=n_features,
        n_actions=n_actions,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.1
    )
    
    print("\n测试期望价值计算...")
    test_state = np.random.randn(n_features)
    
    # 设置一些Q值
    # Set some Q values
    for a in range(n_actions):
        expected_sarsa.weights[a] = np.random.randn(n_features) * 0.1
    
    expected_value = expected_sarsa.get_expected_value(test_state)
    print(f"  期望价值: {expected_value:.3f}")
    
    q_values = [expected_sarsa.get_q_value(test_state, a) for a in range(n_actions)]
    print(f"  Q值: {[f'{q:.3f}' for q in q_values]}")
    print(f"  最大Q值: {max(q_values):.3f}")
    
    # 3. 测试n-step Sarsa
    # 3. Test n-step Sarsa
    print("\n" + "="*60)
    print("3. 半梯度n-step Sarsa")
    print("3. Semi-gradient n-step Sarsa")
    print("="*60)
    
    n_step_sarsa = SemiGradientNStepSarsa(
        n_features=n_features,
        n_actions=n_actions,
        n=4,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.1
    )
    
    print(f"\n使用{n_step_sarsa.n}步回报")
    
    # 填充缓冲
    # Fill buffer
    print("填充n步缓冲...")
    for i in range(6):
        state = np.random.randn(n_features)
        action = n_step_sarsa.select_action(state)
        reward = -1.0  # 典型的步骤惩罚
        
        n_step_sarsa.state_buffer.append(state)
        n_step_sarsa.action_buffer.append(action)
        n_step_sarsa.reward_buffer.append(reward)
    
    # 计算n步回报
    # Compute n-step return
    if len(n_step_sarsa.reward_buffer) >= n_step_sarsa.n:
        G = n_step_sarsa.compute_n_step_return(0)
        print(f"  第0步的{n_step_sarsa.n}步回报: {G:.3f}")
    
    # 4. 测试Mountain Car瓦片编码
    # 4. Test Mountain Car Tile Coding
    print("\n" + "="*60)
    print("4. Mountain Car瓦片编码")
    print("4. Mountain Car Tile Coding")
    print("="*60)
    
    mc_coder = MountainCarTileCoding(
        n_tilings=8,
        tiles_per_dim=8,
        iht_size=512
    )
    
    # 测试几个状态
    # Test some states
    test_states = [
        MountainCarState(position=-0.5, velocity=0.0),  # 中间静止
        MountainCarState(position=0.4, velocity=0.05),   # 接近目标
        MountainCarState(position=-1.0, velocity=-0.05), # 左边界
    ]
    
    print("\n测试状态编码:")
    for i, state in enumerate(test_states):
        active_tiles = mc_coder.get_active_tiles(state)
        features = mc_coder.get_features(state)
        
        print(f"\n状态 {i+1}: pos={state.position:.2f}, vel={state.velocity:.3f}")
        print(f"  活跃瓦片: {active_tiles}")
        print(f"  特征稀疏度: {1 - np.sum(features > 0) / len(features):.1%}")
        print(f"  是否终止: {state.is_terminal}")
    
    # 总结
    # Summary
    print("\n" + "="*80)
    print("回合式控制总结")
    print("Episodic Control Summary")
    print("="*80)
    
    print("""
    关键要点 Key Takeaways:
    =======================
    
    1. 控制需要近似q(s,a)而非v(s)
       Control requires approximating q(s,a) not v(s)
       
    2. 半梯度方法仍然有效
       Semi-gradient methods still work
       
    3. Expected Sarsa通常更稳定
       Expected Sarsa usually more stable
       
    4. n-step方法加速学习
       n-step methods speed up learning
       
    5. 瓦片编码适合连续空间
       Tile coding good for continuous spaces
    
    Mountain Car挑战:
    - 稀疏奖励
      Sparse rewards
    - 需要探索
      Requires exploration
    - 动量积累
      Momentum buildup
    - 函数近似关键
      Function approximation crucial
    """)


if __name__ == "__main__":
    demonstrate_episodic_control()