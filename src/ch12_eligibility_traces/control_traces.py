"""
================================================================================
第12.6-12.8节：控制算法的资格迹
Section 12.6-12.8: Eligibility Traces for Control
================================================================================

将资格迹扩展到控制！
Extending eligibility traces to control!

Sarsa(λ):
- 同策略控制的资格迹版本
  On-policy control with eligibility traces
- 更快的学习速度
  Faster learning

Q(λ):
- Q-learning的资格迹版本
  Q-learning with eligibility traces  
- 需要特殊处理（Watkins's Q(λ), Peng's Q(λ)）
  Requires special handling

关键挑战 Key Challenge:
离策略时资格迹的截断
Trace cutoff for off-policy
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
# 第12.6节：Sarsa(λ)
# Section 12.6: Sarsa(λ)
# ================================================================================

class SarsaLambda:
    """
    Sarsa(λ)算法
    Sarsa(λ) Algorithm
    
    同策略TD控制的资格迹版本
    Eligibility trace version of on-policy TD control
    
    更新规则 Update Rule:
    δ_t = R_{t+1} + γQ(S_{t+1},A_{t+1}) - Q(S_t,A_t)
    e_t = γλe_{t-1} + ∇q̂(S_t,A_t,w_t)
    w_{t+1} = w_t + αδ_t e_t
    
    优势 Advantages:
    - 更快的信用分配
      Faster credit assignment
    - 更好的样本效率
      Better sample efficiency
    """
    
    def __init__(self,
                 n_features: int,
                 n_actions: int,
                 feature_extractor: Callable,
                 lambda_: float = 0.9,
                 alpha: float = 0.01,
                 gamma: float = 0.99,
                 epsilon: float = 0.1,
                 trace_type: str = 'accumulating'):
        """
        初始化Sarsa(λ)
        Initialize Sarsa(λ)
        
        Args:
            n_features: 特征维度
                       Feature dimension
            n_actions: 动作数
                      Number of actions
            feature_extractor: 特征提取器(state, action) -> features
                             Feature extractor
            lambda_: λ参数
                    λ parameter
            alpha: 学习率
                  Learning rate
            gamma: 折扣因子
                  Discount factor
            epsilon: 探索率
                    Exploration rate
            trace_type: 迹类型
                       Trace type
        """
        self.n_features = n_features
        self.n_actions = n_actions
        self.feature_extractor = feature_extractor
        self.lambda_ = lambda_
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.trace_type = trace_type
        
        # 权重和资格迹（每个动作一组）
        # Weights and eligibility traces (one set per action)
        self.weights = np.zeros((n_actions, n_features))
        self.traces = np.zeros((n_actions, n_features))
        
        # 统计
        # Statistics
        self.update_count = 0
        self.episode_count = 0
        self.td_errors = []
        self.episode_returns = []
        
        logger.info(f"初始化Sarsa(λ): λ={lambda_}, ε={epsilon}")
    
    def get_q_value(self, state: Any, action: int) -> float:
        """
        获取动作价值
        Get action value
        
        q̂(s,a,w) = w_a^T φ(s,a)
        """
        features = self.feature_extractor(state, action)
        return np.dot(self.weights[action], features)
    
    def get_all_q_values(self, state: Any) -> np.ndarray:
        """
        获取所有动作的Q值
        Get Q values for all actions
        """
        q_values = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            q_values[a] = self.get_q_value(state, a)
        return q_values
    
    def select_action(self, state: Any, greedy: bool = False) -> int:
        """
        ε-贪婪动作选择
        ε-greedy action selection
        
        Args:
            state: 状态
                  State
            greedy: 是否强制贪婪
                   Whether to force greedy
        
        Returns:
            选择的动作
            Selected action
        """
        if not greedy and np.random.random() < self.epsilon:
            # 探索
            # Explore
            return np.random.randint(self.n_actions)
        else:
            # 利用
            # Exploit
            q_values = self.get_all_q_values(state)
            # 随机打破平局
            # Random tie-breaking
            max_q = np.max(q_values)
            max_actions = np.where(q_values == max_q)[0]
            return np.random.choice(max_actions)
    
    def update_traces(self, state: Any, action: int):
        """
        更新资格迹
        Update eligibility traces
        
        Args:
            state: 状态
                  State
            action: 动作
                   Action
        """
        features = self.feature_extractor(state, action)
        
        if self.trace_type == 'accumulating':
            # 累积迹
            # Accumulating traces
            # 衰减所有迹
            # Decay all traces
            self.traces *= self.gamma * self.lambda_
            # 累积当前状态-动作的迹
            # Accumulate trace for current state-action
            self.traces[action] += features
            
        elif self.trace_type == 'replacing':
            # 替换迹
            # Replacing traces
            # 衰减所有迹
            # Decay all traces
            self.traces *= self.gamma * self.lambda_
            # 替换当前状态-动作的迹
            # Replace trace for current state-action
            active_indices = np.where(features > 0)[0]
            self.traces[action][active_indices] = features[active_indices]
    
    def update(self, state: Any, action: int, reward: float,
               next_state: Any, next_action: int, done: bool):
        """
        Sarsa(λ)更新
        Sarsa(λ) update
        
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
            done: 是否终止
                 Whether done
        """
        # 计算TD误差
        # Compute TD error
        current_q = self.get_q_value(state, action)
        if done:
            td_target = reward
        else:
            next_q = self.get_q_value(next_state, next_action)
            td_target = reward + self.gamma * next_q
        
        td_error = td_target - current_q
        self.td_errors.append(td_error)
        
        # 更新资格迹
        # Update eligibility traces
        self.update_traces(state, action)
        
        # 更新所有权重
        # Update all weights
        self.weights += self.alpha * td_error * self.traces
        
        # 如果终止，重置资格迹
        # Reset traces if terminal
        if done:
            self.traces = np.zeros((self.n_actions, self.n_features))
        
        self.update_count += 1
        
        return td_error
    
    def learn_episode(self, env: Any, max_steps: int = 1000) -> Tuple[float, int]:
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
        
        for step in range(max_steps):
            # 执行动作
            # Execute action
            next_state, reward, done, _ = env.step(action)
            episode_return += reward
            
            # 选择下一动作
            # Select next action
            if not done:
                next_action = self.select_action(next_state)
            else:
                next_action = 0  # 任意值
            
            # Sarsa(λ)更新
            # Sarsa(λ) update
            self.update(state, action, reward, next_state, next_action, done)
            
            if done:
                break
            
            state = next_state
            action = next_action
        
        self.episode_count += 1
        self.episode_returns.append(episode_return)
        
        return episode_return, step + 1


# ================================================================================
# 第12.7节：Watkins's Q(λ)
# Section 12.7: Watkins's Q(λ)
# ================================================================================

class WatkinsQLambda:
    """
    Watkins's Q(λ)
    
    Q-learning的资格迹版本
    Q-learning with eligibility traces
    
    关键思想 Key Idea:
    当选择非贪婪动作时截断资格迹
    Cut off eligibility traces when non-greedy action is selected
    
    保证收敛性但减少了迹的效果
    Guarantees convergence but reduces trace effectiveness
    """
    
    def __init__(self,
                 n_features: int,
                 n_actions: int,
                 feature_extractor: Callable,
                 lambda_: float = 0.9,
                 alpha: float = 0.01,
                 gamma: float = 0.99,
                 epsilon: float = 0.1):
        """
        初始化Watkins's Q(λ)
        Initialize Watkins's Q(λ)
        """
        self.n_features = n_features
        self.n_actions = n_actions
        self.feature_extractor = feature_extractor
        self.lambda_ = lambda_
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # 权重和资格迹
        # Weights and eligibility traces
        self.weights = np.zeros((n_actions, n_features))
        self.traces = np.zeros((n_actions, n_features))
        
        # 统计
        # Statistics
        self.update_count = 0
        self.trace_cuts = 0
        self.td_errors = []
        
        logger.info(f"初始化Watkins's Q(λ): λ={lambda_}")
    
    def get_q_value(self, state: Any, action: int) -> float:
        """获取动作价值"""
        features = self.feature_extractor(state, action)
        return np.dot(self.weights[action], features)
    
    def get_all_q_values(self, state: Any) -> np.ndarray:
        """获取所有动作的Q值"""
        q_values = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            q_values[a] = self.get_q_value(state, a)
        return q_values
    
    def select_action(self, state: Any) -> Tuple[int, bool]:
        """
        选择动作并返回是否贪婪
        Select action and return whether greedy
        
        Returns:
            (动作, 是否贪婪)
            (action, is_greedy)
        """
        q_values = self.get_all_q_values(state)
        max_action = np.argmax(q_values)
        
        if np.random.random() < self.epsilon:
            # 探索
            # Explore
            action = np.random.randint(self.n_actions)
            is_greedy = (action == max_action)
        else:
            # 利用
            # Exploit
            action = max_action
            is_greedy = True
        
        return action, is_greedy
    
    def update(self, state: Any, action: int, reward: float,
               next_state: Any, done: bool, was_greedy: bool):
        """
        Watkins's Q(λ)更新
        Watkins's Q(λ) update
        
        Args:
            was_greedy: 当前动作是否贪婪
                       Whether current action was greedy
        """
        # 获取最大Q值（Q-learning目标）
        # Get max Q value (Q-learning target)
        if done:
            td_target = reward
        else:
            next_q_values = self.get_all_q_values(next_state)
            td_target = reward + self.gamma * np.max(next_q_values)
        
        # TD误差
        # TD error
        current_q = self.get_q_value(state, action)
        td_error = td_target - current_q
        self.td_errors.append(td_error)
        
        # 更新资格迹
        # Update eligibility traces
        features = self.feature_extractor(state, action)
        
        # 如果不是贪婪动作，截断迹
        # Cut traces if not greedy action
        if not was_greedy:
            self.traces = np.zeros((self.n_actions, self.n_features))
            self.trace_cuts += 1
        else:
            # 正常衰减
            # Normal decay
            self.traces *= self.gamma * self.lambda_
        
        # 累积当前状态-动作的迹
        # Accumulate trace for current state-action
        self.traces[action] += features
        
        # 更新权重
        # Update weights
        self.weights += self.alpha * td_error * self.traces
        
        # 如果终止，重置迹
        # Reset traces if terminal
        if done:
            self.traces = np.zeros((self.n_actions, self.n_features))
        
        self.update_count += 1
        
        return td_error


# ================================================================================
# 第12.7.1节：Peng's Q(λ)
# Section 12.7.1: Peng's Q(λ)
# ================================================================================

class PengQLambda:
    """
    Peng's Q(λ)
    
    Q(λ)的混合版本
    Hybrid version of Q(λ)
    
    结合Sarsa和Q-learning的优点
    Combines advantages of Sarsa and Q-learning
    
    使用Q-learning目标但不截断迹
    Uses Q-learning target but doesn't cut traces
    """
    
    def __init__(self,
                 n_features: int,
                 n_actions: int,
                 feature_extractor: Callable,
                 lambda_: float = 0.9,
                 alpha: float = 0.01,
                 gamma: float = 0.99,
                 epsilon: float = 0.1):
        """
        初始化Peng's Q(λ)
        Initialize Peng's Q(λ)
        """
        self.n_features = n_features
        self.n_actions = n_actions
        self.feature_extractor = feature_extractor
        self.lambda_ = lambda_
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # 权重和资格迹
        # Weights and eligibility traces
        self.weights = np.zeros((n_actions, n_features))
        self.traces = np.zeros((n_actions, n_features))
        
        # 额外的期望迹
        # Additional expected trace
        self.expected_traces = np.zeros((n_actions, n_features))
        
        # 统计
        # Statistics
        self.update_count = 0
        
        logger.info(f"初始化Peng's Q(λ): λ={lambda_}")
    
    def get_q_value(self, state: Any, action: int) -> float:
        """获取动作价值"""
        features = self.feature_extractor(state, action)
        return np.dot(self.weights[action], features)
    
    def get_all_q_values(self, state: Any) -> np.ndarray:
        """获取所有动作的Q值"""
        q_values = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            q_values[a] = self.get_q_value(state, a)
        return q_values
    
    def select_action(self, state: Any) -> int:
        """ε-贪婪动作选择"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            q_values = self.get_all_q_values(state)
            return np.argmax(q_values)
    
    def update(self, state: Any, action: int, reward: float,
               next_state: Any, done: bool):
        """
        Peng's Q(λ)更新
        Peng's Q(λ) update
        
        混合更新规则
        Hybrid update rule
        """
        features = self.feature_extractor(state, action)
        
        # 获取下一状态的Q值
        # Get next state Q values
        if done:
            max_next_q = 0.0
            expected_next_q = 0.0
        else:
            next_q_values = self.get_all_q_values(next_state)
            max_next_q = np.max(next_q_values)
            
            # 计算期望Q值（用于Sarsa部分）
            # Compute expected Q value (for Sarsa part)
            greedy_action = np.argmax(next_q_values)
            expected_next_q = 0.0
            for a in range(self.n_actions):
                if a == greedy_action:
                    prob = 1 - self.epsilon + self.epsilon / self.n_actions
                else:
                    prob = self.epsilon / self.n_actions
                expected_next_q += prob * next_q_values[a]
        
        # 两个TD误差
        # Two TD errors
        current_q = self.get_q_value(state, action)
        q_learning_error = reward + self.gamma * max_next_q - current_q
        sarsa_error = reward + self.gamma * expected_next_q - current_q
        
        # 更新标准资格迹（用于Q-learning部分）
        # Update standard eligibility traces (for Q-learning part)
        self.traces *= self.gamma * self.lambda_
        self.traces[action] += features
        
        # 更新期望资格迹（用于Sarsa部分）
        # Update expected eligibility traces (for Sarsa part)
        self.expected_traces *= self.gamma * self.lambda_
        self.expected_traces[action] += features
        
        # 混合更新
        # Hybrid update
        # 使用Q-learning误差但不截断迹
        # Use Q-learning error but don't cut traces
        self.weights += self.alpha * q_learning_error * self.traces
        
        # 如果终止，重置迹
        # Reset traces if terminal
        if done:
            self.traces = np.zeros((self.n_actions, self.n_features))
            self.expected_traces = np.zeros((self.n_actions, self.n_features))
        
        self.update_count += 1
        
        return q_learning_error


# ================================================================================
# 第12.8节：真正的在线Sarsa(λ)
# Section 12.8: True Online Sarsa(λ)
# ================================================================================

class TrueOnlineSarsaLambda:
    """
    真正的在线Sarsa(λ)
    True Online Sarsa(λ)
    
    使用Dutch traces的Sarsa(λ)
    Sarsa(λ) with Dutch traces
    
    更精确的在线更新
    More accurate online updates
    """
    
    def __init__(self,
                 n_features: int,
                 n_actions: int,
                 feature_extractor: Callable,
                 lambda_: float = 0.9,
                 alpha: float = 0.01,
                 gamma: float = 0.99,
                 epsilon: float = 0.1):
        """
        初始化真正的在线Sarsa(λ)
        Initialize True Online Sarsa(λ)
        """
        self.n_features = n_features
        self.n_actions = n_actions
        self.feature_extractor = feature_extractor
        self.lambda_ = lambda_
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # 权重和资格迹
        # Weights and eligibility traces
        self.weights = np.zeros((n_actions, n_features))
        self.traces = np.zeros((n_actions, n_features))
        
        # 上一步的Q值
        # Previous Q value
        self.old_q = 0.0
        
        # 统计
        # Statistics
        self.update_count = 0
        self.episode_count = 0
        
        logger.info(f"初始化真正的在线Sarsa(λ)")
    
    def get_q_value(self, state: Any, action: int) -> float:
        """获取动作价值"""
        features = self.feature_extractor(state, action)
        return np.dot(self.weights[action], features)
    
    def select_action(self, state: Any) -> int:
        """ε-贪婪动作选择"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            q_values = [self.get_q_value(state, a) for a in range(self.n_actions)]
            return np.argmax(q_values)
    
    def update(self, state: Any, action: int, reward: float,
               next_state: Any, next_action: int, done: bool):
        """
        真正的在线Sarsa(λ)更新
        True Online Sarsa(λ) update
        """
        features = self.feature_extractor(state, action)
        
        # 当前和下一Q值
        # Current and next Q values
        current_q = self.get_q_value(state, action)
        if done:
            next_q = 0.0
        else:
            next_q = self.get_q_value(next_state, next_action)
        
        # TD误差
        # TD error
        td_error = reward + self.gamma * next_q - current_q
        
        # Dutch trace更新
        # Dutch trace update
        dutch_factor = 1 - self.alpha * self.gamma * self.lambda_ * np.dot(self.traces[action], features)
        self.traces *= self.gamma * self.lambda_
        self.traces[action] += dutch_factor * features
        
        # 权重更新（包含修正项）
        # Weight update (with correction term)
        self.weights += self.alpha * (td_error + current_q - self.old_q) * self.traces
        self.weights[action] -= self.alpha * (current_q - self.old_q) * features
        
        # 保存当前Q值
        # Save current Q value
        self.old_q = next_q
        
        # 如果终止，重置
        # Reset if terminal
        if done:
            self.traces = np.zeros((self.n_actions, self.n_features))
            self.old_q = 0.0
        
        self.update_count += 1
        
        return td_error


# ================================================================================
# 主函数：演示控制算法的资格迹
# Main Function: Demonstrate Control with Eligibility Traces
# ================================================================================

def demonstrate_control_traces():
    """
    演示控制算法的资格迹
    Demonstrate control algorithms with eligibility traces
    """
    print("\n" + "="*80)
    print("第12.6-12.8节：控制算法的资格迹")
    print("Section 12.6-12.8: Control with Eligibility Traces")
    print("="*80)
    
    # 设置
    # Setup
    n_features = 10
    n_actions = 3
    n_states = 5
    
    # 状态-动作特征提取器
    # State-action feature extractor
    def sa_features(state, action):
        features = np.zeros(n_features)
        if isinstance(state, int):
            # 状态和动作的组合特征
            # Combined features of state and action
            base_idx = (state * n_actions + action) % n_features
            features[base_idx] = 1.0
            features[(base_idx + 1) % n_features] = 0.5
        return features
    
    # 简单环境模拟器
    # Simple environment simulator
    class SimpleEnv:
        def __init__(self):
            self.state = 0
            self.step_count = 0
        
        def reset(self):
            self.state = 0
            self.step_count = 0
            return self.state
        
        def step(self, action):
            # 状态转移
            # State transition
            if action == 0:  # 左
                self.state = max(0, self.state - 1)
            elif action == 1:  # 右
                self.state = min(n_states - 1, self.state + 1)
            else:  # 停留
                pass
            
            # 奖励
            # Reward
            if self.state == n_states - 1:
                reward = 10.0
                done = True
            else:
                reward = -1.0
                done = False
            
            self.step_count += 1
            if self.step_count >= 20:
                done = True
            
            return self.state, reward, done, {}
    
    # 1. 测试Sarsa(λ)
    # 1. Test Sarsa(λ)
    print("\n" + "="*60)
    print("1. Sarsa(λ)")
    print("="*60)
    
    # 累积迹版本
    # Accumulating trace version
    sarsa_lambda_acc = SarsaLambda(
        n_features=n_features,
        n_actions=n_actions,
        feature_extractor=sa_features,
        lambda_=0.9,
        alpha=0.1,
        gamma=0.95,
        epsilon=0.1,
        trace_type='accumulating'
    )
    
    # 替换迹版本
    # Replacing trace version
    sarsa_lambda_rep = SarsaLambda(
        n_features=n_features,
        n_actions=n_actions,
        feature_extractor=sa_features,
        lambda_=0.9,
        alpha=0.1,
        gamma=0.95,
        epsilon=0.1,
        trace_type='replacing'
    )
    
    print("\n训练Sarsa(λ)...")
    env = SimpleEnv()
    
    for ep in range(10):
        # 累积迹
        ret_acc, steps_acc = sarsa_lambda_acc.learn_episode(env, max_steps=50)
        
        # 替换迹
        ret_rep, steps_rep = sarsa_lambda_rep.learn_episode(env, max_steps=50)
        
        if (ep + 1) % 5 == 0:
            print(f"  回合{ep + 1}:")
            print(f"    累积迹 - 回报={ret_acc:.1f}, 步数={steps_acc}")
            print(f"    替换迹 - 回报={ret_rep:.1f}, 步数={steps_rep}")
    
    # 2. 测试Watkins's Q(λ)
    # 2. Test Watkins's Q(λ)
    print("\n" + "="*60)
    print("2. Watkins's Q(λ)")
    print("="*60)
    
    watkins_q = WatkinsQLambda(
        n_features=n_features,
        n_actions=n_actions,
        feature_extractor=sa_features,
        lambda_=0.9,
        alpha=0.1,
        gamma=0.95,
        epsilon=0.1
    )
    
    print("\n训练Watkins's Q(λ)...")
    
    for ep in range(10):
        env = SimpleEnv()
        state = env.reset()
        episode_return = 0.0
        
        for step in range(50):
            action, was_greedy = watkins_q.select_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_return += reward
            
            watkins_q.update(state, action, reward, next_state, done, was_greedy)
            
            if done:
                break
            state = next_state
        
        if (ep + 1) % 5 == 0:
            print(f"  回合{ep + 1}: 回报={episode_return:.1f}, "
                  f"迹截断次数={watkins_q.trace_cuts}")
    
    # 3. 测试Peng's Q(λ)
    # 3. Test Peng's Q(λ)
    print("\n" + "="*60)
    print("3. Peng's Q(λ)")
    print("="*60)
    
    peng_q = PengQLambda(
        n_features=n_features,
        n_actions=n_actions,
        feature_extractor=sa_features,
        lambda_=0.9,
        alpha=0.1,
        gamma=0.95,
        epsilon=0.1
    )
    
    print("\n训练Peng's Q(λ)...")
    
    for ep in range(10):
        env = SimpleEnv()
        state = env.reset()
        episode_return = 0.0
        
        for step in range(50):
            action = peng_q.select_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_return += reward
            
            peng_q.update(state, action, reward, next_state, done)
            
            if done:
                break
            state = next_state
        
        if (ep + 1) % 5 == 0:
            print(f"  回合{ep + 1}: 回报={episode_return:.1f}")
    
    # 4. 测试真正的在线Sarsa(λ)
    # 4. Test True Online Sarsa(λ)
    print("\n" + "="*60)
    print("4. 真正的在线Sarsa(λ)")
    print("4. True Online Sarsa(λ)")
    print("="*60)
    
    true_online_sarsa = TrueOnlineSarsaLambda(
        n_features=n_features,
        n_actions=n_actions,
        feature_extractor=sa_features,
        lambda_=0.9,
        alpha=0.1,
        gamma=0.95,
        epsilon=0.1
    )
    
    print("\n训练真正的在线Sarsa(λ)...")
    
    for ep in range(10):
        env = SimpleEnv()
        state = env.reset()
        action = true_online_sarsa.select_action(state)
        episode_return = 0.0
        
        for step in range(50):
            next_state, reward, done, _ = env.step(action)
            episode_return += reward
            
            if not done:
                next_action = true_online_sarsa.select_action(next_state)
            else:
                next_action = 0
            
            true_online_sarsa.update(state, action, reward, 
                                    next_state, next_action, done)
            
            if done:
                break
            
            state = next_state
            action = next_action
        
        if (ep + 1) % 5 == 0:
            print(f"  回合{ep + 1}: 回报={episode_return:.1f}")
    
    # 5. 比较不同算法
    # 5. Compare different algorithms
    print("\n" + "="*60)
    print("5. 算法比较")
    print("5. Algorithm Comparison")
    print("="*60)
    
    print("\n各算法的Q值估计（状态2的所有动作）:")
    test_state = 2
    
    print("\nSarsa(λ)-累积迹:")
    for a in range(n_actions):
        q_val = sarsa_lambda_acc.get_q_value(test_state, a)
        print(f"  Q({test_state},{a}) = {q_val:.3f}")
    
    print("\nSarsa(λ)-替换迹:")
    for a in range(n_actions):
        q_val = sarsa_lambda_rep.get_q_value(test_state, a)
        print(f"  Q({test_state},{a}) = {q_val:.3f}")
    
    print("\nWatkins's Q(λ):")
    for a in range(n_actions):
        q_val = watkins_q.get_q_value(test_state, a)
        print(f"  Q({test_state},{a}) = {q_val:.3f}")
    
    print("\nPeng's Q(λ):")
    for a in range(n_actions):
        q_val = peng_q.get_q_value(test_state, a)
        print(f"  Q({test_state},{a}) = {q_val:.3f}")
    
    print("\n真正的在线Sarsa(λ):")
    for a in range(n_actions):
        q_val = true_online_sarsa.get_q_value(test_state, a)
        print(f"  Q({test_state},{a}) = {q_val:.3f}")
    
    # 总结
    # Summary
    print("\n" + "="*80)
    print("控制算法资格迹总结")
    print("Control with Eligibility Traces Summary")
    print("="*80)
    
    print("""
    关键要点 Key Takeaways:
    =======================
    
    1. Sarsa(λ)直接扩展同策略控制
       Sarsa(λ) directly extends on-policy control
       
    2. Q(λ)需要特殊处理离策略
       Q(λ) needs special handling for off-policy
       
    3. Watkins截断保证收敛但牺牲效率
       Watkins cutoff ensures convergence but sacrifices efficiency
       
    4. Peng's Q(λ)提供折中方案
       Peng's Q(λ) provides compromise
       
    5. 真正的在线版本最准确
       True online versions most accurate
    
    算法选择 Algorithm Selection:
    - 同策略: Sarsa(λ)或真正的在线Sarsa(λ)
             Sarsa(λ) or True Online Sarsa(λ)
    - 离策略: Watkins's Q(λ)（安全）或Peng's Q(λ)（高效）
             Watkins's Q(λ) (safe) or Peng's Q(λ) (efficient)
    - 稀疏特征: 替换迹
               Replacing traces
    - 稠密特征: 累积迹
               Accumulating traces
    """)


if __name__ == "__main__":
    demonstrate_control_traces()