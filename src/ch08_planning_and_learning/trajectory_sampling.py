"""
================================================================================
第8.6节：轨迹采样 - 聚焦于相关状态
Section 8.6: Trajectory Sampling - Focus on Relevant States
================================================================================

不是均匀更新所有状态，而是沿着轨迹更新！
Not uniform updates over all states, but along trajectories!

核心思想 Core Idea:
从起始状态开始，模拟完整轨迹，只更新访问的状态
Start from start states, simulate full trajectories, update only visited states

两种采样分布 Two Sampling Distributions:
1. 均匀采样 Uniform Sampling:
   - 所有(s,a)等概率
     All (s,a) equally probable
   - 不考虑状态重要性
     Doesn't consider state importance
   
2. 轨迹采样 Trajectory Sampling:
   - 根据策略生成轨迹
     Generate trajectories according to policy
   - 自然聚焦于重要状态
     Naturally focuses on important states

优势 Advantages:
- 聚焦于实际会访问的状态
  Focus on states actually visited
- 避免更新无关状态
  Avoid updating irrelevant states
- 更好的样本效率
  Better sample efficiency

实时动态规划 Real-time Dynamic Programming:
轨迹采样的极端形式，只更新实际访问的状态
Extreme form of trajectory sampling, only update actually visited states
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import time

# 导入基础组件
# Import base components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from ch02_mdp.mdp_framework import State, Action, MDPEnvironment
from ch02_mdp.policies_and_values import (
    Policy, ActionValueFunction, StateValueFunction
)
from ch04_monte_carlo.mc_control import EpsilonGreedyPolicy

from .models_and_planning import DeterministicModel

# 设置日志
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第8.6.1节：轨迹生成器
# Section 8.6.1: Trajectory Generator
# ================================================================================

@dataclass
class Trajectory:
    """
    轨迹
    Trajectory
    
    存储完整的状态-动作-奖励序列
    Stores complete state-action-reward sequence
    """
    states: List[State] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    
    def add_step(self, state: State, action: Optional[Action], reward: float):
        """
        添加一步
        Add one step
        """
        self.states.append(state)
        if action is not None:
            self.actions.append(action)
        self.rewards.append(reward)
    
    @property
    def length(self) -> int:
        """
        轨迹长度
        Trajectory length
        """
        return len(self.states)
    
    @property
    def return_value(self) -> float:
        """
        轨迹回报（无折扣）
        Trajectory return (undiscounted)
        """
        return sum(self.rewards)
    
    def discounted_return(self, gamma: float) -> float:
        """
        折扣回报
        Discounted return
        """
        g = 0.0
        for i, r in enumerate(self.rewards):
            g += (gamma ** i) * r
        return g


class TrajectoryGenerator:
    """
    轨迹生成器
    Trajectory Generator
    
    从模型生成模拟轨迹
    Generate simulated trajectories from model
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 model: DeterministicModel):
        """
        初始化轨迹生成器
        Initialize trajectory generator
        
        Args:
            env: 环境
                Environment
            model: 模型
                  Model
        """
        self.env = env
        self.model = model
        
        logger.info("初始化轨迹生成器")
    
    def generate_trajectory(self,
                          policy: Policy,
                          start_state: Optional[State] = None,
                          max_steps: int = 1000) -> Trajectory:
        """
        生成轨迹
        Generate trajectory
        
        Args:
            policy: 策略
                   Policy
            start_state: 起始状态（None则随机）
                        Start state (None for random)
            max_steps: 最大步数
                      Maximum steps
        
        Returns:
            生成的轨迹
            Generated trajectory
        """
        trajectory = Trajectory()
        
        # 初始状态
        # Initial state
        if start_state is None:
            # 随机选择非终止状态
            # Random non-terminal state
            non_terminal = [s for s in self.env.state_space if not s.is_terminal]
            state = np.random.choice(non_terminal) if non_terminal else self.env.state_space[0]
        else:
            state = start_state
        
        # 生成轨迹
        # Generate trajectory
        for step in range(max_steps):
            # 选择动作
            # Select action
            action = policy.select_action(state)
            
            # 从模型获取下一状态和奖励
            # Get next state and reward from model
            if self.model.is_known(state, action):
                next_state, reward = self.model.sample(state, action)
            else:
                # 未知转移，结束轨迹
                # Unknown transition, end trajectory
                trajectory.add_step(state, action, 0.0)
                break
            
            # 添加到轨迹
            # Add to trajectory
            trajectory.add_step(state, action, reward)
            
            # 检查终止
            # Check termination
            if next_state.is_terminal:
                trajectory.add_step(next_state, None, 0.0)
                break
            
            state = next_state
        
        return trajectory


# ================================================================================
# 第8.6.2节：采样策略
# Section 8.6.2: Sampling Strategies
# ================================================================================

class SamplingStrategy:
    """
    采样策略基类
    Sampling Strategy Base Class
    
    决定如何选择要更新的状态
    Decides how to select states to update
    """
    
    def sample_states(self,
                     env: MDPEnvironment,
                     n_samples: int) -> List[Tuple[State, Action]]:
        """
        采样状态-动作对
        Sample state-action pairs
        
        Args:
            env: 环境
                Environment
            n_samples: 采样数量
                      Number of samples
        
        Returns:
            采样的(s,a)对列表
            List of sampled (s,a) pairs
        """
        raise NotImplementedError


class UniformSampling(SamplingStrategy):
    """
    均匀采样
    Uniform Sampling
    
    所有(s,a)等概率采样
    All (s,a) sampled with equal probability
    
    优势 Advantages:
    - 简单
      Simple
    - 无偏
      Unbiased
    
    劣势 Disadvantages:
    - 浪费计算在无关状态
      Wastes computation on irrelevant states
    - 收敛慢
      Slow convergence
    """
    
    def __init__(self, known_pairs: Set[Tuple[State, Action]]):
        """
        初始化均匀采样
        Initialize uniform sampling
        
        Args:
            known_pairs: 已知的(s,a)对
                        Known (s,a) pairs
        """
        self.known_pairs = list(known_pairs)
        
    def sample_states(self,
                     env: MDPEnvironment,
                     n_samples: int) -> List[Tuple[State, Action]]:
        """
        均匀采样
        Uniform sampling
        """
        if not self.known_pairs:
            return []
        
        # 随机采样（可重复）
        # Random sampling (with replacement)
        indices = np.random.randint(0, len(self.known_pairs), n_samples)
        return [self.known_pairs[i] for i in indices]


class OnPolicySampling(SamplingStrategy):
    """
    同策略采样
    On-Policy Sampling
    
    根据当前策略的分布采样
    Sample according to current policy distribution
    
    更准确地反映策略下的状态分布
    More accurately reflects state distribution under policy
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 model: DeterministicModel,
                 policy: Policy):
        """
        初始化同策略采样
        Initialize on-policy sampling
        
        Args:
            env: 环境
                Environment
            model: 模型
                  Model
            policy: 策略
                   Policy
        """
        self.env = env
        self.model = model
        self.policy = policy
        self.generator = TrajectoryGenerator(env, model)
        
    def sample_states(self,
                     env: MDPEnvironment,
                     n_samples: int) -> List[Tuple[State, Action]]:
        """
        同策略采样
        On-policy sampling
        
        生成轨迹并收集(s,a)对
        Generate trajectories and collect (s,a) pairs
        """
        samples = []
        
        while len(samples) < n_samples:
            # 生成轨迹
            # Generate trajectory
            trajectory = self.generator.generate_trajectory(
                self.policy,
                max_steps=100
            )
            
            # 收集(s,a)对
            # Collect (s,a) pairs
            for i in range(len(trajectory.actions)):
                samples.append((trajectory.states[i], trajectory.actions[i]))
                
                if len(samples) >= n_samples:
                    break
        
        return samples[:n_samples]


# ================================================================================
# 第8.6.3节：轨迹采样算法
# Section 8.6.3: Trajectory Sampling Algorithm
# ================================================================================

class TrajectorySampling:
    """
    轨迹采样算法
    Trajectory Sampling Algorithm
    
    沿着模拟轨迹更新Q值
    Update Q values along simulated trajectories
    
    关键思想 Key Idea:
    从起始状态分布开始，生成完整轨迹，只更新轨迹上的状态
    Start from initial state distribution, generate full trajectories,
    update only states on trajectories
    
    与均匀采样的区别 Difference from Uniform Sampling:
    - 均匀：所有状态等权重
      Uniform: All states equal weight
    - 轨迹：根据访问频率加权
      Trajectory: Weighted by visitation frequency
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 gamma: float = 0.95,
                 alpha: float = 0.1,
                 epsilon: float = 0.1):
        """
        初始化轨迹采样
        Initialize trajectory sampling
        
        Args:
            env: 环境
                Environment
            gamma: 折扣因子
                  Discount factor
            alpha: 学习率
                  Learning rate
            epsilon: 探索率
                    Exploration rate
        """
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        
        # Q函数
        # Q function
        self.Q = ActionValueFunction(
            env.state_space,
            env.action_space,
            initial_value=0.0
        )
        
        # 策略
        # Policy
        self.policy = EpsilonGreedyPolicy(
            self.Q,
            epsilon=epsilon,
            epsilon_decay=1.0,
            epsilon_min=epsilon,
            action_space=env.action_space
        )
        
        # 模型
        # Model
        self.model = DeterministicModel(
            env.state_space,
            env.action_space
        )
        
        # 轨迹生成器
        # Trajectory generator
        self.generator = TrajectoryGenerator(env, self.model)
        
        # 统计
        # Statistics
        self.update_count = 0
        self.trajectory_count = 0
        self.state_visit_counts = defaultdict(int)
        
        logger.info(f"初始化轨迹采样: γ={gamma}, α={alpha}, ε={epsilon}")
    
    def update_along_trajectory(self, trajectory: Trajectory):
        """
        沿轨迹更新Q值
        Update Q values along trajectory
        
        Args:
            trajectory: 轨迹
                       Trajectory
        """
        # 从后向前更新（类似TD）
        # Update backward (similar to TD)
        for t in range(len(trajectory.actions)):
            state = trajectory.states[t]
            action = trajectory.actions[t]
            reward = trajectory.rewards[t]
            
            # 记录访问
            # Record visit
            self.state_visit_counts[state] += 1
            
            # 计算TD目标
            # Compute TD target
            if t + 1 < len(trajectory.states):
                next_state = trajectory.states[t + 1]
                if not next_state.is_terminal:
                    max_q = max(
                        self.Q.get_value(next_state, a)
                        for a in self.env.action_space
                    )
                else:
                    max_q = 0.0
            else:
                max_q = 0.0
            
            td_target = reward + self.gamma * max_q
            
            # 更新Q值
            # Update Q value
            old_q = self.Q.get_value(state, action)
            new_q = old_q + self.alpha * (td_target - old_q)
            self.Q.set_value(state, action, new_q)
            
            self.update_count += 1
    
    def learn_from_real_experience(self,
                                  state: State,
                                  action: Action,
                                  next_state: State,
                                  reward: float):
        """
        从真实经验学习
        Learn from real experience
        
        更新模型并立即规划
        Update model and plan immediately
        
        Args:
            state: 当前状态
                  Current state
            action: 动作
                   Action
            next_state: 下一状态
                       Next state
            reward: 奖励
                   Reward
        """
        # 更新模型
        # Update model
        self.model.update(state, action, next_state, reward)
        
        # 直接Q更新
        # Direct Q update
        if not next_state.is_terminal:
            max_q = max(
                self.Q.get_value(next_state, a)
                for a in self.env.action_space
            )
        else:
            max_q = 0.0
        
        td_target = reward + self.gamma * max_q
        old_q = self.Q.get_value(state, action)
        new_q = old_q + self.alpha * (td_target - old_q)
        self.Q.set_value(state, action, new_q)
        
        self.update_count += 1
    
    def planning_with_trajectories(self, n_trajectories: int = 10):
        """
        使用轨迹进行规划
        Planning with trajectories
        
        Args:
            n_trajectories: 要生成的轨迹数
                          Number of trajectories to generate
        """
        for _ in range(n_trajectories):
            # 生成轨迹
            # Generate trajectory
            trajectory = self.generator.generate_trajectory(
                self.policy,
                max_steps=100
            )
            
            # 沿轨迹更新
            # Update along trajectory
            self.update_along_trajectory(trajectory)
            
            self.trajectory_count += 1
    
    def get_state_distribution(self) -> Dict[State, float]:
        """
        获取状态访问分布
        Get state visitation distribution
        
        Returns:
            状态访问频率
            State visitation frequencies
        """
        total_visits = sum(self.state_visit_counts.values())
        if total_visits == 0:
            return {}
        
        return {
            state: count / total_visits
            for state, count in self.state_visit_counts.items()
        }


# ================================================================================
# 第8.6.4节：实时动态规划
# Section 8.6.4: Real-time Dynamic Programming
# ================================================================================

class RealTimeDynamicProgramming:
    """
    实时动态规划 (RTDP)
    Real-time Dynamic Programming
    
    轨迹采样的极端形式
    Extreme form of trajectory sampling
    
    核心思想 Core Idea:
    只更新实际访问的状态，完全忽略其他状态
    Only update actually visited states, completely ignore others
    
    特点 Features:
    - 最大化相关性
      Maximize relevance
    - 最小化计算浪费
      Minimize computational waste
    - 适合大状态空间
      Suitable for large state spaces
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 gamma: float = 0.95,
                 alpha: float = 0.1):
        """
        初始化RTDP
        Initialize RTDP
        """
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        
        # 价值函数（只存储访问过的状态）
        # Value function (only store visited states)
        self.V = StateValueFunction(
            env.state_space,
            initial_value=0.0
        )
        
        # 访问的状态集合
        # Set of visited states
        self.visited_states: Set[State] = set()
        
        # 统计
        # Statistics
        self.update_count = 0
        
        logger.info(f"初始化RTDP: γ={gamma}, α={alpha}")
    
    def update_state(self, state: State):
        """
        更新单个状态
        Update single state
        
        贝尔曼更新
        Bellman update
        
        Args:
            state: 要更新的状态
                  State to update
        """
        if state.is_terminal:
            return
        
        # 记录访问
        # Record visit
        self.visited_states.add(state)
        
        # 计算最优值（假设知道模型）
        # Compute optimal value (assume known model)
        best_value = float('-inf')
        
        for action in self.env.action_space:
            # 这里简化：假设确定性转移
            # Simplified: assume deterministic transition
            # 实际应该使用期望
            # Should actually use expectation
            next_state, reward, _, _ = self.env.step(action)
            value = reward + self.gamma * self.V.get_value(next_state)
            best_value = max(best_value, value)
        
        # 更新价值
        # Update value
        old_v = self.V.get_value(state)
        new_v = old_v + self.alpha * (best_value - old_v)
        self.V.set_value(state, new_v)
        
        self.update_count += 1
    
    def run_trial(self, start_state: Optional[State] = None,
                 max_steps: int = 100):
        """
        运行一次试验
        Run one trial
        
        从起始状态到终止状态
        From start state to terminal state
        
        Args:
            start_state: 起始状态
                        Start state
            max_steps: 最大步数
                      Maximum steps
        """
        if start_state is None:
            state = self.env.reset()
        else:
            state = start_state
        
        for _ in range(max_steps):
            if state.is_terminal:
                break
            
            # 更新当前状态
            # Update current state
            self.update_state(state)
            
            # 贪婪动作选择（基于当前价值）
            # Greedy action selection (based on current values)
            best_action = None
            best_value = float('-inf')
            
            for action in self.env.action_space:
                next_state, reward, _, _ = self.env.step(action)
                value = reward + self.gamma * self.V.get_value(next_state)
                if value > best_value:
                    best_value = value
                    best_action = action
            
            # 执行最优动作
            # Execute best action
            if best_action is not None:
                next_state, _, done, _ = self.env.step(best_action)
                state = next_state
                
                if done:
                    break


# ================================================================================
# 主函数：演示轨迹采样
# Main Function: Demonstrate Trajectory Sampling
# ================================================================================

def demonstrate_trajectory_sampling():
    """
    演示轨迹采样
    Demonstrate trajectory sampling
    """
    print("\n" + "="*80)
    print("第8.6节：轨迹采样")
    print("Section 8.6: Trajectory Sampling")
    print("="*80)
    
    from ch02_mdp.gridworld import GridWorld
    
    # 创建环境
    # Create environment
    env = GridWorld(rows=5, cols=5,
                   start_pos=(0,0),
                   goal_pos=(4,4),
                   obstacles=[(1,1), (2,2), (3,3)])
    
    print(f"\n创建5×5 GridWorld（带障碍）")
    print(f"  起点: (0,0)")
    print(f"  终点: (4,4)")
    print(f"  障碍: {[(1,1), (2,2), (3,3)]}")
    
    # 1. 演示轨迹生成
    # 1. Demonstrate trajectory generation
    print("\n" + "="*60)
    print("1. 轨迹生成演示")
    print("1. Trajectory Generation Demo")
    print("="*60)
    
    # 创建模型（从经验学习）
    # Create model (learn from experience)
    model = DeterministicModel(env.state_space, env.action_space)
    
    # 收集一些经验
    # Collect some experience
    print("\n收集经验构建模型...")
    for _ in range(50):
        state = env.reset()
        for _ in range(20):
            if state.is_terminal:
                break
            action = np.random.choice(env.action_space)
            next_state, reward, done, _ = env.step(action)
            model.update(state, action, next_state, reward)
            state = next_state
            if done:
                break
    
    # 创建轨迹生成器
    # Create trajectory generator
    generator = TrajectoryGenerator(env, model)
    
    # 创建随机策略
    # Create random policy
    from ch02_mdp.policies_and_values import UniformRandomPolicy
    random_policy = UniformRandomPolicy(env.action_space)
    
    # 生成轨迹
    # Generate trajectory
    print("\n生成示例轨迹...")
    trajectory = generator.generate_trajectory(
        random_policy,
        start_state=env.state_space[0],
        max_steps=20
    )
    
    print(f"  轨迹长度: {trajectory.length}")
    print(f"  总回报: {trajectory.return_value:.2f}")
    print(f"  折扣回报(γ=0.9): {trajectory.discounted_return(0.9):.2f}")
    
    # 显示轨迹前几步
    # Show first few steps
    print("\n  前5步:")
    for i in range(min(5, len(trajectory.actions))):
        print(f"    {trajectory.states[i].id} --{trajectory.actions[i].id}--> "
              f"r={trajectory.rewards[i]:.1f}")
    
    # 2. 比较均匀采样vs轨迹采样
    # 2. Compare uniform vs trajectory sampling
    print("\n" + "="*60)
    print("2. 均匀采样vs轨迹采样")
    print("2. Uniform vs Trajectory Sampling")
    print("="*60)
    
    # 创建轨迹采样算法
    # Create trajectory sampling algorithm
    traj_sampler = TrajectorySampling(env, gamma=0.95, alpha=0.1, epsilon=0.1)
    
    # 从真实经验学习模型
    # Learn model from real experience
    print("\n学习100步真实经验...")
    state = env.reset()
    for step in range(100):
        if state.is_terminal:
            state = env.reset()
        
        action = traj_sampler.policy.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        traj_sampler.learn_from_real_experience(state, action, next_state, reward)
        
        state = next_state
    
    print(f"  模型已知(s,a)对: {len(traj_sampler.model.transitions)}")
    
    # 使用轨迹规划
    # Plan with trajectories
    print("\n使用轨迹规划（10条轨迹）...")
    traj_sampler.planning_with_trajectories(n_trajectories=10)
    
    print(f"  生成轨迹数: {traj_sampler.trajectory_count}")
    print(f"  总更新次数: {traj_sampler.update_count}")
    
    # 显示状态访问分布
    # Show state visitation distribution
    distribution = traj_sampler.get_state_distribution()
    print("\n状态访问分布（前5个最常访问）:")
    sorted_states = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
    for state, freq in sorted_states[:5]:
        print(f"  {state.id}: {freq:.3f}")
    
    # 3. 演示实时动态规划
    # 3. Demonstrate Real-time Dynamic Programming
    print("\n" + "="*60)
    print("3. 实时动态规划(RTDP)")
    print("3. Real-time Dynamic Programming")
    print("="*60)
    
    rtdp = RealTimeDynamicProgramming(env, gamma=0.95, alpha=0.1)
    
    print("\n运行10次RTDP试验...")
    for trial in range(10):
        rtdp.run_trial(max_steps=50)
    
    print(f"  访问状态数: {len(rtdp.visited_states)}/{len(env.state_space)}")
    print(f"  总更新次数: {rtdp.update_count}")
    
    # 显示学习的价值（部分）
    # Show learned values (partial)
    print("\n学习的状态价值（前5个）:")
    for i, state in enumerate(rtdp.visited_states):
        if i >= 5:
            break
        value = rtdp.V.get_value(state)
        print(f"  V({state.id}) = {value:.3f}")
    
    # 4. 效率比较
    # 4. Efficiency comparison
    print("\n" + "="*60)
    print("4. 采样效率比较")
    print("4. Sampling Efficiency Comparison")
    print("="*60)
    
    print("""
    采样策略比较:
    
    均匀采样 Uniform Sampling:
      - 覆盖率: 100%
      - 相关性: 低
      - 适用: 小状态空间
    
    轨迹采样 Trajectory Sampling:
      - 覆盖率: 部分
      - 相关性: 高
      - 适用: 大状态空间
    
    实时DP Real-time DP:
      - 覆盖率: 最小
      - 相关性: 最高
      - 适用: 巨大状态空间
    """)
    
    # 总结
    # Summary
    print("\n" + "="*80)
    print("轨迹采样总结")
    print("Trajectory Sampling Summary")
    print("="*80)
    
    print("""
    关键要点 Key Takeaways:
    =======================
    
    1. 轨迹采样聚焦于相关状态
       Trajectory sampling focuses on relevant states
       
    2. 避免更新无关状态
       Avoids updating irrelevant states
       
    3. 自然的探索-利用平衡
       Natural exploration-exploitation balance
       
    4. RTDP是极端但有效
       RTDP is extreme but effective
       
    5. 适合大规模问题
       Suitable for large-scale problems
    
    实践建议 Practical Tips:
    - 小问题：均匀采样足够
      Small problems: Uniform sampling sufficient
    - 大问题：使用轨迹采样
      Large problems: Use trajectory sampling
    - 巨大问题：考虑RTDP
      Huge problems: Consider RTDP
    """)


if __name__ == "__main__":
    demonstrate_trajectory_sampling()