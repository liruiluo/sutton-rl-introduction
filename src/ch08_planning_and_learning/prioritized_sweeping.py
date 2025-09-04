"""
================================================================================
第8.4节：优先级扫描 - 智能规划
Section 8.4: Prioritized Sweeping - Smart Planning
================================================================================

不是随机规划，而是优先更新重要的状态！
Not random planning, but prioritize important states!

核心思想 Core Idea:
根据价值变化的大小决定更新优先级
Prioritize updates based on magnitude of value changes

优先级度量 Priority Metric:
|Δ| = |r + γ max_a Q(s',a) - Q(s,a)|
TD误差的绝对值
Absolute value of TD error

算法步骤 Algorithm Steps:
1. 初始化Q(s,a), Model(s,a), PQueue
2. 循环：
   Loop:
   a. 与环境交互，获得(s,a,r,s')
      Interact with environment, get (s,a,r,s')
   b. 更新模型Model(s,a) ← (s',r)
      Update model
   c. 计算优先级P = |TD误差|
      Compute priority P = |TD error|
   d. 如果P > θ，插入(s,a)到PQueue
      If P > θ, insert (s,a) into PQueue
   e. 重复n次且PQueue非空：
      Repeat n times while PQueue not empty:
      - 弹出最高优先级的(s,a)
        Pop highest priority (s,a)
      - 更新Q(s,a)
        Update Q(s,a)
      - 对所有可能导致s的(s̄,ā)：
        For all (s̄,ā) that could lead to s:
        计算其优先级并插入PQueue
        Compute priority and insert to PQueue

优势 Advantages:
- 聚焦于重要更新
  Focus on important updates
- 更快收敛
  Faster convergence
- 高效利用计算
  Efficient use of computation

挑战 Challenges:
- 维护优先队列开销
  Priority queue overhead
- 需要反向模型
  Needs reverse model
"""

import numpy as np
import heapq
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import time

# 导入基础组件
# Import base components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.ch03_finite_mdp.mdp_framework import State, Action, MDPEnvironment
from src.ch03_finite_mdp.policies_and_values import ActionValueFunction
from ch04_monte_carlo.mc_control import EpsilonGreedyPolicy

from .models_and_planning import DeterministicModel

# 设置日志
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第8.4.1节：优先队列
# Section 8.4.1: Priority Queue
# ================================================================================

@dataclass(order=True)
class PriorityItem:
    """
    优先级项
    Priority Item
    
    用于优先队列
    For priority queue
    """
    priority: float  # 负数，因为heapq是最小堆
                     # Negative because heapq is min-heap
    state_action: Tuple[State, Action] = field(compare=False)


class PriorityQueue:
    """
    优先队列
    Priority Queue
    
    维护待更新的(s,a)对及其优先级
    Maintains (s,a) pairs to update with priorities
    
    实现细节 Implementation Details:
    - 使用最大堆（通过负优先级实现）
      Use max-heap (implemented via negative priority)
    - 避免重复插入
      Avoid duplicate insertions
    - 支持阈值过滤
      Support threshold filtering
    """
    
    def __init__(self, threshold: float = 1e-5):
        """
        初始化优先队列
        Initialize priority queue
        
        Args:
            threshold: 优先级阈值，低于此值不插入
                      Priority threshold, don't insert if below
        """
        self.threshold = threshold
        self.heap: List[PriorityItem] = []
        self.in_queue: Set[Tuple[State, Action]] = set()
        
        # 统计
        # Statistics
        self.total_insertions = 0
        self.total_pops = 0
        
        logger.info(f"初始化优先队列: θ={threshold}")
    
    def push(self, state: State, action: Action, priority: float):
        """
        插入或更新项
        Insert or update item
        
        Args:
            state: 状态
                  State
            action: 动作
                   Action
            priority: 优先级（TD误差绝对值）
                     Priority (absolute TD error)
        """
        # 检查阈值
        # Check threshold
        if priority <= self.threshold:
            return
        
        key = (state, action)
        
        # 如果已在队列中，不重复插入（简化实现）
        # If already in queue, don't insert again (simplified)
        if key in self.in_queue:
            return
        
        # 插入（使用负优先级实现最大堆）
        # Insert (use negative priority for max-heap)
        item = PriorityItem(-priority, key)
        heapq.heappush(self.heap, item)
        self.in_queue.add(key)
        self.total_insertions += 1
    
    def pop(self) -> Optional[Tuple[State, Action, float]]:
        """
        弹出最高优先级项
        Pop highest priority item
        
        Returns:
            (状态, 动作, 优先级) 或 None
            (state, action, priority) or None
        """
        while self.heap:
            item = heapq.heappop(self.heap)
            state, action = item.state_action
            
            # 从集合中移除
            # Remove from set
            self.in_queue.discard((state, action))
            self.total_pops += 1
            
            # 返回（转换回正优先级）
            # Return (convert back to positive priority)
            return state, action, -item.priority
        
        return None
    
    def is_empty(self) -> bool:
        """
        检查是否为空
        Check if empty
        """
        return len(self.heap) == 0
    
    def size(self) -> int:
        """
        获取队列大小
        Get queue size
        """
        return len(self.heap)
    
    def clear(self):
        """
        清空队列
        Clear queue
        """
        self.heap.clear()
        self.in_queue.clear()


# ================================================================================
# 第8.4.2节：优先级扫描算法
# Section 8.4.2: Prioritized Sweeping Algorithm
# ================================================================================

class PrioritizedSweeping:
    """
    优先级扫描算法
    Prioritized Sweeping Algorithm
    
    智能地选择要更新的状态-动作对
    Intelligently select state-action pairs to update
    
    关键创新 Key Innovation:
    1. 优先更新TD误差大的(s,a)
       Prioritize (s,a) with large TD errors
    2. 传播更新到前驱状态
       Propagate updates to predecessors
    3. 使用阈值避免微小更新
       Use threshold to avoid tiny updates
    
    与Dyna-Q的区别 Difference from Dyna-Q:
    - Dyna-Q: 随机选择(s,a)更新
             Random selection of (s,a) to update
    - PS: 基于优先级选择(s,a)
          Priority-based selection of (s,a)
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 n_planning_steps: int = 10,
                 gamma: float = 0.95,
                 alpha: float = 0.1,
                 epsilon: float = 0.1,
                 threshold: float = 1e-4):
        """
        初始化优先级扫描
        Initialize prioritized sweeping
        
        Args:
            env: 环境
                Environment
            n_planning_steps: 每步规划次数
                             Planning steps per real step
            gamma: 折扣因子
                  Discount factor
            alpha: 学习率
                  Learning rate
            epsilon: 探索率
                    Exploration rate
            threshold: 优先级阈值
                      Priority threshold
        """
        self.env = env
        self.n_planning_steps = n_planning_steps
        self.gamma = gamma
        self.alpha = alpha
        self.threshold = threshold
        
        # Q函数
        # Q function
        self.Q = ActionValueFunction(
            env.state_space,
            env.action_space,
            initial_value=0.0
        )
        
        # ε-贪婪策略
        # ε-greedy policy
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
        
        # 反向模型：记录哪些(s,a)可以到达每个状态
        # Reverse model: record which (s,a) can reach each state
        self.predecessors: Dict[State, Set[Tuple[State, Action]]] = defaultdict(set)
        
        # 优先队列
        # Priority queue
        self.pqueue = PriorityQueue(threshold)
        
        # 统计
        # Statistics
        self.real_steps = 0
        self.planning_steps = 0
        self.episode_count = 0
        self.episode_returns = []
        self.episode_lengths = []
        
        logger.info(f"初始化优先级扫描: n={n_planning_steps}, θ={threshold}")
    
    def compute_td_error(self, state: State, action: Action,
                        next_state: State, reward: float) -> float:
        """
        计算TD误差
        Compute TD error
        
        Args:
            state: 当前状态
                  Current state
            action: 动作
                   Action
            next_state: 下一状态
                       Next state
            reward: 奖励
                   Reward
        
        Returns:
            TD误差
            TD error
        """
        # 当前Q值
        # Current Q value
        current_q = self.Q.get_value(state, action)
        
        # TD目标
        # TD target
        if not next_state.is_terminal:
            max_next_q = max(
                self.Q.get_value(next_state, a)
                for a in self.env.action_space
            )
        else:
            max_next_q = 0.0
        
        td_target = reward + self.gamma * max_next_q
        
        # TD误差
        # TD error
        return td_target - current_q
    
    def update_q(self, state: State, action: Action,
                next_state: State, reward: float):
        """
        更新Q值
        Update Q value
        
        Args:
            state: 状态
                  State
            action: 动作
                   Action
            next_state: 下一状态
                       Next state
            reward: 奖励
                   Reward
        """
        td_error = self.compute_td_error(state, action, next_state, reward)
        old_q = self.Q.get_value(state, action)
        new_q = old_q + self.alpha * td_error
        self.Q.set_value(state, action, new_q)
    
    def planning_step(self):
        """
        执行一步优先级规划
        Execute one prioritized planning step
        """
        # 从优先队列弹出
        # Pop from priority queue
        result = self.pqueue.pop()
        if result is None:
            return
        
        state, action, priority = result
        
        # 从模型获取结果
        # Get result from model
        next_state, reward = self.model.sample(state, action)
        
        # 更新Q值
        # Update Q value
        self.update_q(state, action, next_state, reward)
        
        # 对所有可能导致state的前驱(s̄,ā)
        # For all predecessors (s̄,ā) that could lead to state
        for pred_state, pred_action in self.predecessors.get(state, []):
            # 从模型获取预测
            # Get prediction from model
            pred_next, pred_reward = self.model.sample(pred_state, pred_action)
            
            # 计算前驱的TD误差
            # Compute predecessor's TD error
            pred_td_error = self.compute_td_error(
                pred_state, pred_action, pred_next, pred_reward
            )
            
            # 插入优先队列
            # Insert to priority queue
            pred_priority = abs(pred_td_error)
            self.pqueue.push(pred_state, pred_action, pred_priority)
        
        self.planning_steps += 1
    
    def learn_step(self, state: State, action: Action,
                  next_state: State, reward: float):
        """
        学习一步（包括模型更新和优先级规划）
        Learn one step (including model update and prioritized planning)
        
        Args:
            state: 当前状态
                  Current state
            action: 执行的动作
                   Action taken
            next_state: 下一状态
                       Next state
            reward: 获得的奖励
                   Reward received
        """
        # 更新模型
        # Update model
        self.model.update(state, action, next_state, reward)
        
        # 更新反向模型
        # Update reverse model
        self.predecessors[next_state].add((state, action))
        
        # 计算优先级（TD误差绝对值）
        # Compute priority (absolute TD error)
        td_error = self.compute_td_error(state, action, next_state, reward)
        priority = abs(td_error)
        
        # 如果优先级足够高，插入队列
        # If priority high enough, insert to queue
        if priority > self.threshold:
            self.pqueue.push(state, action, priority)
        
        # 执行n步优先级规划
        # Execute n steps of prioritized planning
        for _ in range(self.n_planning_steps):
            if self.pqueue.is_empty():
                break
            self.planning_step()
        
        self.real_steps += 1
    
    def learn_episode(self) -> Tuple[float, int]:
        """
        学习一个回合
        Learn one episode
        
        Returns:
            (回合回报, 回合长度)
            (episode return, episode length)
        """
        state = self.env.reset()
        episode_return = 0.0
        episode_length = 0
        
        while not state.is_terminal:
            # 选择动作
            # Select action
            action = self.policy.select_action(state)
            
            # 执行动作
            # Execute action
            next_state, reward, done, _ = self.env.step(action)
            
            # 学习
            # Learn
            self.learn_step(state, action, next_state, reward)
            
            # 更新统计
            # Update statistics
            episode_return += reward
            episode_length += 1
            
            state = next_state
            
            if done:
                break
        
        # 记录统计
        # Record statistics
        self.episode_count += 1
        self.episode_returns.append(episode_return)
        self.episode_lengths.append(episode_length)
        
        return episode_return, episode_length
    
    def learn(self,
             n_episodes: int = 100,
             verbose: bool = True) -> ActionValueFunction:
        """
        学习多个回合
        Learn multiple episodes
        
        Args:
            n_episodes: 回合数
                       Number of episodes
            verbose: 是否输出进度
                    Whether to output progress
        
        Returns:
            学习的Q函数
            Learned Q function
        """
        if verbose:
            print(f"\n开始优先级扫描学习: {n_episodes}回合")
            print(f"Starting Prioritized Sweeping learning: {n_episodes} episodes")
            print(f"  参数: n={self.n_planning_steps}, θ={self.threshold}")
        
        for episode in range(n_episodes):
            episode_return, episode_length = self.learn_episode()
            
            if verbose and (episode + 1) % max(1, n_episodes // 10) == 0:
                avg_return = np.mean(self.episode_returns[-10:]) \
                           if len(self.episode_returns) >= 10 \
                           else np.mean(self.episode_returns)
                
                print(f"Episode {episode + 1}/{n_episodes}: "
                      f"Return={episode_return:.2f}, "
                      f"Avg Return={avg_return:.2f}, "
                      f"Queue Size={self.pqueue.size()}, "
                      f"Planning Steps={self.planning_steps}")
        
        if verbose:
            print(f"\n优先级扫描学习完成!")
            print(f"  总真实步数: {self.real_steps}")
            print(f"  总规划步数: {self.planning_steps}")
            print(f"  队列总插入: {self.pqueue.total_insertions}")
            print(f"  队列总弹出: {self.pqueue.total_pops}")
        
        return self.Q


# ================================================================================
# 第8.4.3节：优先级Dyna-Q
# Section 8.4.3: Prioritized Dyna-Q
# ================================================================================

class PrioritizedDynaQ:
    """
    优先级Dyna-Q
    Prioritized Dyna-Q
    
    结合Dyna-Q和优先级扫描
    Combines Dyna-Q with prioritized sweeping
    
    改进 Improvements:
    1. 不随机选择(s,a)更新
       Not random selection of (s,a) to update
    2. 基于TD误差选择
       Based on TD error selection
    3. 更高效的计算利用
       More efficient computation use
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 n_planning_steps: int = 10,
                 gamma: float = 0.95,
                 alpha: float = 0.1,
                 epsilon: float = 0.1,
                 threshold: float = 1e-4):
        """
        初始化优先级Dyna-Q
        Initialize Prioritized Dyna-Q
        """
        self.env = env
        self.n_planning_steps = n_planning_steps
        self.gamma = gamma
        self.alpha = alpha
        self.threshold = threshold
        
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
        
        # 优先队列
        # Priority queue
        self.pqueue = PriorityQueue(threshold)
        
        # 观察的(s,a)对
        # Observed (s,a) pairs
        self.observed_sa: Set[Tuple[State, Action]] = set()
        
        # 统计
        # Statistics
        self.real_steps = 0
        self.planning_steps = 0
        self.episode_returns = []
        
        logger.info(f"初始化优先级Dyna-Q")
    
    def compute_priority(self, state: State, action: Action) -> float:
        """
        计算优先级
        Compute priority
        
        基于模型预测的TD误差
        Based on model-predicted TD error
        """
        if not self.model.is_known(state, action):
            return 0.0
        
        # 从模型获取预测
        # Get prediction from model
        next_state, reward = self.model.sample(state, action)
        
        # 计算TD误差
        # Compute TD error
        current_q = self.Q.get_value(state, action)
        
        if not next_state.is_terminal:
            max_next_q = max(
                self.Q.get_value(next_state, a)
                for a in self.env.action_space
            )
        else:
            max_next_q = 0.0
        
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q
        
        return abs(td_error)
    
    def prioritized_planning_step(self):
        """
        执行优先级规划步骤
        Execute prioritized planning step
        """
        # 如果队列空，重新填充
        # If queue empty, refill
        if self.pqueue.is_empty():
            for state, action in self.observed_sa:
                priority = self.compute_priority(state, action)
                if priority > self.threshold:
                    self.pqueue.push(state, action, priority)
        
        # 从队列弹出并更新
        # Pop from queue and update
        result = self.pqueue.pop()
        if result is None:
            return
        
        state, action, _ = result
        
        # 从模型获取经验
        # Get experience from model
        next_state, reward = self.model.sample(state, action)
        
        # Q-learning更新
        # Q-learning update
        current_q = self.Q.get_value(state, action)
        if not next_state.is_terminal:
            max_next_q = max(
                self.Q.get_value(next_state, a)
                for a in self.env.action_space
            )
        else:
            max_next_q = 0.0
        
        td_target = reward + self.gamma * max_next_q
        new_q = current_q + self.alpha * (td_target - current_q)
        self.Q.set_value(state, action, new_q)
        
        self.planning_steps += 1
    
    def learn_step(self, state: State, action: Action,
                  next_state: State, reward: float):
        """
        学习一步
        Learn one step
        """
        # 直接RL更新
        # Direct RL update
        current_q = self.Q.get_value(state, action)
        if not next_state.is_terminal:
            max_next_q = max(
                self.Q.get_value(next_state, a)
                for a in self.env.action_space
            )
        else:
            max_next_q = 0.0
        
        td_target = reward + self.gamma * max_next_q
        new_q = current_q + self.alpha * (td_target - current_q)
        self.Q.set_value(state, action, new_q)
        
        # 更新模型
        # Update model
        self.model.update(state, action, next_state, reward)
        self.observed_sa.add((state, action))
        
        # 计算并插入优先级
        # Compute and insert priority
        priority = abs(td_target - current_q)
        if priority > self.threshold:
            self.pqueue.push(state, action, priority)
        
        # 优先级规划
        # Prioritized planning
        for _ in range(self.n_planning_steps):
            self.prioritized_planning_step()
        
        self.real_steps += 1


# ================================================================================
# 主函数：演示优先级扫描
# Main Function: Demonstrate Prioritized Sweeping
# ================================================================================

def demonstrate_prioritized_sweeping():
    """
    演示优先级扫描
    Demonstrate prioritized sweeping
    """
    print("\n" + "="*80)
    print("第8.4节：优先级扫描")
    print("Section 8.4: Prioritized Sweeping")
    print("="*80)
    
    from src.ch03_finite_mdp.gridworld import GridWorld
    from .dyna_q import DynaQ
    
    # 创建环境
    # Create environment
    env = GridWorld(rows=6, cols=9,
                   start_pos=(2,0),
                   goal_pos=(0,8),
                   obstacles=[(1,2), (2,2), (3,2), (4,5)])
    
    print(f"\n创建6×9 GridWorld（带障碍）")
    print(f"  起点: (2,0)")
    print(f"  终点: (0,8)")
    print(f"  障碍: {[(1,2), (2,2), (3,2), (4,5)]}")
    
    # 1. 测试优先级扫描
    # 1. Test prioritized sweeping
    print("\n" + "="*60)
    print("1. 优先级扫描测试")
    print("1. Prioritized Sweeping Test")
    print("="*60)
    
    ps = PrioritizedSweeping(
        env, n_planning_steps=10, gamma=0.95,
        alpha=0.1, epsilon=0.1, threshold=1e-4
    )
    
    print("\n学习50回合...")
    Q_ps = ps.learn(n_episodes=50, verbose=True)
    
    # 2. 比较优先级扫描 vs Dyna-Q
    # 2. Compare Prioritized Sweeping vs Dyna-Q
    print("\n" + "="*60)
    print("2. 优先级扫描 vs Dyna-Q比较")
    print("2. Prioritized Sweeping vs Dyna-Q Comparison")
    print("="*60)
    
    n_episodes = 50
    n_runs = 5
    
    ps_returns = []
    dyna_returns = []
    ps_planning = []
    dyna_planning = []
    
    print(f"\n运行{n_runs}次实验，每次{n_episodes}回合...")
    
    for run in range(n_runs):
        # 优先级扫描
        # Prioritized sweeping
        ps = PrioritizedSweeping(
            env, n_planning_steps=10, gamma=0.95,
            alpha=0.1, epsilon=0.1, threshold=1e-4
        )
        ps.learn(n_episodes=n_episodes, verbose=False)
        ps_returns.append(np.mean(ps.episode_returns[-10:]))
        ps_planning.append(ps.planning_steps)
        
        # Dyna-Q
        dyna = DynaQ(
            env, n_planning_steps=10, gamma=0.95,
            alpha=0.1, epsilon=0.1
        )
        dyna.learn(n_episodes=n_episodes, verbose=False)
        dyna_returns.append(np.mean(dyna.episode_returns[-10:]))
        dyna_planning.append(dyna.planning_steps)
        
        print(f"  运行{run+1}: PS回报={ps_returns[-1]:.2f}, "
              f"Dyna回报={dyna_returns[-1]:.2f}")
    
    # 打印比较结果
    # Print comparison results
    print("\n比较结果:")
    print(f"{'算法':<20} {'平均回报':<20} {'平均规划步数':<15}")
    print("-" * 55)
    
    ps_return_mean = np.mean(ps_returns)
    ps_return_std = np.std(ps_returns)
    ps_planning_mean = np.mean(ps_planning)
    
    dyna_return_mean = np.mean(dyna_returns)
    dyna_return_std = np.std(dyna_returns)
    dyna_planning_mean = np.mean(dyna_planning)
    
    print(f"{'优先级扫描':<20} {ps_return_mean:.2f} ± {ps_return_std:.2f}    {ps_planning_mean:.0f}")
    print(f"{'Dyna-Q':<20} {dyna_return_mean:.2f} ± {dyna_return_std:.2f}    {dyna_planning_mean:.0f}")
    
    if ps_return_mean > dyna_return_mean:
        print("\n✓ 优先级扫描表现更好")
    else:
        print("\n✓ Dyna-Q表现更好")
    
    # 3. 演示优先级队列的工作
    # 3. Demonstrate priority queue working
    print("\n" + "="*60)
    print("3. 优先级队列工作演示")
    print("3. Priority Queue Working Demo")
    print("="*60)
    
    # 创建新的PS实例
    # Create new PS instance
    ps_demo = PrioritizedSweeping(
        env, n_planning_steps=5, gamma=0.95,
        alpha=0.1, epsilon=0.1, threshold=1e-4
    )
    
    # 执行几步真实交互
    # Execute some real interactions
    state = env.reset()
    print("\n执行5步真实交互...")
    
    for step in range(5):
        action = ps_demo.policy.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        # 计算TD误差作为优先级
        # Compute TD error as priority
        td_error = ps_demo.compute_td_error(state, action, next_state, reward)
        priority = abs(td_error)
        
        print(f"  步{step+1}: ({state.id}, {action.id}) -> {next_state.id}, "
              f"优先级={priority:.4f}")
        
        ps_demo.learn_step(state, action, next_state, reward)
        
        if done:
            break
        state = next_state
    
    print(f"\n队列大小: {ps_demo.pqueue.size()}")
    print(f"规划步数: {ps_demo.planning_steps}")
    
    # 总结
    # Summary
    print("\n" + "="*80)
    print("优先级扫描总结")
    print("Prioritized Sweeping Summary")
    print("="*80)
    
    print("""
    关键要点 Key Takeaways:
    =======================
    
    1. 优先更新重要的状态-动作
       Prioritize important state-actions
       
    2. TD误差度量重要性
       TD error measures importance
       
    3. 传播更新到前驱
       Propagate updates to predecessors
       
    4. 比随机更新更高效
       More efficient than random updates
       
    5. 需要额外的数据结构开销
       Requires extra data structure overhead
    
    适用场景 Use Cases:
    - 大状态空间
      Large state spaces
    - 稀疏奖励
      Sparse rewards
    - 计算资源有限
      Limited computational resources
    """)


if __name__ == "__main__":
    demonstrate_prioritized_sweeping()