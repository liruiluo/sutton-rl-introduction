"""
================================================================================
第8.1节：模型与规划
Section 8.1: Models and Planning
================================================================================

模型是环境的内部表示！
Models are internal representations of the environment!

核心概念 Core Concepts:
1. 分布模型 Distribution Model:
   p(s',r|s,a) - 完全的概率分布
   
2. 样本模型 Sample Model:
   能生成样本但不知道概率
   Can generate samples but doesn't know probabilities
   
3. 规划 Planning:
   使用模型改进策略而不与真实环境交互
   Use model to improve policy without real interaction

模型的用途 Uses of Models:
- 模拟经验 Simulate experience
- 规划未来 Plan ahead
- 反事实推理 Counterfactual reasoning

关键权衡 Key Tradeoff:
模型准确性 vs 计算成本
Model accuracy vs computational cost
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from abc import ABC, abstractmethod
import logging

# 导入基础组件
# Import base components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.ch03_finite_mdp.mdp_framework import State, Action, MDPEnvironment
from src.ch03_finite_mdp.policies_and_values import (
    Policy, StateValueFunction, ActionValueFunction
)

# 设置日志
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第8.1.1节：表格模型
# Section 8.1.1: Tabular Model
# ================================================================================

class TabularModel(ABC):
    """
    表格模型基类
    Tabular Model Base Class
    
    存储转移概率和奖励的表格
    Stores transition probabilities and rewards in tables
    
    关键功能 Key Features:
    1. 学习环境动态 Learn environment dynamics
    2. 生成模拟经验 Generate simulated experience
    3. 支持规划 Support planning
    """
    
    def __init__(self,
                 state_space: List[State],
                 action_space: List[Action]):
        """
        初始化表格模型
        Initialize tabular model
        
        Args:
            state_space: 状态空间
                        State space
            action_space: 动作空间
                         Action space
        """
        self.state_space = state_space
        self.action_space = action_space
        
        # 经验计数器
        # Experience counter
        self.experience_count = 0
        
        # 模型更新次数
        # Model update count
        self.update_count = 0
        
        logger.info(f"初始化表格模型: |S|={len(state_space)}, |A|={len(action_space)}")
    
    @abstractmethod
    def update(self, state: State, action: Action,
              next_state: State, reward: float):
        """
        根据经验更新模型
        Update model based on experience
        
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
        pass
    
    @abstractmethod
    def sample(self, state: State, action: Action) -> Tuple[State, float]:
        """
        从模型中采样
        Sample from model
        
        Args:
            state: 状态
                  State
            action: 动作
                   Action
        
        Returns:
            (下一状态, 奖励)
            (next state, reward)
        """
        pass
    
    @abstractmethod
    def get_probability(self, state: State, action: Action,
                       next_state: State) -> float:
        """
        获取转移概率
        Get transition probability
        
        Args:
            state: 当前状态
                  Current state
            action: 动作
                   Action
            next_state: 下一状态
                       Next state
        
        Returns:
            转移概率 p(s'|s,a)
            Transition probability
        """
        pass
    
    @abstractmethod
    def get_expected_reward(self, state: State, action: Action) -> float:
        """
        获取期望奖励
        Get expected reward
        
        Args:
            state: 状态
                  State
            action: 动作
                   Action
        
        Returns:
            期望奖励 E[R|s,a]
            Expected reward
        """
        pass
    
    def is_known(self, state: State, action: Action) -> bool:
        """
        检查(s,a)是否已知
        Check if (s,a) is known
        
        Args:
            state: 状态
                  State
            action: 动作
                   Action
        
        Returns:
            是否有该状态-动作的经验
            Whether have experience for this state-action
        """
        return self.experience_count > 0


# ================================================================================
# 第8.1.2节：确定性模型
# Section 8.1.2: Deterministic Model
# ================================================================================

class DeterministicModel(TabularModel):
    """
    确定性表格模型
    Deterministic Tabular Model
    
    假设环境是确定性的
    Assumes environment is deterministic
    
    简化假设 Simplifying Assumption:
    对每个(s,a)，只有一个可能的s'和r
    For each (s,a), only one possible s' and r
    
    适用场景 Use Cases:
    - 确定性环境
      Deterministic environments
    - 近似确定性环境
      Approximately deterministic environments
    """
    
    def __init__(self,
                 state_space: List[State],
                 action_space: List[Action]):
        """
        初始化确定性模型
        Initialize deterministic model
        """
        super().__init__(state_space, action_space)
        
        # 转移表: (s,a) -> s'
        # Transition table
        self.transitions: Dict[Tuple[State, Action], State] = {}
        
        # 奖励表: (s,a) -> r
        # Reward table
        self.rewards: Dict[Tuple[State, Action], float] = {}
        
        # 访问计数: (s,a) -> count
        # Visit count
        self.visit_counts: Dict[Tuple[State, Action], int] = defaultdict(int)
        
        logger.info("初始化确定性模型")
    
    def update(self, state: State, action: Action,
              next_state: State, reward: float):
        """
        更新确定性模型
        Update deterministic model
        
        直接记录最新的转移
        Simply record the latest transition
        """
        key = (state, action)
        
        # 更新转移和奖励
        # Update transition and reward
        self.transitions[key] = next_state
        self.rewards[key] = reward
        
        # 更新计数
        # Update counts
        self.visit_counts[key] += 1
        self.experience_count += 1
        self.update_count += 1
    
    def sample(self, state: State, action: Action) -> Tuple[State, float]:
        """
        从确定性模型采样
        Sample from deterministic model
        
        返回记录的转移
        Return recorded transition
        """
        key = (state, action)
        
        if key not in self.transitions:
            # 未知的(s,a)，返回随机
            # Unknown (s,a), return random
            next_state = np.random.choice(self.state_space)
            reward = 0.0
        else:
            next_state = self.transitions[key]
            reward = self.rewards[key]
        
        return next_state, reward
    
    def get_probability(self, state: State, action: Action,
                       next_state: State) -> float:
        """
        获取确定性转移概率
        Get deterministic transition probability
        
        要么1要么0
        Either 1 or 0
        """
        key = (state, action)
        
        if key not in self.transitions:
            return 0.0
        
        return 1.0 if self.transitions[key] == next_state else 0.0
    
    def get_expected_reward(self, state: State, action: Action) -> float:
        """
        获取期望奖励
        Get expected reward
        
        确定性情况下就是记录的奖励
        In deterministic case, it's the recorded reward
        """
        key = (state, action)
        return self.rewards.get(key, 0.0)
    
    def is_known(self, state: State, action: Action) -> bool:
        """
        检查是否已知
        Check if known
        """
        return (state, action) in self.transitions


# ================================================================================
# 第8.1.3节：随机模型
# Section 8.1.3: Stochastic Model
# ================================================================================

@dataclass
class TransitionStats:
    """
    转移统计
    Transition Statistics
    
    记录(s,a)的所有可能结果
    Records all possible outcomes for (s,a)
    """
    # 下一状态计数
    # Next state counts
    next_state_counts: Dict[State, int] = field(default_factory=lambda: defaultdict(int))
    
    # 奖励累积
    # Reward accumulation
    reward_sum: float = 0.0
    reward_squared_sum: float = 0.0
    
    # 总计数
    # Total count
    total_count: int = 0
    
    def update(self, next_state: State, reward: float):
        """
        更新统计
        Update statistics
        """
        self.next_state_counts[next_state] += 1
        self.reward_sum += reward
        self.reward_squared_sum += reward * reward
        self.total_count += 1
    
    def get_next_state_probabilities(self) -> Dict[State, float]:
        """
        获取下一状态概率分布
        Get next state probability distribution
        """
        if self.total_count == 0:
            return {}
        
        return {
            s: count / self.total_count
            for s, count in self.next_state_counts.items()
        }
    
    def get_expected_reward(self) -> float:
        """
        获取期望奖励
        Get expected reward
        """
        if self.total_count == 0:
            return 0.0
        return self.reward_sum / self.total_count
    
    def get_reward_variance(self) -> float:
        """
        获取奖励方差
        Get reward variance
        """
        if self.total_count == 0:
            return 0.0
        
        mean = self.get_expected_reward()
        mean_squared = self.reward_squared_sum / self.total_count
        return mean_squared - mean * mean


class StochasticModel(TabularModel):
    """
    随机表格模型
    Stochastic Tabular Model
    
    完整的概率模型
    Full probabilistic model
    
    维护完整分布 Maintains Full Distribution:
    p(s',r|s,a) 通过计数估计
    Estimated through counting
    
    优势 Advantages:
    - 准确建模随机性
      Accurately models stochasticity
    - 支持期望更新
      Supports expected updates
    
    代价 Cost:
    - 需要更多存储
      Requires more storage
    - 需要更多样本收敛
      Needs more samples to converge
    """
    
    def __init__(self,
                 state_space: List[State],
                 action_space: List[Action]):
        """
        初始化随机模型
        Initialize stochastic model
        """
        super().__init__(state_space, action_space)
        
        # 转移统计: (s,a) -> TransitionStats
        # Transition statistics
        self.transition_stats: Dict[Tuple[State, Action], TransitionStats] = \
            defaultdict(TransitionStats)
        
        logger.info("初始化随机模型")
    
    def update(self, state: State, action: Action,
              next_state: State, reward: float):
        """
        更新随机模型
        Update stochastic model
        
        增量更新统计
        Incrementally update statistics
        """
        key = (state, action)
        
        # 更新统计
        # Update statistics
        self.transition_stats[key].update(next_state, reward)
        
        # 更新计数
        # Update counts
        self.experience_count += 1
        self.update_count += 1
    
    def sample(self, state: State, action: Action) -> Tuple[State, float]:
        """
        从随机模型采样
        Sample from stochastic model
        
        根据学习的分布采样
        Sample according to learned distribution
        """
        key = (state, action)
        
        if key not in self.transition_stats:
            # 未知的(s,a)
            # Unknown (s,a)
            next_state = np.random.choice(self.state_space)
            reward = 0.0
        else:
            stats = self.transition_stats[key]
            
            # 根据概率分布采样下一状态
            # Sample next state according to distribution
            probs = stats.get_next_state_probabilities()
            if probs:
                states = list(probs.keys())
                probabilities = list(probs.values())
                next_state = np.random.choice(states, p=probabilities)
            else:
                next_state = np.random.choice(self.state_space)
            
            # 奖励用期望值（简化）
            # Use expected reward (simplification)
            reward = stats.get_expected_reward()
        
        return next_state, reward
    
    def get_probability(self, state: State, action: Action,
                       next_state: State) -> float:
        """
        获取转移概率
        Get transition probability
        """
        key = (state, action)
        
        if key not in self.transition_stats:
            return 0.0
        
        stats = self.transition_stats[key]
        probs = stats.get_next_state_probabilities()
        
        return probs.get(next_state, 0.0)
    
    def get_expected_reward(self, state: State, action: Action) -> float:
        """
        获取期望奖励
        Get expected reward
        """
        key = (state, action)
        
        if key not in self.transition_stats:
            return 0.0
        
        return self.transition_stats[key].get_expected_reward()
    
    def is_known(self, state: State, action: Action) -> bool:
        """
        检查是否已知
        Check if known
        """
        key = (state, action)
        return key in self.transition_stats and \
               self.transition_stats[key].total_count > 0
    
    def get_all_next_states(self, state: State, action: Action) -> List[State]:
        """
        获取所有可能的下一状态
        Get all possible next states
        
        用于期望更新
        For expected updates
        """
        key = (state, action)
        
        if key not in self.transition_stats:
            return []
        
        stats = self.transition_stats[key]
        return list(stats.next_state_counts.keys())


# ================================================================================
# 第8.1.4节：规划智能体
# Section 8.1.4: Planning Agent
# ================================================================================

class PlanningAgent:
    """
    规划智能体
    Planning Agent
    
    使用模型进行规划
    Uses model for planning
    
    规划的本质 Essence of Planning:
    使用模型生成模拟经验，然后用这些经验学习
    Use model to generate simulated experience, then learn from it
    
    规划步骤 Planning Steps:
    1. 从模型采样(s,a)
       Sample (s,a) from model
    2. 使用模型生成(s',r)
       Use model to generate (s',r)
    3. 用模拟经验更新价值函数
       Update value function with simulated experience
    """
    
    def __init__(self,
                 model: TabularModel,
                 state_space: List[State],
                 action_space: List[Action],
                 gamma: float = 0.99,
                 alpha: float = 0.1):
        """
        初始化规划智能体
        Initialize planning agent
        
        Args:
            model: 环境模型
                  Environment model
            state_space: 状态空间
                        State space
            action_space: 动作空间
                         Action space
            gamma: 折扣因子
                  Discount factor
            alpha: 学习率
                  Learning rate
        """
        self.model = model
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.alpha = alpha
        
        # Q函数
        # Q function
        self.Q = ActionValueFunction(
            state_space,
            action_space,
            initial_value=0.0
        )
        
        # 规划步数计数
        # Planning step counter
        self.planning_steps = 0
        
        logger.info(f"初始化规划智能体: γ={gamma}, α={alpha}")
    
    def planning_step(self):
        """
        执行一步规划
        Execute one planning step
        
        随机选择(s,a)并更新
        Randomly select (s,a) and update
        """
        # 随机选择已知的(s,a)
        # Randomly select known (s,a)
        known_pairs = [
            (s, a)
            for s in self.state_space
            for a in self.action_space
            if self.model.is_known(s, a)
        ]
        
        if not known_pairs:
            return
        
        # 随机选择
        # Random selection
        state, action = known_pairs[np.random.randint(len(known_pairs))]
        
        # 从模型生成经验
        # Generate experience from model
        next_state, reward = self.model.sample(state, action)
        
        # Q-learning更新
        # Q-learning update
        if not next_state.is_terminal:
            max_q = max(
                self.Q.get_value(next_state, a)
                for a in self.action_space
            )
        else:
            max_q = 0.0
        
        old_q = self.Q.get_value(state, action)
        td_target = reward + self.gamma * max_q
        new_q = old_q + self.alpha * (td_target - old_q)
        
        self.Q.set_value(state, action, new_q)
        
        self.planning_steps += 1
    
    def plan(self, n_steps: int):
        """
        执行多步规划
        Execute multiple planning steps
        
        Args:
            n_steps: 规划步数
                    Number of planning steps
        """
        for _ in range(n_steps):
            self.planning_step()
    
    def expected_planning_step(self):
        """
        执行期望规划步骤
        Execute expected planning step
        
        使用完整分布而不是采样
        Use full distribution instead of sampling
        """
        if not isinstance(self.model, StochasticModel):
            # 退化为普通规划
            # Degenerate to normal planning
            self.planning_step()
            return
        
        # 随机选择已知的(s,a)
        # Randomly select known (s,a)
        known_pairs = [
            (s, a)
            for s in self.state_space
            for a in self.action_space
            if self.model.is_known(s, a)
        ]
        
        if not known_pairs:
            return
        
        state, action = known_pairs[np.random.randint(len(known_pairs))]
        
        # 计算期望更新
        # Compute expected update
        expected_value = 0.0
        
        # 遍历所有可能的下一状态
        # Iterate over all possible next states
        for next_state in self.model.get_all_next_states(state, action):
            prob = self.model.get_probability(state, action, next_state)
            
            if not next_state.is_terminal:
                max_q = max(
                    self.Q.get_value(next_state, a)
                    for a in self.action_space
                )
            else:
                max_q = 0.0
            
            expected_value += prob * max_q
        
        # 期望奖励
        # Expected reward
        expected_reward = self.model.get_expected_reward(state, action)
        
        # 更新Q值
        # Update Q value
        old_q = self.Q.get_value(state, action)
        td_target = expected_reward + self.gamma * expected_value
        new_q = old_q + self.alpha * (td_target - old_q)
        
        self.Q.set_value(state, action, new_q)
        
        self.planning_steps += 1


# ================================================================================
# 主函数：演示模型与规划
# Main Function: Demonstrate Models and Planning
# ================================================================================

def demonstrate_models_and_planning():
    """
    演示模型与规划
    Demonstrate models and planning
    """
    print("\n" + "="*80)
    print("第8.1节：模型与规划")
    print("Section 8.1: Models and Planning")
    print("="*80)
    
    from src.ch03_finite_mdp.gridworld import GridWorld
    
    # 创建环境
    # Create environment
    env = GridWorld(rows=3, cols=3,
                   start_pos=(0,0),
                   goal_pos=(2,2))
    
    print(f"\n创建3×3 GridWorld")
    print(f"  起点: (0,0)")
    print(f"  终点: (2,2)")
    
    # 1. 测试确定性模型
    # 1. Test deterministic model
    print("\n" + "="*60)
    print("1. 确定性模型测试")
    print("1. Deterministic Model Test")
    print("="*60)
    
    det_model = DeterministicModel(env.state_space, env.action_space)
    
    # 添加一些经验
    # Add some experiences
    state = env.state_space[0]
    for action in env.action_space[:2]:
        next_state = env.state_space[1]
        reward = -1.0
        det_model.update(state, action, next_state, reward)
        print(f"  更新: ({state.id}, {action.id}) -> ({next_state.id}, {reward})")
    
    # 测试采样
    # Test sampling
    print("\n从模型采样:")
    for action in env.action_space[:2]:
        next_state, reward = det_model.sample(state, action)
        print(f"  采样: ({state.id}, {action.id}) -> ({next_state.id}, {reward})")
    
    # 2. 测试随机模型
    # 2. Test stochastic model
    print("\n" + "="*60)
    print("2. 随机模型测试")
    print("2. Stochastic Model Test")
    print("="*60)
    
    stoch_model = StochasticModel(env.state_space, env.action_space)
    
    # 添加多个经验模拟随机性
    # Add multiple experiences to simulate stochasticity
    state = env.state_space[0]
    action = env.action_space[0]
    
    # 模拟不同的结果
    # Simulate different outcomes
    outcomes = [
        (env.state_space[1], -1.0),
        (env.state_space[1], -1.0),
        (env.state_space[2], -2.0),
    ]
    
    for next_state, reward in outcomes:
        stoch_model.update(state, action, next_state, reward)
        print(f"  更新: ({state.id}, {action.id}) -> ({next_state.id}, {reward})")
    
    # 显示学习的分布
    # Show learned distribution
    print(f"\n学习的分布 p(s'|{state.id},{action.id}):")
    for s in env.state_space[:3]:
        prob = stoch_model.get_probability(state, action, s)
        if prob > 0:
            print(f"  p({s.id}) = {prob:.2f}")
    
    expected_r = stoch_model.get_expected_reward(state, action)
    print(f"期望奖励: E[R|{state.id},{action.id}] = {expected_r:.2f}")
    
    # 3. 测试规划智能体
    # 3. Test planning agent
    print("\n" + "="*60)
    print("3. 规划智能体测试")
    print("3. Planning Agent Test")
    print("="*60)
    
    planner = PlanningAgent(
        det_model,
        env.state_space,
        env.action_space,
        gamma=0.9,
        alpha=0.1
    )
    
    # 执行规划
    # Execute planning
    print("\n执行10步规划...")
    planner.plan(10)
    print(f"  完成{planner.planning_steps}步规划")
    
    # 显示一些Q值
    # Show some Q values
    print("\n学习的Q值:")
    for s in env.state_space[:2]:
        for a in env.action_space[:2]:
            q = planner.Q.get_value(s, a)
            if abs(q) > 0.001:
                print(f"  Q({s.id}, {a.id}) = {q:.3f}")
    
    # 总结
    # Summary
    print("\n" + "="*80)
    print("模型与规划总结")
    print("Models and Planning Summary")
    print("="*80)
    
    print("""
    关键要点 Key Takeaways:
    =======================
    
    1. 模型是环境的内部表示
       Models are internal representations of environment
       
    2. 分布模型 vs 样本模型
       Distribution model vs sample model
       
    3. 规划 = 使用模型生成经验 + 学习
       Planning = Generate experience with model + Learn
       
    4. 模型可以显著提高样本效率
       Models can significantly improve sample efficiency
       
    5. 模型误差会影响规划质量
       Model errors affect planning quality
    """)


if __name__ == "__main__":
    demonstrate_models_and_planning()