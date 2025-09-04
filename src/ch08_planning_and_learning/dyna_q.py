"""
================================================================================
第8.2节：Dyna-Q算法 - 集成规划、动作和学习
Section 8.2: Dyna-Q - Integrating Planning, Acting, and Learning
================================================================================

Dyna架构统一了所有功能！
Dyna architecture unifies all functions!

Dyna的核心思想 Core Ideas of Dyna:
1. 直接RL：从真实经验学习
   Direct RL: Learn from real experience
2. 模型学习：从真实经验学习模型
   Model Learning: Learn model from real experience  
3. 规划：从模拟经验学习
   Planning: Learn from simulated experience

算法流程 Algorithm Flow:
对每个时间步 For each timestep:
  (a) 与环境交互，获得真实经验
      Interact with environment, get real experience
  (b) 直接RL：用真实经验更新Q
      Direct RL: Update Q with real experience
  (c) 模型学习：用真实经验更新模型
      Model learning: Update model with real experience
  (d) 规划：重复n次
      Planning: Repeat n times
      - 从模型采样之前观察的(s,a)
        Sample previously observed (s,a) from model
      - 用模型生成(s',r)
        Generate (s',r) with model
      - 用模拟经验更新Q
        Update Q with simulated experience

优势 Advantages:
- 高效利用经验
  Efficient use of experience
- 快速适应
  Fast adaptation
- 在线学习
  Online learning
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import time
import matplotlib.pyplot as plt

# 导入基础组件
# Import base components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.ch03_finite_mdp.mdp_framework import State, Action, MDPEnvironment
from src.ch03_finite_mdp.policies_and_values import (
    Policy, ActionValueFunction
)
from ch04_monte_carlo.mc_control import EpsilonGreedyPolicy

from .models_and_planning import DeterministicModel, TabularModel

# 设置日志
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第8.2.1节：模拟经验
# Section 8.2.1: Simulated Experience
# ================================================================================

@dataclass
class SimulatedExperience:
    """
    模拟经验
    Simulated Experience
    
    从模型生成的经验
    Experience generated from model
    """
    state: State
    action: Action
    next_state: State
    reward: float
    is_terminal: bool
    source: str = "simulated"  # "real" or "simulated"


# ================================================================================
# 第8.2.2节：Dyna-Q算法
# Section 8.2.2: Dyna-Q Algorithm
# ================================================================================

class DynaQ:
    """
    Dyna-Q算法
    Dyna-Q Algorithm
    
    最简单的Dyna架构实现
    Simplest implementation of Dyna architecture
    
    关键组件 Key Components:
    1. Q函数：动作价值函数
       Q function: Action value function
    2. 模型：环境的内部表示
       Model: Internal representation of environment
    3. 规划步数n：每个真实步骤后的规划步数
       Planning steps n: Planning steps after each real step
    
    算法步骤 Algorithm Steps:
    1. 初始化Q(s,a)和Model(s,a)
       Initialize Q(s,a) and Model(s,a)
    2. 循环（对每个回合）：
       Loop (for each episode):
       a. 初始化S
          Initialize S
       b. 循环（对每个步骤）：
          Loop (for each step):
          i. 选择A from S using ε-greedy(Q)
             Choose A from S using ε-greedy(Q)
          ii. 执行A，观察R,S'
              Take A, observe R,S'
          iii. Q(S,A) ← Q(S,A) + α[R + γmax_a Q(S',a) - Q(S,A)]
          iv. Model(S,A) ← (S',R) (确定性情况)
          v. 重复n次（规划）：
             Repeat n times (planning):
             - 随机选择之前观察的S,A
               Randomly select previously observed S,A
             - S',R ← Model(S,A)
             - Q(S,A) ← Q(S,A) + α[R + γmax_a Q(S',a) - Q(S,A)]
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 n_planning_steps: int = 5,
                 gamma: float = 0.95,
                 alpha: float = 0.1,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 1.0,
                 epsilon_min: float = 0.01):
        """
        初始化Dyna-Q
        Initialize Dyna-Q
        
        Args:
            env: 环境
                Environment
            n_planning_steps: 每个真实步骤后的规划步数
                             Planning steps after each real step
            gamma: 折扣因子
                  Discount factor
            alpha: 学习率
                  Learning rate
            epsilon: 探索率
                    Exploration rate
            epsilon_decay: ε衰减率
                         ε decay rate
            epsilon_min: 最小ε
                        Minimum ε
        """
        self.env = env
        self.n_planning_steps = n_planning_steps
        self.gamma = gamma
        self.alpha = alpha
        
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
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            action_space=env.action_space
        )
        
        # 模型（使用确定性模型）
        # Model (using deterministic model)
        self.model = DeterministicModel(
            env.state_space,
            env.action_space
        )
        
        # 记录观察过的(s,a)对
        # Record observed (s,a) pairs
        self.observed_sa_pairs: Set[Tuple[State, Action]] = set()
        
        # 统计
        # Statistics
        self.real_steps = 0
        self.planning_steps = 0
        self.episode_count = 0
        self.episode_returns = []
        self.episode_lengths = []
        
        logger.info(f"初始化Dyna-Q: n={n_planning_steps}, γ={gamma}, α={alpha}, ε={epsilon}")
    
    def q_learning_update(self, state: State, action: Action,
                         next_state: State, reward: float):
        """
        Q-learning更新
        Q-learning update
        
        TD(0)更新规则
        TD(0) update rule
        
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
        # 计算TD目标
        # Compute TD target
        if not next_state.is_terminal:
            max_q = max(
                self.Q.get_value(next_state, a)
                for a in self.env.action_space
            )
        else:
            max_q = 0.0
        
        td_target = reward + self.gamma * max_q
        
        # 更新Q值
        # Update Q value
        old_q = self.Q.get_value(state, action)
        new_q = old_q + self.alpha * (td_target - old_q)
        self.Q.set_value(state, action, new_q)
    
    def planning_step(self):
        """
        执行一步规划
        Execute one planning step
        
        从模型采样并更新Q
        Sample from model and update Q
        """
        if not self.observed_sa_pairs:
            return
        
        # 随机选择之前观察的(s,a)
        # Randomly select previously observed (s,a)
        state, action = list(self.observed_sa_pairs)[
            np.random.randint(len(self.observed_sa_pairs))
        ]
        
        # 从模型生成经验
        # Generate experience from model
        next_state, reward = self.model.sample(state, action)
        
        # Q-learning更新
        # Q-learning update
        self.q_learning_update(state, action, next_state, reward)
        
        self.planning_steps += 1
    
    def learn_step(self, state: State, action: Action,
                  next_state: State, reward: float):
        """
        学习一步（包括直接RL、模型学习和规划）
        Learn one step (including direct RL, model learning, and planning)
        
        这是Dyna-Q的核心
        This is the core of Dyna-Q
        
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
        # (a) 直接RL：用真实经验更新Q
        # (a) Direct RL: Update Q with real experience
        self.q_learning_update(state, action, next_state, reward)
        
        # (b) 模型学习：更新模型
        # (b) Model learning: Update model
        self.model.update(state, action, next_state, reward)
        
        # 记录观察的(s,a)
        # Record observed (s,a)
        self.observed_sa_pairs.add((state, action))
        
        # (c) 规划：执行n步规划
        # (c) Planning: Execute n planning steps
        for _ in range(self.n_planning_steps):
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
            
            # 学习（直接RL + 模型学习 + 规划）
            # Learn (direct RL + model learning + planning)
            self.learn_step(state, action, next_state, reward)
            
            # 更新统计
            # Update statistics
            episode_return += reward
            episode_length += 1
            
            # 下一状态
            # Next state
            state = next_state
            
            if done:
                break
        
        # 衰减ε
        # Decay ε
        self.policy.decay_epsilon()
        
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
            print(f"\n开始Dyna-Q学习: {n_episodes}回合")
            print(f"Starting Dyna-Q learning: {n_episodes} episodes")
            print(f"  参数: n={self.n_planning_steps}, γ={self.gamma}, α={self.alpha}")
        
        for episode in range(n_episodes):
            episode_return, episode_length = self.learn_episode()
            
            if verbose and (episode + 1) % max(1, n_episodes // 10) == 0:
                avg_return = np.mean(self.episode_returns[-10:]) \
                           if len(self.episode_returns) >= 10 \
                           else np.mean(self.episode_returns)
                
                print(f"Episode {episode + 1}/{n_episodes}: "
                      f"Return={episode_return:.2f}, "
                      f"Avg Return={avg_return:.2f}, "
                      f"Length={episode_length}, "
                      f"Real Steps={self.real_steps}, "
                      f"Planning Steps={self.planning_steps}")
        
        if verbose:
            print(f"\nDyna-Q学习完成!")
            print(f"  总真实步数: {self.real_steps}")
            print(f"  总规划步数: {self.planning_steps}")
            print(f"  规划/真实比率: {self.planning_steps/max(1,self.real_steps):.1f}")
        
        return self.Q


# ================================================================================
# 第8.2.3节：Dyna-Q+算法
# Section 8.2.3: Dyna-Q+ Algorithm
# ================================================================================

class DynaQPlus(DynaQ):
    """
    Dyna-Q+算法
    Dyna-Q+ Algorithm
    
    处理变化的环境
    Handles changing environments
    
    核心创新 Core Innovation:
    给长时间未访问的(s,a)添加探索奖励
    Add exploration bonus to long-unvisited (s,a)
    
    奖励修正 Reward Modification:
    r + κ√τ
    其中 where:
    - κ: 小常数
         Small constant
    - τ: 自上次访问以来的时间
         Time since last visit
    
    鼓励探索旧的状态-动作对
    Encourages exploration of old state-action pairs
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 n_planning_steps: int = 5,
                 gamma: float = 0.95,
                 alpha: float = 0.1,
                 epsilon: float = 0.1,
                 kappa: float = 0.001):
        """
        初始化Dyna-Q+
        Initialize Dyna-Q+
        
        Args:
            env: 环境
                Environment
            n_planning_steps: 规划步数
                             Planning steps
            gamma: 折扣因子
                  Discount factor
            alpha: 学习率
                  Learning rate
            epsilon: 探索率
                    Exploration rate
            kappa: 探索奖励系数
                  Exploration bonus coefficient
        """
        super().__init__(env, n_planning_steps, gamma, alpha, epsilon)
        
        self.kappa = kappa
        
        # 记录每个(s,a)的最后访问时间
        # Record last visit time for each (s,a)
        self.last_visit_time: Dict[Tuple[State, Action], int] = defaultdict(int)
        
        # 当前时间步
        # Current time step
        self.current_time = 0
        
        logger.info(f"初始化Dyna-Q+: κ={kappa}")
    
    def get_exploration_bonus(self, state: State, action: Action) -> float:
        """
        获取探索奖励
        Get exploration bonus
        
        Args:
            state: 状态
                  State
            action: 动作
                   Action
        
        Returns:
            探索奖励 κ√τ
            Exploration bonus
        """
        key = (state, action)
        time_since_visit = self.current_time - self.last_visit_time[key]
        return self.kappa * np.sqrt(time_since_visit)
    
    def planning_step(self):
        """
        执行一步规划（带探索奖励）
        Execute one planning step (with exploration bonus)
        """
        if not self.observed_sa_pairs:
            return
        
        # 随机选择之前观察的(s,a)
        # Randomly select previously observed (s,a)
        state, action = list(self.observed_sa_pairs)[
            np.random.randint(len(self.observed_sa_pairs))
        ]
        
        # 从模型生成经验
        # Generate experience from model
        next_state, reward = self.model.sample(state, action)
        
        # 添加探索奖励
        # Add exploration bonus
        reward += self.get_exploration_bonus(state, action)
        
        # Q-learning更新
        # Q-learning update
        self.q_learning_update(state, action, next_state, reward)
        
        self.planning_steps += 1
    
    def learn_step(self, state: State, action: Action,
                  next_state: State, reward: float):
        """
        学习一步（更新访问时间）
        Learn one step (update visit time)
        """
        # 更新访问时间
        # Update visit time
        self.last_visit_time[(state, action)] = self.current_time
        self.current_time += 1
        
        # 调用父类方法
        # Call parent method
        super().learn_step(state, action, next_state, reward)


# ================================================================================
# 第8.2.4节：Dyna-Q比较器
# Section 8.2.4: Dyna-Q Comparator
# ================================================================================

class DynaQComparator:
    """
    Dyna-Q比较器
    Dyna-Q Comparator
    
    比较不同规划步数的效果
    Compare effects of different planning steps
    
    实验设计 Experimental Design:
    1. 固定环境和参数
       Fix environment and parameters
    2. 改变规划步数n
       Vary planning steps n
    3. 比较学习曲线
       Compare learning curves
    
    关键指标 Key Metrics:
    - 收敛速度
      Convergence speed
    - 最终性能
      Final performance
    - 计算成本
      Computational cost
    """
    
    def __init__(self, env: MDPEnvironment):
        """
        初始化比较器
        Initialize comparator
        
        Args:
            env: 环境
                Environment
        """
        self.env = env
        self.results = {}
        
        logger.info("初始化Dyna-Q比较器")
    
    def compare_planning_steps(self,
                              n_values: List[int] = [0, 5, 50],
                              n_episodes: int = 50,
                              n_runs: int = 10,
                              verbose: bool = True) -> Dict:
        """
        比较不同规划步数
        Compare different planning steps
        
        Args:
            n_values: 规划步数列表
                     List of planning steps
            n_episodes: 每次运行的回合数
                       Episodes per run
            n_runs: 运行次数
                   Number of runs
            verbose: 是否输出进度
                    Whether to output progress
        
        Returns:
            比较结果
            Comparison results
        """
        if verbose:
            print("\n" + "="*80)
            print("Dyna-Q规划步数比较实验")
            print("Dyna-Q Planning Steps Comparison Experiment")
            print("="*80)
            print(f"比较n值: {n_values}")
            print(f"实验: {n_episodes}回合 × {n_runs}次运行")
        
        results = {n: {
            'returns': [],
            'lengths': [],
            'real_steps': [],
            'planning_steps': []
        } for n in n_values}
        
        for run in range(n_runs):
            if verbose:
                print(f"\n运行 {run + 1}/{n_runs}:")
            
            for n in n_values:
                # 创建Dyna-Q算法
                # Create Dyna-Q algorithm
                dyna_q = DynaQ(
                    self.env,
                    n_planning_steps=n,
                    gamma=0.95,
                    alpha=0.1,
                    epsilon=0.1
                )
                
                # 学习
                # Learn
                dyna_q.learn(n_episodes=n_episodes, verbose=False)
                
                # 记录结果
                # Record results
                results[n]['returns'].append(dyna_q.episode_returns)
                results[n]['lengths'].append(dyna_q.episode_lengths)
                results[n]['real_steps'].append(dyna_q.real_steps)
                results[n]['planning_steps'].append(dyna_q.planning_steps)
                
                if verbose:
                    avg_return = np.mean(dyna_q.episode_returns[-10:])
                    print(f"  n={n}: 最终平均回报={avg_return:.2f}, "
                          f"真实步数={dyna_q.real_steps}, "
                          f"规划步数={dyna_q.planning_steps}")
        
        # 分析结果
        # Analyze results
        self.results = self._analyze_results(results)
        
        if verbose:
            self._print_comparison_summary()
        
        return self.results
    
    def _analyze_results(self, results: Dict) -> Dict:
        """
        分析结果
        Analyze results
        """
        analyzed = {}
        
        for n, data in results.items():
            returns_array = np.array(data['returns'])
            lengths_array = np.array(data['lengths'])
            
            analyzed[n] = {
                'mean_returns': np.mean(returns_array, axis=0),
                'std_returns': np.std(returns_array, axis=0),
                'mean_lengths': np.mean(lengths_array, axis=0),
                'final_return_mean': np.mean([r[-1] for r in data['returns']]),
                'final_return_std': np.std([r[-1] for r in data['returns']]),
                'total_real_steps': np.mean(data['real_steps']),
                'total_planning_steps': np.mean(data['planning_steps'])
            }
        
        return analyzed
    
    def _print_comparison_summary(self):
        """
        打印比较摘要
        Print comparison summary
        """
        print("\n" + "="*80)
        print("Dyna-Q比较结果摘要")
        print("Dyna-Q Comparison Results Summary")
        print("="*80)
        
        print(f"\n{'n值':<10} {'最终回报':<20} {'真实步数':<15} {'规划步数':<15}")
        print("-" * 60)
        
        for n, data in sorted(self.results.items()):
            final_return = f"{data['final_return_mean']:.2f} ± {data['final_return_std']:.2f}"
            real_steps = f"{data['total_real_steps']:.0f}"
            planning_steps = f"{data['total_planning_steps']:.0f}"
            
            print(f"{n:<10} {final_return:<20} {real_steps:<15} {planning_steps:<15}")
        
        print("""
        典型观察 Typical Observations:
        - n=0: 纯Q-learning，无规划
               Pure Q-learning, no planning
        - n=5: 适度规划，平衡好
               Moderate planning, good balance
        - n=50: 大量规划，收敛快但计算密集
                Heavy planning, fast convergence but computationally intensive
        """)


# ================================================================================
# 主函数：演示Dyna-Q
# Main Function: Demonstrate Dyna-Q
# ================================================================================

def demonstrate_dyna_q():
    """
    演示Dyna-Q算法
    Demonstrate Dyna-Q algorithm
    """
    print("\n" + "="*80)
    print("第8.2节：Dyna-Q算法")
    print("Section 8.2: Dyna-Q Algorithm")
    print("="*80)
    
    from src.ch03_finite_mdp.gridworld import GridWorld
    
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
    
    # 1. 测试基本Dyna-Q
    # 1. Test basic Dyna-Q
    print("\n" + "="*60)
    print("1. 基本Dyna-Q测试")
    print("1. Basic Dyna-Q Test")
    print("="*60)
    
    dyna_q = DynaQ(env, n_planning_steps=5, gamma=0.95, alpha=0.1, epsilon=0.1)
    
    print("\n学习50回合...")
    Q = dyna_q.learn(n_episodes=50, verbose=True)
    
    # 显示一些Q值
    # Show some Q values
    print("\n学习的Q值（起点附近）:")
    start_state = env.state_space[env.get_state_index((2,0))]
    for action in env.action_space:
        q_value = Q.get_value(start_state, action)
        print(f"  Q(start, {action.id}) = {q_value:.3f}")
    
    # 2. 比较不同规划步数
    # 2. Compare different planning steps
    print("\n" + "="*60)
    print("2. 比较不同规划步数")
    print("2. Compare Different Planning Steps")
    print("="*60)
    
    comparator = DynaQComparator(env)
    results = comparator.compare_planning_steps(
        n_values=[0, 5, 50],
        n_episodes=50,
        n_runs=5,
        verbose=True
    )
    
    # 3. 测试Dyna-Q+（处理变化环境）
    # 3. Test Dyna-Q+ (handling changing environments)
    print("\n" + "="*60)
    print("3. Dyna-Q+测试（探索奖励）")
    print("3. Dyna-Q+ Test (Exploration Bonus)")
    print("="*60)
    
    dyna_q_plus = DynaQPlus(
        env, n_planning_steps=5, gamma=0.95, 
        alpha=0.1, epsilon=0.1, kappa=0.001
    )
    
    print("\n学习50回合...")
    Q_plus = dyna_q_plus.learn(n_episodes=50, verbose=False)
    
    avg_return = np.mean(dyna_q_plus.episode_returns[-10:])
    print(f"Dyna-Q+最终平均回报: {avg_return:.2f}")
    
    # 显示探索奖励的效果
    # Show effect of exploration bonus
    print("\n探索奖励示例:")
    sample_state = env.state_space[0]
    sample_action = env.action_space[0]
    
    # 模拟不同的访问时间
    # Simulate different visit times
    for time_diff in [0, 10, 100, 1000]:
        dyna_q_plus.current_time = time_diff
        dyna_q_plus.last_visit_time[(sample_state, sample_action)] = 0
        bonus = dyna_q_plus.get_exploration_bonus(sample_state, sample_action)
        print(f"  时间差={time_diff}: 探索奖励={bonus:.4f}")
    
    # 总结
    # Summary
    print("\n" + "="*80)
    print("Dyna-Q算法总结")
    print("Dyna-Q Algorithm Summary")
    print("="*80)
    
    print("""
    关键要点 Key Takeaways:
    =======================
    
    1. Dyna统一了规划和学习
       Dyna unifies planning and learning
       
    2. 规划显著提高样本效率
       Planning significantly improves sample efficiency
       
    3. n控制计算-性能权衡
       n controls computation-performance tradeoff
       
    4. Dyna-Q+处理非平稳环境
       Dyna-Q+ handles non-stationary environments
       
    5. 模型误差会累积
       Model errors can accumulate
    
    实践建议 Practical Tips:
    - n=5-10通常是好的起点
      n=5-10 is usually a good starting point
    - 模型质量很关键
      Model quality is crucial
    - 考虑计算预算
      Consider computational budget
    """)


if __name__ == "__main__":
    demonstrate_dyna_q()