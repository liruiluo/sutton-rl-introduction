"""
================================================================================
第4.3节：蒙特卡洛控制 - 从经验中学习最优策略
Section 4.3: Monte Carlo Control - Learning Optimal Policy from Experience
================================================================================

MC控制将MC预测扩展到寻找最优策略的问题。
MC control extends MC prediction to the problem of finding optimal policy.

核心挑战：探索-利用权衡
Core challenge: Exploration-Exploitation Trade-off
- 需要探索所有动作来准确估计Q
  Need to explore all actions to estimate Q accurately
- 需要利用当前知识来改进策略
  Need to exploit current knowledge to improve policy

两种主要方法：
Two main approaches:
1. On-Policy MC控制：评估和改进同一个策略
   On-Policy MC Control: Evaluate and improve the same policy
   - 使用ε-贪婪策略保证探索
     Use ε-greedy policy to ensure exploration
   - 简单但可能收敛到次优
     Simple but may converge to suboptimal

2. Off-Policy MC控制：行为策略≠目标策略
   Off-Policy MC Control: Behavior policy ≠ Target policy
   - 行为策略探索，目标策略贪婪
     Behavior policy explores, target policy is greedy
   - 需要重要性采样但可以找到最优
     Needs importance sampling but can find optimal

特殊技巧：
Special techniques:
- 探索性起始（Exploring Starts）：从所有(s,a)对开始
  Exploring Starts: Start from all (s,a) pairs
- ε-贪婪策略：以ε概率随机探索
  ε-greedy policy: Explore randomly with probability ε
- 软策略（Soft Policy）：所有动作都有非零概率
  Soft Policy: All actions have non-zero probability

这是通向Q-learning的重要一步！
This is an important step towards Q-learning!
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from scipy import stats
import time
import random

# 导入基础组件
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from ch02_mdp.mdp_framework import State, Action, MDPEnvironment
from ch02_mdp.policies_and_values import (
    Policy, StateValueFunction, ActionValueFunction,
    StochasticPolicy, DeterministicPolicy
)
from ch04_monte_carlo.mc_foundations import (
    Episode, Experience, Return, MCStatistics
)
from ch04_monte_carlo.mc_prediction import MCPrediction, FirstVisitMC

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第4.3.1节：ε-贪婪策略
# Section 4.3.1: Epsilon-Greedy Policy
# ================================================================================

class EpsilonGreedyPolicy(StochasticPolicy):
    """
    ε-贪婪策略
    Epsilon-Greedy Policy
    
    平衡探索与利用的经典方法
    Classic method to balance exploration and exploitation
    
    策略定义：
    Policy definition:
    π(a|s) = {
        1 - ε + ε/|A|    if a = argmax_a Q(s,a)  (贪婪动作)
        ε/|A|            otherwise                 (探索动作)
    }
    
    关键性质：
    Key properties:
    1. 保证探索：每个动作至少有ε/|A|的概率
       Ensures exploration: Each action has at least ε/|A| probability
    2. 主要利用：贪婪动作有1-ε+ε/|A|的概率
       Mainly exploits: Greedy action has 1-ε+ε/|A| probability
    3. 软策略：π(a|s) > 0 对所有a
       Soft policy: π(a|s) > 0 for all a
    4. 可以退火：ε可以随时间减小
       Can be annealed: ε can decrease over time
    
    为什么需要ε-贪婪？
    Why need ε-greedy?
    - 纯贪婪会陷入局部最优
      Pure greedy gets stuck in local optimum
    - 完全随机学习太慢
      Fully random learns too slowly
    - ε-贪婪是简单有效的折中
      ε-greedy is simple and effective compromise
    
    类比：餐厅选择
    Analogy: Restaurant selection
    - 通常去最喜欢的餐厅（利用）
      Usually go to favorite restaurant (exploit)
    - 偶尔尝试新餐厅（探索）
      Occasionally try new restaurants (explore)
    """
    
    def __init__(self, 
                 Q: ActionValueFunction,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 1.0,
                 epsilon_min: float = 0.01,
                 action_space: Optional[List[Action]] = None):
        """
        初始化ε-贪婪策略
        Initialize ε-greedy policy
        
        Args:
            Q: 动作价值函数
               Action-value function
            epsilon: 初始探索率
                    Initial exploration rate
            epsilon_decay: 衰减因子（每回合乘以此值）
                          Decay factor (multiply each episode)
            epsilon_min: 最小探索率
                        Minimum exploration rate
            action_space: 动作空间（可选，如果不提供则从Q推断）
                        Action space (optional, inferred from Q if not provided)
        
        设计选择：
        Design choices:
        - epsilon=0.1是常见选择（10%探索）
          epsilon=0.1 is common choice (10% exploration)
        - 衰减帮助后期收敛
          Decay helps late convergence
        - 保持最小值避免完全贪婪
          Keep minimum to avoid pure greedy
        """
        # 初始化基类，传入空的policy_probs（我们动态计算）
        # Initialize base class with empty policy_probs (we calculate dynamically)
        super().__init__(policy_probs={})
        self.Q = Q
        self.epsilon = epsilon
        self.epsilon_initial = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # 存储动作空间以符合基类Policy接口
        # Store action space to comply with base Policy interface
        self.action_space = action_space if action_space else Q.actions
        
        # 记录选择历史（用于分析）
        # Record selection history (for analysis)
        self.selection_history = []
        self.exploration_count = 0
        self.exploitation_count = 0
        
        logger.info(f"初始化ε-贪婪策略: ε={epsilon}")
    
    def get_action_probabilities(self, state: State, 
                                action_space: Optional[List[Action]] = None) -> Dict[Action, float]:
        """
        获取动作概率分布
        Get action probability distribution
        
        实现ε-贪婪的概率分配
        Implement ε-greedy probability assignment
        
        兼容基类接口但也支持传入action_space
        Compatible with base interface but also supports passing action_space
        """
        # 使用提供的动作空间或存储的动作空间
        # Use provided action space or stored action space
        actions = action_space if action_space else self.action_space
        probs = {}
        
        # 找到贪婪动作（Q值最大的）
        # Find greedy action (max Q-value)
        q_values = {a: self.Q.get_value(state, a) for a in actions}
        max_q = max(q_values.values())
        
        # 可能有多个最优动作（打破平局）
        # May have multiple optimal actions (tie-breaking)
        greedy_actions = [a for a, q in q_values.items() if q == max_q]
        n_greedy = len(greedy_actions)
        n_actions = len(actions)
        
        # 计算概率
        # Calculate probabilities
        for action in actions:
            if action in greedy_actions:
                # 贪婪动作：基础探索概率 + 额外的利用概率
                # Greedy action: base exploration + extra exploitation
                probs[action] = self.epsilon / n_actions + (1 - self.epsilon) / n_greedy
            else:
                # 非贪婪动作：只有探索概率
                # Non-greedy action: only exploration probability
                probs[action] = self.epsilon / n_actions
        
        return probs
    
    def select_action(self, state: State) -> Action:
        """
        选择动作
        Select action
        
        使用ε-贪婪策略
        Use ε-greedy strategy
        
        遵循基类Policy接口，不需要action_space参数
        Follow base Policy interface, no action_space parameter needed
        """
        # ε概率随机探索
        # Random exploration with probability ε
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
            self.exploration_count += 1
            self.selection_history.append(('explore', state.id, action.id))
        else:
            # 1-ε概率选择贪婪动作
            # Select greedy action with probability 1-ε
            q_values = {a: self.Q.get_value(state, a) for a in self.action_space}
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if q == max_q]
            action = np.random.choice(best_actions)  # 随机打破平局
            self.exploitation_count += 1
            self.selection_history.append(('exploit', state.id, action.id))
        
        return action
    
    def decay_epsilon(self):
        """
        衰减探索率
        Decay exploration rate
        
        用于退火策略
        Used for annealing schedule
        """
        old_epsilon = self.epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        
        if old_epsilon != self.epsilon:
            logger.debug(f"ε衰减: {old_epsilon:.4f} → {self.epsilon:.4f}")
    
    def reset_epsilon(self):
        """
        重置探索率
        Reset exploration rate
        """
        self.epsilon = self.epsilon_initial
        self.exploration_count = 0
        self.exploitation_count = 0
        self.selection_history.clear()
    
    def get_exploration_stats(self) -> Dict[str, Any]:
        """
        获取探索统计
        Get exploration statistics
        """
        total = self.exploration_count + self.exploitation_count
        if total == 0:
            return {
                'epsilon': self.epsilon,
                'exploration_ratio': 0,
                'exploitation_ratio': 0,
                'total_selections': 0
            }
        
        return {
            'epsilon': self.epsilon,
            'exploration_ratio': self.exploration_count / total,
            'exploitation_ratio': self.exploitation_count / total,
            'total_selections': total,
            'exploration_count': self.exploration_count,
            'exploitation_count': self.exploitation_count
        }
    
    def analyze_exploration_pattern(self):
        """
        分析探索模式
        Analyze exploration pattern
        
        展示探索-利用的平衡
        Show exploration-exploitation balance
        """
        print("\n" + "="*60)
        print("ε-贪婪策略探索分析")
        print("ε-Greedy Policy Exploration Analysis")
        print("="*60)
        
        stats = self.get_exploration_stats()
        
        print(f"\n当前ε: {stats['epsilon']:.4f}")
        print(f"总选择次数: {stats['total_selections']}")
        print(f"探索次数: {stats['exploration_count']} ({stats['exploration_ratio']:.2%})")
        print(f"利用次数: {stats['exploitation_count']} ({stats['exploitation_ratio']:.2%})")
        
        # 理论vs实际
        # Theory vs Actual
        print(f"\n理论探索率: {self.epsilon:.2%}")
        print(f"实际探索率: {stats['exploration_ratio']:.2%}")
        
        # 分析最近的模式
        # Analyze recent pattern
        if len(self.selection_history) >= 100:
            recent = self.selection_history[-100:]
            recent_explore = sum(1 for t, _, _ in recent if t == 'explore')
            print(f"\n最近100次:")
            print(f"  探索: {recent_explore}%")
            print(f"  利用: {100 - recent_explore}%")


# ================================================================================
# 第4.3.2节：MC控制基类
# Section 4.3.2: MC Control Base Class
# ================================================================================

class MCControl(ABC):
    """
    蒙特卡洛控制基类
    Monte Carlo Control Base Class
    
    定义MC控制算法的共同结构
    Defines common structure for MC control algorithms
    
    控制 = 预测 + 改进
    Control = Prediction + Improvement
    
    通用流程：
    General flow:
    1. 初始化Q(s,a)和π
       Initialize Q(s,a) and π
    2. 重复：
       Repeat:
       a. 用π生成回合
          Generate episode using π
       b. 更新Q基于回合（预测）
          Update Q based on episode (prediction)
       c. 改进π基于Q（控制）
          Improve π based on Q (control)
    3. 直到收敛
       Until convergence
    
    这就是广义策略迭代（GPI）在MC中的体现！
    This is Generalized Policy Iteration (GPI) in MC!
    
    关键设计决策：
    Key design decisions:
    - 使用Q而不是V（不需要模型）
      Use Q not V (no model needed)
    - 软策略保证探索
      Soft policy ensures exploration
    - 增量更新节省内存
      Incremental updates save memory
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 gamma: float = 1.0,
                 visit_type: str = 'first'):
        """
        初始化MC控制
        Initialize MC Control
        
        Args:
            env: MDP环境
            gamma: 折扣因子
            visit_type: 'first' 或 'every'
        """
        self.env = env
        self.gamma = gamma
        self.visit_type = visit_type
        
        # 初始化Q函数（随机小值避免对称性）
        # Initialize Q function (small random to break symmetry)
        self.Q = ActionValueFunction(
            env.state_space, 
            env.action_space,
            initial_value=0.0
        )
        
        # 添加小的随机噪声打破对称
        # Add small random noise to break symmetry
        for state in env.state_space:
            for action in env.action_space:
                noise = np.random.randn() * 0.01
                self.Q.set_value(state, action, noise)
        
        # 统计收集
        # Statistics collection
        self.statistics = MCStatistics()
        
        # 访问计数
        # Visit counts
        self.sa_visits = defaultdict(int)
        
        # 回合历史
        # Episode history
        self.episodes = []
        
        # 学习曲线
        # Learning curve
        self.learning_curve = []
        self.policy_changes = []
        
        logger.info(f"初始化MC控制: γ={gamma}, visit_type={visit_type}")
    
    @abstractmethod
    def learn(self, n_episodes: int, verbose: bool = True) -> Policy:
        """
        学习最优策略（子类实现）
        Learn optimal policy (implemented by subclasses)
        """
        pass
    
    def generate_episode(self, policy: Policy, 
                        max_steps: int = 1000,
                        exploring_starts: bool = False) -> Episode:
        """
        生成回合
        Generate episode
        
        Args:
            policy: 当前策略
            max_steps: 最大步数
            exploring_starts: 是否使用探索性起始
                            Whether to use exploring starts
        
        探索性起始的重要性：
        Importance of exploring starts:
        - 保证所有(s,a)对被访问
          Ensures all (s,a) pairs are visited
        - 解决探索问题的替代方案
          Alternative solution to exploration problem
        - 但实践中可能不可行
          But may not be feasible in practice
        """
        episode = Episode()
        
        if exploring_starts:
            # 随机选择起始状态和动作
            # Randomly select starting state and action
            non_terminal_states = [s for s in self.env.state_space 
                                  if not s.is_terminal]
            if non_terminal_states:
                state = np.random.choice(non_terminal_states)
                action = np.random.choice(self.env.action_space)
                
                # 强制执行这个动作
                # Force execute this action
                self.env.current_state = state
                next_state, reward, done, _ = self.env.step(action)
                
                exp = Experience(state, action, reward, next_state, done)
                episode.add_experience(exp)
                
                state = next_state
                
                if done:
                    return episode
        else:
            state = self.env.reset()
        
        # 继续正常的回合生成
        # Continue normal episode generation
        for t in range(max_steps):
            action = policy.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            
            exp = Experience(state, action, reward, next_state, done)
            episode.add_experience(exp)
            
            state = next_state
            
            if done:
                break
        
        return episode
    
    def update_Q(self, episode: Episode):
        """
        更新Q函数
        Update Q function
        
        使用MC方法从回合中学习
        Learn from episode using MC method
        """
        returns = episode.compute_returns(self.gamma)
        
        if self.visit_type == 'first':
            # First-visit更新
            sa_pairs_seen = set()
            
            for t, exp in enumerate(episode.experiences):
                sa_pair = (exp.state.id, exp.action.id)
                
                if sa_pair not in sa_pairs_seen:
                    sa_pairs_seen.add(sa_pair)
                    G = returns[t]
                    
                    # 增量更新Q
                    self.sa_visits[sa_pair] += 1
                    n = self.sa_visits[sa_pair]
                    
                    old_q = self.Q.get_value(exp.state, exp.action)
                    new_q = old_q + (G - old_q) / n
                    self.Q.set_value(exp.state, exp.action, new_q)
                    
                    # 更新统计
                    self.statistics.update_action_value(exp.state, exp.action, G)
        
        else:  # every-visit
            for t, exp in enumerate(episode.experiences):
                sa_pair = (exp.state.id, exp.action.id)
                G = returns[t]
                
                # 增量更新Q
                self.sa_visits[sa_pair] += 1
                n = self.sa_visits[sa_pair]
                
                old_q = self.Q.get_value(exp.state, exp.action)
                new_q = old_q + (G - old_q) / n
                self.Q.set_value(exp.state, exp.action, new_q)
                
                # 更新统计
                self.statistics.update_action_value(exp.state, exp.action, G)
    
    def create_greedy_policy(self) -> DeterministicPolicy:
        """
        创建贪婪策略
        Create greedy policy
        
        π(s) = argmax_a Q(s,a)
        """
        policy_map = {}
        
        for state in self.env.state_space:
            if not state.is_terminal:
                # 找最优动作
                best_action = None
                best_value = float('-inf')
                
                for action in self.env.action_space:
                    q_value = self.Q.get_value(state, action)
                    if q_value > best_value:
                        best_value = q_value
                        best_action = action
                
                if best_action:
                    policy_map[state] = best_action
        
        return DeterministicPolicy(policy_map)
    
    def evaluate_policy(self, policy: Policy, n_episodes: int = 100) -> float:
        """
        评估策略
        Evaluate policy
        
        运行多个回合计算平均回报
        Run multiple episodes to compute average return
        """
        total_return = 0.0
        
        for _ in range(n_episodes):
            episode = self.generate_episode(policy, exploring_starts=False)
            if episode.experiences:
                returns = episode.compute_returns(self.gamma)
                total_return += returns[0] if returns else 0
        
        return total_return / n_episodes
    
    def analyze_learning(self):
        """
        分析学习过程
        Analyze learning process
        """
        print("\n" + "="*60)
        print("MC控制学习分析")
        print("MC Control Learning Analysis")
        print("="*60)
        
        print(f"\n总回合数: {len(self.episodes)}")
        print(f"访问的(s,a)对: {len(self.sa_visits)}")
        print(f"平均访问次数: {np.mean(list(self.sa_visits.values())):.1f}")
        
        # 访问频率分布
        # Visit frequency distribution
        visits = list(self.sa_visits.values())
        if visits:
            print(f"\n访问统计:")
            print(f"  最少: {min(visits)}")
            print(f"  最多: {max(visits)}")
            print(f"  中位数: {np.median(visits):.0f}")
            
            # 找出访问很少的(s,a)对
            # Find rarely visited (s,a) pairs
            rare_pairs = sum(1 for v in visits if v < 5)
            print(f"  访问<5次的对: {rare_pairs} ({rare_pairs/len(visits):.1%})")


# ================================================================================
# 第4.3.3节：On-Policy MC控制
# Section 4.3.3: On-Policy MC Control
# ================================================================================

class OnPolicyMCControl(MCControl):
    """
    On-Policy蒙特卡洛控制
    On-Policy Monte Carlo Control
    
    评估和改进同一个策略
    Evaluate and improve the same policy
    
    核心思想：使用软策略（如ε-贪婪）
    Core idea: Use soft policy (e.g., ε-greedy)
    - 策略必须探索（软）
      Policy must explore (soft)
    - 评估这个软策略
      Evaluate this soft policy
    - 改进也保持软
      Improvement also stays soft
    
    算法流程：
    Algorithm flow:
    1. 初始化Q(s,a)任意，π为ε-贪婪
       Initialize Q(s,a) arbitrarily, π as ε-greedy
    2. 重复每个回合：
       Repeat for each episode:
       a. 用π生成回合
          Generate episode using π
       b. 对回合中每个(s,a)：
          For each (s,a) in episode:
          - 计算回报G
            Compute return G
          - 更新Q(s,a)向G
            Update Q(s,a) toward G
       c. 对回合中每个s：
          For each s in episode:
          - 更新π(s)为关于Q的ε-贪婪
            Update π(s) to be ε-greedy w.r.t. Q
    
    收敛性质：
    Convergence properties:
    - 收敛到ε-贪婪策略中的最优
      Converges to best among ε-greedy policies
    - 不是全局最优（因为必须保持探索）
      Not globally optimal (must maintain exploration)
    - 但接近最优当ε很小时
      But near-optimal when ε is small
    
    为什么叫"On-Policy"？
    Why called "On-Policy"?
    因为改进的策略就是生成数据的策略
    Because the policy being improved is the one generating data
    
    类比：自我改进
    Analogy: Self-improvement
    像一个人通过实践自己的方法来改进
    Like a person improving by practicing their own method
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 gamma: float = 1.0,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 visit_type: str = 'first'):
        """
        初始化On-Policy MC控制
        Initialize On-Policy MC Control
        
        Args:
            env: 环境
            gamma: 折扣因子
            epsilon: 探索率
            epsilon_decay: 探索率衰减
            epsilon_min: 最小探索率
            visit_type: 访问类型
        """
        super().__init__(env, gamma, visit_type)
        
        # 创建ε-贪婪策略
        # Create ε-greedy policy
        self.policy = EpsilonGreedyPolicy(
            self.Q, epsilon, epsilon_decay, epsilon_min, env.action_space
        )
        
        # 记录策略改进历史
        # Record policy improvement history
        self.improvement_history = []
        
        logger.info(f"初始化On-Policy MC控制: ε={epsilon}")
    
    def learn(self, n_episodes: int = 1000, 
             verbose: bool = True) -> Policy:
        """
        学习最优策略
        Learn optimal policy
        
        实现On-Policy MC控制算法
        Implement On-Policy MC control algorithm
        """
        if verbose:
            print("\n" + "="*60)
            print("On-Policy MC控制")
            print("On-Policy MC Control")
            print("="*60)
            print(f"  环境: {self.env.name}")
            print(f"  回合数: {n_episodes}")
            print(f"  初始ε: {self.policy.epsilon}")
            print(f"  访问类型: {self.visit_type}")
        
        start_time = time.time()
        
        for episode_num in range(n_episodes):
            # 生成回合（使用当前ε-贪婪策略）
            # Generate episode (using current ε-greedy policy)
            episode = self.generate_episode(self.policy)
            self.episodes.append(episode)
            
            # 从回合中学习（更新Q）
            # Learn from episode (update Q)
            self.update_Q(episode)
            
            # 策略已经通过Q的更新而隐式改进
            # Policy is already implicitly improved through Q update
            # （ε-贪婪自动跟随Q的变化）
            # (ε-greedy automatically follows Q changes)
            
            # 衰减探索率
            # Decay exploration rate
            self.policy.decay_epsilon()
            
            # 记录学习进度
            # Record learning progress
            if episode.experiences:
                episode_return = episode.compute_returns(self.gamma)[0]
                self.learning_curve.append(episode_return)
            
            # 定期输出进度
            # Periodically output progress
            if verbose and (episode_num + 1) % 100 == 0:
                avg_return = np.mean(self.learning_curve[-100:]) if self.learning_curve else 0
                print(f"  Episode {episode_num + 1}/{n_episodes}: "
                      f"平均回报={avg_return:.2f}, ε={self.policy.epsilon:.4f}")
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\n学习完成:")
            print(f"  总时间: {total_time:.2f}秒")
            print(f"  最终ε: {self.policy.epsilon:.4f}")
            print(f"  访问的(s,a)对: {len(self.sa_visits)}")
            
            # 探索统计
            stats = self.policy.get_exploration_stats()
            print(f"  总探索率: {stats['exploration_ratio']:.2%}")
        
        # 返回最终的ε-贪婪策略
        # Return final ε-greedy policy
        return self.policy
    
    def demonstrate_gpi(self):
        """
        演示广义策略迭代
        Demonstrate Generalized Policy Iteration
        
        展示评估和改进的交互
        Show interaction of evaluation and improvement
        """
        print("\n" + "="*60)
        print("On-Policy MC中的GPI")
        print("GPI in On-Policy MC")
        print("="*60)
        
        print("""
        🔄 广义策略迭代 Generalized Policy Iteration
        =============================================
        
        在On-Policy MC中：
        In On-Policy MC:
        
        评估 Evaluation:           改进 Improvement:
        Q^π ← Q                    π ← ε-greedy(Q)
             ↘                    ↙
              ↘                  ↙
               ↘                ↙
                Q^π* ←→ π*
                (最优点 Optimal point)
        
        特点 Characteristics:
        ---------------------
        1. 不完全评估：
           Incomplete evaluation:
           每个回合只更新访问的(s,a)
           Each episode only updates visited (s,a)
        
        2. 隐式改进：
           Implicit improvement:
           ε-贪婪自动跟随Q的变化
           ε-greedy automatically follows Q changes
        
        3. 软收敛：
           Soft convergence:
           收敛到ε-软最优策略
           Converges to ε-soft optimal policy
        
        与DP的区别 Difference from DP:
        -------------------------------
        DP:  完全评估 → 完全改进
             Complete evaluation → Complete improvement
        
        MC:  部分评估 → 隐式改进
             Partial evaluation → Implicit improvement
        
        效果 Effect:
        -----------
        - 更高效（不需要遍历所有状态）
          More efficient (no need to sweep all states)
        - 可能更慢收敛（采样方差）
          May converge slower (sampling variance)
        - 实际可行（不需要模型）
          Practically feasible (no model needed)
        """)


# ================================================================================
# 第4.3.4节：Off-Policy MC控制
# Section 4.3.4: Off-Policy MC Control
# ================================================================================

class OffPolicyMCControl(MCControl):
    """
    Off-Policy蒙特卡洛控制
    Off-Policy Monte Carlo Control
    
    使用不同的策略来探索和学习
    Use different policies for exploration and learning
    
    两个策略：
    Two policies:
    1. 行为策略b(a|s)：生成数据，必须探索
       Behavior policy b(a|s): Generates data, must explore
    2. 目标策略π(a|s)：要学习的策略，可以确定性
       Target policy π(a|s): Policy to learn, can be deterministic
    
    核心技术：重要性采样
    Core technique: Importance Sampling
    - 用b的数据估计π的价值
      Use b's data to estimate π's value
    - 需要重要性采样比率
      Need importance sampling ratio
    
    算法流程：
    Algorithm flow:
    1. 初始化Q(s,a)，π为贪婪
       Initialize Q(s,a), π as greedy
    2. 重复：
       Repeat:
       a. 用b生成回合
          Generate episode using b
       b. 计算重要性采样比率
          Compute importance sampling ratio
       c. 用加权回报更新Q
          Update Q with weighted returns
       d. 更新π为关于Q贪婪
          Update π to be greedy w.r.t. Q
    
    优势：
    Advantages:
    - 可以学习最优确定性策略
      Can learn optimal deterministic policy
    - 可以重用任何策略的数据
      Can reuse data from any policy
    - 更灵活的探索策略
      More flexible exploration strategy
    
    劣势：
    Disadvantages:
    - 高方差（重要性采样）
      High variance (importance sampling)
    - 需要b(a|s) > 0当π(a|s) > 0
      Need b(a|s) > 0 when π(a|s) > 0
    - 收敛可能很慢
      Convergence can be slow
    
    为什么叫"Off-Policy"？
    Why called "Off-Policy"?
    因为目标策略"离线"学习，不直接生成数据
    Because target policy learns "offline", not directly generating data
    
    类比：观察学习
    Analogy: Observational learning
    像通过观察别人来学习最优行为
    Like learning optimal behavior by observing others
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 gamma: float = 1.0,
                 behavior_epsilon: float = 0.3,
                 visit_type: str = 'first'):
        """
        初始化Off-Policy MC控制
        Initialize Off-Policy MC Control
        
        Args:
            env: 环境
            gamma: 折扣因子
            behavior_epsilon: 行为策略的探索率
                            Exploration rate for behavior policy
            visit_type: 访问类型
        """
        super().__init__(env, gamma, visit_type)
        
        # 行为策略（ε-贪婪，探索）
        # Behavior policy (ε-greedy, explores)
        self.behavior_policy = EpsilonGreedyPolicy(
            self.Q, 
            epsilon=behavior_epsilon,
            epsilon_decay=1.0,  # 不衰减，保持探索
            epsilon_min=behavior_epsilon,
            action_space=env.action_space
        )
        
        # 目标策略（贪婪，确定性）
        # Target policy (greedy, deterministic)
        self.target_policy = self.create_greedy_policy()
        
        # 累积分母（用于加权重要性采样）
        # Cumulative denominator (for weighted importance sampling)
        self.C = defaultdict(float)
        
        # 记录重要性采样比率
        # Record importance sampling ratios
        self.importance_ratios = []
        
        logger.info(f"初始化Off-Policy MC控制: 行为ε={behavior_epsilon}")
    
    def compute_importance_ratio(self, episode: Episode, t: int) -> float:
        """
        计算重要性采样比率
        Compute importance sampling ratio
        
        ρ_{t:T-1} = ∏_{k=t}^{T-1} [π(A_k|S_k) / b(A_k|S_k)]
        
        这是off-policy的关键！
        This is the key to off-policy!
        
        Args:
            episode: 回合
            t: 起始时间步
        
        Returns:
            重要性采样比率
            Importance sampling ratio
        
        数学原理：
        Mathematical principle:
        - 期望的变换
          Transformation of expectation
        - E_b[ρ × G] = E_π[G]
        - 使b的数据无偏估计π
          Makes b's data unbiased for π
        """
        ratio = 1.0
        
        for k in range(t, len(episode.experiences)):
            exp = episode.experiences[k]
            
            # 获取目标策略概率
            # Get target policy probability
            if isinstance(self.target_policy, DeterministicPolicy):
                if exp.state in self.target_policy.policy_map:
                    target_action = self.target_policy.policy_map[exp.state]
                    target_prob = 1.0 if exp.action.id == target_action.id else 0.0
                else:
                    target_prob = 0.0
            else:
                target_probs = self.target_policy.get_action_probabilities(
                    exp.state
                )
                target_prob = target_probs.get(exp.action, 0.0)
            
            # 获取行为策略概率
            # Get behavior policy probability
            behavior_probs = self.behavior_policy.get_action_probabilities(
                exp.state
            )
            behavior_prob = behavior_probs.get(exp.action, 1e-10)  # 避免除零
            
            # 累积比率
            # Accumulate ratio
            ratio *= target_prob / behavior_prob
            
            # 如果比率为0，后续都是0
            # If ratio is 0, all subsequent are 0
            if ratio == 0:
                break
        
        return ratio
    
    def learn(self, n_episodes: int = 1000,
             verbose: bool = True) -> Policy:
        """
        学习最优策略
        Learn optimal policy
        
        实现Off-Policy MC控制（加权重要性采样）
        Implement Off-Policy MC Control (weighted importance sampling)
        """
        if verbose:
            print("\n" + "="*60)
            print("Off-Policy MC控制")
            print("Off-Policy MC Control")
            print("="*60)
            print(f"  环境: {self.env.name}")
            print(f"  回合数: {n_episodes}")
            print(f"  行为策略ε: {self.behavior_policy.epsilon}")
            print(f"  目标策略: 贪婪（确定性）")
        
        start_time = time.time()
        
        for episode_num in range(n_episodes):
            # 用行为策略生成回合
            # Generate episode using behavior policy
            episode = self.generate_episode(self.behavior_policy)
            self.episodes.append(episode)
            
            # 计算回报
            # Compute returns
            returns = episode.compute_returns(self.gamma)
            
            # 反向处理回合（为了累积重要性比率）
            # Process episode backward (to accumulate importance ratio)
            W = 1.0  # 累积重要性比率
            
            for t in reversed(range(len(episode.experiences))):
                exp = episode.experiences[t]
                sa_pair = (exp.state.id, exp.action.id)
                G = returns[t]
                
                # 更新累积分母
                # Update cumulative denominator
                self.C[sa_pair] += W
                
                # 加权更新Q
                # Weighted Q update
                if self.C[sa_pair] > 0:
                    old_q = self.Q.get_value(exp.state, exp.action)
                    # 加权增量更新
                    # Weighted incremental update
                    new_q = old_q + (W / self.C[sa_pair]) * (G - old_q)
                    self.Q.set_value(exp.state, exp.action, new_q)
                
                # 更新目标策略（贪婪）
                # Update target policy (greedy)
                self.target_policy = self.create_greedy_policy()
                
                # 如果动作不是目标策略会选的，终止
                # If action is not what target would choose, terminate
                if isinstance(self.target_policy, DeterministicPolicy):
                    if exp.state in self.target_policy.policy_map:
                        target_action = self.target_policy.policy_map[exp.state]
                        if exp.action.id != target_action.id:
                            break  # 重要性比率后续为0
                
                # 更新W（重要性比率）
                # Update W (importance ratio)
                behavior_probs = self.behavior_policy.get_action_probabilities(
                    exp.state, self.env.action_space
                )
                behavior_prob = behavior_probs.get(exp.action, 1e-10)
                
                W = W / behavior_prob  # 目标策略是确定性的，分子是1
                
                # 记录比率
                # Record ratio
                self.importance_ratios.append(W)
            
            # 记录学习进度
            # Record learning progress
            if episode.experiences:
                episode_return = returns[0] if returns else 0
                self.learning_curve.append(episode_return)
            
            # 定期输出进度
            # Periodically output progress
            if verbose and (episode_num + 1) % 100 == 0:
                avg_return = np.mean(self.learning_curve[-100:]) if self.learning_curve else 0
                avg_ratio = np.mean(self.importance_ratios[-1000:]) if self.importance_ratios else 0
                
                print(f"  Episode {episode_num + 1}/{n_episodes}: "
                      f"平均回报={avg_return:.2f}, "
                      f"平均IS比率={avg_ratio:.2f}")
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\n学习完成:")
            print(f"  总时间: {total_time:.2f}秒")
            print(f"  访问的(s,a)对: {len(self.sa_visits)}")
            print(f"  平均IS比率: {np.mean(self.importance_ratios):.2f}")
            print(f"  IS比率标准差: {np.std(self.importance_ratios):.2f}")
        
        # 返回学习到的目标策略
        # Return learned target policy
        return self.target_policy
    
    def analyze_importance_sampling(self):
        """
        分析重要性采样
        Analyze importance sampling
        
        展示off-policy的挑战
        Show challenges of off-policy
        """
        print("\n" + "="*60)
        print("重要性采样分析")
        print("Importance Sampling Analysis")
        print("="*60)
        
        if not self.importance_ratios:
            print("没有重要性采样数据")
            return
        
        ratios = np.array(self.importance_ratios)
        
        print(f"\n重要性比率统计:")
        print(f"  样本数: {len(ratios)}")
        print(f"  均值: {np.mean(ratios):.2f}")
        print(f"  标准差: {np.std(ratios):.2f}")
        print(f"  最小值: {np.min(ratios):.4f}")
        print(f"  最大值: {np.max(ratios):.2f}")
        print(f"  中位数: {np.median(ratios):.2f}")
        
        # 分析极端值
        # Analyze extreme values
        extreme_threshold = 10.0
        extreme_count = np.sum(ratios > extreme_threshold)
        print(f"\n极端值 (>{extreme_threshold}): {extreme_count} ({extreme_count/len(ratios):.1%})")
        
        # 有效样本大小
        # Effective sample size
        if len(ratios) > 0:
            # ESS = (Σw)² / Σw²
            sum_w = np.sum(ratios)
            sum_w2 = np.sum(ratios ** 2)
            if sum_w2 > 0:
                ess = (sum_w ** 2) / sum_w2
                print(f"\n有效样本大小 (ESS): {ess:.1f} / {len(ratios)}")
                print(f"效率: {ess/len(ratios):.1%}")
        
        print("\n" + "="*40)
        print("问题诊断:")
        print("Problem Diagnosis:")
        print("="*40)
        
        if np.std(ratios) > np.mean(ratios):
            print("⚠️ 高方差问题：标准差 > 均值")
            print("   High variance: std > mean")
            print("   建议：减小行为策略的ε")
            print("   Suggestion: Reduce behavior policy ε")
        
        if extreme_count / len(ratios) > 0.1:
            print("⚠️ 过多极端值")
            print("   Too many extreme values")
            print("   建议：使用更相似的行为策略")
            print("   Suggestion: Use more similar behavior policy")
        
        print("""
        理论背景 Theoretical Background:
        ================================
        
        重要性采样的方差：
        Variance of importance sampling:
        Var[ρG] = E_b[(ρG)²] - (E_b[ρG])²
        
        当b和π差异大时：
        When b and π differ greatly:
        - ρ的方差爆炸
          Variance of ρ explodes
        - 估计变得不可靠
          Estimates become unreliable
        - 收敛极慢
          Convergence extremely slow
        
        解决方案：
        Solutions:
        1. 加权重要性采样（已使用）
           Weighted importance sampling (already used)
        2. 截断重要性比率
           Truncate importance ratios
        3. 使用更接近的行为策略
           Use closer behavior policy
        """)


# ================================================================================
# 第4.3.5节：探索性起始
# Section 4.3.5: Exploring Starts
# ================================================================================

class ExploringStarts:
    """
    探索性起始方法
    Exploring Starts Method
    
    保证探索的另一种方式
    Another way to ensure exploration
    
    核心思想：
    Core idea:
    - 每个回合从随机的(s,a)对开始
      Each episode starts from random (s,a) pair
    - 之后可以用确定性策略
      Can use deterministic policy afterwards
    - 保证所有(s,a)对被无限访问
      Ensures all (s,a) pairs visited infinitely
    
    算法（MC ES）：
    Algorithm (MC ES):
    1. 初始化Q(s,a)和π(s)任意
       Initialize Q(s,a) and π(s) arbitrarily
    2. 重复：
       Repeat:
       a. 选择随机S₀∈S, A₀∈A(S₀)
          Choose random S₀∈S, A₀∈A(S₀)
       b. 从S₀,A₀开始生成回合
          Generate episode starting from S₀,A₀
       c. 更新Q使用回合
          Update Q using episode
       d. 对每个s，π(s) ← argmax_a Q(s,a)
          For each s, π(s) ← argmax_a Q(s,a)
    
    优势：
    Advantages:
    - 可以学习确定性最优策略
      Can learn deterministic optimal policy
    - 不需要ε-贪婪的次优性
      No suboptimality of ε-greedy
    
    劣势：
    Disadvantages:
    - 需要能指定起始状态
      Need ability to specify starting state
    - 实践中常常不可行
      Often infeasible in practice
    - 不适用于继续性任务
      Not applicable to continuing tasks
    
    这是理论上优雅但实践受限的方法
    This is theoretically elegant but practically limited
    """
    
    @staticmethod
    def demonstrate(env: MDPEnvironment, 
                   n_episodes: int = 1000,
                   gamma: float = 1.0):
        """
        演示探索性起始
        Demonstrate exploring starts
        """
        print("\n" + "="*60)
        print("探索性起始MC控制 (MC ES)")
        print("Exploring Starts MC Control (MC ES)")
        print("="*60)
        
        # 初始化
        Q = ActionValueFunction(env.state_space, env.action_space, initial_value=0.0)
        sa_visits = defaultdict(int)
        
        # 创建初始策略（随机）
        # Create initial policy (random)
        policy_map = {}
        for state in env.state_space:
            if not state.is_terminal:
                policy_map[state] = np.random.choice(env.action_space)
        
        policy = DeterministicPolicy(policy_map)
        
        print(f"运行{n_episodes}个回合...")
        
        for episode_num in range(n_episodes):
            # 探索性起始：随机选择起始(s,a)
            # Exploring start: randomly choose starting (s,a)
            non_terminal_states = [s for s in env.state_space if not s.is_terminal]
            if not non_terminal_states:
                break
            
            start_state = np.random.choice(non_terminal_states)
            start_action = np.random.choice(env.action_space)
            
            # 生成回合
            # Generate episode
            episode = Episode()
            
            # 强制第一步
            # Force first step
            env.current_state = start_state
            next_state, reward, done, _ = env.step(start_action)
            
            exp = Experience(start_state, start_action, reward, next_state, done)
            episode.add_experience(exp)
            
            # 继续回合（用当前策略）
            # Continue episode (using current policy)
            state = next_state
            while not done:
                if state in policy.policy_map:
                    action = policy.policy_map[state]
                else:
                    action = np.random.choice(env.action_space)
                
                next_state, reward, done, _ = env.step(action)
                exp = Experience(state, action, reward, next_state, done)
                episode.add_experience(exp)
                state = next_state
            
            # 更新Q（first-visit）
            # Update Q (first-visit)
            returns = episode.compute_returns(gamma)
            sa_pairs_seen = set()
            
            for t, exp in enumerate(episode.experiences):
                sa_pair = (exp.state.id, exp.action.id)
                
                if sa_pair not in sa_pairs_seen:
                    sa_pairs_seen.add(sa_pair)
                    G = returns[t]
                    
                    sa_visits[sa_pair] += 1
                    n = sa_visits[sa_pair]
                    
                    old_q = Q.get_value(exp.state, exp.action)
                    new_q = old_q + (G - old_q) / n
                    Q.set_value(exp.state, exp.action, new_q)
            
            # 改进策略（贪婪）
            # Improve policy (greedy)
            for state in env.state_space:
                if not state.is_terminal:
                    best_action = None
                    best_value = float('-inf')
                    
                    for action in env.action_space:
                        q_value = Q.get_value(state, action)
                        if q_value > best_value:
                            best_value = q_value
                            best_action = action
                    
                    if best_action:
                        policy.policy_map[state] = best_action
            
            if (episode_num + 1) % 100 == 0:
                print(f"  Episode {episode_num + 1}: 访问{len(sa_visits)}个(s,a)对")
        
        print(f"\n结果:")
        print(f"  总访问(s,a)对: {len(sa_visits)}")
        print(f"  平均访问次数: {np.mean(list(sa_visits.values())):.1f}")
        
        # 分析覆盖率
        # Analyze coverage
        total_sa_pairs = sum(1 for s in env.state_space 
                           for a in env.action_space 
                           if not s.is_terminal)
        coverage = len(sa_visits) / total_sa_pairs if total_sa_pairs > 0 else 0
        
        print(f"  (s,a)对覆盖率: {coverage:.1%}")
        
        print("\n" + "="*40)
        print("探索性起始的特点:")
        print("Characteristics of Exploring Starts:")
        print("="*40)
        print("""
        ✓ 优点 Advantages:
        ------------------
        1. 理论保证：
           Theoretical guarantee:
           所有(s,a)对都被访问 → 收敛到最优
           All (s,a) pairs visited → Converges to optimal
        
        2. 无需软策略：
           No need for soft policy:
           可以学习确定性最优策略
           Can learn deterministic optimal policy
        
        3. 简单清晰：
           Simple and clear:
           算法逻辑直接
           Algorithm logic is straightforward
        
        ✗ 缺点 Disadvantages:
        ---------------------
        1. 实践限制：
           Practical limitations:
           很多环境不能任意设置起始状态
           Many environments can't set arbitrary start state
        
        2. 覆盖困难：
           Coverage difficulty:
           状态空间大时难以覆盖所有(s,a)
           Hard to cover all (s,a) when state space is large
        
        3. 不自然：
           Unnatural:
           随机起始可能不符合问题设定
           Random starts may not fit problem setting
        
        因此实践中更常用ε-贪婪或off-policy方法
        Therefore ε-greedy or off-policy more common in practice
        """)
        
        return policy, Q


# ================================================================================
# 第4.3.6节：MC控制可视化器
# Section 4.3.6: MC Control Visualizer
# ================================================================================

class MCControlVisualizer:
    """
    MC控制可视化器
    MC Control Visualizer
    
    提供丰富的可视化来理解MC控制
    Provides rich visualizations to understand MC control
    """
    
    @staticmethod
    def plot_learning_curves(controllers: Dict[str, MCControl]):
        """
        绘制学习曲线比较
        Plot learning curves comparison
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        colors = {'On-Policy': 'blue', 'Off-Policy': 'red', 'MC ES': 'green'}
        
        # 图1：回报曲线
        # Plot 1: Return curves
        ax1 = axes[0, 0]
        ax1.set_title('Learning Curves (Episode Returns)')
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Episode Return')
        
        for name, controller in controllers.items():
            if controller.learning_curve:
                # 平滑曲线
                # Smooth curve
                window = 50
                if len(controller.learning_curve) >= window:
                    smoothed = np.convolve(controller.learning_curve, 
                                          np.ones(window)/window, 
                                          mode='valid')
                    x = np.arange(len(smoothed))
                    ax1.plot(x, smoothed, color=colors.get(name, 'gray'),
                           label=name, linewidth=2, alpha=0.7)
        
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 图2：Q值演化
        # Plot 2: Q-value evolution
        ax2 = axes[0, 1]
        ax2.set_title('Q-value Statistics')
        
        for idx, (name, controller) in enumerate(controllers.items()):
            # 获取所有Q值
            # Get all Q values
            q_values = []
            for state in controller.env.state_space:
                if not state.is_terminal:
                    for action in controller.env.action_space:
                        q_values.append(controller.Q.get_value(state, action))
            
            if q_values:
                # 箱线图
                # Box plot
                bp = ax2.boxplot(q_values, positions=[idx], widths=0.6,
                                patch_artist=True, labels=[name])
                bp['boxes'][0].set_facecolor(colors.get(name, 'gray'))
                bp['boxes'][0].set_alpha(0.5)
        
        ax2.set_ylabel('Q-values')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 图3：探索统计（仅On-Policy）
        # Plot 3: Exploration statistics (On-Policy only)
        ax3 = axes[1, 0]
        ax3.set_title('Exploration vs Exploitation (On-Policy)')
        
        for name, controller in controllers.items():
            if hasattr(controller, 'policy') and isinstance(controller.policy, EpsilonGreedyPolicy):
                stats = controller.policy.get_exploration_stats()
                
                # 饼图
                # Pie chart
                sizes = [stats['exploration_count'], stats['exploitation_count']]
                labels = ['Exploration', 'Exploitation']
                colors_pie = ['lightcoral', 'lightblue']
                
                if sum(sizes) > 0:
                    ax3.pie(sizes, labels=labels, colors=colors_pie,
                           autopct='%1.1f%%', startangle=90)
                    ax3.set_title(f'{name}: ε={controller.policy.epsilon:.3f}')
                    break  # 只显示一个
        
        # 图4：访问频率分布
        # Plot 4: Visit frequency distribution
        ax4 = axes[1, 1]
        ax4.set_title('State-Action Visit Frequencies')
        ax4.set_xlabel('Visit Count')
        ax4.set_ylabel('Number of (s,a) pairs')
        
        for name, controller in controllers.items():
            if controller.sa_visits:
                visits = list(controller.sa_visits.values())
                ax4.hist(visits, bins=30, alpha=0.5, 
                        label=name, color=colors.get(name, 'gray'))
        
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('MC Control Methods Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_policy_evolution(controller: MCControl, 
                            sample_states: Optional[List[State]] = None):
        """
        绘制策略演化
        Plot policy evolution
        
        展示策略如何随学习改变
        Show how policy changes with learning
        """
        if not hasattr(controller, 'policy'):
            print("控制器没有策略属性")
            return None
        
        policy = controller.policy
        
        # 如果没指定，选择一些状态
        # If not specified, select some states
        if sample_states is None:
            non_terminal = [s for s in controller.env.state_space 
                          if not s.is_terminal]
            sample_states = non_terminal[:min(4, len(non_terminal))]
        
        if not sample_states:
            print("没有可显示的状态")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, state in enumerate(sample_states):
            if idx >= 4:
                break
            
            ax = axes[idx]
            
            # 获取动作概率
            # Get action probabilities
            if isinstance(policy, StochasticPolicy):
                probs = policy.get_action_probabilities(state)
                
                # 条形图
                # Bar plot
                actions = list(probs.keys())
                probabilities = list(probs.values())
                
                x = np.arange(len(actions))
                bars = ax.bar(x, probabilities, alpha=0.7, color='steelblue')
                
                # 高亮最优动作
                # Highlight best action
                best_idx = np.argmax(probabilities)
                bars[best_idx].set_color('red')
                bars[best_idx].set_alpha(1.0)
                
                ax.set_xticks(x)
                ax.set_xticklabels([a.id for a in actions], rotation=45)
                ax.set_ylabel('Probability')
                ax.set_ylim([0, 1])
                ax.set_title(f'State: {state.id}')
                
                # 添加Q值作为参考
                # Add Q-values as reference
                q_values = [controller.Q.get_value(state, a) for a in actions]
                ax2 = ax.twinx()
                ax2.plot(x, q_values, 'go-', linewidth=2, markersize=8, alpha=0.5)
                ax2.set_ylabel('Q-value', color='g')
                ax2.tick_params(axis='y', labelcolor='g')
                
            elif isinstance(policy, DeterministicPolicy):
                # 对于确定性策略，显示Q值
                # For deterministic policy, show Q-values
                q_values = []
                action_labels = []
                
                for action in controller.env.action_space:
                    q_values.append(controller.Q.get_value(state, action))
                    action_labels.append(action.id)
                
                x = np.arange(len(q_values))
                bars = ax.bar(x, q_values, alpha=0.7, color='steelblue')
                
                # 高亮选择的动作
                # Highlight selected action
                if state in policy.policy_map:
                    selected_action = policy.policy_map[state]
                    for i, action in enumerate(controller.env.action_space):
                        if action.id == selected_action.id:
                            bars[i].set_color('red')
                            bars[i].set_alpha(1.0)
                            break
                
                ax.set_xticks(x)
                ax.set_xticklabels(action_labels, rotation=45)
                ax.set_ylabel('Q-value')
                ax.set_title(f'State: {state.id}')
        
        plt.suptitle('Policy and Q-values by State', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_importance_sampling_analysis(off_policy_controller: OffPolicyMCControl):
        """
        绘制重要性采样分析
        Plot importance sampling analysis
        """
        if not off_policy_controller.importance_ratios:
            print("没有重要性采样数据")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        ratios = np.array(off_policy_controller.importance_ratios)
        
        # 图1：比率分布
        # Plot 1: Ratio distribution
        ax1 = axes[0, 0]
        ax1.hist(np.clip(ratios, 0, 20), bins=50, alpha=0.7, color='blue')
        ax1.set_xlabel('Importance Ratio (clipped at 20)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Importance Ratios')
        ax1.axvline(x=1.0, color='red', linestyle='--', label='Ratio=1')
        ax1.legend()
        
        # 图2：比率随时间变化
        # Plot 2: Ratios over time
        ax2 = axes[0, 1]
        window = min(100, len(ratios) // 10)
        if len(ratios) >= window:
            smoothed = np.convolve(ratios, np.ones(window)/window, mode='valid')
            ax2.plot(smoothed, alpha=0.7, color='green')
            ax2.set_xlabel('Sample Index')
            ax2.set_ylabel('Importance Ratio (smoothed)')
            ax2.set_title(f'IS Ratios Over Time (window={window})')
            ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        
        # 图3：有效样本大小
        # Plot 3: Effective sample size
        ax3 = axes[1, 0]
        
        # 计算累积ESS
        # Compute cumulative ESS
        ess_history = []
        for i in range(100, len(ratios), 100):
            batch = ratios[:i]
            sum_w = np.sum(batch)
            sum_w2 = np.sum(batch ** 2)
            if sum_w2 > 0:
                ess = (sum_w ** 2) / sum_w2
                ess_history.append(ess / i)  # 归一化
        
        if ess_history:
            ax3.plot(np.arange(100, len(ratios), 100), ess_history, 'bo-')
            ax3.set_xlabel('Number of Samples')
            ax3.set_ylabel('ESS / n (Efficiency)')
            ax3.set_title('Effective Sample Size Efficiency')
            ax3.axhline(y=1.0, color='red', linestyle='--', label='Perfect efficiency')
            ax3.axhline(y=0.5, color='orange', linestyle='--', label='50% efficiency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 图4：极端值分析
        # Plot 4: Extreme values analysis
        ax4 = axes[1, 1]
        
        thresholds = [1, 2, 5, 10, 20, 50]
        extreme_counts = [np.sum(ratios > t) / len(ratios) * 100 for t in thresholds]
        
        ax4.bar(range(len(thresholds)), extreme_counts, alpha=0.7, color='coral')
        ax4.set_xticks(range(len(thresholds)))
        ax4.set_xticklabels([str(t) for t in thresholds])
        ax4.set_xlabel('Threshold')
        ax4.set_ylabel('Percentage of Samples (%)')
        ax4.set_title('Percentage of Samples Exceeding Threshold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Importance Sampling Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig


# ================================================================================
# 第4.3.7节：MC控制综合演示
# Section 4.3.7: MC Control Comprehensive Demo
# ================================================================================

def demonstrate_mc_control():
    """
    综合演示MC控制方法
    Comprehensive demonstration of MC control methods
    """
    print("\n" + "="*80)
    print("蒙特卡洛控制方法综合演示")
    print("Monte Carlo Control Methods Comprehensive Demo")
    print("="*80)
    
    # 创建测试环境
    # Create test environment
    from ch02_mdp.gridworld import GridWorld
    
    env = GridWorld(rows=4, cols=4,
                   start_pos=(0,0),
                   goal_pos=(3,3),
                   obstacles={(1,1), (2,2)})
    
    print(f"\n测试环境: {env.name}")
    print(f"  状态数: {len(env.state_space)}")
    print(f"  动作数: {len(env.action_space)}")
    print(f"  起点: (0,0), 终点: (3,3)")
    print(f"  障碍物: (1,1), (2,2)")
    
    # 训练回合数
    # Number of training episodes
    n_episodes = 500
    
    # 1. On-Policy MC控制
    # 1. On-Policy MC Control
    print("\n" + "="*60)
    print("1. On-Policy MC控制")
    on_policy = OnPolicyMCControl(
        env, gamma=0.9,
        epsilon=0.2,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    on_policy_result = on_policy.learn(n_episodes, verbose=True)
    on_policy.analyze_learning()
    on_policy.demonstrate_gpi()
    on_policy.policy.analyze_exploration_pattern()
    
    # 2. Off-Policy MC控制
    # 2. Off-Policy MC Control
    print("\n" + "="*60)
    print("2. Off-Policy MC控制")
    off_policy = OffPolicyMCControl(
        env, gamma=0.9,
        behavior_epsilon=0.3
    )
    
    off_policy_result = off_policy.learn(n_episodes, verbose=True)
    off_policy.analyze_learning()
    off_policy.analyze_importance_sampling()
    
    # 3. 探索性起始
    # 3. Exploring Starts
    print("\n" + "="*60)
    print("3. 探索性起始")
    es_policy, es_Q = ExploringStarts.demonstrate(env, n_episodes=500, gamma=0.9)
    
    # 比较结果
    # Compare results
    print("\n" + "="*80)
    print("方法比较")
    print("Method Comparison")
    print("="*80)
    
    # 评估最终策略
    # Evaluate final policies
    print("\n最终策略评估 (100回合平均):")
    print("Final Policy Evaluation (100 episode average):")
    
    on_return = on_policy.evaluate_policy(on_policy_result, 100)
    off_return = off_policy.evaluate_policy(off_policy_result, 100)
    
    # 为ES创建临时控制器来评估
    # Create temporary controller for ES evaluation
    es_controller = MCControl(env, gamma=0.9)
    es_return = es_controller.evaluate_policy(es_policy, 100)
    
    print(f"  On-Policy: {on_return:.2f}")
    print(f"  Off-Policy: {off_return:.2f}")
    print(f"  Exploring Starts: {es_return:.2f}")
    
    # 分析覆盖率
    # Analyze coverage
    print("\n状态-动作对覆盖率:")
    print("State-Action Pair Coverage:")
    
    total_sa = sum(1 for s in env.state_space 
                  for a in env.action_space 
                  if not s.is_terminal)
    
    on_coverage = len(on_policy.sa_visits) / total_sa * 100
    off_coverage = len(off_policy.sa_visits) / total_sa * 100
    
    print(f"  On-Policy: {on_coverage:.1f}%")
    print(f"  Off-Policy: {off_coverage:.1f}%")
    
    # 可视化
    # Visualization
    print("\n生成可视化...")
    
    controllers = {
        'On-Policy': on_policy,
        'Off-Policy': off_policy
    }
    
    # 学习曲线比较
    # Learning curves comparison
    fig1 = MCControlVisualizer.plot_learning_curves(controllers)
    
    # 策略可视化
    # Policy visualization
    fig2 = MCControlVisualizer.plot_policy_evolution(on_policy)
    
    # 重要性采样分析
    # Importance sampling analysis
    fig3 = MCControlVisualizer.plot_importance_sampling_analysis(off_policy)
    
    # 总结
    # Summary
    print("\n" + "="*80)
    print("关键要点")
    print("Key Takeaways")
    print("="*80)
    print("""
    1. On-Policy MC控制:
       - 简单直接，容易实现
         Simple and straightforward
       - ε-贪婪平衡探索和利用
         ε-greedy balances exploration and exploitation
       - 收敛到ε-软最优策略
         Converges to ε-soft optimal policy
    
    2. Off-Policy MC控制:
       - 可以学习最优确定性策略
         Can learn optimal deterministic policy
       - 重要性采样带来高方差
         Importance sampling brings high variance
       - 数据效率可能更高
         May be more data efficient
    
    3. 探索性起始:
       - 理论优雅但实践受限
         Theoretically elegant but practically limited
       - 保证所有(s,a)对被访问
         Ensures all (s,a) pairs are visited
       - 需要环境支持任意起始
         Needs environment support for arbitrary starts
    
    4. 共同特点:
       - 都是无模型方法
         All are model-free methods
       - 都需要完整回合
         All need complete episodes
       - 都基于GPI框架
         All based on GPI framework
    
    5. 向TD方法的过渡:
       - MC的高方差激发了TD方法
         High variance of MC motivated TD methods
       - TD结合了MC和DP的优点
         TD combines advantages of MC and DP
       - 下一步：学习TD控制（Q-learning, SARSA）
         Next: Learn TD control (Q-learning, SARSA)
    """)
    print("="*80)
    
    plt.show()


# ================================================================================
# 主函数
# Main Function  
# ================================================================================

def main():
    """
    运行MC控制演示
    Run MC Control Demo
    """
    demonstrate_mc_control()


if __name__ == "__main__":
    main()