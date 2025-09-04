"""
================================================================================
第7.4节：n步Tree Backup算法 - 无需重要性采样的Off-Policy方法
Section 7.4: n-step Tree Backup Algorithm - Off-Policy Without Importance Sampling
================================================================================

Tree Backup是off-policy学习的突破！
Tree Backup is a breakthrough in off-policy learning!

核心思想 Core Idea:
使用所有可能动作的期望而不是采样动作，避免重要性采样
Use expectation over all possible actions instead of sampled action, avoid importance sampling

Tree结构 Tree Structure:
         S_t
        / | \ 
       a1 a2 a3  <- 考虑所有动作
       |
      S_{t+1}
     / | \
    a1 a2 a3     <- 递归考虑

n步Tree Backup回报 n-step Tree Backup Return:
G_t:t+n = R_{t+1} + γΣ_a π(a|S_{t+1})Q(S_{t+1},a) 
         - γπ(A_{t+1}|S_{t+1})Q(S_{t+1},A_{t+1})
         + γπ(A_{t+1}|S_{t+1})G_{t+1:t+n}

关键特性 Key Features:
1. 完全off-policy
   Fully off-policy
2. 无需重要性采样
   No importance sampling
3. 低方差
   Low variance  
4. 可以从任意策略学习
   Can learn from any policy

优势 Advantages:
- 稳定性好
  Good stability
- 无需知道行为策略
  No need to know behavior policy
- 没有高方差问题
  No high variance issue

代价 Cost:
- 计算更复杂
  More complex computation
- 需要遍历所有动作
  Need to iterate all actions
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from scipy import stats
import time

# 导入基础组件
# Import base components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.ch03_finite_mdp.mdp_framework import State, Action, MDPEnvironment, MDPAgent
from src.ch03_finite_mdp.policies_and_values import (
    Policy, StateValueFunction, ActionValueFunction,
    StochasticPolicy, DeterministicPolicy
)
from ch04_monte_carlo.mc_control import EpsilonGreedyPolicy
from ch05_temporal_difference.td_foundations import TDError, TDErrorAnalyzer

# 设置日志
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第7.4.1节：Tree Backup节点
# Section 7.4.1: Tree Backup Node
# ================================================================================

@dataclass
class TreeBackupNode:
    """
    Tree Backup树节点
    Tree Backup Tree Node
    
    表示树中的一个决策点
    Represents a decision point in the tree
    """
    state: State
    depth: int
    is_leaf: bool = False
    expected_value: float = 0.0
    action_values: Dict[Action, float] = field(default_factory=dict)
    action_probabilities: Dict[Action, float] = field(default_factory=dict)
    taken_action: Optional[Action] = None  # 实际采取的动作
    received_reward: float = 0.0
    
    def compute_backup_value(self, gamma: float) -> float:
        """
        计算backup值
        Compute backup value
        
        考虑所有可能动作的期望
        Consider expectation over all possible actions
        
        Args:
            gamma: 折扣因子
                  Discount factor
        
        Returns:
            Backup值
            Backup value
        """
        if self.is_leaf:
            return self.expected_value
        
        # 计算期望（除了实际采取的动作）
        # Compute expectation (except taken action)
        backup = 0.0
        
        for action, q_value in self.action_values.items():
            prob = self.action_probabilities.get(action, 0.0)
            
            if action == self.taken_action:
                # 实际采取的动作使用递归值
                # Taken action uses recursive value
                continue
            else:
                # 其他动作使用Q值
                # Other actions use Q values
                backup += prob * q_value
        
        return backup


# ================================================================================
# 第7.4.2节：n步Tree Backup算法
# Section 7.4.2: n-step Tree Backup Algorithm
# ================================================================================

class NStepTreeBackup:
    """
    n步Tree Backup算法
    n-step Tree Backup Algorithm
    
    完全off-policy，无需重要性采样
    Fully off-policy without importance sampling
    
    算法步骤 Algorithm Steps:
    1. 收集n步轨迹
       Collect n-step trajectory
    2. 构建backup树
       Build backup tree
    3. 递归计算tree backup值
       Recursively compute tree backup value
    4. 更新Q函数
       Update Q function
    
    更新规则 Update Rule:
    Q(S_t, A_t) ← Q(S_t, A_t) + α[G_t^{tree} - Q(S_t, A_t)]
    
    其中G_t^{tree}是tree backup回报
    where G_t^{tree} is tree backup return
    
    关键创新 Key Innovation:
    - 不沿着采样路径backup
      Don't backup along sampled path
    - 而是考虑所有可能的动作
      Instead consider all possible actions
    - 这避免了重要性采样
      This avoids importance sampling
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 n: int = 4,
                 gamma: float = 0.99,
                 alpha: Union[float, Callable] = 0.1,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        初始化n步Tree Backup
        Initialize n-step Tree Backup
        
        Args:
            env: 环境
                Environment
            n: 步数
               Number of steps
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
        self.n = n
        self.gamma = gamma
        
        # 学习率
        # Learning rate
        if callable(alpha):
            self.alpha_func = alpha
        else:
            self.alpha_func = lambda t: alpha
        
        # Q函数
        # Q function
        self.Q = ActionValueFunction(
            env.state_space,
            env.action_space,
            initial_value=0.0
        )
        
        # 目标策略（学习的策略）
        # Target policy (policy being learned)
        self.target_policy = EpsilonGreedyPolicy(
            self.Q,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            action_space=env.action_space
        )
        
        # TD误差分析
        # TD error analysis
        self.td_analyzer = TDErrorAnalyzer()
        
        # 统计
        # Statistics
        self.episode_count = 0
        self.step_count = 0
        self.episode_returns = []
        self.episode_lengths = []
        self.tree_backup_values = []
        
        logger.info(f"初始化{n}步Tree Backup: γ={gamma}, ε={epsilon}")
    
    def get_action_probability(self, state: State, action: Action) -> float:
        """
        获取目标策略的动作概率
        Get action probability for target policy
        
        Args:
            state: 状态
                  State
            action: 动作
                   Action
        
        Returns:
            动作概率π(a|s)
            Action probability
        """
        # ε-贪婪策略
        # ε-greedy policy
        q_values = [self.Q.get_value(state, a) for a in self.env.action_space]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(self.env.action_space, q_values) if q == max_q]
        
        if action in best_actions:
            return (1 - self.target_policy.epsilon) / len(best_actions) + \
                   self.target_policy.epsilon / len(self.env.action_space)
        else:
            return self.target_policy.epsilon / len(self.env.action_space)
    
    def compute_tree_backup_return(self,
                                   states: List[State],
                                   actions: List[Action],
                                   rewards: List[float],
                                   tau: int,
                                   T: int) -> float:
        """
        计算tree backup回报
        Compute tree backup return
        
        这是算法的核心
        This is the core of the algorithm
        
        Args:
            states: 状态序列
                   State sequence
            actions: 动作序列
                    Action sequence
            rewards: 奖励序列
                    Reward sequence
            tau: 当前时刻
                Current time
            T: 终止时刻
               Terminal time
        
        Returns:
            Tree backup回报
            Tree backup return
        """
        # 如果是终止状态
        # If terminal state
        if tau >= T - 1:
            return rewards[tau] if tau < len(rewards) else 0.0
        
        # 计算n步tree backup
        # Compute n-step tree backup
        n_actual = min(self.n, T - tau)
        
        # 递归计算tree backup值
        # Recursively compute tree backup value
        g = rewards[tau]
        
        # 从tau+1开始构建tree
        # Build tree from tau+1
        for k in range(1, n_actual):
            t = tau + k
            
            if t < len(states) and not states[t].is_terminal:
                # 对于时刻t，考虑所有可能的动作
                # For time t, consider all possible actions
                
                if t < len(actions):
                    # 实际采取的动作
                    # Actually taken action
                    actual_action = actions[t]
                    actual_prob = self.get_action_probability(states[t], actual_action)
                    
                    # 计算期望值（除了实际动作）
                    # Compute expected value (except actual action)
                    expected_value = 0.0
                    for action in self.env.action_space:
                        prob = self.get_action_probability(states[t], action)
                        q_value = self.Q.get_value(states[t], action)
                        
                        if action != actual_action:
                            expected_value += prob * q_value
                    
                    # 添加实际动作的贡献
                    # Add contribution of actual action
                    if t < len(rewards):
                        # 实际动作继续递归
                        # Actual action continues recursion
                        g += (self.gamma ** k) * (
                            expected_value + 
                            actual_prob * (rewards[t] - self.Q.get_value(states[t], actual_action))
                        )
                    else:
                        g += (self.gamma ** k) * expected_value
                else:
                    # 没有更多动作，使用期望
                    # No more actions, use expectation
                    expected_value = 0.0
                    for action in self.env.action_space:
                        prob = self.get_action_probability(states[t], action)
                        q_value = self.Q.get_value(states[t], action)
                        expected_value += prob * q_value
                    
                    g += (self.gamma ** k) * expected_value
        
        # 添加最后的bootstrap
        # Add final bootstrap
        if tau + n_actual < len(states) and not states[tau + n_actual].is_terminal:
            expected_value = 0.0
            for action in self.env.action_space:
                prob = self.get_action_probability(states[tau + n_actual], action)
                q_value = self.Q.get_value(states[tau + n_actual], action)
                expected_value += prob * q_value
            
            g += (self.gamma ** n_actual) * expected_value
        
        return g
    
    def learn_episode(self, behavior_policy: Optional[Policy] = None) -> Tuple[float, int]:
        """
        学习一个回合
        Learn one episode
        
        Args:
            behavior_policy: 行为策略（可选，默认使用目标策略）
                           Behavior policy (optional, default to target policy)
        
        Returns:
            (回合回报, 回合长度)
            (episode return, episode length)
        """
        if behavior_policy is None:
            behavior_policy = self.target_policy
        
        # 初始化
        # Initialize
        states = []
        actions = []
        rewards = []
        
        state = self.env.reset()
        states.append(state)
        
        t = 0
        T = float('inf')
        episode_return = 0.0
        
        # 收集轨迹
        # Collect trajectory
        while True:
            # τ时刻需要更新
            # Time τ to update
            tau = t - self.n + 1
            
            if t < T:
                # 选择动作（使用行为策略）
                # Select action (using behavior policy)
                action = behavior_policy.select_action(state)
                actions.append(action)
                
                # 执行动作
                # Execute action
                next_state, reward, done, _ = self.env.step(action)
                
                states.append(next_state)
                rewards.append(reward)
                episode_return += reward * (self.gamma ** t)
                
                if done:
                    T = t + 1
                
                state = next_state
            
            # 更新
            # Update
            if tau >= 0 and tau < len(actions):
                # 计算tree backup回报
                # Compute tree backup return
                g_tree = self.compute_tree_backup_return(
                    states, actions, rewards, tau, T
                )
                
                self.tree_backup_values.append(g_tree)
                
                # 更新Q函数
                # Update Q function
                update_state = states[tau]
                update_action = actions[tau]
                old_q = self.Q.get_value(update_state, update_action)
                alpha = self.alpha_func(self.step_count)
                new_q = old_q + alpha * (g_tree - old_q)
                self.Q.set_value(update_state, update_action, new_q)
                
                # 记录TD误差
                # Record TD error
                td_error = g_tree - old_q
                td_err_obj = TDError(
                    value=td_error,
                    timestep=self.step_count,
                    state=update_state,
                    next_state=states[tau + 1] if tau + 1 < len(states) else None,
                    reward=rewards[tau] if tau < len(rewards) else 0,
                    state_value=old_q,
                    next_state_value=g_tree
                )
                self.td_analyzer.add_error(td_err_obj)
                
                self.step_count += 1
            
            t += 1
            
            # 检查是否结束
            # Check if done
            if tau == T - 1:
                break
        
        # 衰减ε
        # Decay ε
        self.target_policy.decay_epsilon()
        
        # 记录统计
        # Record statistics
        self.episode_count += 1
        self.episode_returns.append(episode_return)
        self.episode_lengths.append(t)
        
        return episode_return, t
    
    def learn(self,
             n_episodes: int = 1000,
             behavior_policy: Optional[Policy] = None,
             verbose: bool = True) -> ActionValueFunction:
        """
        学习Q函数
        Learn Q function
        
        Args:
            n_episodes: 回合数
                       Number of episodes
            behavior_policy: 行为策略
                           Behavior policy
            verbose: 是否输出进度
                    Whether to output progress
        
        Returns:
            学习的Q函数
            Learned Q function
        """
        if verbose:
            print(f"\n开始{self.n}步Tree Backup学习: {n_episodes}回合")
            print(f"Starting {self.n}-step Tree Backup learning: {n_episodes} episodes")
            print(f"  参数: γ={self.gamma}, n={self.n}")
            print(f"  初始ε: {self.target_policy.epsilon:.3f}")
        
        for episode in range(n_episodes):
            episode_return, episode_length = self.learn_episode(behavior_policy)
            
            if verbose and (episode + 1) % max(1, n_episodes // 10) == 0:
                avg_return = np.mean(self.episode_returns[-100:]) \
                           if len(self.episode_returns) >= 100 \
                           else np.mean(self.episode_returns)
                avg_length = np.mean(self.episode_lengths[-100:]) \
                           if len(self.episode_lengths) >= 100 \
                           else np.mean(self.episode_lengths)
                
                stats = self.td_analyzer.get_statistics()
                
                print(f"Episode {episode + 1}/{n_episodes}: "
                      f"Return={episode_return:.2f}, "
                      f"Avg Return={avg_return:.2f}, "
                      f"Avg Length={avg_length:.1f}, "
                      f"ε={self.target_policy.epsilon:.3f}, "
                      f"TD Error={stats.get('recent_abs_mean', 0):.4f}")
        
        if verbose:
            print(f"\n{self.n}步Tree Backup学习完成!")
            print(f"  最终ε: {self.target_policy.epsilon:.3f}")
            print(f"  总步数: {self.step_count}")
            if self.tree_backup_values:
                print(f"  平均tree backup值: {np.mean(self.tree_backup_values):.3f}")
        
        return self.Q


# ================================================================================
# 第7.4.3节：Tree Backup可视化
# Section 7.4.3: Tree Backup Visualization
# ================================================================================

class TreeBackupVisualizer:
    """
    Tree Backup可视化器
    Tree Backup Visualizer
    
    展示tree backup的计算过程
    Show tree backup computation process
    """
    
    @staticmethod
    def visualize_tree_structure():
        """
        可视化tree backup结构
        Visualize tree backup structure
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # 树的层级
        # Tree levels
        levels = 4
        
        # 绘制节点
        # Draw nodes
        node_positions = {}
        y_spacing = 1.0
        
        # 根节点
        # Root node
        ax.scatter([0], [levels * y_spacing], s=200, c='red', zorder=5)
        ax.text(0, levels * y_spacing + 0.15, 'S_t, A_t', ha='center', fontsize=10, fontweight='bold')
        node_positions[(0, 0)] = (0, levels * y_spacing)
        
        # 绘制树
        # Draw tree
        for level in range(1, levels):
            y = (levels - level) * y_spacing
            n_nodes = 3 ** level  # 每层3^level个节点（3个动作）
            x_spacing = 6.0 / (n_nodes + 1)
            
            for i in range(n_nodes):
                x = -3.0 + (i + 1) * x_spacing
                
                # 判断节点类型
                # Determine node type
                if i % 3 == 0:
                    color = 'blue'  # 采样路径
                    label = f'S_{{{level}}}'
                else:
                    color = 'green'  # 期望路径
                    label = f'Q_{{{level}}}'
                
                ax.scatter([x], [y], s=100, c=color, alpha=0.7, zorder=5)
                
                # 连接到父节点
                # Connect to parent node
                parent_idx = i // 3
                if (level - 1, parent_idx) in node_positions:
                    parent_x, parent_y = node_positions[(level - 1, parent_idx)]
                    
                    if i % 3 == 0:
                        # 实际路径
                        # Actual path
                        ax.plot([parent_x, x], [parent_y, y], 'b-', linewidth=2, alpha=0.5)
                    else:
                        # 期望路径
                        # Expected path
                        ax.plot([parent_x, x], [parent_y, y], 'g--', linewidth=1, alpha=0.3)
                
                node_positions[(level, i)] = (x, y)
        
        # 添加图例
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(color='red', label='Current State-Action'),
            Patch(color='blue', label='Sampled Path'),
            Patch(color='green', label='Expected Values')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_xlim(-4, 4)
        ax.set_ylim(-0.5, levels * y_spacing + 0.5)
        ax.set_title('Tree Backup Structure\n(Considers all actions at each state)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Action Space')
        ax.set_ylabel('Time Steps')
        ax.grid(True, alpha=0.3)
        
        # 添加说明
        # Add explanation
        ax.text(0, -0.3, 
               'Tree Backup backs up along all possible actions,\n'
               'not just the sampled path, avoiding importance sampling',
               ha='center', fontsize=10, style='italic')
        
        return fig


# ================================================================================
# 主函数：演示Tree Backup
# Main Function: Demonstrate Tree Backup
# ================================================================================

def demonstrate_tree_backup():
    """
    演示Tree Backup算法
    Demonstrate Tree Backup algorithm
    """
    print("\n" + "="*80)
    print("第7.4节：n步Tree Backup算法")
    print("Section 7.4: n-step Tree Backup Algorithm")
    print("="*80)
    
    from src.ch03_finite_mdp.gridworld import GridWorld
    from src.ch03_finite_mdp.policies_and_values import UniformRandomPolicy
    
    # 创建环境
    # Create environment
    env = GridWorld(rows=4, cols=4,
                   start_pos=(0,0),
                   goal_pos=(3,3),
                   obstacles=[(1,1), (2,2)])
    
    print(f"\n创建4×4 GridWorld（含障碍）")
    print(f"  起点: (0,0)")
    print(f"  终点: (3,3)")
    print(f"  障碍: (1,1), (2,2)")
    
    # 1. 可视化Tree Backup结构
    # 1. Visualize Tree Backup structure
    print("\n" + "="*60)
    print("1. Tree Backup结构")
    print("1. Tree Backup Structure")
    print("="*60)
    
    print("\nTree Backup的关键思想:")
    print("- 不沿着采样路径backup")
    print("- 而是考虑所有可能的动作")
    print("- 使用期望值而不是采样值")
    print("- 这避免了重要性采样的高方差问题")
    
    fig = TreeBackupVisualizer.visualize_tree_structure()
    
    # 2. 测试Tree Backup算法
    # 2. Test Tree Backup algorithm
    print("\n" + "="*60)
    print("2. n步Tree Backup学习")
    print("2. n-step Tree Backup Learning")
    print("="*60)
    
    # 测试不同的n值
    # Test different n values
    n_values = [1, 2, 4, 8]
    
    for n in n_values:
        print(f"\n测试{n}步Tree Backup:")
        tree_backup = NStepTreeBackup(
            env, n=n, gamma=0.99, alpha=0.1, epsilon=0.1
        )
        
        Q = tree_backup.learn(n_episodes=200, verbose=False)
        
        # 显示结果
        # Show results
        avg_return = np.mean(tree_backup.episode_returns[-50:])
        avg_length = np.mean(tree_backup.episode_lengths[-50:])
        
        print(f"  最终平均回报: {avg_return:.2f}")
        print(f"  最终平均长度: {avg_length:.1f}")
        
        # 显示一些Q值
        # Show some Q values
        sample_state = env.state_space[0]
        if not sample_state.is_terminal:
            print(f"  Q值示例:")
            for action in env.action_space[:2]:
                q_value = Q.get_value(sample_state, action)
                print(f"    Q(s0, {action.id}) = {q_value:.3f}")
    
    # 3. 比较Tree Backup与其他方法
    # 3. Compare Tree Backup with other methods
    print("\n" + "="*60)
    print("3. Tree Backup vs 其他n步方法比较")
    print("3. Tree Backup vs Other n-step Methods Comparison")
    print("="*60)
    
    from ch07_n_step_bootstrapping.n_step_sarsa import NStepSARSA
    from ch07_n_step_bootstrapping.off_policy_n_step import OffPolicyNStepSARSA
    
    n = 4
    n_runs = 5
    n_episodes = 200
    
    methods = {
        'n-step SARSA': NStepSARSA,
        'Off-policy n-step SARSA': OffPolicyNStepSARSA,
        'n-step Tree Backup': NStepTreeBackup
    }
    
    results = {name: [] for name in methods}
    
    print(f"\n运行{n_runs}次实验，每次{n_episodes}回合...")
    
    for run in range(n_runs):
        for name, AlgoClass in methods.items():
            if name == 'Off-policy n-step SARSA':
                algo = AlgoClass(env, n=n, gamma=0.99, alpha=0.1,
                                epsilon_behavior=0.3, epsilon_target=0.1)
                # 学习
                for _ in range(n_episodes):
                    algo.learn_episode()
            else:
                algo = AlgoClass(env, n=n, gamma=0.99, alpha=0.1, epsilon=0.1)
                algo.learn(n_episodes=n_episodes, verbose=False)
            
            # 记录最终性能
            # Record final performance
            final_return = np.mean(algo.episode_returns[-50:])
            results[name].append(final_return)
    
    # 打印结果
    # Print results
    print("\n方法比较结果:")
    print(f"{'方法':<30} {'平均回报':<20} {'标准差':<15}")
    print("-" * 65)
    
    for name, returns in results.items():
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        print(f"{name:<30} {mean_return:<20.2f} {std_return:<15.2f}")
    
    # 找最佳方法
    # Find best method
    best_method = max(results.items(), key=lambda x: np.mean(x[1]))
    print(f"\n最佳方法: {best_method[0]}")
    
    # 4. Tree Backup的off-policy学习
    # 4. Off-policy learning with Tree Backup
    print("\n" + "="*60)
    print("4. Tree Backup的Off-Policy学习")
    print("4. Off-Policy Learning with Tree Backup")
    print("="*60)
    
    # 创建随机行为策略
    # Create random behavior policy
    behavior_policy = UniformRandomPolicy(env.action_space)
    
    tree_backup_offpolicy = NStepTreeBackup(
        env, n=4, gamma=0.99, alpha=0.1, epsilon=0.05  # 目标策略更贪婪
    )
    
    print("\n使用随机行为策略学习贪婪目标策略...")
    print("Learning greedy target policy from random behavior policy...")
    
    Q_offpolicy = tree_backup_offpolicy.learn(
        n_episodes=300, 
        behavior_policy=behavior_policy,
        verbose=False
    )
    
    avg_return = np.mean(tree_backup_offpolicy.episode_returns[-50:])
    print(f"\n最终平均回报: {avg_return:.2f}")
    print("✓ Tree Backup成功从随机策略学习（无需重要性采样）")
    print("✓ Tree Backup successfully learned from random policy (without IS)")
    
    # 总结
    # Summary
    print("\n" + "="*80)
    print("Tree Backup算法总结")
    print("Tree Backup Algorithm Summary")
    print("="*80)
    
    print("""
    关键要点 Key Takeaways:
    =======================
    
    1. Tree Backup是完全off-policy的
       Tree Backup is fully off-policy
       
    2. 无需重要性采样，避免高方差
       No importance sampling, avoid high variance
       
    3. 考虑所有动作的期望
       Consider expectation over all actions
       
    4. 计算复杂度较高
       Higher computational complexity
       
    5. 稳定性优于基于IS的方法
       More stable than IS-based methods
    
    适用场景 Use Cases:
    - Off-policy学习
      Off-policy learning
    - 行为策略未知
      Unknown behavior policy
    - 需要低方差
      Need low variance
    - 动作空间不太大
      Action space not too large
    """)
    
    plt.show()


if __name__ == "__main__":
    demonstrate_tree_backup()