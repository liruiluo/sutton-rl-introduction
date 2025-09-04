"""
================================================================================
第4.2节：策略迭代 - 通过评估和改进找到最优策略
Section 4.2: Policy Iteration - Finding Optimal Policy through Evaluation and Improvement
================================================================================

策略迭代是动态规划的经典算法之一，它通过不断交替进行策略评估和策略改进来找到最优策略。
Policy Iteration is one of the classic DP algorithms, finding optimal policy by alternating 
between policy evaluation and policy improvement.

算法流程就像爬山：
The algorithm is like hill climbing:
1. 评估当前位置的高度（策略评估）
   Evaluate current position height (policy evaluation)
2. 找到更高的方向（策略改进）
   Find higher direction (policy improvement)
3. 移动到新位置（更新策略）
   Move to new position (update policy)
4. 重复直到到达山顶（最优策略）
   Repeat until reach peak (optimal policy)

为什么这个方法有效？
Why does this work?
- 策略评估给出准确的v_π
  Policy evaluation gives exact v_π
- 策略改进保证新策略不更差
  Policy improvement guarantees new policy not worse
- 有限MDP只有有限个策略，必然收敛
  Finite MDP has finite policies, must converge
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from collections import defaultdict
import time

# 导入基础组件
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.ch03_finite_mdp.mdp_framework import (
    State, Action, MDPEnvironment
)
from src.ch03_finite_mdp.policies_and_values import (
    Policy, StateValueFunction, ActionValueFunction,
    DeterministicPolicy, StochasticPolicy
)
from .dp_foundations import (
    PolicyEvaluationDP, PolicyImprovementDP,
    BellmanOperator
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第4.2.1节：策略迭代算法
# Section 4.2.1: Policy Iteration Algorithm
# ================================================================================

class PolicyIteration:
    """
    策略迭代算法
    Policy Iteration Algorithm
    
    这是求解MDP的第一个完整算法！
    This is the first complete algorithm for solving MDPs!
    
    算法伪代码：
    Algorithm Pseudocode:
    ```
    1. 初始化
       Initialization
       对所有s∈S，任意初始化V(s)和π(s)
       For all s∈S, arbitrarily initialize V(s) and π(s)
    
    2. 策略评估（Policy Evaluation）
       重复
       Repeat
         Δ ← 0
         对每个s∈S：
         For each s∈S:
           v ← V(s)
           V(s) ← Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γV(s')]
           Δ ← max(Δ, |v - V(s)|)
       直到Δ < θ（一个小的阈值）
       until Δ < θ (a small threshold)
    
    3. 策略改进（Policy Improvement）
       policy_stable ← true
       对每个s∈S：
       For each s∈S:
         old_action ← π(s)
         π(s) ← argmax_a Σ_{s',r} p(s',r|s,a)[r + γV(s')]
         如果old_action ≠ π(s)，则policy_stable ← false
         If old_action ≠ π(s), then policy_stable ← false
       
    4. 如果policy_stable，则停止；否则回到2
       If policy_stable, then stop; else go to 2
    ```
    
    关键性质：
    Key Properties:
    1. 有限步收敛到最优策略
       Converges to optimal policy in finite steps
    2. 每次迭代策略严格改进（除非已最优）
       Each iteration strictly improves policy (unless optimal)
    3. 计算量大但精确
       Computationally expensive but exact
    """
    
    def __init__(self, mdp_env: MDPEnvironment, gamma: float = 0.99):
        """
        初始化策略迭代
        Initialize Policy Iteration
        
        Args:
            mdp_env: MDP环境（需要完整模型）
                    MDP environment (needs complete model)
            gamma: 折扣因子
                  Discount factor
        
        设计考虑：
        Design Considerations:
        - 使用组合而非继承，保持模块独立
          Use composition over inheritance, keep modules independent
        - 记录详细历史用于分析和可视化
          Record detailed history for analysis and visualization
        """
        self.env = mdp_env
        self.gamma = gamma
        
        # 创建评估器和改进器
        self.evaluator = PolicyEvaluationDP(mdp_env, gamma)
        self.improver = PolicyImprovementDP(mdp_env, gamma)
        
        # 记录迭代历史
        self.iteration_history = []
        
        # 性能统计
        self.total_evaluations = 0
        self.total_improvements = 0
        self.total_time = 0.0
        
        logger.info(f"初始化策略迭代，γ={gamma}")
    
    def solve(self, 
             initial_policy: Optional[Policy] = None,
             theta: float = 1e-6,
             max_iterations: int = 100,
             verbose: bool = True) -> Tuple[Policy, StateValueFunction]:
        """
        运行策略迭代算法
        Run Policy Iteration Algorithm
        
        这是算法的主入口，协调评估和改进的循环
        This is the main entry point, coordinating evaluation and improvement loop
        
        Args:
            initial_policy: 初始策略（None则使用随机策略）
                          Initial policy (None for random policy)
            theta: 策略评估的收敛阈值
                  Convergence threshold for policy evaluation
            max_iterations: 最大迭代次数（防止无限循环）
                          Maximum iterations (prevent infinite loop)
            verbose: 是否打印详细信息
                    Whether to print detailed info
        
        Returns:
            (最优策略, 最优价值函数)
            (optimal policy, optimal value function)
        
        算法复杂度分析：
        Complexity Analysis:
        - 每次策略评估: O(|S|²|A|) × 收敛所需迭代次数
          Each evaluation: O(|S|²|A|) × iterations to converge
        - 每次策略改进: O(|S||A|)
          Each improvement: O(|S||A|)
        - 总迭代次数: 通常很少（<10）对于小问题
          Total iterations: Usually few (<10) for small problems
        
        教学要点：
        Teaching Points:
        1. 注意策略是如何逐步改进的
           Notice how policy gradually improves
        2. 评估需要多次迭代，改进只需一次扫描
           Evaluation needs many iterations, improvement needs one sweep
        3. 当策略不再改变时，我们找到了最优策略
           When policy stops changing, we found optimal policy
        """
        # 清空历史记录
        self.iteration_history = []
        self.total_evaluations = 0
        self.total_improvements = 0
        
        # 开始计时
        start_time = time.time()
        
        # 初始化策略
        if initial_policy is None:
            # 使用均匀随机策略作为初始策略
            from src.ch03_finite_mdp.policies_and_values import UniformRandomPolicy
            policy = UniformRandomPolicy(self.env.action_space)
            logger.info("使用均匀随机策略初始化")
        else:
            policy = initial_policy
            logger.info("使用提供的初始策略")
        
        # 初始化价值函数
        V = StateValueFunction(self.env.state_space, initial_value=0.0)
        
        if verbose:
            print("\n" + "="*60)
            print("开始策略迭代")
            print("Starting Policy Iteration")
            print("="*60)
        
        # 主循环：交替进行评估和改进
        for iteration in range(max_iterations):
            if verbose:
                print(f"\n--- 迭代 {iteration + 1} ---")
                print(f"--- Iteration {iteration + 1} ---")
            
            # ============ 步骤1：策略评估 ============
            # Step 1: Policy Evaluation
            if verbose:
                print("执行策略评估...")
                print("Performing policy evaluation...")
            
            eval_start = time.time()
            V = self.evaluator.evaluate(policy, theta=theta)
            eval_time = time.time() - eval_start
            self.total_evaluations += 1
            
            if verbose:
                print(f"  评估完成，用时 {eval_time:.3f}秒")
                print(f"  Evaluation done in {eval_time:.3f}s")
                
                # 显示一些状态的价值
                sample_states = self.env.state_space[:min(3, len(self.env.state_space))]
                for state in sample_states:
                    print(f"    V({state.id}) = {V.get_value(state):.3f}")
            
            # ============ 步骤2：策略改进 ============
            # Step 2: Policy Improvement
            if verbose:
                print("执行策略改进...")
                print("Performing policy improvement...")
            
            improve_start = time.time()
            
            # 记录旧策略（用于比较）
            old_policy_actions = {}
            if isinstance(policy, DeterministicPolicy):
                old_policy_actions = policy.policy_map.copy()
            
            # 改进策略
            new_policy, policy_changed = self.improver.improve(V)
            improve_time = time.time() - improve_start
            self.total_improvements += 1
            
            # 检查策略是否改变
            policy_stable = True
            changes_count = 0
            
            for state in self.env.state_space:
                if state.is_terminal:
                    continue
                
                # 比较新旧策略
                if isinstance(new_policy, DeterministicPolicy) and isinstance(policy, DeterministicPolicy):
                    if state in new_policy.policy_map and state in old_policy_actions:
                        if new_policy.policy_map[state] != old_policy_actions[state]:
                            policy_stable = False
                            changes_count += 1
                            
                            if verbose and changes_count <= 3:  # 只显示前3个变化
                                old_action = old_policy_actions[state]
                                new_action = new_policy.policy_map[state]
                                print(f"    状态 {state.id}: {old_action.id} → {new_action.id}")
            
            if verbose:
                print(f"  改进完成，用时 {improve_time:.3f}秒")
                print(f"  Improvement done in {improve_time:.3f}s")
                print(f"  策略变化: {changes_count}个状态")
                print(f"  Policy changes: {changes_count} states")
            
            # 记录本次迭代
            iteration_data = {
                'iteration': iteration + 1,
                'value_function': V,
                'policy': new_policy,
                'policy_stable': policy_stable,
                'changes_count': changes_count,
                'eval_time': eval_time,
                'improve_time': improve_time
            }
            self.iteration_history.append(iteration_data)
            
            # 更新策略
            policy = new_policy
            
            # ============ 步骤3：检查收敛 ============
            # Step 3: Check Convergence
            if policy_stable:
                if verbose:
                    print("\n" + "="*60)
                    print(f"✓ 策略迭代收敛！")
                    print(f"✓ Policy Iteration Converged!")
                    print(f"  总迭代次数: {iteration + 1}")
                    print(f"  Total iterations: {iteration + 1}")
                    print("="*60)
                
                logger.info(f"策略迭代在第{iteration + 1}次迭代收敛")
                break
        else:
            # 达到最大迭代次数
            logger.warning(f"达到最大迭代次数 {max_iterations}")
            if verbose:
                print(f"\n⚠ 达到最大迭代次数 {max_iterations}，可能未完全收敛")
                print(f"⚠ Reached max iterations {max_iterations}, may not fully converged")
        
        # 记录总时间
        self.total_time = time.time() - start_time
        
        # 打印最终统计
        if verbose:
            self._print_statistics()
        
        return policy, V
    
    def _print_statistics(self):
        """
        打印算法统计信息
        Print Algorithm Statistics
        
        帮助理解算法的计算成本
        Helps understand computational cost of algorithm
        """
        print("\n算法统计 Algorithm Statistics:")
        print("-" * 40)
        print(f"总运行时间: {self.total_time:.3f}秒")
        print(f"Total runtime: {self.total_time:.3f}s")
        print(f"策略评估次数: {self.total_evaluations}")
        print(f"Policy evaluations: {self.total_evaluations}")
        print(f"策略改进次数: {self.total_improvements}")
        print(f"Policy improvements: {self.total_improvements}")
        
        if self.iteration_history:
            total_eval_time = sum(it['eval_time'] for it in self.iteration_history)
            total_improve_time = sum(it['improve_time'] for it in self.iteration_history)
            print(f"评估总时间: {total_eval_time:.3f}秒 ({total_eval_time/self.total_time*100:.1f}%)")
            print(f"改进总时间: {total_improve_time:.3f}秒 ({total_improve_time/self.total_time*100:.1f}%)")


# ================================================================================
# 第4.2.2节：策略迭代可视化
# Section 4.2.2: Policy Iteration Visualization
# ================================================================================

class PolicyIterationVisualizer:
    """
    策略迭代可视化器
    Policy Iteration Visualizer
    
    可视化是理解算法的关键！
    Visualization is key to understanding algorithms!
    
    展示内容：
    What to show:
    1. 策略演化：策略如何逐步改进
       Policy evolution: how policy improves step by step
    2. 价值函数变化：V(s)如何收敛
       Value function changes: how V(s) converges
    3. 收敛过程：迭代次数与改进关系
       Convergence process: iterations vs improvements
    """
    
    @staticmethod
    def visualize_convergence(policy_iter: PolicyIteration):
        """
        可视化收敛过程
        Visualize Convergence Process
        
        展示策略迭代的收敛特性
        Show convergence characteristics of policy iteration
        """
        if not policy_iter.iteration_history:
            logger.warning("没有迭代历史可视化")
            return None
        
        history = policy_iter.iteration_history
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # ========== 图1：策略变化数量 ==========
        # Chart 1: Number of Policy Changes
        ax1 = axes[0, 0]
        iterations = [h['iteration'] for h in history]
        changes = [h['changes_count'] for h in history]
        
        ax1.bar(iterations, changes, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Iteration / 迭代')
        ax1.set_ylabel('Policy Changes / 策略变化数')
        ax1.set_title('Policy Changes per Iteration / 每次迭代的策略变化')
        ax1.grid(True, alpha=0.3)
        
        # 标注收敛点
        if changes[-1] == 0:
            ax1.axvline(x=iterations[-1], color='red', linestyle='--', 
                       label='Converged / 收敛')
            ax1.legend()
        
        # ========== 图2：计算时间分布 ==========
        # Chart 2: Computation Time Distribution
        ax2 = axes[0, 1]
        eval_times = [h['eval_time'] for h in history]
        improve_times = [h['improve_time'] for h in history]
        
        width = 0.35
        x_pos = np.arange(len(iterations))
        
        bars1 = ax2.bar(x_pos - width/2, eval_times, width, 
                       label='Evaluation / 评估', color='lightblue')
        bars2 = ax2.bar(x_pos + width/2, improve_times, width,
                       label='Improvement / 改进', color='lightcoral')
        
        ax2.set_xlabel('Iteration / 迭代')
        ax2.set_ylabel('Time (seconds) / 时间（秒）')
        ax2.set_title('Computation Time per Step / 每步计算时间')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(iterations)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # ========== 图3：价值函数演化（热力图） ==========
        # Chart 3: Value Function Evolution (Heatmap)
        ax3 = axes[1, 0]
        
        # 收集所有迭代的价值函数
        states = policy_iter.env.state_space
        n_states = min(10, len(states))  # 最多显示10个状态
        sample_states = states[:n_states]
        
        value_matrix = np.zeros((len(history), n_states))
        for i, h in enumerate(history):
            V = h['value_function']
            for j, state in enumerate(sample_states):
                value_matrix[i, j] = V.get_value(state)
        
        # 绘制热力图
        im = ax3.imshow(value_matrix.T, aspect='auto', cmap='coolwarm')
        ax3.set_xlabel('Iteration / 迭代')
        ax3.set_ylabel('State / 状态')
        ax3.set_title('Value Function Evolution / 价值函数演化')
        ax3.set_xticks(range(len(history)))
        ax3.set_xticklabels([h['iteration'] for h in history])
        ax3.set_yticks(range(n_states))
        ax3.set_yticklabels([s.id for s in sample_states])
        
        # 添加颜色条
        plt.colorbar(im, ax=ax3, label='State Value / 状态价值')
        
        # ========== 图4：算法流程示意 ==========
        # Chart 4: Algorithm Flow Diagram
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # 绘制流程图
        PolicyIterationVisualizer._draw_flow_diagram(ax4)
        
        plt.suptitle('Policy Iteration Analysis / 策略迭代分析', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def _draw_flow_diagram(ax):
        """
        绘制算法流程图
        Draw Algorithm Flow Diagram
        
        帮助理解算法的逻辑流程
        Helps understand algorithm logic flow
        """
        # 设置坐标范围
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        # 定义方框样式
        box_style = "round,pad=0.3"
        
        # 1. 初始化
        init_box = FancyBboxPatch((1, 8), 3, 1,
                                  boxstyle=box_style,
                                  facecolor='lightgreen',
                                  edgecolor='black',
                                  linewidth=2)
        ax.add_patch(init_box)
        ax.text(2.5, 8.5, 'Initialize π', ha='center', va='center', fontweight='bold')
        
        # 2. 策略评估
        eval_box = FancyBboxPatch((1, 5.5), 3, 1,
                                  boxstyle=box_style,
                                  facecolor='lightblue',
                                  edgecolor='black',
                                  linewidth=2)
        ax.add_patch(eval_box)
        ax.text(2.5, 6, 'Policy\nEvaluation', ha='center', va='center', fontweight='bold')
        
        # 3. 策略改进
        improve_box = FancyBboxPatch((6, 5.5), 3, 1,
                                     boxstyle=box_style,
                                     facecolor='lightcoral',
                                     edgecolor='black',
                                     linewidth=2)
        ax.add_patch(improve_box)
        ax.text(7.5, 6, 'Policy\nImprovement', ha='center', va='center', fontweight='bold')
        
        # 4. 检查收敛
        check_box = FancyBboxPatch((3.5, 2.5), 3, 1,
                                   boxstyle=box_style,
                                   facecolor='lightyellow',
                                   edgecolor='black',
                                   linewidth=2)
        ax.add_patch(check_box)
        ax.text(5, 3, 'Converged?', ha='center', va='center', fontweight='bold')
        
        # 5. 输出
        output_box = FancyBboxPatch((3.5, 0.5), 3, 0.8,
                                    boxstyle=box_style,
                                    facecolor='lightgreen',
                                    edgecolor='black',
                                    linewidth=2)
        ax.add_patch(output_box)
        ax.text(5, 0.9, 'Output π*', ha='center', va='center', fontweight='bold')
        
        # 绘制箭头
        # 初始化 -> 评估
        ax.arrow(2.5, 7.9, 0, -1.3, head_width=0.2, head_length=0.1, 
                fc='black', ec='black')
        
        # 评估 -> 改进
        ax.arrow(4.1, 6, 1.8, 0, head_width=0.2, head_length=0.1,
                fc='black', ec='black')
        
        # 改进 -> 检查
        ax.arrow(7.5, 5.4, -2, -1.8, head_width=0.2, head_length=0.1,
                fc='black', ec='black')
        
        # 检查 -> 评估（循环）
        ax.arrow(3.4, 3, -1, 2.4, head_width=0.2, head_length=0.1,
                fc='blue', ec='blue')
        ax.text(2, 4.2, 'No', color='blue', fontweight='bold')
        
        # 检查 -> 输出（收敛）
        ax.arrow(5, 2.4, 0, -1, head_width=0.2, head_length=0.1,
                fc='green', ec='green')
        ax.text(5.5, 1.7, 'Yes', color='green', fontweight='bold')
        
        # 添加标题
        ax.text(5, 9.5, 'Policy Iteration Flow', ha='center', fontsize=12, fontweight='bold')
        ax.text(5, 9, '策略迭代流程', ha='center', fontsize=10)
    
    @staticmethod
    def visualize_policy_evolution(policy_iter: PolicyIteration, 
                                  grid_env=None):
        """
        可视化策略演化（适用于网格世界）
        Visualize Policy Evolution (for Grid World)
        
        展示策略在网格世界中如何逐步改进
        Show how policy improves step by step in grid world
        """
        if not policy_iter.iteration_history:
            logger.warning("没有迭代历史可视化")
            return None
        
        if grid_env is None:
            logger.warning("需要网格世界环境进行策略可视化")
            return None
        
        history = policy_iter.iteration_history
        n_iterations = len(history)
        
        # 创建子图
        n_cols = min(4, n_iterations)
        n_rows = (n_iterations + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_iterations == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # 为每个迭代绘制策略
        for idx, h in enumerate(history):
            ax = axes[idx] if n_iterations > 1 else axes[0]
            
            # 绘制网格
            PolicyIterationVisualizer._draw_grid_policy(
                ax, grid_env, h['policy'], h['value_function']
            )
            
            ax.set_title(f'Iteration {h["iteration"]}\n'
                        f'Changes: {h["changes_count"]}')
        
        # 隐藏多余的子图
        for idx in range(n_iterations, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Policy Evolution / 策略演化', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def _draw_grid_policy(ax, grid_env, policy, value_function):
        """
        在网格上绘制策略
        Draw Policy on Grid
        
        使用箭头表示动作，颜色表示价值
        Use arrows for actions, colors for values
        """
        rows, cols = grid_env.rows, grid_env.cols
        
        # 设置坐标
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(-0.5, rows - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        # 绘制网格线
        for i in range(rows + 1):
            ax.axhline(y=i - 0.5, color='gray', linewidth=0.5)
        for j in range(cols + 1):
            ax.axvline(x=j - 0.5, color='gray', linewidth=0.5)
        
        # 动作到箭头的映射
        action_arrows = {
            'up': (0, -0.3),
            'down': (0, 0.3),
            'left': (-0.3, 0),
            'right': (0.3, 0)
        }
        
        # 绘制每个格子
        for i in range(rows):
            for j in range(cols):
                pos = (i, j)
                
                # 检查是否是障碍物
                if pos in grid_env.obstacles:
                    rect = patches.Rectangle((j-0.5, i-0.5), 1, 1,
                                           facecolor='gray', alpha=0.8)
                    ax.add_patch(rect)
                    continue
                
                # 获取状态
                if pos in grid_env.pos_to_state:
                    state = grid_env.pos_to_state[pos]
                    
                    # 获取价值并着色
                    value = value_function.get_value(state)
                    # 归一化价值用于着色
                    norm_value = (value - value_function.V.values().min()) / \
                                (value_function.V.values().max() - value_function.V.values().min() + 1e-10)
                    color = plt.cm.coolwarm(norm_value)
                    
                    rect = patches.Rectangle((j-0.5, i-0.5), 1, 1,
                                           facecolor=color, alpha=0.3)
                    ax.add_patch(rect)
                    
                    # 显示价值
                    ax.text(j, i-0.3, f'{value:.1f}', 
                           ha='center', va='center', fontsize=8)
                    
                    # 绘制策略箭头
                    if not state.is_terminal and isinstance(policy, DeterministicPolicy):
                        if state in policy.policy_map:
                            action = policy.policy_map[state]
                            if action.id in action_arrows:
                                dx, dy = action_arrows[action.id]
                                ax.arrow(j, i, dx, dy, head_width=0.15, 
                                       head_length=0.1, fc='black', ec='black')
                
                # 标记特殊位置
                if pos == grid_env.start_pos:
                    ax.text(j, i+0.3, 'S', ha='center', va='center',
                           fontweight='bold', color='green')
                elif pos == grid_env.goal_pos:
                    ax.text(j, i+0.3, 'G', ha='center', va='center',
                           fontweight='bold', color='red')
        
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.grid(True, alpha=0.3)


# ================================================================================
# 第4.2.3节：策略迭代分析
# Section 4.2.3: Policy Iteration Analysis  
# ================================================================================

class PolicyIterationAnalysis:
    """
    策略迭代理论分析
    Policy Iteration Theoretical Analysis
    
    深入分析算法的性质和性能
    Deep analysis of algorithm properties and performance
    """
    
    @staticmethod
    def analyze_convergence():
        """
        分析收敛性
        Analyze Convergence
        
        策略迭代的收敛性分析
        Convergence analysis of policy iteration
        """
        print("\n" + "="*80)
        print("策略迭代收敛性分析")
        print("Policy Iteration Convergence Analysis")
        print("="*80)
        
        print("""
        📊 1. 收敛性保证
        Convergence Guarantee
        ================================
        
        定理：对于有限MDP，策略迭代在有限步内收敛到最优策略。
        Theorem: For finite MDPs, policy iteration converges to optimal policy in finite steps.
        
        证明要点 Proof Outline:
        1. 每次改进要么严格改进策略，要么策略不变
           Each improvement either strictly improves or keeps policy unchanged
        2. 有限MDP只有有限个确定性策略：|A|^|S|
           Finite MDP has finite deterministic policies: |A|^|S|
        3. 不会重复访问同一策略（单调改进）
           Won't revisit same policy (monotonic improvement)
        4. 因此必在有限步内达到最优
           Therefore must reach optimal in finite steps
        
        📊 2. 收敛速度
        Convergence Speed
        ================================
        
        实践观察：
        Empirical Observations:
        - 通常收敛很快（<10次迭代）
          Usually converges quickly (<10 iterations)
        - 远少于策略总数|A|^|S|
          Much less than total policies |A|^|S|
        
        原因 Reasons:
        1. 策略改进通常改变多个状态的动作
           Policy improvement usually changes actions for multiple states
        2. 向最优策略的"捷径"
           "Shortcuts" toward optimal policy
        3. 好的初始策略加速收敛
           Good initial policy speeds convergence
        
        📊 3. 计算复杂度
        Computational Complexity
        ================================
        
        每次迭代：
        Per iteration:
        - 策略评估: O(|S|²|A|) × 评估迭代次数
          Policy evaluation: O(|S|²|A|) × evaluation iterations
        - 策略改进: O(|S||A||S|) = O(|S|²|A|)
          Policy improvement: O(|S||A||S|) = O(|S|²|A|)
        
        总复杂度：
        Total complexity:
        O(K × |S|²|A| × I)
        其中 where:
        - K: 策略迭代次数（通常很小）
          K: policy iterations (usually small)
        - I: 每次策略评估的迭代次数
          I: iterations per policy evaluation
        
        📊 4. vs 其他算法
        vs Other Algorithms
        ================================
        
        | 算法 Algorithm | 每步计算 Per Step | 收敛速度 Convergence | 精确性 Exactness |
        |----------------|------------------|---------------------|-----------------|
        | 策略迭代 PI     | 高 High          | 快 Fast             | 精确 Exact      |
        | 价值迭代 VI     | 中 Medium        | 慢 Slow             | 精确 Exact      |
        | 修改的PI       | 低 Low           | 中 Medium           | 精确 Exact      |
        | Q-学习         | 低 Low           | 慢 Slow             | 近似 Approx     |
        
        策略迭代的优势：
        Advantages of Policy Iteration:
        ✓ 收敛步数少
          Few convergence steps
        ✓ 每步都有明确的策略
          Clear policy at each step
        ✓ 理论保证强
          Strong theoretical guarantees
        
        劣势：
        Disadvantages:
        ✗ 每步计算量大（完整策略评估）
          High computation per step (full evaluation)
        ✗ 需要完整模型
          Needs complete model
        ✗ 不适合大状态空间
          Not suitable for large state spaces
        """)
    
    @staticmethod
    def compare_with_initial_policies(env):
        """
        比较不同初始策略的影响
        Compare Impact of Different Initial Policies
        
        展示初始策略如何影响收敛速度
        Show how initial policy affects convergence speed
        """
        print("\n" + "="*80)
        print("初始策略对比实验")
        print("Initial Policy Comparison Experiment")
        print("="*80)
        
        from src.ch03_finite_mdp.policies_and_values import UniformRandomPolicy
        
        # 不同的初始策略
        initial_policies = []
        
        # 1. 随机策略
        random_policy = UniformRandomPolicy(env.action_space)
        initial_policies.append(("Random", random_policy))
        
        # 2. 总是向右的策略（可能不错的启发式）
        right_policy_map = {}
        right_action = None
        for action in env.action_space:
            if action.id == 'right':
                right_action = action
                break
        
        if right_action:
            for state in env.state_space:
                if not state.is_terminal:
                    right_policy_map[state] = right_action
            right_policy = DeterministicPolicy(right_policy_map)
            initial_policies.append(("Always Right", right_policy))
        
        # 3. 总是向上的策略（可能较差的启发式）
        up_policy_map = {}
        up_action = None
        for action in env.action_space:
            if action.id == 'up':
                up_action = action
                break
        
        if up_action:
            for state in env.state_space:
                if not state.is_terminal:
                    up_policy_map[state] = up_action
            up_policy = DeterministicPolicy(up_policy_map)
            initial_policies.append(("Always Up", up_policy))
        
        # 运行实验
        results = []
        
        for name, init_policy in initial_policies:
            print(f"\n测试初始策略: {name}")
            print(f"Testing initial policy: {name}")
            
            # 运行策略迭代
            pi = PolicyIteration(env, gamma=0.9)
            policy, V = pi.solve(initial_policy=init_policy, verbose=False)
            
            result = {
                'name': name,
                'iterations': len(pi.iteration_history),
                'total_time': pi.total_time,
                'final_value': sum(V.get_value(s) for s in env.state_space) / len(env.state_space)
            }
            results.append(result)
            
            print(f"  收敛迭代数: {result['iterations']}")
            print(f"  总时间: {result['total_time']:.3f}秒")
            print(f"  平均价值: {result['final_value']:.3f}")
        
        # 可视化结果
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        names = [r['name'] for r in results]
        iterations = [r['iterations'] for r in results]
        times = [r['total_time'] for r in results]
        values = [r['final_value'] for r in results]
        
        # 迭代次数
        ax1 = axes[0]
        bars1 = ax1.bar(names, iterations, color='steelblue', alpha=0.7)
        ax1.set_ylabel('Iterations to Converge')
        ax1.set_title('Convergence Speed')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 标注数值
        for bar, val in zip(bars1, iterations):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val}', ha='center', va='bottom')
        
        # 运行时间
        ax2 = axes[1]
        bars2 = ax2.bar(names, times, color='lightcoral', alpha=0.7)
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Runtime')
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars2, times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom')
        
        # 最终价值
        ax3 = axes[2]
        bars3 = ax3.bar(names, values, color='lightgreen', alpha=0.7)
        ax3.set_ylabel('Average State Value')
        ax3.set_title('Final Policy Quality')
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars3, values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom')
        
        plt.suptitle('Impact of Initial Policy / 初始策略的影响', fontweight='bold')
        plt.tight_layout()
        
        return fig


# ================================================================================
# 主函数：演示策略迭代
# Main Function: Demonstrate Policy Iteration
# ================================================================================

def main():
    """
    运行策略迭代完整演示
    Run Complete Policy Iteration Demo
    """
    print("\n" + "="*80)
    print("第4.2节：策略迭代")
    print("Section 4.2: Policy Iteration")
    print("="*80)
    
    # 创建测试环境
    from src.ch03_finite_mdp.gridworld import GridWorld
    
    # 创建4x4网格世界（稍大一些展示算法）
    env = GridWorld(
        rows=4, 
        cols=4,
        start_pos=(0, 0),
        goal_pos=(3, 3),
        obstacles={(1, 1), (2, 2)}  # 添加一些障碍物
    )
    
    print(f"\n创建 {env.rows}×{env.cols} 网格世界")
    print(f"起点: {env.start_pos}, 终点: {env.goal_pos}")
    print(f"障碍物: {env.obstacles}")
    
    # 1. 运行策略迭代
    print("\n" + "="*60)
    print("运行策略迭代算法")
    print("Running Policy Iteration Algorithm")
    print("="*60)
    
    pi = PolicyIteration(env, gamma=0.9)
    optimal_policy, optimal_V = pi.solve(verbose=True)
    
    # 2. 可视化收敛过程
    print("\n可视化收敛过程...")
    visualizer = PolicyIterationVisualizer()
    fig1 = visualizer.visualize_convergence(pi)
    
    # 3. 可视化策略演化
    print("\n可视化策略演化...")
    fig2 = visualizer.visualize_policy_evolution(pi, env)
    
    # 4. 理论分析
    PolicyIterationAnalysis.analyze_convergence()
    
    # 5. 初始策略对比
    print("\n运行初始策略对比实验...")
    fig3 = PolicyIterationAnalysis.compare_with_initial_policies(env)
    
    # 显示最优策略的一些信息
    print("\n" + "="*60)
    print("最优策略分析")
    print("Optimal Policy Analysis")
    print("="*60)
    
    # 显示几个状态的最优动作
    print("\n部分状态的最优动作:")
    print("Optimal actions for some states:")
    
    sample_positions = [(0, 0), (0, 1), (1, 0), (2, 1), (3, 2)]
    for pos in sample_positions:
        if pos in env.pos_to_state:
            state = env.pos_to_state[pos]
            if isinstance(optimal_policy, DeterministicPolicy) and state in optimal_policy.policy_map:
                action = optimal_policy.policy_map[state]
                value = optimal_V.get_value(state)
                print(f"  位置 {pos}: {action.id} (V={value:.2f})")
    
    print("\n" + "="*80)
    print("策略迭代演示完成！")
    print("Policy Iteration Demo Complete!")
    print("\n关键要点 Key Takeaways:")
    print("1. 策略迭代交替进行评估和改进")
    print("   Policy iteration alternates evaluation and improvement")
    print("2. 通常收敛很快（<10次迭代）")
    print("   Usually converges quickly (<10 iterations)")
    print("3. 每次迭代都保证不会变差")
    print("   Each iteration guaranteed not worse")
    print("4. 适合小到中等规模的问题")
    print("   Suitable for small to medium problems")
    print("="*80)
    
    plt.show()


if __name__ == "__main__":
    main()