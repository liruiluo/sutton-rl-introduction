"""
================================================================================
第3.4节：广义策略迭代 - 强化学习的核心模式
Section 3.4: Generalized Policy Iteration - The Core Pattern of RL
================================================================================

广义策略迭代(GPI)是几乎所有强化学习方法的底层模式。
Generalized Policy Iteration (GPI) is the underlying pattern of almost all RL methods.

核心思想：同时维护一个策略π和一个价值函数V，两者相互改进
Core idea: Maintain both a policy π and a value function V, improving each other

两个过程的交互：
Interaction of two processes:
1. 策略评估：使V接近v_π（让价值函数更准确）
   Policy Evaluation: Make V closer to v_π (make value function more accurate)
2. 策略改进：使π对V贪婪（让策略更好）
   Policy Improvement: Make π greedy w.r.t. V (make policy better)

这两个过程既竞争又合作：
These two processes both compete and cooperate:
- 竞争：一个的改变会让另一个不准确
  Competition: Change in one makes the other inaccurate
- 合作：共同向最优解前进
  Cooperation: Together they move toward optimal solution

比喻：像两个登山者用绳子相连
Analogy: Like two climbers connected by a rope
- 一个找更高的路（策略改进）
  One finds higher path (policy improvement)
- 一个稳定位置（策略评估）
  One stabilizes position (policy evaluation)
- 最终都到达山顶（最优策略）
  Eventually both reach peak (optimal policy)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.animation as animation
import seaborn as sns
from collections import defaultdict, deque
import time
from abc import ABC, abstractmethod

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
from dp_foundations import (
    BellmanOperator, PolicyEvaluationDP, PolicyImprovementDP
)
from policy_iteration import PolicyIteration
from value_iteration import ValueIteration

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第3.4.1节：GPI模式定义
# Section 3.4.1: GPI Pattern Definition
# ================================================================================

class GPIPattern(Enum):
    """
    GPI的不同模式
    Different Patterns of GPI
    
    这些是GPI的特殊情况，展示了评估和改进的不同平衡
    These are special cases of GPI, showing different balance of evaluation and improvement
    """
    
    # 经典模式
    POLICY_ITERATION = "policy_iteration"  # 完全评估 + 贪婪改进
                                           # Full evaluation + greedy improvement
    VALUE_ITERATION = "value_iteration"    # 一步评估 + 贪婪改进
                                           # One-step evaluation + greedy improvement
    
    # 修改的策略迭代
    MODIFIED_PI_2 = "modified_pi_2"        # 2步评估 + 贪婪改进
                                           # 2-step evaluation + greedy improvement
    MODIFIED_PI_K = "modified_pi_k"        # k步评估 + 贪婪改进
                                           # k-step evaluation + greedy improvement
    
    # 异步模式
    ASYNC_IN_PLACE = "async_in_place"      # 原地异步更新
                                           # In-place asynchronous update
    PRIORITIZED = "prioritized"            # 优先级扫描
                                           # Prioritized sweeping
    
    # 近似模式（为后续章节预留）
    APPROXIMATE = "approximate"             # 函数近似下的GPI
                                           # GPI with function approximation


@dataclass
class GPIState:
    """
    GPI过程的状态
    State of GPI Process
    
    记录GPI在某一时刻的完整状态
    Records complete state of GPI at a moment
    
    这个数据结构帮助我们理解GPI的动态过程
    This data structure helps us understand GPI dynamics
    """
    iteration: int                          # 当前迭代次数
                                           # Current iteration number
    policy: Policy                         # 当前策略
                                           # Current policy
    value_function: StateValueFunction      # 当前价值函数
                                           # Current value function
    evaluation_error: float                 # 评估误差 ||V - v_π||
                                           # Evaluation error
    improvement_delta: int                  # 策略改变的状态数
                                           # Number of states with policy change
    is_optimal: bool = False               # 是否已达到最优
                                           # Whether optimal reached
    
    # 性能指标
    evaluation_steps: int = 0              # 评估步数
                                           # Evaluation steps
    improvement_steps: int = 0             # 改进步数
                                           # Improvement steps
    computation_time: float = 0.0          # 计算时间
                                           # Computation time


# ================================================================================
# 第3.4.2节：广义策略迭代算法
# Section 3.4.2: Generalized Policy Iteration Algorithm
# ================================================================================

class GeneralizedPolicyIteration:
    """
    广义策略迭代 - RL的统一框架
    Generalized Policy Iteration - Unified Framework of RL
    
    这个类展示了所有DP算法都是GPI的特例
    This class shows all DP algorithms are special cases of GPI
    
    关键洞察：
    Key Insights:
    1. 不需要完全评估就可以改进策略
       Don't need full evaluation to improve policy
    2. 不需要完全贪婪就可以改进策略
       Don't need full greediness to improve policy
    3. 评估和改进可以交织进行
       Evaluation and improvement can be interleaved
    
    数学基础：
    Mathematical Foundation:
    - 单调性：每次改进不会变差
      Monotonicity: Each improvement not worse
    - 收敛性：最终收敛到最优
      Convergence: Eventually converges to optimal
    - 灵活性：可以在任何时候停止
      Flexibility: Can stop at any time
    """
    
    def __init__(self, mdp_env: MDPEnvironment, gamma: float = 0.99):
        """
        初始化GPI
        Initialize GPI
        
        Args:
            mdp_env: MDP环境
            gamma: 折扣因子
        
        设计思考：
        Design Consideration:
        将GPI设计成可配置的框架，支持不同的评估和改进策略
        Design GPI as configurable framework supporting different evaluation and improvement strategies
        """
        self.env = mdp_env
        self.gamma = gamma
        
        # 核心组件
        self.bellman_op = BellmanOperator(mdp_env, gamma)
        self.evaluator = PolicyEvaluationDP(mdp_env, gamma)
        self.improver = PolicyImprovementDP(mdp_env, gamma)
        
        # GPI历史记录
        self.gpi_history: List[GPIState] = []
        
        # 统计信息
        self.total_iterations = 0
        self.total_eval_steps = 0
        self.total_improve_steps = 0
        self.total_time = 0.0
        
        logger.info(f"初始化广义策略迭代，γ={gamma}")
    
    def solve(self,
             pattern: GPIPattern = GPIPattern.POLICY_ITERATION,
             initial_policy: Optional[Policy] = None,
             evaluation_steps: Union[int, str] = "full",
             improvement_type: str = "greedy",
             theta: float = 1e-6,
             max_iterations: int = 1000,
             verbose: bool = True) -> Tuple[Policy, StateValueFunction]:
        """
        运行GPI算法
        Run GPI Algorithm
        
        这是GPI的核心实现，展示了不同算法如何作为参数配置
        This is the core implementation of GPI, showing how different algorithms are parameter configurations
        
        Args:
            pattern: GPI模式
                    GPI pattern
            initial_policy: 初始策略
                          Initial policy
            evaluation_steps: 评估步数
                            - "full": 完全评估（策略迭代）
                            - 1: 一步评估（价值迭代）
                            - k: k步评估（修改的策略迭代）
                            Evaluation steps
                            - "full": full evaluation (policy iteration)
                            - 1: one step (value iteration)
                            - k: k steps (modified policy iteration)
            improvement_type: 改进类型
                            - "greedy": 完全贪婪
                            - "epsilon_greedy": ε-贪婪
                            - "soft": 软改进
                            Improvement type
            theta: 收敛阈值
                  Convergence threshold
            max_iterations: 最大迭代次数
                          Maximum iterations
            verbose: 是否打印详细信息
                    Whether to print details
        
        Returns:
            (最优策略, 最优价值函数)
            (optimal policy, optimal value function)
        
        算法框架：
        Algorithm Framework:
        ```
        初始化 π, V
        Initialize π, V
        重复：
        Repeat:
            部分评估：V → v_π的方向移动
            Partial evaluation: V moves toward v_π
            部分改进：π → π'使得π'对V更贪婪
            Partial improvement: π → π' to be more greedy w.r.t. V
        直到收敛
        Until convergence
        ```
        """
        # 清空历史
        self.gpi_history = []
        
        # 开始计时
        start_time = time.time()
        
        # 初始化策略
        if initial_policy is None:
            from src.ch03_finite_mdp.policies_and_values import UniformRandomPolicy
            policy = UniformRandomPolicy(self.env.action_space)
        else:
            policy = initial_policy
        
        # 初始化价值函数
        V = StateValueFunction(self.env.state_space, initial_value=0.0)
        
        if verbose:
            print("\n" + "="*80)
            print(f"开始广义策略迭代 (模式: {pattern.value})")
            print(f"Starting Generalized Policy Iteration (pattern: {pattern.value})")
            print("="*80)
            print(f"评估步数: {evaluation_steps}")
            print(f"改进类型: {improvement_type}")
            print(f"收敛阈值: {theta}")
        
        # 根据模式配置参数
        eval_steps, improve_type = self._configure_pattern(pattern, evaluation_steps, improvement_type)
        
        # GPI主循环
        for iteration in range(max_iterations):
            iter_start = time.time()
            
            if verbose and iteration % 10 == 0:
                print(f"\n--- GPI迭代 {iteration + 1} ---")
            
            # ========== 策略评估阶段 ==========
            # Policy Evaluation Phase
            V_old = StateValueFunction(self.env.state_space)
            for state in self.env.state_space:
                V_old.set_value(state, V.get_value(state))
            
            # 执行部分评估
            V, eval_error, eval_steps_taken = self._partial_evaluation(
                policy, V, eval_steps, theta
            )
            self.total_eval_steps += eval_steps_taken
            
            if verbose and iteration % 10 == 0:
                print(f"  评估: {eval_steps_taken}步, 误差={eval_error:.2e}")
            
            # ========== 策略改进阶段 ==========
            # Policy Improvement Phase
            new_policy, improvement_delta = self._policy_improvement(
                V, policy, improve_type
            )
            self.total_improve_steps += 1
            
            if verbose and iteration % 10 == 0:
                print(f"  改进: {improvement_delta}个状态改变")
            
            # 检查是否达到最优
            is_optimal = (improvement_delta == 0 and eval_error < theta)
            
            # 记录GPI状态
            gpi_state = GPIState(
                iteration=iteration + 1,
                policy=new_policy,
                value_function=V,
                evaluation_error=eval_error,
                improvement_delta=improvement_delta,
                is_optimal=is_optimal,
                evaluation_steps=eval_steps_taken,
                improvement_steps=1,
                computation_time=time.time() - iter_start
            )
            self.gpi_history.append(gpi_state)
            
            # 更新策略
            policy = new_policy
            
            # 检查收敛
            if is_optimal:
                self.total_iterations = iteration + 1
                if verbose:
                    print(f"\n✓ GPI收敛！")
                    print(f"  迭代次数: {self.total_iterations}")
                    print(f"  总评估步数: {self.total_eval_steps}")
                    print(f"  总改进步数: {self.total_improve_steps}")
                break
            
            # 早停检查（如果策略稳定但评估未完全收敛）
            if improvement_delta == 0 and eval_error < theta * 10:
                if verbose:
                    print(f"\n✓ 策略稳定，提前停止")
                self.total_iterations = iteration + 1
                break
        else:
            # 达到最大迭代
            self.total_iterations = max_iterations
            if verbose:
                print(f"\n⚠ 达到最大迭代次数 {max_iterations}")
        
        self.total_time = time.time() - start_time
        
        if verbose:
            self._print_statistics()
            self._analyze_convergence()
        
        return policy, V
    
    def _configure_pattern(self, pattern: GPIPattern, 
                          eval_steps: Union[int, str],
                          improve_type: str) -> Tuple[Union[int, str], str]:
        """
        根据GPI模式配置参数
        Configure parameters based on GPI pattern
        
        展示不同算法如何映射到GPI参数
        Shows how different algorithms map to GPI parameters
        """
        if pattern == GPIPattern.POLICY_ITERATION:
            # 策略迭代：完全评估 + 贪婪改进
            return "full", "greedy"
        elif pattern == GPIPattern.VALUE_ITERATION:
            # 价值迭代：一步评估 + 贪婪改进
            return 1, "greedy"
        elif pattern == GPIPattern.MODIFIED_PI_2:
            # 修改的策略迭代(m=2)
            return 2, "greedy"
        elif pattern == GPIPattern.MODIFIED_PI_K:
            # 修改的策略迭代(m=k)
            return eval_steps if isinstance(eval_steps, int) else 5, "greedy"
        elif pattern == GPIPattern.ASYNC_IN_PLACE:
            # 异步原地更新
            return 1, "greedy"
        elif pattern == GPIPattern.PRIORITIZED:
            # 优先级扫描
            return 1, "greedy"
        else:
            # 默认使用提供的参数
            return eval_steps, improve_type
    
    def _partial_evaluation(self, policy: Policy, V: StateValueFunction,
                           steps: Union[int, str], theta: float) -> Tuple[StateValueFunction, float, int]:
        """
        部分策略评估
        Partial Policy Evaluation
        
        这是GPI的关键：不需要完全评估！
        This is key to GPI: Don't need full evaluation!
        
        Args:
            policy: 当前策略
                   Current policy
            V: 当前价值函数
              Current value function
            steps: 评估步数
                  Evaluation steps
            theta: 收敛阈值
                  Convergence threshold
        
        Returns:
            (更新的价值函数, 评估误差, 实际步数)
            (updated value function, evaluation error, actual steps)
        """
        if steps == "full":
            # 完全评估（策略迭代模式）
            V_new = self.evaluator.evaluate(policy, theta=theta)
            steps_taken = len(self.evaluator.evaluation_history)
            
            # 计算误差
            error = 0.0
            for state in self.env.state_space:
                error = max(error, abs(V_new.get_value(state) - V.get_value(state)))
            
            return V_new, error, steps_taken
        
        else:
            # 部分评估（k步）
            steps_taken = 0
            max_error = 0.0
            
            for _ in range(steps):
                # 应用一次贝尔曼期望算子
                V_new = self.bellman_op.bellman_expectation_operator(V, policy)
                
                # 计算这一步的误差
                step_error = 0.0
                for state in self.env.state_space:
                    old_val = V.get_value(state)
                    new_val = V_new.get_value(state)
                    step_error = max(step_error, abs(new_val - old_val))
                
                max_error = max(max_error, step_error)
                V = V_new
                steps_taken += 1
                
                # 如果已经收敛，提前停止
                if step_error < theta:
                    break
            
            return V, max_error, steps_taken
    
    def _policy_improvement(self, V: StateValueFunction, 
                           current_policy: Policy,
                           improve_type: str) -> Tuple[Policy, int]:
        """
        策略改进
        Policy Improvement
        
        根据改进类型执行不同的策略更新
        Execute different policy updates based on improvement type
        
        Args:
            V: 价值函数
              Value function
            current_policy: 当前策略
                          Current policy
            improve_type: 改进类型
                        Improvement type
        
        Returns:
            (新策略, 改变的状态数)
            (new policy, number of changed states)
        """
        if improve_type == "greedy":
            # 完全贪婪改进
            new_policy, _ = self.improver.improve(V)
            
            # 计算改变的状态数
            changes = 0
            if isinstance(new_policy, DeterministicPolicy) and isinstance(current_policy, DeterministicPolicy):
                for state in self.env.state_space:
                    if not state.is_terminal:
                        if state in new_policy.policy_map and state in current_policy.policy_map:
                            if new_policy.policy_map[state] != current_policy.policy_map[state]:
                                changes += 1
            
            return new_policy, changes
        
        elif improve_type == "epsilon_greedy":
            # ε-贪婪改进（为后续章节预留）
            # 这里简化为贪婪改进
            return self._policy_improvement(V, current_policy, "greedy")
        
        elif improve_type == "soft":
            # 软改进（为后续章节预留）
            # 这里简化为贪婪改进
            return self._policy_improvement(V, current_policy, "greedy")
        
        else:
            raise ValueError(f"未知的改进类型: {improve_type}")
    
    def _print_statistics(self):
        """
        打印GPI统计信息
        Print GPI Statistics
        
        帮助理解不同GPI变体的性能特征
        Helps understand performance characteristics of different GPI variants
        """
        print("\n" + "-"*40)
        print("GPI统计信息")
        print("GPI Statistics")
        print("-"*40)
        
        print(f"总迭代次数: {self.total_iterations}")
        print(f"总评估步数: {self.total_eval_steps}")
        print(f"总改进步数: {self.total_improve_steps}")
        print(f"总运行时间: {self.total_time:.3f}秒")
        
        if self.total_eval_steps > 0:
            print(f"平均评估步数/迭代: {self.total_eval_steps/self.total_iterations:.1f}")
        
        print(f"平均时间/迭代: {self.total_time/self.total_iterations:.4f}秒")
    
    def _analyze_convergence(self):
        """
        分析收敛过程
        Analyze Convergence Process
        
        展示GPI的收敛特性
        Show convergence characteristics of GPI
        """
        if not self.gpi_history:
            return
        
        print("\n" + "-"*40)
        print("收敛分析")
        print("Convergence Analysis")
        print("-"*40)
        
        # 分析评估误差下降
        eval_errors = [state.evaluation_error for state in self.gpi_history]
        print(f"初始评估误差: {eval_errors[0]:.2e}")
        print(f"最终评估误差: {eval_errors[-1]:.2e}")
        
        # 分析策略改变
        policy_changes = [state.improvement_delta for state in self.gpi_history]
        stable_iteration = None
        for i, changes in enumerate(policy_changes):
            if changes == 0:
                stable_iteration = i + 1
                break
        
        if stable_iteration:
            print(f"策略稳定于迭代: {stable_iteration}")
        
        # 分析计算效率
        total_comp_time = sum(state.computation_time for state in self.gpi_history)
        eval_time = sum(state.computation_time * state.evaluation_steps / 
                       (state.evaluation_steps + state.improvement_steps)
                       for state in self.gpi_history)
        improve_time = total_comp_time - eval_time
        
        print(f"评估时间占比: {eval_time/total_comp_time*100:.1f}%")
        print(f"改进时间占比: {improve_time/total_comp_time*100:.1f}%")


# ================================================================================
# 第3.4.3节：GPI可视化
# Section 3.4.3: GPI Visualization
# ================================================================================

class GPIVisualizer:
    """
    GPI可视化器
    GPI Visualizer
    
    展示GPI的动态过程和收敛特性
    Show dynamics and convergence of GPI
    
    可视化帮助理解：
    Visualization helps understand:
    1. 评估和改进的相互作用
       Interaction between evaluation and improvement
    2. 不同GPI变体的收敛速度
       Convergence speed of different GPI variants
    3. 策略和价值函数的协同演化
       Co-evolution of policy and value function
    """
    
    @staticmethod
    def visualize_gpi_process(gpi: GeneralizedPolicyIteration):
        """
        可视化GPI过程
        Visualize GPI Process
        
        展示评估和改进的交替过程
        Show alternating process of evaluation and improvement
        """
        if not gpi.gpi_history:
            logger.warning("没有GPI历史可视化")
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # ========== 图1：评估误差和策略变化 ==========
        ax1 = axes[0, 0]
        
        iterations = [state.iteration for state in gpi.gpi_history]
        eval_errors = [state.evaluation_error for state in gpi.gpi_history]
        policy_changes = [state.improvement_delta for state in gpi.gpi_history]
        
        # 双Y轴
        ax1_twin = ax1.twinx()
        
        # 评估误差（对数尺度）
        line1 = ax1.semilogy(iterations, eval_errors, 'b-', 
                            label='Evaluation Error', linewidth=2)
        ax1.set_xlabel('Iteration / 迭代')
        ax1.set_ylabel('Evaluation Error (log) / 评估误差（对数）', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # 策略变化
        line2 = ax1_twin.plot(iterations, policy_changes, 'r--', 
                             label='Policy Changes', linewidth=2, alpha=0.7)
        ax1_twin.set_ylabel('Policy Changes / 策略变化', color='r')
        ax1_twin.tick_params(axis='y', labelcolor='r')
        
        ax1.set_title('GPI Convergence / GPI收敛')
        ax1.grid(True, alpha=0.3)
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
        
        # ========== 图2：评估步数分布 ==========
        ax2 = axes[0, 1]
        
        eval_steps = [state.evaluation_steps for state in gpi.gpi_history]
        
        ax2.bar(iterations, eval_steps, color='lightblue', alpha=0.7)
        ax2.set_xlabel('Iteration / 迭代')
        ax2.set_ylabel('Evaluation Steps / 评估步数')
        ax2.set_title('Evaluation Effort / 评估工作量')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加平均线
        avg_steps = np.mean(eval_steps)
        ax2.axhline(y=avg_steps, color='red', linestyle='--', 
                   label=f'Average: {avg_steps:.1f}')
        ax2.legend()
        
        # ========== 图3：计算时间分析 ==========
        ax3 = axes[0, 2]
        
        comp_times = [state.computation_time for state in gpi.gpi_history]
        cumulative_times = np.cumsum(comp_times)
        
        ax3.plot(iterations, cumulative_times, 'g-', linewidth=2)
        ax3.fill_between(iterations, 0, cumulative_times, alpha=0.3, color='green')
        ax3.set_xlabel('Iteration / 迭代')
        ax3.set_ylabel('Cumulative Time (s) / 累积时间（秒）')
        ax3.set_title('Computational Cost / 计算成本')
        ax3.grid(True, alpha=0.3)
        
        # ========== 图4：GPI概念图 ==========
        ax4 = axes[1, 0]
        ax4.axis('off')
        
        GPIVisualizer._draw_gpi_diagram(ax4)
        
        # ========== 图5：算法比较 ==========
        ax5 = axes[1, 1]
        ax5.axis('off')
        
        # 创建比较表格
        comparison_data = {
            'Algorithm': ['Policy Iteration', 'Value Iteration', 'Modified PI', 'Async GPI'],
            'Eval Steps': ['Full (~100)', '1', 'k (2-10)', '1'],
            'Convergence': ['Fast (<10)', 'Slow (>50)', 'Medium', 'Variable'],
            'Memory': ['High', 'Low', 'Medium', 'Low'],
            'Stability': ['Stable', 'Stable', 'Stable', 'Less Stable']
        }
        
        # 转置数据用于表格
        table_data = []
        headers = list(comparison_data.keys())
        for i in range(len(comparison_data['Algorithm'])):
            row = [comparison_data[key][i] for key in headers]
            table_data.append(row)
        
        table = ax5.table(cellText=table_data,
                         colLabels=headers,
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.25, 0.2, 0.2, 0.15, 0.2])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # 设置表格样式
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # 标题行
                cell.set_facecolor('#40466e')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else '#ffffff')
        
        ax5.set_title('GPI Variants Comparison / GPI变体比较', pad=20)
        
        # ========== 图6：收敛轨迹 ==========
        ax6 = axes[1, 2]
        
        # 在策略-价值空间中绘制轨迹
        # 这里用一个简化的2D投影表示
        
        # 计算策略"质量"（用改变数的反向作为代理）
        max_states = len(gpi.env.state_space)
        policy_quality = [1 - (state.improvement_delta / max_states) 
                         for state in gpi.gpi_history]
        
        # 计算价值函数"准确度"（用误差的反向作为代理）
        max_error = max(eval_errors) if eval_errors else 1.0
        value_accuracy = [1 - (err / max_error) for err in eval_errors]
        
        # 绘制轨迹
        ax6.plot(value_accuracy, policy_quality, 'o-', markersize=4, alpha=0.7)
        
        # 标记起点和终点
        ax6.plot(value_accuracy[0], policy_quality[0], 'go', markersize=10, 
                label='Start')
        ax6.plot(value_accuracy[-1], policy_quality[-1], 'r*', markersize=15, 
                label='End (Optimal)')
        
        ax6.set_xlabel('Value Accuracy / 价值准确度')
        ax6.set_ylabel('Policy Quality / 策略质量')
        ax6.set_title('GPI Trajectory / GPI轨迹')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 添加理想路径（对角线）
        ax6.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Ideal Path')
        
        plt.suptitle('Generalized Policy Iteration Analysis / 广义策略迭代分析',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def _draw_gpi_diagram(ax):
        """
        绘制GPI概念图
        Draw GPI Conceptual Diagram
        
        展示评估和改进的相互作用
        Show interaction between evaluation and improvement
        """
        # 设置坐标范围
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        # 策略空间
        policy_box = FancyBboxPatch((1, 6), 3, 2,
                                   boxstyle="round,pad=0.1",
                                   facecolor='lightblue',
                                   edgecolor='blue',
                                   linewidth=2)
        ax.add_patch(policy_box)
        ax.text(2.5, 7, 'Policy π', ha='center', va='center', 
               fontweight='bold', fontsize=12)
        
        # 价值空间
        value_box = FancyBboxPatch((6, 6), 3, 2,
                                  boxstyle="round,pad=0.1",
                                  facecolor='lightgreen',
                                  edgecolor='green',
                                  linewidth=2)
        ax.add_patch(value_box)
        ax.text(7.5, 7, 'Value V', ha='center', va='center',
               fontweight='bold', fontsize=12)
        
        # 评估箭头
        eval_arrow = FancyArrowPatch((4, 7.5), (6, 7.5),
                                   connectionstyle="arc3,rad=0",
                                   arrowstyle="->",
                                   mutation_scale=20,
                                   linewidth=2,
                                   color='blue')
        ax.add_patch(eval_arrow)
        ax.text(5, 8, 'Evaluation', ha='center', fontsize=10)
        ax.text(5, 7.8, 'V → v_π', ha='center', fontsize=9, style='italic')
        
        # 改进箭头
        improve_arrow = FancyArrowPatch((6, 6.5), (4, 6.5),
                                      connectionstyle="arc3,rad=0",
                                      arrowstyle="->",
                                      mutation_scale=20,
                                      linewidth=2,
                                      color='green')
        ax.add_patch(improve_arrow)
        ax.text(5, 6, 'Improvement', ha='center', fontsize=10)
        ax.text(5, 5.8, 'π → greedy(V)', ha='center', fontsize=9, style='italic')
        
        # 最优点
        optimal_point = plt.Circle((5, 3), 0.5, color='red', alpha=0.7)
        ax.add_patch(optimal_point)
        ax.text(5, 3, 'π* = v*', ha='center', va='center',
               color='white', fontweight='bold')
        
        # 添加螺旋轨迹表示收敛
        theta_spiral = np.linspace(0, 4*np.pi, 100)
        r_spiral = np.linspace(2, 0.5, 100)
        x_spiral = 5 + r_spiral * np.cos(theta_spiral)
        y_spiral = 3 + r_spiral * np.sin(theta_spiral) * 0.5
        ax.plot(x_spiral, y_spiral, 'k--', alpha=0.3, linewidth=1)
        
        # 添加说明
        ax.text(5, 9.5, 'GPI: Competing & Cooperating', 
               ha='center', fontsize=11, fontweight='bold')
        ax.text(5, 9, '竞争与合作', ha='center', fontsize=10)
        
        ax.text(5, 1, 'Eventually converge to optimal\n最终收敛到最优',
               ha='center', fontsize=9, style='italic')
    
    @staticmethod
    def compare_gpi_variants(env: MDPEnvironment, n_runs: int = 3):
        """
        比较不同GPI变体
        Compare Different GPI Variants
        
        实验展示不同平衡点的效果
        Experiment shows effects of different balance points
        """
        print("\n" + "="*80)
        print("GPI变体比较实验")
        print("GPI Variants Comparison Experiment")
        print("="*80)
        
        patterns = [
            (GPIPattern.POLICY_ITERATION, "full", "Policy Iteration"),
            (GPIPattern.VALUE_ITERATION, 1, "Value Iteration"),
            (GPIPattern.MODIFIED_PI_2, 2, "Modified PI (m=2)"),
            (GPIPattern.MODIFIED_PI_K, 5, "Modified PI (m=5)"),
        ]
        
        results = {name: [] for _, _, name in patterns}
        
        for run in range(n_runs):
            print(f"\n运行 {run + 1}/{n_runs}")
            
            for pattern, eval_steps, name in patterns:
                print(f"  测试 {name}...")
                
                gpi = GeneralizedPolicyIteration(env, gamma=0.9)
                
                # 配置评估步数
                if pattern == GPIPattern.POLICY_ITERATION:
                    eval_steps_config = "full"
                else:
                    eval_steps_config = eval_steps
                
                # 运行GPI
                policy, V = gpi.solve(
                    pattern=pattern,
                    evaluation_steps=eval_steps_config,
                    max_iterations=200,
                    verbose=False
                )
                
                # 记录结果
                results[name].append({
                    'iterations': gpi.total_iterations,
                    'eval_steps': gpi.total_eval_steps,
                    'time': gpi.total_time
                })
                
                print(f"    完成: {gpi.total_iterations}次迭代, "
                     f"{gpi.total_eval_steps}次评估步")
        
        # 可视化比较结果
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        names = list(results.keys())
        colors = ['steelblue', 'lightcoral', 'lightgreen', 'gold']
        
        # 图1：迭代次数
        ax1 = axes[0]
        avg_iterations = [np.mean([r['iterations'] for r in results[name]]) 
                         for name in names]
        bars1 = ax1.bar(range(len(names)), avg_iterations, color=colors, alpha=0.7)
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.set_ylabel('Iterations to Converge')
        ax1.set_title('Convergence Speed')
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars1, avg_iterations):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.0f}', ha='center', va='bottom')
        
        # 图2：总评估步数
        ax2 = axes[1]
        avg_eval_steps = [np.mean([r['eval_steps'] for r in results[name]])
                         for name in names]
        bars2 = ax2.bar(range(len(names)), avg_eval_steps, color=colors, alpha=0.7)
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.set_ylabel('Total Evaluation Steps')
        ax2.set_title('Evaluation Effort')
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars2, avg_eval_steps):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.0f}', ha='center', va='bottom')
        
        # 图3：运行时间
        ax3 = axes[2]
        avg_times = [np.mean([r['time'] for r in results[name]])
                    for name in names]
        bars3 = ax3.bar(range(len(names)), avg_times, color=colors, alpha=0.7)
        ax3.set_xticks(range(len(names)))
        ax3.set_xticklabels(names, rotation=45, ha='right')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_title('Runtime')
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars3, avg_times):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom')
        
        plt.suptitle('GPI Variants Performance Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # 打印总结
        print("\n" + "="*60)
        print("实验总结")
        print("="*60)
        
        print("\n性能排名（按运行时间）：")
        time_ranking = sorted([(name, np.mean([r['time'] for r in results[name]])) 
                             for name in names], key=lambda x: x[1])
        for i, (name, time) in enumerate(time_ranking, 1):
            print(f"{i}. {name}: {time:.3f}秒")
        
        print("\n关键发现：")
        print("1. 策略迭代：迭代少但每步计算多")
        print("2. 价值迭代：迭代多但每步计算少")
        print("3. 修改的策略迭代：在两者间取得平衡")
        print("4. 最优的m值取决于具体问题")
        
        return fig


# ================================================================================
# 第3.4.4节：GPI理论分析
# Section 3.4.4: GPI Theoretical Analysis
# ================================================================================

class GPITheory:
    """
    GPI理论分析
    GPI Theoretical Analysis
    
    深入理解GPI的数学性质
    Deep understanding of GPI mathematical properties
    """
    
    @staticmethod
    def explain_gpi_theory():
        """
        解释GPI理论
        Explain GPI Theory
        """
        print("\n" + "="*80)
        print("广义策略迭代理论")
        print("Generalized Policy Iteration Theory")
        print("="*80)
        
        print("""
        📚 1. GPI的数学基础
        Mathematical Foundation of GPI
        ================================
        
        GPI维护两个近似：
        GPI maintains two approximations:
        - 价值函数V ≈ v_π
          Value function V ≈ v_π
        - 策略π ≈ greedy(V)
          Policy π ≈ greedy(V)
        
        稳定条件（GPI不动点）：
        Stability condition (GPI fixed point):
        V = v_π 且 π = greedy(V)
        
        这恰好是贝尔曼最优方程的条件！
        This is exactly the Bellman optimality condition!
        
        📚 2. 收敛性证明要点
        Convergence Proof Outline
        ================================
        
        定理：GPI收敛到最优策略π*和最优价值v*
        Theorem: GPI converges to optimal policy π* and optimal value v*
        
        证明思路：
        Proof idea:
        1. 每次改进单调不减：v_{π_{k+1}} ≥ v_{π_k}
           Each improvement is monotonic
        2. 价值函数有上界：v_π ≤ v* for all π
           Value functions are bounded
        3. 单调有界序列必收敛
           Monotonic bounded sequence converges
        4. 收敛点满足贝尔曼最优条件
           Convergence point satisfies Bellman optimality
        
        📚 3. GPI的普遍性
        Universality of GPI
        ================================
        
        几乎所有RL方法都是GPI的实例：
        Almost all RL methods are instances of GPI:
        
        | 方法 Method | 评估 Evaluation | 改进 Improvement |
        |------------|----------------|-----------------|
        | DP | 贝尔曼期望算子 | 贪婪 |
        | MC | 采样平均 | 贪婪 |
        | TD | 自举更新 | ε-贪婪 |
        | Q-Learning | TD(0) | 贪婪 |
        | Actor-Critic | Critic | Actor |
        
        📚 4. 评估-改进的权衡
        Evaluation-Improvement Tradeoff
        ================================
        
        关键洞察：不需要完美！
        Key insight: Don't need perfection!
        
        - 完全评估（策略迭代）：
          Full evaluation (Policy Iteration):
          ✓ 稳定，迭代少
          ✓ Stable, few iterations
          ✗ 每步计算量大
          ✗ High computation per step
        
        - 最小评估（价值迭代）：
          Minimal evaluation (Value Iteration):
          ✓ 每步计算量小
          ✓ Low computation per step
          ✗ 需要更多迭代
          ✗ Need more iterations
        
        - 平衡点（修改的策略迭代）：
          Balance (Modified Policy Iteration):
          ✓ 可调节的权衡
          ✓ Adjustable tradeoff
          ✓ 通常最实用
          ✓ Often most practical
        
        📚 5. GPI与强化学习的统一视角
        GPI as Unifying View of RL
        ================================
        
        GPI提供了理解所有RL算法的框架：
        GPI provides framework to understand all RL:
        
        维度1：评估方法
        Dimension 1: Evaluation method
        - 基于模型（DP）
          Model-based (DP)
        - 无模型（MC, TD）
          Model-free (MC, TD)
        - 函数近似
          Function approximation
        
        维度2：改进方法
        Dimension 2: Improvement method
        - 贪婪
          Greedy
        - ε-贪婪
          ε-greedy
        - 软策略
          Soft policy
        - 策略梯度
          Policy gradient
        
        维度3：交替模式
        Dimension 3: Alternation pattern
        - 完全交替
          Full alternation
        - 部分交替
          Partial alternation
        - 异步更新
          Asynchronous update
        
        这个统一视角是理解整个RL领域的关键！
        This unified view is key to understanding all of RL!
        """)


# ================================================================================
# 主函数：演示GPI
# Main Function: Demonstrate GPI
# ================================================================================

def main():
    """
    运行广义策略迭代演示
    Run Generalized Policy Iteration Demo
    """
    print("\n" + "="*80)
    print("第3.4节：广义策略迭代")
    print("Section 3.4: Generalized Policy Iteration")
    print("="*80)
    
    # 创建测试环境
    from src.ch03_finite_mdp.gridworld import GridWorld
    
    # 创建4x4网格世界
    env = GridWorld(
        rows=4,
        cols=4,
        start_pos=(0, 0),
        goal_pos=(3, 3),
        obstacles={(1, 1), (2, 2)}
    )
    
    print(f"\n创建 {env.rows}×{env.cols} 网格世界")
    print(f"Create {env.rows}×{env.cols} Grid World")
    
    # 1. 演示标准GPI（策略迭代模式）
    print("\n" + "="*60)
    print("1. 标准GPI（策略迭代模式）")
    print("1. Standard GPI (Policy Iteration Mode)")
    print("="*60)
    
    gpi = GeneralizedPolicyIteration(env, gamma=0.9)
    policy_pi, V_pi = gpi.solve(
        pattern=GPIPattern.POLICY_ITERATION,
        verbose=True
    )
    
    # 2. 演示GPI（价值迭代模式）
    print("\n" + "="*60)
    print("2. GPI（价值迭代模式）")
    print("2. GPI (Value Iteration Mode)")
    print("="*60)
    
    gpi_vi = GeneralizedPolicyIteration(env, gamma=0.9)
    policy_vi, V_vi = gpi_vi.solve(
        pattern=GPIPattern.VALUE_ITERATION,
        verbose=True
    )
    
    # 3. 演示修改的策略迭代
    print("\n" + "="*60)
    print("3. 修改的策略迭代 (m=3)")
    print("3. Modified Policy Iteration (m=3)")
    print("="*60)
    
    gpi_mod = GeneralizedPolicyIteration(env, gamma=0.9)
    policy_mod, V_mod = gpi_mod.solve(
        pattern=GPIPattern.MODIFIED_PI_K,
        evaluation_steps=3,
        verbose=True
    )
    
    # 4. 可视化GPI过程
    print("\n4. 可视化GPI过程")
    print("4. Visualize GPI Process")
    visualizer = GPIVisualizer()
    fig1 = visualizer.visualize_gpi_process(gpi)
    
    # 5. 比较不同GPI变体
    print("\n5. 比较GPI变体")
    print("5. Compare GPI Variants")
    fig2 = GPIVisualizer.compare_gpi_variants(env, n_runs=3)
    
    # 6. 理论分析
    GPITheory.explain_gpi_theory()
    
    # 7. 验证所有方法收敛到相同的最优策略
    print("\n" + "="*60)
    print("验证收敛结果")
    print("Verify Convergence Results")
    print("="*60)
    
    # 比较价值函数
    print("\n价值函数比较（部分状态）：")
    print("Value Function Comparison (sample states):")
    
    sample_states = env.state_space[:5]
    for state in sample_states:
        v_pi = V_pi.get_value(state)
        v_vi = V_vi.get_value(state)
        v_mod = V_mod.get_value(state)
        
        print(f"  State {state.id}:")
        print(f"    Policy Iter: {v_pi:.3f}")
        print(f"    Value Iter:  {v_vi:.3f}")
        print(f"    Modified PI: {v_mod:.3f}")
        
        # 检查是否收敛到相同值
        if abs(v_pi - v_vi) < 0.01 and abs(v_pi - v_mod) < 0.01:
            print(f"    ✓ 收敛一致")
        else:
            print(f"    ⚠ 值不一致")
    
    print("\n" + "="*80)
    print("广义策略迭代演示完成！")
    print("Generalized Policy Iteration Demo Complete!")
    print("\n关键要点 Key Takeaways:")
    print("1. GPI是几乎所有RL方法的核心模式")
    print("   GPI is the core pattern of almost all RL methods")
    print("2. 评估和改进相互竞争又相互合作")
    print("   Evaluation and improvement compete and cooperate")
    print("3. 不需要完美的评估或改进就能收敛")
    print("   Don't need perfect evaluation or improvement to converge")
    print("4. 不同的GPI变体在计算效率上有不同权衡")
    print("   Different GPI variants have different computational tradeoffs")
    print("5. 理解GPI是理解整个RL的关键")
    print("   Understanding GPI is key to understanding all of RL")
    print("="*80)
    
    plt.show()
    
    return policy_pi, V_pi


if __name__ == "__main__":
    main()