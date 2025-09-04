"""
================================================================================
第3.3节：价值迭代 - 直接寻找最优价值函数
Section 3.3: Value Iteration - Finding Optimal Value Function Directly
================================================================================

价值迭代是另一种动态规划算法，它跳过了显式的策略表示，直接寻找最优价值函数。
Value Iteration is another DP algorithm that skips explicit policy representation
and finds optimal value function directly.

核心思想：反复应用贝尔曼最优算子
Core idea: Repeatedly apply Bellman optimality operator
v_{k+1}(s) = max_a Σ_{s',r} p(s',r|s,a)[r + γv_k(s')]

这可以看作是"截断的策略迭代"，每次只做一步策略评估就立即改进。
This can be viewed as "truncated policy iteration" with only one step of evaluation before improvement.

为什么叫"价值迭代"？
Why called "Value Iteration"?
因为我们直接迭代价值函数，策略是隐含的（从价值函数贪婪导出）。
Because we directly iterate value function, policy is implicit (derived greedily from values).

优势 vs 策略迭代：
Advantages vs Policy Iteration:
- 每次迭代计算量更小（不需要完整策略评估）
  Less computation per iteration (no full policy evaluation)
- 实现更简单（不需要存储策略）
  Simpler implementation (no need to store policy)
- 可以随时停止得到近似解
  Can stop anytime to get approximation

劣势：
Disadvantages:
- 需要更多迭代次数才能收敛
  Needs more iterations to converge
- 中间过程没有明确的策略
  No explicit policy during process
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import logging
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, FancyBboxPatch
import seaborn as sns
from collections import defaultdict
import time
from IPython.display import HTML

# 导入基础组件
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.ch03_finite_mdp.mdp_framework import (
    State, Action, MDPEnvironment
)
from src.ch03_finite_mdp.policies_and_values import (
    Policy, StateValueFunction, ActionValueFunction,
    DeterministicPolicy
)
from .dp_foundations import BellmanOperator

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第3.3.1节：价值迭代算法
# Section 3.3.1: Value Iteration Algorithm
# ================================================================================

class ValueIteration:
    """
    价值迭代算法
    Value Iteration Algorithm
    
    算法伪代码：
    Algorithm Pseudocode:
    ```
    初始化 Initialize:
    对所有s∈S，V(s) = 0（除了终止状态）
    For all s∈S, V(s) = 0 (except terminal states)
    
    重复 Repeat:
        Δ ← 0
        对每个s∈S：
        For each s∈S:
            v ← V(s)
            V(s) ← max_a Σ_{s',r} p(s',r|s,a)[r + γV(s')]
            Δ ← max(Δ, |v - V(s)|)
    直到Δ < θ
    until Δ < θ
    
    输出确定性策略：
    Output deterministic policy:
    π(s) = argmax_a Σ_{s',r} p(s',r|s,a)[r + γV(s')]
    ```
    
    数学原理：
    Mathematical Principle:
    - 贝尔曼最优算子T*是γ-收缩映射
      Bellman optimality operator T* is γ-contraction
    - 有唯一不动点v*（最优价值函数）
      Has unique fixed point v* (optimal value function)
    - 从任意初始值开始都会收敛到v*
      Converges to v* from any initial value
    
    收敛速度：
    Convergence Rate:
    ||v_{k+1} - v*||∞ ≤ γ||v_k - v*||∞
    
    这意味着误差以γ的速度指数衰减！
    This means error decays exponentially at rate γ!
    """
    
    def __init__(self, mdp_env: MDPEnvironment, gamma: float = 0.99):
        """
        初始化价值迭代
        Initialize Value Iteration
        
        Args:
            mdp_env: MDP环境
            gamma: 折扣因子
        
        为什么gamma重要？
        Why is gamma important?
        - γ接近1：考虑长远，收敛慢
          γ close to 1: long-term focus, slow convergence
        - γ接近0：短视，收敛快
          γ close to 0: myopic, fast convergence
        - γ决定了收缩速度和最终价值大小
          γ determines contraction rate and value magnitude
        """
        self.env = mdp_env
        self.gamma = gamma
        self.bellman_op = BellmanOperator(mdp_env, gamma)
        
        # 记录迭代历史
        self.iteration_history = []
        self.convergence_history = []
        
        # 性能统计
        self.total_iterations = 0
        self.total_time = 0.0
        
        logger.info(f"初始化价值迭代，γ={gamma}")
        logger.info(f"状态空间大小: {len(mdp_env.state_space)}")
        logger.info(f"动作空间大小: {len(mdp_env.action_space)}")
    
    def solve(self,
             theta: float = 1e-6,
             max_iterations: int = 1000,
             initial_v: Optional[StateValueFunction] = None,
             verbose: bool = True) -> Tuple[Policy, StateValueFunction]:
        """
        运行价值迭代算法
        Run Value Iteration Algorithm
        
        与策略迭代的关键区别：
        Key difference from Policy Iteration:
        - 不维护显式策略，只维护价值函数
          No explicit policy, only value function
        - 每次迭代都是一步贝尔曼最优更新
          Each iteration is one Bellman optimality update
        - 最后从价值函数提取策略
          Extract policy from values at the end
        
        Args:
            theta: 收敛阈值
                  Convergence threshold
            max_iterations: 最大迭代次数
                          Maximum iterations
            initial_v: 初始价值函数
                      Initial value function
            verbose: 是否打印详细信息
                    Whether to print details
        
        Returns:
            (最优策略, 最优价值函数)
            (optimal policy, optimal value function)
        
        实现细节：
        Implementation Details:
        1. 同步更新：需要两个数组存储新旧值
           Synchronous update: need two arrays for old and new values
        2. 原地更新也可以（Gauss-Seidel风格），可能更快收敛
           In-place update also works (Gauss-Seidel style), may converge faster
        3. 终止状态的价值始终为0
           Terminal states always have value 0
        """
        # 清空历史
        self.iteration_history = []
        self.convergence_history = []
        
        # 开始计时
        start_time = time.time()
        
        # 初始化价值函数
        if initial_v is None:
            V = StateValueFunction(self.env.state_space, initial_value=0.0)
        else:
            V = initial_v
        
        if verbose:
            print("\n" + "="*60)
            print("开始价值迭代")
            print("Starting Value Iteration")
            print("="*60)
            print(f"收敛阈值 θ = {theta}")
            print(f"折扣因子 γ = {self.gamma}")
            print(f"状态数量 |S| = {len(self.env.state_space)}")
            print(f"动作数量 |A| = {len(self.env.action_space)}")
        
        # 主循环
        for iteration in range(max_iterations):
            # 记录当前价值函数（深拷贝）
            V_old = StateValueFunction(self.env.state_space)
            for state in self.env.state_space:
                V_old.set_value(state, V.get_value(state))
            
            # 应用贝尔曼最优算子
            # Apply Bellman optimality operator
            V_new = self.bellman_op.bellman_optimality_operator(V)
            
            # 计算最大变化（收敛判断）
            # Calculate maximum change (convergence check)
            delta = 0.0
            state_changes = {}  # 记录每个状态的变化
            
            for state in self.env.state_space:
                old_value = V.get_value(state)
                new_value = V_new.get_value(state)
                change = abs(old_value - new_value)
                delta = max(delta, change)
                state_changes[state] = change
            
            # 更新价值函数
            V = V_new
            
            # 记录历史
            self.iteration_history.append({
                'iteration': iteration + 1,
                'value_function': V_old,  # 记录更新前的值
                'delta': delta,
                'max_change_state': max(state_changes, key=state_changes.get) if state_changes else None
            })
            self.convergence_history.append(delta)
            
            # 打印进度
            if verbose and (iteration % 10 == 0 or delta < theta):
                print(f"迭代 {iteration + 1}: Δ = {delta:.2e}")
                
                # 显示变化最大的状态
                if state_changes:
                    max_change_state = max(state_changes, key=state_changes.get)
                    print(f"  变化最大的状态: {max_change_state.id} "
                          f"(Δ = {state_changes[max_change_state]:.2e})")
                
                # 显示几个状态的价值
                if iteration % 50 == 0:
                    sample_states = self.env.state_space[:min(3, len(self.env.state_space))]
                    print("  示例状态价值:")
                    for s in sample_states:
                        print(f"    V({s.id}) = {V.get_value(s):.3f}")
            
            # 检查收敛
            if delta < theta:
                self.total_iterations = iteration + 1
                if verbose:
                    print(f"\n✓ 价值迭代收敛！")
                    print(f"  迭代次数: {self.total_iterations}")
                    print(f"  最终 Δ: {delta:.2e}")
                
                logger.info(f"价值迭代在第{self.total_iterations}次迭代收敛")
                break
        else:
            # 达到最大迭代
            self.total_iterations = max_iterations
            logger.warning(f"达到最大迭代次数 {max_iterations}，Δ = {delta:.2e}")
            if verbose:
                print(f"\n⚠ 达到最大迭代次数，可能未完全收敛")
                print(f"  当前 Δ = {delta:.2e} > θ = {theta}")
        
        # 记录总时间
        self.total_time = time.time() - start_time
        
        # 从最优价值函数提取最优策略
        # Extract optimal policy from optimal value function
        if verbose:
            print("\n提取最优策略...")
            print("Extracting optimal policy...")
        
        optimal_policy = self._extract_policy(V)
        
        if verbose:
            print(f"\n总运行时间: {self.total_time:.3f}秒")
            print(f"平均每次迭代: {self.total_time/self.total_iterations:.4f}秒")
            
            # 理论分析
            self._print_theoretical_analysis()
        
        return optimal_policy, V
    
    def _extract_policy(self, V: StateValueFunction) -> Policy:
        """
        从价值函数提取贪婪策略
        Extract greedy policy from value function
        
        π*(s) = argmax_a Σ_{s',r} p(s',r|s,a)[r + γV(s')]
        
        这是价值迭代的关键步骤！
        This is the key step of value iteration!
        
        注意：只有在V接近v*时，提取的策略才接近π*
        Note: Only when V is close to v*, extracted policy is close to π*
        """
        policy_map = {}
        P = self.bellman_op.P
        
        for state in self.env.state_space:
            if state.is_terminal:
                continue
            
            # 计算每个动作的Q值
            action_values = {}
            for action in self.env.action_space:
                q_value = self.bellman_op._compute_q_value(state, action, V)
                action_values[action] = q_value
            
            # 选择最佳动作
            if action_values:
                best_action = max(action_values, key=action_values.get)
                policy_map[state] = best_action
                
                logger.debug(f"State {state.id}: "
                           f"Q-values = {{{', '.join(f'{a.id}:{q:.2f}' for a, q in action_values.items())}}}, "
                           f"Best = {best_action.id}")
        
        return DeterministicPolicy(policy_map)
    
    def _print_theoretical_analysis(self):
        """
        打印理论分析
        Print Theoretical Analysis
        
        帮助理解算法的收敛性质
        Helps understand convergence properties
        """
        print("\n" + "-"*40)
        print("理论分析 Theoretical Analysis")
        print("-"*40)
        
        # 估计收敛速度
        if len(self.convergence_history) > 10:
            # 计算实际收缩率
            recent_deltas = self.convergence_history[-10:]
            ratios = [recent_deltas[i+1]/recent_deltas[i] 
                     for i in range(len(recent_deltas)-1) 
                     if recent_deltas[i] > 0]
            if ratios:
                avg_ratio = np.mean(ratios)
                print(f"实际收缩率: {avg_ratio:.3f} (理论上界: {self.gamma})")
                print(f"Actual contraction: {avg_ratio:.3f} (theoretical bound: {self.gamma})")
        
        # 估计到最优的距离
        final_delta = self.convergence_history[-1] if self.convergence_history else 0
        if final_delta > 0:
            # 使用误差界：||v_k - v*|| ≤ γ^k/(1-γ) * ||v_1 - v_0||
            estimated_error = final_delta / (1 - self.gamma)
            print(f"估计误差上界: {estimated_error:.2e}")
            print(f"Estimated error bound: {estimated_error:.2e}")
        
        # 计算效率
        total_updates = self.total_iterations * len(self.env.state_space)
        print(f"总状态更新次数: {total_updates}")
        print(f"Total state updates: {total_updates}")
    
    def get_value_evolution(self, state_indices: List[int] = None) -> np.ndarray:
        """
        获取价值函数演化轨迹
        Get value function evolution trajectory
        
        用于可视化分析
        For visualization and analysis
        
        Args:
            state_indices: 要跟踪的状态索引
        
        Returns:
            形状为 (n_iterations, n_states) 的数组
            Array of shape (n_iterations, n_states)
        """
        if not self.iteration_history:
            return np.array([])
        
        states = self.env.state_space
        if state_indices is not None:
            states = [states[i] for i in state_indices if i < len(states)]
        
        n_iterations = len(self.iteration_history)
        n_states = len(states)
        
        evolution = np.zeros((n_iterations, n_states))
        
        for i, hist in enumerate(self.iteration_history):
            V = hist['value_function']
            for j, state in enumerate(states):
                evolution[i, j] = V.get_value(state)
        
        return evolution


# ================================================================================
# 第3.3.2节：异步价值迭代
# Section 3.3.2: Asynchronous Value Iteration
# ================================================================================

class AsynchronousValueIteration(ValueIteration):
    """
    异步价值迭代
    Asynchronous Value Iteration
    
    与同步版本的区别：
    Difference from synchronous version:
    - 同步：所有状态同时更新（需要两个数组）
      Synchronous: all states updated simultaneously (needs two arrays)
    - 异步：状态按某种顺序逐个更新（只需一个数组）
      Asynchronous: states updated one by one in some order (needs one array)
    
    优势：
    Advantages:
    - 内存效率更高（只需一个数组）
      More memory efficient (one array)
    - 可能收敛更快（新信息立即传播）
      May converge faster (new info propagates immediately)
    - 更灵活（可以优先更新重要状态）
      More flexible (can prioritize important states)
    
    变体：
    Variants:
    1. Gauss-Seidel：固定顺序更新
       Fixed order update
    2. 随机选择：随机选择状态更新
       Random selection
    3. 优先级扫描：优先更新变化大的状态
       Prioritized sweeping: update high-change states first
    """
    
    def __init__(self, mdp_env: MDPEnvironment, gamma: float = 0.99,
                 update_mode: str = 'random'):
        """
        初始化异步价值迭代
        
        Args:
            update_mode: 更新模式
                - 'sequential': 顺序更新
                - 'random': 随机更新
                - 'prioritized': 优先级更新
        """
        super().__init__(mdp_env, gamma)
        self.update_mode = update_mode
        
        # 优先级队列（用于优先级扫描）
        self.priority_queue = []
        
        logger.info(f"初始化异步价值迭代，模式: {update_mode}")
    
    def solve(self,
             theta: float = 1e-6,
             max_iterations: int = 10000,
             updates_per_iteration: int = None,
             verbose: bool = True) -> Tuple[Policy, StateValueFunction]:
        """
        运行异步价值迭代
        Run Asynchronous Value Iteration
        
        注意：迭代的定义不同
        Note: Different definition of iteration
        - 同步：一次迭代 = 更新所有状态
          Synchronous: one iteration = update all states
        - 异步：一次迭代 = 更新一个（或几个）状态
          Asynchronous: one iteration = update one (or few) states
        
        Args:
            updates_per_iteration: 每次迭代更新的状态数
                                  Number of states to update per iteration
        """
        # 每次迭代更新的状态数
        if updates_per_iteration is None:
            updates_per_iteration = len(self.env.state_space)
        
        # 初始化
        V = StateValueFunction(self.env.state_space, initial_value=0.0)
        start_time = time.time()
        
        if verbose:
            print("\n" + "="*60)
            print(f"开始异步价值迭代 (模式: {self.update_mode})")
            print(f"Starting Asynchronous Value Iteration (mode: {self.update_mode})")
            print("="*60)
        
        # 初始化优先级（如果使用优先级扫描）
        if self.update_mode == 'prioritized':
            self._initialize_priorities(V)
        
        # 记录更新次数
        total_updates = 0
        max_delta_history = []
        
        # 主循环
        for iteration in range(max_iterations):
            iteration_delta = 0.0
            
            # 选择要更新的状态
            states_to_update = self._select_states_to_update(
                V, updates_per_iteration
            )
            
            # 更新选中的状态
            for state in states_to_update:
                if state.is_terminal:
                    continue
                
                # 计算新值（贝尔曼最优更新）
                old_value = V.get_value(state)
                
                # 计算max_a Q(s,a)
                max_q_value = float('-inf')
                for action in self.env.action_space:
                    q_value = self.bellman_op._compute_q_value(state, action, V)
                    max_q_value = max(max_q_value, q_value)
                
                # 原地更新
                V.set_value(state, max_q_value)
                
                # 记录变化
                delta = abs(old_value - max_q_value)
                iteration_delta = max(iteration_delta, delta)
                
                # 更新优先级（如果使用）
                if self.update_mode == 'prioritized':
                    self._update_priority(state, delta)
                
                total_updates += 1
            
            max_delta_history.append(iteration_delta)
            
            # 定期检查收敛
            if iteration % 100 == 0:
                # 计算所有状态的最大变化
                global_delta = self._compute_global_delta(V)
                
                if verbose and iteration % 1000 == 0:
                    print(f"迭代 {iteration}: "
                          f"局部Δ = {iteration_delta:.2e}, "
                          f"全局Δ = {global_delta:.2e}")
                
                # 检查收敛
                if global_delta < theta:
                    if verbose:
                        print(f"\n✓ 异步价值迭代收敛！")
                        print(f"  总更新次数: {total_updates}")
                        print(f"  迭代次数: {iteration + 1}")
                    break
        
        self.total_time = time.time() - start_time
        self.total_iterations = iteration + 1
        
        # 提取策略
        optimal_policy = self._extract_policy(V)
        
        if verbose:
            print(f"\n总运行时间: {self.total_time:.3f}秒")
            print(f"总状态更新: {total_updates}")
            print(f"平均每秒更新: {total_updates/self.total_time:.0f}")
        
        return optimal_policy, V
    
    def _select_states_to_update(self, V: StateValueFunction, 
                                 n: int) -> List[State]:
        """
        选择要更新的状态
        Select states to update
        
        根据update_mode选择不同策略
        Different strategies based on update_mode
        """
        non_terminal_states = [s for s in self.env.state_space 
                              if not s.is_terminal]
        
        if self.update_mode == 'sequential':
            # 循环顺序选择
            if not hasattr(self, '_sequential_index'):
                self._sequential_index = 0
            
            states = []
            for _ in range(min(n, len(non_terminal_states))):
                states.append(non_terminal_states[self._sequential_index])
                self._sequential_index = (self._sequential_index + 1) % len(non_terminal_states)
            return states
            
        elif self.update_mode == 'random':
            # 随机选择
            n = min(n, len(non_terminal_states))
            return np.random.choice(non_terminal_states, n, replace=False).tolist()
            
        elif self.update_mode == 'prioritized':
            # 优先级选择
            return self._select_by_priority(n)
        
        else:
            raise ValueError(f"未知的更新模式: {self.update_mode}")
    
    def _initialize_priorities(self, V: StateValueFunction):
        """
        初始化优先级队列
        Initialize priority queue
        """
        self.priorities = {}
        for state in self.env.state_space:
            if not state.is_terminal:
                # 初始优先级设为无穷大（确保所有状态至少更新一次）
                self.priorities[state] = float('inf')
    
    def _update_priority(self, state: State, delta: float):
        """
        更新状态优先级
        Update state priority
        """
        if self.update_mode == 'prioritized':
            self.priorities[state] = delta
    
    def _select_by_priority(self, n: int) -> List[State]:
        """
        根据优先级选择状态
        Select states by priority
        """
        if not self.priorities:
            return []
        
        # 选择优先级最高的n个状态
        sorted_states = sorted(self.priorities.items(), 
                             key=lambda x: x[1], 
                             reverse=True)
        return [state for state, _ in sorted_states[:n]]
    
    def _compute_global_delta(self, V: StateValueFunction) -> float:
        """
        计算全局最大变化
        Compute global maximum change
        
        用于判断真正的收敛
        For determining true convergence
        """
        max_delta = 0.0
        
        for state in self.env.state_space:
            if state.is_terminal:
                continue
            
            old_value = V.get_value(state)
            
            # 计算应该的新值
            max_q_value = float('-inf')
            for action in self.env.action_space:
                q_value = self.bellman_op._compute_q_value(state, action, V)
                max_q_value = max(max_q_value, q_value)
            
            delta = abs(old_value - max_q_value)
            max_delta = max(max_delta, delta)
        
        return max_delta


# ================================================================================
# 第3.3.3节：价值迭代可视化
# Section 3.3.3: Value Iteration Visualization
# ================================================================================

class ValueIterationVisualizer:
    """
    价值迭代可视化器
    Value Iteration Visualizer
    
    展示价值迭代的收敛过程和特性
    Show convergence process and properties of value iteration
    """
    
    @staticmethod
    def visualize_convergence(vi: ValueIteration):
        """
        可视化收敛过程
        Visualize Convergence Process
        
        展示价值迭代如何逐步收敛到最优
        Show how value iteration converges to optimal
        """
        if not vi.convergence_history:
            logger.warning("没有收敛历史可视化")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # ========== 图1：收敛曲线（对数尺度） ==========
        ax1 = axes[0, 0]
        iterations = range(1, len(vi.convergence_history) + 1)
        ax1.semilogy(iterations, vi.convergence_history, 'b-', linewidth=2)
        ax1.set_xlabel('Iteration / 迭代')
        ax1.set_ylabel('Max Change Δ (log scale) / 最大变化（对数尺度）')
        ax1.set_title('Convergence Rate / 收敛速度')
        ax1.grid(True, alpha=0.3)
        
        # 添加理论界限
        if len(vi.convergence_history) > 1:
            initial_delta = vi.convergence_history[0]
            theoretical_bound = [initial_delta * (vi.gamma ** i) 
                               for i in range(len(vi.convergence_history))]
            ax1.semilogy(iterations, theoretical_bound, 'r--', 
                        alpha=0.5, label=f'γ^k bound (γ={vi.gamma})')
            ax1.legend()
        
        # 标记收敛点
        ax1.axhline(y=1e-6, color='g', linestyle='--', alpha=0.5, label='θ=1e-6')
        
        # ========== 图2：收缩率分析 ==========
        ax2 = axes[0, 1]
        if len(vi.convergence_history) > 1:
            # 计算相邻迭代的比率
            ratios = []
            for i in range(1, len(vi.convergence_history)):
                if vi.convergence_history[i-1] > 0:
                    ratio = vi.convergence_history[i] / vi.convergence_history[i-1]
                    ratios.append(ratio)
            
            if ratios:
                ax2.plot(range(2, len(vi.convergence_history) + 1), ratios, 
                        'o-', markersize=4, alpha=0.7)
                ax2.axhline(y=vi.gamma, color='r', linestyle='--', 
                           label=f'γ = {vi.gamma}')
                ax2.set_xlabel('Iteration / 迭代')
                ax2.set_ylabel('Contraction Ratio / 收缩比率')
                ax2.set_title('Actual vs Theoretical Contraction / 实际vs理论收缩')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim([0, 1])
        
        # ========== 图3：价值函数演化 ==========
        ax3 = axes[1, 0]
        
        # 选择几个状态展示
        n_states_to_show = min(5, len(vi.env.state_space))
        state_indices = np.linspace(0, len(vi.env.state_space)-1, 
                                   n_states_to_show, dtype=int)
        
        evolution = vi.get_value_evolution(state_indices)
        
        if evolution.size > 0:
            for i, idx in enumerate(state_indices):
                state = vi.env.state_space[idx]
                ax3.plot(evolution[:, i], label=f'State {state.id}', alpha=0.7)
            
            ax3.set_xlabel('Iteration / 迭代')
            ax3.set_ylabel('State Value / 状态价值')
            ax3.set_title('Value Function Evolution / 价值函数演化')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # ========== 图4：算法比较 ==========
        ax4 = axes[1, 1]
        
        # 创建比较表格
        comparison_data = {
            'Property': ['收敛速度\nConvergence', '每步计算\nPer Step', 
                        '内存需求\nMemory', '实现难度\nImplementation'],
            'Value Iteration': ['慢 Slow\nO(γ^k)', '低 Low\nO(|S||A|)', 
                              '低 Low\nO(|S|)', '简单 Simple'],
            'Policy Iteration': ['快 Fast\n<10 iterations', '高 High\nO(|S|²|A|×I)', 
                               '高 High\nO(|S|+|A|)', '复杂 Complex']
        }
        
        # 清空坐标轴
        ax4.axis('tight')
        ax4.axis('off')
        
        # 创建表格
        table = ax4.table(cellText=[[comparison_data[col][i] 
                                    for col in comparison_data.keys()] 
                                   for i in range(len(comparison_data['Property']))],
                         colLabels=list(comparison_data.keys()),
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.3, 0.35, 0.35])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # 设置表格样式
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # 标题行
                cell.set_facecolor('#40466e')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if j % 2 == 0 else '#ffffff')
        
        ax4.set_title('Algorithm Comparison / 算法比较', pad=20)
        
        plt.suptitle('Value Iteration Analysis / 价值迭代分析', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def create_animation(vi: ValueIteration, grid_env=None):
        """
        创建价值迭代动画
        Create Value Iteration Animation
        
        动态展示价值函数如何传播
        Dynamically show how value function propagates
        """
        if not vi.iteration_history or grid_env is None:
            logger.warning("需要迭代历史和网格环境创建动画")
            return None
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        def animate(frame):
            ax.clear()
            
            # 获取当前迭代的价值函数
            if frame < len(vi.iteration_history):
                V = vi.iteration_history[frame]['value_function']
                iteration = vi.iteration_history[frame]['iteration']
                delta = vi.iteration_history[frame]['delta']
            else:
                return
            
            # 绘制网格和价值
            ValueIterationVisualizer._draw_grid_values(ax, grid_env, V)
            
            ax.set_title(f'Value Iteration - Iteration {iteration}\n'
                        f'Δ = {delta:.2e}', fontsize=12)
        
        anim = animation.FuncAnimation(
            fig, animate,
            frames=len(vi.iteration_history),
            interval=200,  # 每帧200ms
            repeat=True
        )
        
        plt.close()  # 防止显示静态图
        return anim
    
    @staticmethod
    def _draw_grid_values(ax, grid_env, V: StateValueFunction):
        """
        在网格上绘制价值函数
        Draw value function on grid
        
        使用热力图展示价值分布
        Use heatmap to show value distribution
        """
        rows, cols = grid_env.rows, grid_env.cols
        
        # 创建价值矩阵
        value_matrix = np.zeros((rows, cols))
        
        for i in range(rows):
            for j in range(cols):
                pos = (i, j)
                if pos in grid_env.pos_to_state:
                    state = grid_env.pos_to_state[pos]
                    value_matrix[i, j] = V.get_value(state)
                elif pos in grid_env.obstacles:
                    value_matrix[i, j] = np.nan  # 障碍物
        
        # 绘制热力图
        im = ax.imshow(value_matrix, cmap='coolwarm', aspect='equal')
        
        # 添加数值标签
        for i in range(rows):
            for j in range(cols):
                pos = (i, j)
                if pos in grid_env.pos_to_state:
                    state = grid_env.pos_to_state[pos]
                    value = V.get_value(state)
                    
                    # 根据值的大小调整文本颜色
                    text_color = 'white' if value < np.nanmean(value_matrix) else 'black'
                    ax.text(j, i, f'{value:.1f}', ha='center', va='center',
                           color=text_color, fontweight='bold')
                
                # 标记特殊位置
                if pos == grid_env.start_pos:
                    ax.add_patch(Rectangle((j-0.45, i-0.45), 0.9, 0.9,
                                         fill=False, edgecolor='green', linewidth=3))
                    ax.text(j, i-0.35, 'S', ha='center', va='center',
                           color='green', fontweight='bold', fontsize=12)
                elif pos == grid_env.goal_pos:
                    ax.add_patch(Rectangle((j-0.45, i-0.45), 0.9, 0.9,
                                         fill=False, edgecolor='red', linewidth=3))
                    ax.text(j, i-0.35, 'G', ha='center', va='center',
                           color='red', fontweight='bold', fontsize=12)
                elif pos in grid_env.obstacles:
                    ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1,
                                         facecolor='gray', alpha=0.8))
        
        # 设置坐标轴
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.grid(True, color='black', linewidth=0.5)
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, label='State Value')


# ================================================================================
# 第3.3.4节：价值迭代分析
# Section 3.3.4: Value Iteration Analysis
# ================================================================================

class ValueIterationAnalysis:
    """
    价值迭代理论与实验分析
    Value Iteration Theoretical and Experimental Analysis
    """
    
    @staticmethod
    def theoretical_analysis():
        """
        理论分析
        Theoretical Analysis
        """
        print("\n" + "="*80)
        print("价值迭代理论分析")
        print("Value Iteration Theoretical Analysis")
        print("="*80)
        
        print("""
        📚 1. 收敛性证明
        Convergence Proof
        ================================
        
        贝尔曼最优算子T*的性质：
        Properties of Bellman optimality operator T*:
        
        (1) 单调性 Monotonicity:
            v ≤ w ⟹ T*v ≤ T*w
        
        (2) 收缩性 Contraction:
            ||T*v - T*w||∞ ≤ γ||v - w||∞
        
        由Banach不动点定理：
        By Banach fixed-point theorem:
        - T*有唯一不动点v*
          T* has unique fixed point v*
        - 从任意v_0开始，v_k → v* as k → ∞
          Starting from any v_0, v_k → v* as k → ∞
        - 收敛速度：||v_k - v*|| ≤ γ^k ||v_0 - v*||
          Convergence rate: ||v_k - v*|| ≤ γ^k ||v_0 - v*||
        
        📚 2. 误差界
        Error Bounds
        ================================
        
        k步后的误差上界：
        Error bound after k steps:
        
        ||v_k - v*||∞ ≤ γ^k/(1-γ) · ||v_1 - v_0||∞
        
        这告诉我们：
        This tells us:
        - γ越小，收敛越快
          Smaller γ, faster convergence
        - 初始值的选择影响有限
          Initial value choice has limited impact
        - 可以预估需要的迭代次数
          Can estimate required iterations
        
        要达到ε精度，需要迭代次数：
        To reach ε accuracy, need iterations:
        k ≥ log(ε(1-γ)/||v_1-v_0||) / log(γ)
        
        📚 3. vs 策略迭代
        vs Policy Iteration
        ================================
        
        价值迭代 = 修改的策略迭代(m=1)
        Value Iteration = Modified Policy Iteration (m=1)
        
        | 方面 Aspect | 价值迭代 VI | 策略迭代 PI |
        |------------|------------|-------------|
        | 迭代次数    | 多 Many     | 少 Few      |
        | 每步计算    | 少 Less     | 多 More     |
        | 内存需求    | 小 Small    | 大 Large    |
        | 中间策略    | 无 None     | 有 Yes      |
        | 适用场景    | γ小 Small γ | γ大 Large γ |
        
        📚 4. 实践技巧
        Practical Tips
        ================================
        
        加速收敛：
        Speed up convergence:
        
        1. 好的初始值：
           Good initial values:
           - 使用启发式（如最短路径）
             Use heuristics (e.g., shortest path)
           - 从相似问题的解开始
             Start from similar problem's solution
        
        2. 异步更新：
           Asynchronous updates:
           - Gauss-Seidel比Jacobi快
             Gauss-Seidel faster than Jacobi
           - 优先级扫描更高效
             Prioritized sweeping more efficient
        
        3. 早停：
           Early stopping:
           - 不需要完全收敛就能得到好策略
             Don't need full convergence for good policy
           - ε-最优策略可能就够了
             ε-optimal policy may be enough
        """)
    
    @staticmethod
    def compare_sync_async(env, n_runs: int = 5):
        """
        比较同步和异步价值迭代
        Compare Synchronous and Asynchronous Value Iteration
        
        实验展示两种方法的性能差异
        Experiment shows performance difference between two methods
        """
        print("\n" + "="*80)
        print("同步 vs 异步价值迭代")
        print("Synchronous vs Asynchronous Value Iteration")
        print("="*80)
        
        results = {
            'Synchronous': [],
            'Async-Random': [],
            'Async-Sequential': []
        }
        
        for run in range(n_runs):
            print(f"\n运行 {run + 1}/{n_runs}")
            
            # 同步版本
            vi_sync = ValueIteration(env, gamma=0.9)
            _, _ = vi_sync.solve(theta=1e-4, verbose=False)
            results['Synchronous'].append({
                'iterations': vi_sync.total_iterations,
                'time': vi_sync.total_time,
                'updates': vi_sync.total_iterations * len(env.state_space)
            })
            
            # 异步随机版本
            vi_async_random = AsynchronousValueIteration(env, gamma=0.9, 
                                                        update_mode='random')
            _, _ = vi_async_random.solve(theta=1e-4, verbose=False)
            results['Async-Random'].append({
                'iterations': vi_async_random.total_iterations,
                'time': vi_async_random.total_time,
                'updates': vi_async_random.total_iterations
            })
            
            # 异步顺序版本
            vi_async_seq = AsynchronousValueIteration(env, gamma=0.9,
                                                     update_mode='sequential')
            _, _ = vi_async_seq.solve(theta=1e-4, verbose=False)
            results['Async-Sequential'].append({
                'iterations': vi_async_seq.total_iterations,
                'time': vi_async_seq.total_time,
                'updates': vi_async_seq.total_iterations
            })
        
        # 统计和可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        methods = list(results.keys())
        colors = ['steelblue', 'lightcoral', 'lightgreen']
        
        # 图1：迭代次数
        ax1 = axes[0]
        avg_iterations = [np.mean([r['iterations'] for r in results[m]]) 
                         for m in methods]
        std_iterations = [np.std([r['iterations'] for r in results[m]]) 
                         for m in methods]
        
        bars1 = ax1.bar(methods, avg_iterations, yerr=std_iterations,
                       color=colors, alpha=0.7, capsize=5)
        ax1.set_ylabel('Iterations')
        ax1.set_title('Iterations to Converge')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加数值
        for bar, val, std in zip(bars1, avg_iterations, std_iterations):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.0f}±{std:.0f}', ha='center', va='bottom', fontsize=9)
        
        # 图2：运行时间
        ax2 = axes[1]
        avg_times = [np.mean([r['time'] for r in results[m]]) for m in methods]
        std_times = [np.std([r['time'] for r in results[m]]) for m in methods]
        
        bars2 = ax2.bar(methods, avg_times, yerr=std_times,
                       color=colors, alpha=0.7, capsize=5)
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Runtime')
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, val, std in zip(bars2, avg_times, std_times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 图3：总更新次数
        ax3 = axes[2]
        avg_updates = [np.mean([r['updates'] for r in results[m]]) for m in methods]
        
        bars3 = ax3.bar(methods, avg_updates, color=colors, alpha=0.7)
        ax3.set_ylabel('Total State Updates')
        ax3.set_title('Total Updates')
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars3, avg_updates):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.0f}', ha='center', va='bottom')
        
        plt.suptitle('Synchronous vs Asynchronous Comparison', fontweight='bold')
        plt.tight_layout()
        
        # 打印总结
        print("\n实验总结 Experiment Summary:")
        print("-" * 40)
        for method in methods:
            print(f"\n{method}:")
            print(f"  平均迭代: {np.mean([r['iterations'] for r in results[method]]):.1f}")
            print(f"  平均时间: {np.mean([r['time'] for r in results[method]]):.3f}s")
            print(f"  平均更新: {np.mean([r['updates'] for r in results[method]]):.0f}")
        
        return fig


# ================================================================================
# 主函数：演示价值迭代
# Main Function: Demonstrate Value Iteration
# ================================================================================

def main():
    """
    运行价值迭代完整演示
    Run Complete Value Iteration Demo
    """
    print("\n" + "="*80)
    print("第3.3节：价值迭代")
    print("Section 3.3: Value Iteration")
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
    
    # 1. 运行标准价值迭代
    print("\n" + "="*60)
    print("1. 标准（同步）价值迭代")
    print("1. Standard (Synchronous) Value Iteration")
    print("="*60)
    
    vi = ValueIteration(env, gamma=0.9)
    optimal_policy, optimal_V = vi.solve(theta=1e-6, verbose=True)
    
    # 2. 可视化收敛过程
    print("\n2. 可视化收敛过程")
    print("2. Visualize Convergence Process")
    visualizer = ValueIterationVisualizer()
    fig1 = visualizer.visualize_convergence(vi)
    
    # 3. 创建动画（如果可能）
    print("\n3. 创建价值传播动画")
    print("3. Create Value Propagation Animation")
    anim = visualizer.create_animation(vi, env)
    if anim:
        print("动画创建成功（在Jupyter中可以播放）")
        print("Animation created successfully (can play in Jupyter)")
    
    # 4. 理论分析
    ValueIterationAnalysis.theoretical_analysis()
    
    # 5. 比较同步和异步版本
    print("\n5. 比较同步和异步版本")
    print("5. Compare Synchronous and Asynchronous Versions")
    fig2 = ValueIterationAnalysis.compare_sync_async(env, n_runs=3)
    
    # 6. 展示最优策略
    print("\n" + "="*60)
    print("最优策略和价值")
    print("Optimal Policy and Values")
    print("="*60)
    
    # 显示关键位置的价值和动作
    key_positions = [
        (0, 0),  # 起点
        (0, 2),  # 右上
        (2, 0),  # 左下
        (2, 3),  # 目标附近
        (3, 2)   # 目标附近
    ]
    
    print("\n关键位置的最优决策:")
    print("Optimal decisions at key positions:")
    for pos in key_positions:
        if pos in env.pos_to_state:
            state = env.pos_to_state[pos]
            value = optimal_V.get_value(state)
            
            if isinstance(optimal_policy, DeterministicPolicy) and state in optimal_policy.policy_map:
                action = optimal_policy.policy_map[state]
                print(f"  位置 {pos}: {action.id} (V={value:.2f})")
    
    # 比较收敛速度
    print(f"\n收敛统计 Convergence Statistics:")
    print(f"  价值迭代迭代次数: {vi.total_iterations}")
    print(f"  总状态更新次数: {vi.total_iterations * len(env.state_space)}")
    print(f"  平均收缩率: {np.mean(vi.convergence_history[i+1]/vi.convergence_history[i] for i in range(len(vi.convergence_history)-1) if vi.convergence_history[i] > 0):.3f}")
    
    print("\n" + "="*80)
    print("价值迭代演示完成！")
    print("Value Iteration Demo Complete!")
    print("\n关键要点 Key Takeaways:")
    print("1. 价值迭代直接寻找最优价值函数")
    print("   Value iteration directly finds optimal value function")
    print("2. 每次迭代应用贝尔曼最优算子")
    print("   Each iteration applies Bellman optimality operator")
    print("3. 收敛速度取决于γ（指数收敛）")
    print("   Convergence speed depends on γ (exponential)")
    print("4. 异步版本可能更高效")
    print("   Asynchronous version may be more efficient")
    print("5. 适合需要近似解的场景（可随时停止）")
    print("   Good for scenarios needing approximation (can stop anytime)")
    print("="*80)
    
    plt.show()
    
    return optimal_policy, optimal_V


if __name__ == "__main__":
    main()