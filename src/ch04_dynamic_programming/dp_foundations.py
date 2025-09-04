"""
================================================================================
第3.1节：动态规划基础 - RL算法的理论根基
Section 3.1: Dynamic Programming Foundations - Theoretical Basis of RL Algorithms
================================================================================

动态规划(Dynamic Programming, DP)这个名字听起来很酷，但它的核心思想其实很简单：
The name "Dynamic Programming" sounds cool, but its core idea is actually simple:
将复杂问题分解成更简单的子问题，然后组合子问题的解来得到原问题的解。
Break complex problems into simpler subproblems, then combine solutions to get the original solution.

在RL中，DP利用了价值函数的递归结构（贝尔曼方程）来寻找最优策略。
In RL, DP exploits the recursive structure of value functions (Bellman equations) to find optimal policies.

为什么叫"动态规划"？
Why is it called "Dynamic Programming"?
- "动态"：问题具有时序结构，需要做序列决策
  "Dynamic": Problems have temporal structure, requiring sequential decisions
- "规划"：通过计算来优化决策
  "Programming": Optimize decisions through computation

历史趣闻：Richard Bellman在1950年代创造这个术语时，故意选了一个
听起来很厉害但又模糊的名字，以避免他的研究被当时的国防部长否决。
Historical note: Richard Bellman coined this term in the 1950s, deliberately choosing
an impressive but vague name to avoid his research being rejected by the Secretary of Defense.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# 导入第2章的组件
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from ch02_mdp.mdp_framework import (
    State, Action, MDPEnvironment,
    TransitionProbability, RewardFunction
)
from ch02_mdp.policies_and_values import (
    Policy, StateValueFunction, ActionValueFunction,
    DeterministicPolicy, StochasticPolicy
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第3.1.1节：动态规划的核心思想
# Section 3.1.1: Core Ideas of Dynamic Programming
# ================================================================================

class DynamicProgrammingFoundations:
    """
    动态规划基础理论
    Dynamic Programming Foundations
    
    DP的两个关键性质：
    Two key properties for DP:
    
    1. 最优子结构(Optimal Substructure)
       - 最优解可以通过子问题的最优解构造
       - Optimal solution can be constructed from optimal solutions of subproblems
       - 在RL中：最优策略的价值函数满足贝尔曼最优方程
       - In RL: Value function of optimal policy satisfies Bellman optimality equation
    
    2. 重叠子问题(Overlapping Subproblems)
       - 子问题会被多次求解
       - Subproblems are solved multiple times
       - 在RL中：同一个状态会被多次访问和更新
       - In RL: Same state is visited and updated multiple times
    
    DP vs 其他方法：
    DP vs Other Methods:
    
    | 方法 Method | 需要模型? Need Model? | 计算方式 Computation | 适用场景 Use Case |
    |-------------|---------------------|-------------------|-----------------|
    | DP | 是 Yes | 全宽度更新 Full-width | 小状态空间 Small state space |
    | MC | 否 No | 采样 Sampling | 回合式任务 Episodic tasks |
    | TD | 否 No | 自举 Bootstrapping | 在线学习 Online learning |
    """
    
    @staticmethod
    def explain_dp_principles():
        """
        详细解释DP原理
        Detailed Explanation of DP Principles
        
        这是一个教学函数，通过具体例子帮助理解DP
        This is a teaching function that helps understand DP through concrete examples
        """
        print("\n" + "="*80)
        print("动态规划核心原理")
        print("Core Principles of Dynamic Programming")
        print("="*80)
        
        print("""
        📚 1. 什么是动态规划？
        What is Dynamic Programming?
        ================================
        
        想象你要从家到公司，有多条路线可选：
        Imagine going from home to office with multiple route options:
        
        家 Home ──┐
                 ├─→ 路口A Junction A ─┐
                 │                    ├─→ 公司 Office
                 └─→ 路口B Junction B ─┘
        
        动态规划的思路：
        DP approach:
        1. 先计算从各路口到公司的最短时间
           First calculate shortest time from each junction to office
        2. 然后选择：家到哪个路口 + 该路口到公司的时间最短
           Then choose: home to which junction + that junction to office is shortest
        
        这就是"最优子结构"：整体最优解包含子问题的最优解
        This is "optimal substructure": overall optimal solution contains optimal solutions of subproblems
        
        📚 2. DP在RL中的应用
        DP in RL
        ================================
        
        贝尔曼方程就是DP的递归关系：
        Bellman equation is the recursive relation of DP:
        
        v(s) = max_a [r(s,a) + γ Σ_s' p(s'|s,a) v(s')]
               ↑                    ↑
               当前奖励              未来价值
               immediate reward     future value
        
        这个方程说明：
        This equation shows:
        - 一个状态的价值 = 立即奖励 + 折扣的未来价值
          Value of a state = immediate reward + discounted future value
        - 这是一个递归定义，可以用DP求解
          This is a recursive definition, solvable by DP
        
        📚 3. DP的计算模式
        Computation Pattern of DP
        ================================
        
        同步更新 Synchronous Update:
        ┌──────────┐      ┌──────────┐
        │ V_k(s1)  │      │ V_{k+1}  │
        │ V_k(s2)  │ ───→ │   (s1)   │
        │ V_k(s3)  │      │ V_{k+1}  │
        │   ...    │      │   (s2)   │
        └──────────┘      └──────────┘
        使用所有V_k        计算所有V_{k+1}
        Use all V_k       Compute all V_{k+1}
        
        异步更新 Asynchronous Update:
        随时更新任意状态，更灵活但收敛性分析更复杂
        Update any state at any time, more flexible but convergence analysis is complex
        
        📚 4. DP的优势与局限
        Advantages and Limitations of DP
        ================================
        
        优势 Advantages:
        ✓ 数学优雅，理论保证收敛到最优
          Mathematically elegant, guaranteed to converge to optimal
        ✓ 充分利用模型信息，样本效率高
          Fully utilizes model information, high sample efficiency
        ✓ 可以离线计算，不需要与环境交互
          Can compute offline, no environment interaction needed
        
        局限 Limitations:
        ✗ 需要完整的环境模型（转移概率和奖励函数）
          Requires complete environment model (transition probabilities and reward function)
        ✗ 计算复杂度高：O(|S|²|A|) per iteration
          High computational complexity: O(|S|²|A|) per iteration
        ✗ 维度诅咒：状态空间大时不可行
          Curse of dimensionality: infeasible for large state spaces
        
        这就是为什么我们需要MC和TD方法！
        This is why we need MC and TD methods!
        """)
        
        # 创建可视化
        DynamicProgrammingFoundations._visualize_dp_concept()
    
    @staticmethod
    def _visualize_dp_concept():
        """
        可视化DP概念
        Visualize DP Concepts
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 图1：最优子结构
        ax1 = axes[0]
        ax1.set_title("Optimal Substructure / 最优子结构")
        
        # 绘制树形结构
        positions = {
            'root': (0.5, 0.8),
            'left': (0.3, 0.5),
            'right': (0.7, 0.5),
            'leaf1': (0.2, 0.2),
            'leaf2': (0.4, 0.2),
            'leaf3': (0.6, 0.2),
            'leaf4': (0.8, 0.2)
        }
        
        # 画节点
        for node, (x, y) in positions.items():
            circle = plt.Circle((x, y), 0.05, color='lightblue', ec='black')
            ax1.add_patch(circle)
            if node == 'root':
                ax1.text(x, y, 'v(s)', ha='center', va='center', fontweight='bold')
            else:
                ax1.text(x, y, 'v', ha='center', va='center')
        
        # 画边
        edges = [
            ('root', 'left'), ('root', 'right'),
            ('left', 'leaf1'), ('left', 'leaf2'),
            ('right', 'leaf3'), ('right', 'leaf4')
        ]
        for start, end in edges:
            x1, y1 = positions[start]
            x2, y2 = positions[end]
            ax1.plot([x1, x2], [y1, y2], 'k-', alpha=0.5)
        
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        ax1.text(0.5, 0.05, "最优解依赖子问题最优解\nOptimal depends on suboptimal", 
                ha='center', fontsize=10)
        
        # 图2：迭代过程
        ax2 = axes[1]
        ax2.set_title("Iterative Process / 迭代过程")
        
        # 模拟价值函数收敛
        iterations = 20
        x = np.arange(iterations)
        
        # 不同状态的价值收敛曲线
        np.random.seed(42)
        for i in range(3):
            true_value = np.random.uniform(5, 10)
            values = [0]
            for t in range(1, iterations):
                # 模拟收敛过程
                values.append(true_value * (1 - np.exp(-0.3 * t)) + np.random.normal(0, 0.1))
            ax2.plot(x, values, marker='o', markersize=3, label=f'State {i+1}')
            ax2.axhline(y=true_value, color='gray', linestyle='--', alpha=0.3)
        
        ax2.set_xlabel('Iteration / 迭代次数')
        ax2.set_ylabel('Value / 价值')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.text(10, 2, "价值函数逐渐收敛\nValues converge gradually", 
                ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
        
        # 图3：同步更新示意
        ax3 = axes[2]
        ax3.set_title("Synchronous Update / 同步更新")
        
        # 创建网格表示状态
        grid_size = 4
        old_values = np.random.rand(grid_size, grid_size) * 5
        new_values = old_values * 1.2 + np.random.rand(grid_size, grid_size)
        
        # 显示旧值
        im1 = ax3.imshow(old_values, cmap='coolwarm', alpha=0.5, extent=[0, 2, 0, 2])
        
        # 添加箭头
        ax3.arrow(2.2, 1, 0.6, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        # 显示新值
        im2 = ax3.imshow(new_values, cmap='coolwarm', alpha=0.5, extent=[3, 5, 0, 2])
        
        ax3.text(1, -0.3, "V_k", ha='center', fontsize=12, fontweight='bold')
        ax3.text(4, -0.3, "V_{k+1}", ha='center', fontsize=12, fontweight='bold')
        ax3.text(2.5, 1, "Bellman\nUpdate", ha='center', va='center', fontsize=10)
        
        ax3.set_xlim(-0.5, 5.5)
        ax3.set_ylim(-0.5, 2.5)
        ax3.axis('off')
        
        # 添加颜色条
        plt.colorbar(im2, ax=ax3, orientation='horizontal', pad=0.1, fraction=0.05)
        
        plt.suptitle("Dynamic Programming Concepts / 动态规划概念", fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig


# ================================================================================
# 第3.1.2节：贝尔曼算子
# Section 3.1.2: Bellman Operators
# ================================================================================

class BellmanOperator:
    """
    贝尔曼算子 - DP算法的数学核心
    Bellman Operators - Mathematical Core of DP Algorithms
    
    贝尔曼算子是作用在价值函数上的映射，将一个价值函数映射到另一个价值函数。
    Bellman operators are mappings on value functions, mapping one value function to another.
    
    为什么重要？
    Why important?
    1. DP算法本质上是反复应用贝尔曼算子
       DP algorithms essentially apply Bellman operators repeatedly
    2. 算子的性质（如收缩性）保证了算法收敛
       Properties of operators (like contraction) guarantee algorithm convergence
    3. 不动点就是我们要找的解
       Fixed point is the solution we're looking for
    
    数学表示：
    Mathematical representation:
    
    贝尔曼期望算子 Bellman Expectation Operator:
    T^π(v)(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γv(s')]
    
    贝尔曼最优算子 Bellman Optimality Operator:
    T*(v)(s) = max_a Σ_{s',r} p(s',r|s,a)[r + γv(s')]
    
    关键性质 Key Properties:
    1. 单调性(Monotonicity): v ≤ w ⟹ Tv ≤ Tw
    2. 收缩性(Contraction): ||Tv - Tw||∞ ≤ γ||v - w||∞
    3. 唯一不动点(Unique Fixed Point): v* = Tv*
    """
    
    def __init__(self, mdp_env: MDPEnvironment, gamma: float = 0.99):
        """
        初始化贝尔曼算子
        Initialize Bellman Operator
        
        Args:
            mdp_env: MDP环境（需要知道模型）
            gamma: 折扣因子
        
        设计思考：
        Design Consideration:
        我们把算子设计成类，是因为它需要记住MDP的参数（P, R, γ）
        We design operator as a class because it needs to remember MDP parameters
        """
        self.env = mdp_env
        self.gamma = gamma
        self.P, self.R = mdp_env.get_dynamics()
        
        logger.info(f"初始化贝尔曼算子，γ={gamma}")
    
    def bellman_expectation_operator(self, 
                                    v: StateValueFunction,
                                    policy: Policy) -> StateValueFunction:
        """
        贝尔曼期望算子 T^π
        Bellman Expectation Operator T^π
        
        这个算子用于策略评估：给定策略π，计算其价值函数
        This operator is used for policy evaluation: given policy π, compute its value function
        
        T^π(v)(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γv(s')]
        
        为什么叫"期望"算子？
        Why called "expectation" operator?
        因为它计算的是遵循策略π的期望回报
        Because it computes expected return following policy π
        
        Args:
            v: 当前价值函数估计
               Current value function estimate
            policy: 策略π
                   Policy π
        
        Returns:
            新的价值函数 T^π(v)
            New value function T^π(v)
        
        时间复杂度 Time Complexity: O(|S|²|A|)
        空间复杂度 Space Complexity: O(|S|)
        """
        # 创建新的价值函数
        v_new = StateValueFunction(self.env.state_space, initial_value=0.0)
        
        # 对每个状态应用算子
        for state in self.env.state_space:
            if state.is_terminal:
                # 终止状态的价值为0
                # Terminal state has value 0
                v_new.set_value(state, 0.0)
                continue
            
            # 计算期望价值
            # Compute expected value
            value = 0.0
            
            # 获取策略在该状态的动作分布
            action_probs = policy.get_action_probabilities(state)
            
            for action, action_prob in action_probs.items():
                # 计算选择该动作的价值
                q_value = self._compute_q_value(state, action, v)
                
                # 用策略概率加权
                value += action_prob * q_value
            
            v_new.set_value(state, value)
            
            logger.debug(f"T^π(v)({state.id}) = {value:.3f}")
        
        return v_new
    
    def bellman_optimality_operator(self, 
                                   v: StateValueFunction) -> StateValueFunction:
        """
        贝尔曼最优算子 T*
        Bellman Optimality Operator T*
        
        这个算子用于寻找最优价值函数
        This operator is used to find optimal value function
        
        T*(v)(s) = max_a Σ_{s',r} p(s',r|s,a)[r + γv(s')]
        
        为什么叫"最优"算子？
        Why called "optimality" operator?
        因为它总是选择最好的动作（贪婪）
        Because it always chooses the best action (greedy)
        
        Args:
            v: 当前价值函数估计
               Current value function estimate
        
        Returns:
            新的价值函数 T*(v)
            New value function T*(v)
        
        注意：这个算子的不动点就是最优价值函数v*！
        Note: The fixed point of this operator is the optimal value function v*!
        """
        v_new = StateValueFunction(self.env.state_space, initial_value=0.0)
        
        for state in self.env.state_space:
            if state.is_terminal:
                v_new.set_value(state, 0.0)
                continue
            
            # 找最大价值
            # Find maximum value
            max_value = float('-inf')
            
            for action in self.env.action_space:
                q_value = self._compute_q_value(state, action, v)
                max_value = max(max_value, q_value)
            
            v_new.set_value(state, max_value)
            
            logger.debug(f"T*(v)({state.id}) = {max_value:.3f}")
        
        return v_new
    
    def _compute_q_value(self, state: State, action: Action, 
                        v: StateValueFunction) -> float:
        """
        计算Q值：q(s,a) = Σ_{s',r} p(s',r|s,a)[r + γv(s')]
        Compute Q-value: q(s,a) = Σ_{s',r} p(s',r|s,a)[r + γv(s')]
        
        这是贝尔曼算子的核心计算
        This is the core computation of Bellman operators
        
        Args:
            state: 状态s
            action: 动作a
            v: 价值函数
        
        Returns:
            动作价值q(s,a)
        """
        q_value = 0.0
        
        # 获取所有可能的转移
        transitions = self.P.get_transitions(state, action)
        
        for next_state, reward, prob in transitions:
            # 贝尔曼方程的核心：立即奖励 + 折扣的未来价值
            # Core of Bellman equation: immediate reward + discounted future value
            q_value += prob * (reward + self.gamma * v.get_value(next_state))
        
        return q_value
    
    def verify_contraction_property(self, v1: StateValueFunction, 
                                   v2: StateValueFunction) -> float:
        """
        验证收缩性质
        Verify Contraction Property
        
        收缩映射定理保证了价值迭代的收敛性
        Contraction mapping theorem guarantees convergence of value iteration
        
        性质：||Tv - Tw||∞ ≤ γ||v - w||∞
        Property: ||Tv - Tw||∞ ≤ γ||v - w||∞
        
        这意味着每次迭代，误差至少缩小到原来的γ倍
        This means each iteration reduces error by at least factor γ
        
        Args:
            v1, v2: 两个价值函数
        
        Returns:
            收缩因子（应该≤γ）
            Contraction factor (should be ≤γ)
        """
        # 计算原始距离
        original_dist = self._compute_max_norm_distance(v1, v2)
        
        # 应用算子
        tv1 = self.bellman_optimality_operator(v1)
        tv2 = self.bellman_optimality_operator(v2)
        
        # 计算新距离
        new_dist = self._compute_max_norm_distance(tv1, tv2)
        
        # 计算收缩因子
        contraction_factor = new_dist / original_dist if original_dist > 0 else 0
        
        logger.info(f"收缩验证: ||Tv-Tw||={new_dist:.4f}, ||v-w||={original_dist:.4f}, "
                   f"factor={contraction_factor:.4f} (应≤{self.gamma})")
        
        return contraction_factor
    
    def _compute_max_norm_distance(self, v1: StateValueFunction, 
                                   v2: StateValueFunction) -> float:
        """
        计算最大范数距离 ||v1 - v2||∞
        Compute max norm distance
        
        这是衡量两个价值函数差异的标准方法
        This is the standard way to measure difference between value functions
        """
        max_diff = 0.0
        for state in self.env.state_space:
            diff = abs(v1.get_value(state) - v2.get_value(state))
            max_diff = max(max_diff, diff)
        return max_diff


# ================================================================================
# 第3.1.3节：策略评估（预测问题）
# Section 3.1.3: Policy Evaluation (Prediction Problem)
# ================================================================================

class PolicyEvaluationDP:
    """
    策略评估 - 动态规划版本
    Policy Evaluation - Dynamic Programming Version
    
    问题：给定策略π，计算其价值函数v_π
    Problem: Given policy π, compute its value function v_π
    
    这是"预测问题"：预测遵循策略π能获得多少回报
    This is the "prediction problem": predict how much return we get following policy π
    
    算法：迭代应用贝尔曼期望算子
    Algorithm: Iteratively apply Bellman expectation operator
    v_{k+1} = T^π(v_k)
    
    为什么会收敛？
    Why does it converge?
    因为T^π是γ-收缩映射，有唯一不动点v_π
    Because T^π is a γ-contraction mapping with unique fixed point v_π
    
    收敛速度：O(γ^k)，指数收敛！
    Convergence rate: O(γ^k), exponential convergence!
    """
    
    def __init__(self, mdp_env: MDPEnvironment, gamma: float = 0.99):
        """
        初始化策略评估器
        Initialize Policy Evaluator
        
        Args:
            mdp_env: MDP环境
            gamma: 折扣因子
        """
        self.env = mdp_env
        self.gamma = gamma
        self.bellman_op = BellmanOperator(mdp_env, gamma)
        
        # 记录评估历史（用于可视化）
        self.evaluation_history = []
        
        logger.info("初始化策略评估器(DP)")
    
    def evaluate(self, policy: Policy, 
                theta: float = 1e-6,
                max_iterations: int = 1000,
                initial_v: Optional[StateValueFunction] = None) -> StateValueFunction:
        """
        迭代策略评估
        Iterative Policy Evaluation
        
        这是最基础的DP算法，理解它是理解所有DP算法的关键！
        This is the most basic DP algorithm, understanding it is key to understanding all DP algorithms!
        
        算法流程：
        Algorithm Flow:
        1. 初始化V(s)任意（通常为0）
           Initialize V(s) arbitrarily (usually 0)
        2. 重复直到收敛：
           Repeat until convergence:
           对每个状态s：
           For each state s:
             v(s) ← Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γV(s')]
        3. 返回收敛的V
           Return converged V
        
        Args:
            policy: 要评估的策略
                   Policy to evaluate
            theta: 收敛阈值（最大值变化小于此值时停止）
                  Convergence threshold
            max_iterations: 最大迭代次数
                          Maximum iterations
            initial_v: 初始价值函数（可选）
                      Initial value function (optional)
        
        Returns:
            策略的价值函数v_π
            Value function of policy v_π
        
        教学要点：
        Teaching Points:
        1. 这是同步更新：所有状态同时更新
           This is synchronous update: all states updated simultaneously
        2. 需要两个数组：旧值和新值
           Need two arrays: old values and new values
        3. 收敛判断基于最大变化（无穷范数）
           Convergence based on max change (infinity norm)
        """
        # 初始化价值函数
        if initial_v is None:
            v = StateValueFunction(self.env.state_space, initial_value=0.0)
        else:
            v = initial_v
        
        # 清空历史记录
        self.evaluation_history = []
        
        logger.info(f"开始策略评估，theta={theta}")
        
        # 迭代评估
        for iteration in range(max_iterations):
            # 记录当前价值函数（深拷贝）
            v_snapshot = StateValueFunction(self.env.state_space)
            for state in self.env.state_space:
                v_snapshot.set_value(state, v.get_value(state))
            self.evaluation_history.append(v_snapshot)
            
            # 应用贝尔曼期望算子
            v_new = self.bellman_op.bellman_expectation_operator(v, policy)
            
            # 计算最大变化（判断收敛）
            delta = 0.0
            for state in self.env.state_space:
                old_value = v.get_value(state)
                new_value = v_new.get_value(state)
                delta = max(delta, abs(old_value - new_value))
            
            # 更新价值函数
            v = v_new
            
            # 日志记录
            if iteration % 10 == 0:
                logger.debug(f"迭代 {iteration}: delta = {delta:.6f}")
            
            # 检查收敛
            if delta < theta:
                logger.info(f"策略评估收敛！迭代次数: {iteration + 1}, 最终delta: {delta:.6f}")
                
                # 记录最终状态
                self.evaluation_history.append(v)
                break
        else:
            logger.warning(f"达到最大迭代次数 {max_iterations}，可能未完全收敛")
        
        return v
    
    def evaluate_with_trace(self, policy: Policy, 
                           theta: float = 1e-6) -> Tuple[StateValueFunction, List[float]]:
        """
        带轨迹的策略评估
        Policy Evaluation with Trace
        
        记录每次迭代的误差，用于分析收敛性
        Record error at each iteration for convergence analysis
        
        Returns:
            (最终价值函数, 误差轨迹)
            (final value function, error trace)
        """
        v = StateValueFunction(self.env.state_space, initial_value=0.0)
        errors = []
        
        for iteration in range(1000):
            v_new = self.bellman_op.bellman_expectation_operator(v, policy)
            
            # 计算误差
            delta = 0.0
            for state in self.env.state_space:
                delta = max(delta, abs(v.get_value(state) - v_new.get_value(state)))
            
            errors.append(delta)
            v = v_new
            
            if delta < theta:
                break
        
        return v, errors
    
    def demonstrate_convergence(self, policy: Policy):
        """
        演示收敛过程
        Demonstrate Convergence Process
        
        这个函数展示了价值函数如何逐步收敛到真实值
        This function shows how value function gradually converges to true values
        """
        print("\n" + "="*60)
        print("策略评估收敛演示")
        print("Policy Evaluation Convergence Demo")
        print("="*60)
        
        # 运行评估
        v_final, errors = self.evaluate_with_trace(policy)
        
        # 绘制收敛曲线
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 左图：误差下降
        ax1 = axes[0]
        ax1.semilogy(errors, 'b-', linewidth=2)
        ax1.set_xlabel('Iteration / 迭代')
        ax1.set_ylabel('Max Error (log scale) / 最大误差（对数尺度）')
        ax1.set_title('Convergence of Policy Evaluation / 策略评估收敛')
        ax1.grid(True, alpha=0.3)
        
        # 标注关键点
        ax1.axhline(y=1e-6, color='r', linestyle='--', alpha=0.5, label='θ=1e-6')
        convergence_iter = len(errors)
        ax1.plot(convergence_iter-1, errors[-1], 'ro', markersize=8)
        ax1.text(convergence_iter-1, errors[-1], f'  Converged at {convergence_iter}', 
                ha='left', va='center')
        ax1.legend()
        
        # 右图：价值函数演化
        ax2 = axes[1]
        
        # 选择几个状态展示
        sample_states = self.env.state_space[:min(5, len(self.env.state_space))]
        
        for state in sample_states:
            values = [vh.get_value(state) for vh in self.evaluation_history[::5]]  # 每5步采样
            ax2.plot(range(0, len(self.evaluation_history), 5), values, 
                    marker='o', markersize=3, label=f'State {state.id}')
        
        ax2.set_xlabel('Iteration / 迭代')
        ax2.set_ylabel('State Value / 状态价值')
        ax2.set_title('Value Function Evolution / 价值函数演化')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 打印收敛统计
        print(f"\n收敛统计 Convergence Statistics:")
        print(f"  迭代次数 Iterations: {len(errors)}")
        print(f"  最终误差 Final error: {errors[-1]:.2e}")
        print(f"  收敛速度 Convergence rate: ~{self.gamma:.3f} per iteration")
        
        # 打印最终价值
        print(f"\n最终价值函数 Final Value Function:")
        for state in self.env.state_space[:5]:  # 显示前5个状态
            print(f"  V({state.id}) = {v_final.get_value(state):.3f}")
        
        return fig


# ================================================================================
# 第3.1.4节：策略改进（控制问题的一部分）
# Section 3.1.4: Policy Improvement (Part of Control Problem)
# ================================================================================

class PolicyImprovementDP:
    """
    策略改进 - 基于价值函数改进策略
    Policy Improvement - Improve Policy Based on Value Function
    
    核心思想：如果我们知道v_π，就可以通过贪婪化得到更好的策略
    Core idea: If we know v_π, we can get a better policy by being greedy
    
    策略改进定理：
    Policy Improvement Theorem:
    如果对所有s，q_π(s, π'(s)) ≥ v_π(s)
    则对所有s，v_π'(s) ≥ v_π(s)
    
    这保证了贪婪策略不会更差！
    This guarantees greedy policy is not worse!
    
    数学原理：
    Mathematical Principle:
    π'(s) = argmax_a q_π(s,a) = argmax_a Σ_{s',r} p(s',r|s,a)[r + γv_π(s')]
    """
    
    def __init__(self, mdp_env: MDPEnvironment, gamma: float = 0.99):
        """
        初始化策略改进器
        Initialize Policy Improver
        """
        self.env = mdp_env
        self.gamma = gamma
        self.P, self.R = mdp_env.get_dynamics()
        
        logger.info("初始化策略改进器(DP)")
    
    def improve(self, v: StateValueFunction) -> Tuple[Policy, bool]:
        """
        基于价值函数改进策略
        Improve Policy Based on Value Function
        
        这是策略迭代的关键步骤！
        This is the key step in policy iteration!
        
        算法：
        Algorithm:
        对每个状态s：
        For each state s:
            π'(s) = argmax_a Σ_{s',r} p(s',r|s,a)[r + γv(s')]
        
        Args:
            v: 当前策略的价值函数
               Value function of current policy
        
        Returns:
            (改进的策略, 策略是否改变)
            (improved policy, whether policy changed)
        
        教学要点：
        Teaching Points:
        1. 这创建了一个确定性策略（每个状态选最佳动作）
           This creates a deterministic policy (best action for each state)
        2. 如果策略不变，说明已经是最优策略
           If policy unchanged, it's optimal policy
        3. 改进是单调的：新策略不会更差
           Improvement is monotonic: new policy not worse
        """
        # 存储新策略
        policy_map = {}
        policy_changed = False
        
        # 对每个状态找最佳动作
        for state in self.env.state_space:
            if state.is_terminal:
                continue
            
            # 计算每个动作的Q值
            action_values = {}
            for action in self.env.action_space:
                q_value = self._compute_q_value(state, action, v)
                action_values[action] = q_value
            
            # 选择最佳动作（贪婪）
            best_action = max(action_values, key=action_values.get)
            policy_map[state] = best_action
            
            # 记录详细信息用于教学
            logger.debug(f"State {state.id}: "
                        f"Q-values = {{{', '.join(f'{a.id}:{q:.2f}' for a, q in action_values.items())}}}, "
                        f"Best = {best_action.id}")
        
        # 创建新的确定性策略
        new_policy = DeterministicPolicy(policy_map)
        
        return new_policy, policy_changed
    
    def _compute_q_value(self, state: State, action: Action, 
                        v: StateValueFunction) -> float:
        """
        计算动作价值Q(s,a)
        Compute Action Value Q(s,a)
        
        q_π(s,a) = Σ_{s',r} p(s',r|s,a)[r + γv_π(s')]
        
        这是选择最佳动作的依据
        This is the basis for selecting best action
        """
        q_value = 0.0
        transitions = self.P.get_transitions(state, action)
        
        for next_state, reward, prob in transitions:
            q_value += prob * (reward + self.gamma * v.get_value(next_state))
        
        return q_value
    
    def demonstrate_improvement(self, initial_policy: Policy):
        """
        演示策略改进过程
        Demonstrate Policy Improvement Process
        
        展示一次策略改进如何产生更好的策略
        Show how one policy improvement produces a better policy
        """
        print("\n" + "="*60)
        print("策略改进演示")
        print("Policy Improvement Demonstration")
        print("="*60)
        
        # 评估初始策略
        evaluator = PolicyEvaluationDP(self.env, self.gamma)
        v_old = evaluator.evaluate(initial_policy)
        
        print("\n初始策略价值 Initial Policy Values:")
        for state in self.env.state_space[:3]:
            print(f"  V({state.id}) = {v_old.get_value(state):.3f}")
        
        # 改进策略
        new_policy, _ = self.improve(v_old)
        
        # 评估新策略
        v_new = evaluator.evaluate(new_policy)
        
        print("\n改进后策略价值 Improved Policy Values:")
        for state in self.env.state_space[:3]:
            old_val = v_old.get_value(state)
            new_val = v_new.get_value(state)
            improvement = new_val - old_val
            print(f"  V({state.id}) = {new_val:.3f} "
                  f"({'↑' if improvement > 0 else '='} {improvement:+.3f})")
        
        # 验证策略改进定理
        print("\n策略改进定理验证 Policy Improvement Theorem Verification:")
        all_improved = True
        for state in self.env.state_space:
            if v_new.get_value(state) < v_old.get_value(state) - 1e-6:
                all_improved = False
                break
        
        print(f"  所有状态价值不减少: {'✓ 是' if all_improved else '✗ 否'}")
        print(f"  All state values non-decreasing: {'✓ Yes' if all_improved else '✗ No'}")


# ================================================================================
# 主函数：演示DP基础
# Main Function: Demonstrate DP Foundations
# ================================================================================

def main():
    """
    运行动态规划基础演示
    Run Dynamic Programming Foundations Demo
    
    这个演示展示了DP的核心概念和算法
    This demo shows core concepts and algorithms of DP
    """
    print("\n" + "="*80)
    print("第3.1节：动态规划基础")
    print("Section 3.1: Dynamic Programming Foundations")
    print("="*80)
    
    # 1. 解释DP原理
    DynamicProgrammingFoundations.explain_dp_principles()
    
    # 2. 创建简单环境测试
    print("\n" + "="*80)
    print("创建测试环境")
    print("Creating Test Environment")
    print("="*80)
    
    # 使用第2章的网格世界
    from ch02_mdp.gridworld import GridWorld
    
    # 创建3x3网格世界
    env = GridWorld(rows=3, cols=3, start_pos=(0, 0), goal_pos=(2, 2))
    print(f"创建 {env.rows}×{env.cols} 网格世界")
    
    # 3. 测试贝尔曼算子
    print("\n" + "="*80)
    print("测试贝尔曼算子")
    print("Testing Bellman Operators")
    print("="*80)
    
    bellman_op = BellmanOperator(env, gamma=0.9)
    
    # 创建两个不同的价值函数
    v1 = StateValueFunction(env.state_space, initial_value=0.0)
    v2 = StateValueFunction(env.state_space, initial_value=1.0)
    
    # 验证收缩性
    contraction_factor = bellman_op.verify_contraction_property(v1, v2)
    print(f"收缩因子: {contraction_factor:.3f} (应该 ≤ 0.9)")
    
    # 4. 演示策略评估
    print("\n" + "="*80)
    print("演示策略评估")
    print("Demonstrating Policy Evaluation")
    print("="*80)
    
    # 创建随机策略
    from ch02_mdp.policies_and_values import UniformRandomPolicy
    random_policy = UniformRandomPolicy(env.action_space)
    
    # 评估策略
    evaluator = PolicyEvaluationDP(env, gamma=0.9)
    evaluator.demonstrate_convergence(random_policy)
    
    # 5. 演示策略改进
    print("\n" + "="*80)
    print("演示策略改进")
    print("Demonstrating Policy Improvement")
    print("="*80)
    
    improver = PolicyImprovementDP(env, gamma=0.9)
    improver.demonstrate_improvement(random_policy)
    
    print("\n" + "="*80)
    print("动态规划基础演示完成！")
    print("Dynamic Programming Foundations Demo Complete!")
    print("\n关键要点 Key Takeaways:")
    print("1. DP利用贝尔曼方程的递归结构")
    print("   DP exploits recursive structure of Bellman equations")
    print("2. 贝尔曼算子是收缩映射，保证收敛")
    print("   Bellman operators are contraction mappings, guaranteeing convergence")
    print("3. 策略评估计算v_π，策略改进得到更好的π")
    print("   Policy evaluation computes v_π, policy improvement gets better π")
    print("4. 这些是策略迭代和价值迭代的基础")
    print("   These are foundations of policy iteration and value iteration")
    print("="*80)
    
    plt.show()


if __name__ == "__main__":
    main()