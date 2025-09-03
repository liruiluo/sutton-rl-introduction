"""
================================================================================
第6.1节：资格迹基础 - 统一MC和TD的优雅方法
Section 6.1: Eligibility Traces - Elegant Unification of MC and TD
================================================================================

资格迹是强化学习最优雅的思想之一！
Eligibility traces are one of the most elegant ideas in RL!

核心思想 Core Idea:
对于每个状态分配一个"资格"（eligibility），表示该状态对当前TD误差负多少"责任"
Assign an "eligibility" to each state, representing how much "credit" it should get for current TD error

数学定义 Mathematical Definition:
e_t(s) = γλe_{t-1}(s) + 1(S_t = s)

其中 where:
- γ: 折扣因子 Discount factor
- λ: 迹衰减参数 Trace decay parameter (0 ≤ λ ≤ 1)
- 1(·): 指示函数 Indicator function

λ的作用 Role of λ:
- λ = 0: TD(0), 只更新当前状态
         Only update current state
- λ = 1: MC, 更新整个轨迹
         Update entire trajectory
- 0 < λ < 1: 介于TD和MC之间
            Between TD and MC

两种视角 Two Views:
1. 前向视角 Forward View:
   - λ-回报: G_t^λ = (1-λ)Σ_{n=1}^∞ λ^{n-1} G_t^{(n)}
   - 所有n-step回报的加权平均
     Weighted average of all n-step returns

2. 后向视角 Backward View:
   - 使用资格迹在线更新
     Online update using eligibility traces
   - 计算效率高
     Computationally efficient

资格迹类型 Types of Traces:
1. 累积迹 Accumulating Traces:
   e_t(s) = γλe_{t-1}(s) + 1(S_t = s)
   
2. 替换迹 Replacing Traces:
   e_t(s) = max(γλe_{t-1}(s), 1(S_t = s))
   
3. Dutch迹 Dutch Traces:
   e_t(s) = γλe_{t-1}(s) + α(1 - γλe_{t-1}(s))1(S_t = s)
   
生物学意义 Biological Significance:
资格迹类似于突触可塑性的时间窗口！
Eligibility traces resemble synaptic plasticity time windows!

当多巴胺信号（TD误差）到达时，根据资格迹更新突触强度
When dopamine signal (TD error) arrives, update synaptic strength based on eligibility trace

应用 Applications:
- TD(λ): 预测问题
         Prediction problems
- SARSA(λ): On-policy控制
            On-policy control
- Q(λ): Off-policy控制
        Off-policy control
- Actor-Critic with eligibility traces
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union, Set
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

from ch02_mdp.mdp_framework import State, Action, MDPEnvironment, MDPAgent
from ch02_mdp.policies_and_values import (
    Policy, StateValueFunction, ActionValueFunction,
    StochasticPolicy, DeterministicPolicy
)
from ch05_temporal_difference.td_foundations import TDError, TDErrorAnalyzer

# 设置日志
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第6.1.1节：资格迹的数学理论
# Section 6.1.1: Mathematical Theory of Eligibility Traces
# ================================================================================

class EligibilityTrace:
    """
    资格迹
    Eligibility Trace
    
    记录状态或状态-动作对的"资格"
    Track "eligibility" of states or state-action pairs
    
    核心公式 Core Formula:
    e_t(s) = γλe_{t-1}(s) + 1(S_t = s)
    
    物理意义 Physical Meaning:
    - 记忆的衰减过程
      Decay process of memory
    - 信用分配的时间窗口
      Time window for credit assignment
    - 学习信号的传播路径
      Propagation path of learning signal
    
    实现技巧 Implementation Tricks:
    1. 稀疏表示：只存储非零迹
       Sparse representation: Only store non-zero traces
    2. 阈值截断：当迹小于ε时置零
       Threshold truncation: Set to zero when trace < ε
    3. 迹重置：新回合开始时清零
       Trace reset: Clear at episode start
    """
    
    def __init__(self,
                 gamma: float = 0.99,
                 lambda_: float = 0.9,
                 trace_type: str = "accumulating",
                 threshold: float = 1e-5):
        """
        初始化资格迹
        Initialize eligibility trace
        
        Args:
            gamma: 折扣因子
                  Discount factor
            lambda_: 迹衰减参数，决定信用分配的时间范围
                    Trace decay parameter, determines credit assignment horizon
            trace_type: 迹类型 ("accumulating", "replacing", "dutch")
                       Trace type
            threshold: 截断阈值，低于此值的迹置零
                      Truncation threshold, traces below this are zeroed
        """
        self.gamma = gamma
        self.lambda_ = lambda_
        self.trace_type = trace_type
        self.threshold = threshold
        
        # 存储非零迹（稀疏表示）
        # Store non-zero traces (sparse representation)
        self.traces: Dict[Any, float] = {}
        
        # 统计信息
        # Statistics
        self.total_updates = 0
        self.max_trace_value = 0.0
        self.active_traces_history = []
        
        logger.info(f"初始化{trace_type}资格迹: γ={gamma}, λ={lambda_}")
    
    def update(self, current_state: Any, alpha: float = 1.0):
        """
        更新资格迹
        Update eligibility trace
        
        执行一步迹更新
        Perform one step of trace update
        
        Args:
            current_state: 当前状态
                         Current state
            alpha: 学习率（用于Dutch迹）
                  Learning rate (for Dutch traces)
        """
        # 衰减所有现有迹
        # Decay all existing traces
        decay_factor = self.gamma * self.lambda_
        
        # 使用列表复制避免在迭代时修改字典
        # Use list copy to avoid modifying dict during iteration
        for state in list(self.traces.keys()):
            self.traces[state] *= decay_factor
            
            # 截断小迹（稀疏性）
            # Truncate small traces (sparsity)
            if self.traces[state] < self.threshold:
                del self.traces[state]
        
        # 更新当前状态的迹
        # Update trace for current state
        if self.trace_type == "accumulating":
            # 累积迹：e(s) = γλe(s) + 1
            # Accumulating trace
            self.traces[current_state] = self.traces.get(current_state, 0.0) + 1.0
            
        elif self.trace_type == "replacing":
            # 替换迹：e(s) = max(γλe(s), 1)
            # Replacing trace
            self.traces[current_state] = 1.0
            
        elif self.trace_type == "dutch":
            # Dutch迹：e(s) = γλe(s) + α(1 - γλe(s))
            # Dutch trace
            old_trace = self.traces.get(current_state, 0.0)
            self.traces[current_state] = old_trace + alpha * (1.0 - old_trace)
        
        else:
            raise ValueError(f"未知迹类型: {self.trace_type}")
        
        # 更新统计
        # Update statistics
        self.total_updates += 1
        if self.traces:
            self.max_trace_value = max(self.max_trace_value, max(self.traces.values()))
        self.active_traces_history.append(len(self.traces))
    
    def get(self, state: Any) -> float:
        """
        获取状态的资格迹
        Get eligibility trace for state
        
        Args:
            state: 状态
                  State
        
        Returns:
            资格迹值
            Eligibility trace value
        """
        return self.traces.get(state, 0.0)
    
    def reset(self):
        """
        重置所有迹
        Reset all traces
        
        新回合开始时调用
        Called at start of new episode
        """
        self.traces.clear()
    
    def get_active_states(self) -> Set[Any]:
        """
        获取有非零迹的状态集合
        Get set of states with non-zero traces
        
        Returns:
            活跃状态集合
            Set of active states
        """
        return set(self.traces.keys())
    
    def analyze_traces(self):
        """
        分析迹的分布和特性
        Analyze trace distribution and characteristics
        """
        print("\n" + "="*60)
        print("资格迹分析 Eligibility Trace Analysis")
        print("="*60)
        
        print(f"\n迹类型 Trace Type: {self.trace_type}")
        print(f"参数 Parameters: γ={self.gamma}, λ={self.lambda_}")
        print(f"总更新次数 Total Updates: {self.total_updates}")
        print(f"最大迹值 Max Trace Value: {self.max_trace_value:.4f}")
        
        if self.traces:
            trace_values = list(self.traces.values())
            print(f"\n当前活跃迹 Current Active Traces: {len(self.traces)}")
            print(f"平均迹值 Mean Trace Value: {np.mean(trace_values):.4f}")
            print(f"迹值标准差 Std of Traces: {np.std(trace_values):.4f}")
            
            # 显示前5个最大的迹
            # Show top 5 largest traces
            sorted_traces = sorted(self.traces.items(), 
                                 key=lambda x: x[1], reverse=True)[:5]
            print("\n最大的5个迹 Top 5 Traces:")
            for state, trace_val in sorted_traces:
                print(f"  State {state}: {trace_val:.4f}")
        else:
            print("\n没有活跃迹 No active traces")
        
        # 分析迹的时间演化
        # Analyze temporal evolution of traces
        if self.active_traces_history:
            print(f"\n活跃迹数量历史 Active Traces History:")
            print(f"  最大 Max: {max(self.active_traces_history)}")
            print(f"  平均 Mean: {np.mean(self.active_traces_history):.1f}")
            print(f"  最近 Recent: {self.active_traces_history[-1]}")


class LambdaReturn:
    """
    λ-回报计算
    Lambda-Return Computation
    
    前向视角的核心概念
    Core concept of forward view
    
    λ-回报定义 Lambda-Return Definition:
    G_t^λ = (1-λ) Σ_{n=1}^{T-t-1} λ^{n-1} G_t^{(n)} + λ^{T-t-1} G_t
    
    其中 where:
    - G_t^{(n)}: n步回报 n-step return
    - T: 终止时间 Terminal time
    - λ: 加权参数 Weighting parameter
    
    直觉理解 Intuitive Understanding:
    - λ=0: 只用1步回报（TD(0)）
           Only 1-step return
    - λ=1: 用完整回报（MC）
           Full return
    - 0<λ<1: 指数加权平均
            Exponentially weighted average
    
    几何级数求和 Geometric Series:
    权重和 = (1-λ)(1 + λ + λ² + ... + λ^{T-t-2}) + λ^{T-t-1}
           = 1 (归一化)
             (normalized)
    """
    
    @staticmethod
    def compute_lambda_return(rewards: List[float],
                             values: List[float],
                             gamma: float,
                             lambda_: float) -> List[float]:
        """
        计算λ-回报
        Compute lambda-returns
        
        离线计算，需要完整轨迹
        Offline computation, needs complete trajectory
        
        Args:
            rewards: 奖励序列 R_1, R_2, ..., R_T
                    Reward sequence
            values: 价值估计 V(S_1), V(S_2), ..., V(S_{T+1})
                   Value estimates for next states
            gamma: 折扣因子
                  Discount factor
            lambda_: λ参数
                    Lambda parameter
        
        Returns:
            每个时刻的λ-回报
            Lambda-return for each timestep
        """
        T = len(rewards)
        lambda_returns = []
        
        for t in range(T):
            # 特殊处理λ=0的情况（TD(0)）
            # Special handling for λ=0 (TD(0))
            if lambda_ == 0.0:
                # TD(0): R_{t+1} + γV(S_{t+1})
                if t < len(values):
                    g_lambda = rewards[t] + gamma * values[t]
                else:
                    g_lambda = rewards[t]
            else:
                # 计算所有n步回报
                # Compute all n-step returns
                n_step_returns = []
                
                for n in range(1, T - t + 1):
                    # n步回报：前n步用真实奖励，然后bootstrap
                    # n-step return: n steps of actual rewards, then bootstrap
                    g_n = 0.0
                    
                    # 累积n步奖励
                    # Accumulate n steps of rewards
                    for k in range(min(n, T - t)):
                        g_n += (gamma ** k) * rewards[t + k]
                    
                    # 添加bootstrap值（如果不是终止）
                    # Add bootstrap value (if not terminal)
                    if t + n < len(values):
                        g_n += (gamma ** n) * values[t + n]
                    
                    n_step_returns.append(g_n)
                
                # 计算λ-加权平均
                # Compute λ-weighted average
                g_lambda = 0.0
                
                if len(n_step_returns) == 1:
                    # 只有一步可用
                    # Only one step available
                    g_lambda = n_step_returns[0]
                else:
                    # 一般情况：加权求和
                    # General case: weighted sum
                    for i, g_n in enumerate(n_step_returns[:-1]):
                        weight = (1 - lambda_) * (lambda_ ** i)
                        g_lambda += weight * g_n
                    
                    # 添加最后一项（完整回报）
                    # Add last term (complete return)
                    if n_step_returns:
                        weight = lambda_ ** (len(n_step_returns) - 1)
                        g_lambda += weight * n_step_returns[-1]
            
            lambda_returns.append(g_lambda)
        
        return lambda_returns
    
    @staticmethod
    def demonstrate_lambda_effect():
        """
        演示λ参数的效果
        Demonstrate effect of λ parameter
        
        展示不同λ值如何影响回报计算
        Show how different λ values affect return computation
        """
        print("\n" + "="*60)
        print("λ参数效果演示 Lambda Parameter Effect Demo")
        print("="*60)
        
        # 创建示例轨迹
        # Create example trajectory
        rewards = [0, 0, 1, 0, 0]  # 第3步获得奖励
        values = [0.1, 0.2, 0.5, 0.3, 0.1, 0]  # 状态价值估计
        gamma = 0.9
        
        print("\n示例轨迹 Example Trajectory:")
        print(f"奖励 Rewards: {rewards}")
        print(f"价值 Values: {values}")
        print(f"折扣 Gamma: {gamma}")
        
        # 测试不同λ值
        # Test different λ values
        lambda_values = [0.0, 0.5, 0.9, 1.0]
        
        print("\nλ-回报 Lambda-Returns:")
        print("-" * 40)
        
        for lam in lambda_values:
            returns = LambdaReturn.compute_lambda_return(
                rewards, values[1:], gamma, lam
            )
            
            print(f"\nλ = {lam}:")
            for t, g in enumerate(returns):
                print(f"  G_{t}^λ = {g:.3f}")
            
            # 解释
            # Explanation
            if lam == 0.0:
                print("  (TD(0): 只用1步预测)")
                print("  (TD(0): Only 1-step lookahead)")
            elif lam == 1.0:
                print("  (MC: 用完整回报)")
                print("  (MC: Use complete return)")
            else:
                print(f"  (混合: {1-lam:.1%} TD + {lam:.1%} MC)")
                print(f"  (Mixed: {1-lam:.1%} TD + {lam:.1%} MC)")


# ================================================================================
# 第6.1.2节：前向视角与后向视角的等价性
# Section 6.1.2: Equivalence of Forward and Backward Views
# ================================================================================

class ForwardBackwardEquivalence:
    """
    前向与后向视角的等价性
    Equivalence of Forward and Backward Views
    
    TD(λ)的优雅之处：
    The elegance of TD(λ):
    两种完全不同的视角产生相同的更新！
    Two completely different views produce the same updates!
    
    前向视角 Forward View:
    - 理论上优雅，使用λ-回报
      Theoretically elegant, uses λ-return
    - 需要等到回合结束（离线）
      Needs to wait until episode end (offline)
    - 更新：ΔV(s) = α[G^λ - V(s)]
      Update: ΔV(s) = α[G^λ - V(s)]
    
    后向视角 Backward View:
    - 计算高效，使用资格迹
      Computationally efficient, uses eligibility traces
    - 可以在线更新（每步）
      Can update online (every step)
    - 更新：ΔV(s) = αδe(s)
      Update: ΔV(s) = αδe(s)
    
    等价性定理 Equivalence Theorem:
    离线λ-回报算法 = 在线资格迹算法（在回合结束时）
    Offline λ-return = Online eligibility traces (at episode end)
    
    这个等价性是TD(λ)的核心洞察！
    This equivalence is the key insight of TD(λ)!
    """
    
    @staticmethod
    def demonstrate_equivalence():
        """
        演示前向和后向视角的等价性
        Demonstrate equivalence of forward and backward views
        
        通过简单例子展示两种方法产生相同结果
        Show that both methods produce same results with simple example
        """
        print("\n" + "="*80)
        print("前向与后向视角等价性演示")
        print("Forward-Backward View Equivalence Demo")
        print("="*80)
        
        # 创建简单的回合
        # Create simple episode
        states = ['A', 'B', 'C', 'D']  # 状态序列
        rewards = [0, 0, 1]  # 奖励序列（到达D获得1）
        
        # 参数
        # Parameters
        gamma = 0.9
        lambda_ = 0.8
        alpha = 0.1
        
        # 初始价值函数
        # Initial value function
        V = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}
        
        print("\n初始设置 Initial Setup:")
        print(f"状态序列 State sequence: {' → '.join(states)}")
        print(f"奖励序列 Reward sequence: {rewards}")
        print(f"参数 Parameters: γ={gamma}, λ={lambda_}, α={alpha}")
        print(f"初始价值 Initial values: {V}")
        
        # ===== 方法1：前向视角（离线λ-回报）=====
        # Method 1: Forward View (Offline λ-return)
        print("\n" + "="*60)
        print("方法1：前向视角（λ-回报）")
        print("Method 1: Forward View (λ-return)")
        print("="*60)
        
        V_forward = V.copy()
        
        # 计算每个状态的λ-回报
        # Compute λ-return for each state
        for t, state in enumerate(states[:-1]):  # 不包括终止状态
            # 计算G_t^λ
            g_lambda = 0.0
            
            # 计算所有n步回报的加权和
            # Compute weighted sum of all n-step returns
            remaining = len(states) - t - 1
            
            for n in range(1, remaining + 1):
                # n步回报
                # n-step return
                g_n = 0.0
                for k in range(min(n, len(rewards) - t)):
                    g_n += (gamma ** k) * rewards[t + k]
                
                # Bootstrap（如果需要）
                # Bootstrap (if needed)
                if t + n < len(states):
                    g_n += (gamma ** n) * V_forward[states[t + n]]
                
                # 加权
                # Weighting
                if n < remaining:
                    weight = (1 - lambda_) * (lambda_ ** (n - 1))
                else:
                    weight = lambda_ ** (n - 1)
                
                g_lambda += weight * g_n
                
                print(f"  {state}: G^({n}) = {g_n:.3f}, weight = {weight:.3f}")
            
            print(f"  {state}: G^λ = {g_lambda:.3f}")
            
            # 更新价值
            # Update value
            V_forward[state] += alpha * (g_lambda - V_forward[state])
            print(f"  {state}: V_new = {V_forward[state]:.3f}")
        
        # ===== 方法2：后向视角（资格迹）=====
        # Method 2: Backward View (Eligibility Traces)
        print("\n" + "="*60)
        print("方法2：后向视角（资格迹）")
        print("Method 2: Backward View (Eligibility Traces)")
        print("="*60)
        
        V_backward = V.copy()
        
        # 初始化资格迹
        # Initialize eligibility traces
        e = {s: 0.0 for s in V.keys()}
        
        # 在线更新（每步）
        # Online update (every step)
        for t in range(len(states) - 1):
            state = states[t]
            next_state = states[t + 1]
            reward = rewards[t]
            
            # 更新当前状态的迹
            # Update trace for current state
            e[state] += 1.0
            
            # 计算TD误差
            # Compute TD error
            v_next = V_backward[next_state] if t < len(states) - 1 else 0
            delta = reward + gamma * v_next - V_backward[state]
            
            print(f"\n步骤{t+1} Step {t+1}: {state} → {next_state}")
            print(f"  TD误差 TD error δ = {delta:.3f}")
            print(f"  迹 Traces: {dict((k, f'{v:.3f}') for k, v in e.items() if v > 0)}")
            
            # 更新所有状态的价值（根据迹）
            # Update all state values (based on traces)
            for s in e.keys():
                if e[s] > 0:
                    update = alpha * delta * e[s]
                    V_backward[s] += update
                    if abs(update) > 0.001:
                        print(f"    更新 Update V({s}): +{update:.4f} = {V_backward[s]:.3f}")
            
            # 衰减所有迹
            # Decay all traces
            for s in e.keys():
                e[s] *= gamma * lambda_
        
        # ===== 比较结果 =====
        # Compare Results
        print("\n" + "="*60)
        print("结果比较 Results Comparison")
        print("="*60)
        
        print("\n最终价值函数 Final Value Functions:")
        print(f"{'State':<10} {'Initial':<10} {'Forward':<10} {'Backward':<10} {'Difference':<10}")
        print("-" * 50)
        
        for state in V.keys():
            diff = abs(V_forward[state] - V_backward[state])
            print(f"{state:<10} {V[state]:<10.3f} {V_forward[state]:<10.3f} "
                  f"{V_backward[state]:<10.3f} {diff:<10.6f}")
        
        # 验证等价性
        # Verify equivalence
        max_diff = max(abs(V_forward[s] - V_backward[s]) for s in V.keys())
        
        print(f"\n最大差异 Max Difference: {max_diff:.6f}")
        
        if max_diff < 0.001:
            print("✅ 前向和后向视角产生相同结果！")
            print("✅ Forward and backward views produce same results!")
        else:
            print("⚠️ 存在数值差异（可能由于浮点精度）")
            print("⚠️ Numerical differences exist (likely due to floating point)")
        
        print("\n关键洞察 Key Insights:")
        print("-" * 40)
        print("""
        1. 前向视角需要完整轨迹
           Forward view needs complete trajectory
           
        2. 后向视角可以在线更新
           Backward view can update online
           
        3. 两者在数学上等价
           Both are mathematically equivalent
           
        4. 资格迹使在线学习成为可能
           Eligibility traces enable online learning
           
        5. TD(λ)统一了TD和MC
           TD(λ) unifies TD and MC
        """)


# ================================================================================
# 第6.1.3节：资格迹的可视化
# Section 6.1.3: Visualization of Eligibility Traces
# ================================================================================

class TraceVisualizer:
    """
    资格迹可视化器
    Eligibility Trace Visualizer
    
    直观展示资格迹的动态变化
    Intuitively show dynamics of eligibility traces
    """
    
    @staticmethod
    def plot_trace_evolution(trace_history: Dict[str, List[float]],
                            td_errors: List[float] = None):
        """
        绘制迹的演化过程
        Plot evolution of traces
        
        Args:
            trace_history: 每个状态的迹历史
                         Trace history for each state
            td_errors: TD误差序列（可选）
                      TD error sequence (optional)
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), 
                                gridspec_kw={'height_ratios': [3, 1]})
        
        # 图1：资格迹演化
        # Plot 1: Eligibility trace evolution
        ax1 = axes[0]
        
        # 使用不同颜色绘制每个状态的迹
        # Plot each state's trace with different color
        colors = plt.cm.tab10(np.linspace(0, 1, len(trace_history)))
        
        for (state, history), color in zip(trace_history.items(), colors):
            ax1.plot(history, label=f'State {state}', 
                    color=color, linewidth=2, alpha=0.8)
        
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Eligibility Trace e(s)')
        ax1.set_title('Evolution of Eligibility Traces')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([-0.1, max(1.5, 
                               max(max(h) for h in trace_history.values()) * 1.1)])
        
        # 添加关键事件标记
        # Add markers for key events
        for t, state_traces in enumerate(zip(*trace_history.values())):
            if any(trace > 0.9 for trace in state_traces):
                ax1.axvline(x=t, color='red', linestyle='--', 
                           alpha=0.3, linewidth=0.5)
        
        # 图2：TD误差（如果提供）
        # Plot 2: TD errors (if provided)
        if td_errors:
            ax2 = axes[1]
            ax2.plot(td_errors, color='green', linewidth=2, alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('TD Error δ')
            ax2.set_title('TD Errors Over Time')
            ax2.grid(True, alpha=0.3)
            
            # 标记正负误差区域
            # Mark positive/negative error regions
            ax2.fill_between(range(len(td_errors)), 0, td_errors,
                           where=[d > 0 for d in td_errors],
                           color='green', alpha=0.2, label='Positive')
            ax2.fill_between(range(len(td_errors)), 0, td_errors,
                           where=[d < 0 for d in td_errors],
                           color='red', alpha=0.2, label='Negative')
            ax2.legend()
        
        plt.suptitle('Eligibility Traces Dynamics', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def visualize_trace_types():
        """
        可视化不同类型的资格迹
        Visualize different types of eligibility traces
        
        比较累积迹、替换迹和Dutch迹
        Compare accumulating, replacing, and Dutch traces
        """
        print("\n" + "="*60)
        print("不同类型资格迹比较")
        print("Comparison of Different Trace Types")
        print("="*60)
        
        # 参数
        # Parameters
        gamma = 0.9
        lambda_ = 0.8
        alpha = 0.1
        n_steps = 20
        
        # 创建状态访问序列（有重复访问）
        # Create state visit sequence (with revisits)
        visit_sequence = ['A', 'B', 'C', 'B', 'C', 'D', 
                         'A', 'B', 'C', 'D'] + ['E'] * 10
        
        # 初始化不同类型的迹
        # Initialize different trace types
        traces = {
            'Accumulating': EligibilityTrace(gamma, lambda_, 'accumulating'),
            'Replacing': EligibilityTrace(gamma, lambda_, 'replacing'),
            'Dutch': EligibilityTrace(gamma, lambda_, 'dutch')
        }
        
        # 记录历史
        # Record history
        history = {name: {state: [] for state in set(visit_sequence)} 
                  for name in traces.keys()}
        
        # 模拟更新
        # Simulate updates
        for step, state in enumerate(visit_sequence[:n_steps]):
            for name, trace in traces.items():
                # 更新迹
                # Update trace
                if name == 'Dutch':
                    trace.update(state, alpha)
                else:
                    trace.update(state)
                
                # 记录所有状态的迹值
                # Record trace values for all states
                for s in set(visit_sequence):
                    history[name][s].append(trace.get(s))
        
        # 绘图
        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, (name, hist) in enumerate(history.items()):
            ax = axes[idx]
            
            # 只绘制被访问过的状态
            # Only plot visited states
            visited_states = [s for s in hist.keys() 
                            if any(v > 0 for v in hist[s])]
            
            for state in visited_states:
                ax.plot(hist[state], label=f'State {state}', 
                       linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Eligibility Trace')
            ax.set_title(f'{name} Traces')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([-0.1, 2.5])
            
            # 标记状态访问
            # Mark state visits
            for t, s in enumerate(visit_sequence[:n_steps]):
                if t < len(hist[s]) and hist[s][t] > 0.9:
                    ax.scatter(t, hist[s][t], s=50, zorder=5,
                             color='red', alpha=0.5)
        
        plt.suptitle(f'Comparison of Trace Types (γ={gamma}, λ={lambda_})',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # 打印分析
        # Print analysis
        print("\n关键差异 Key Differences:")
        print("-" * 40)
        print("""
        1. 累积迹 Accumulating:
           - 重复访问时迹值累加
             Trace accumulates on revisits
           - 可能超过1
             Can exceed 1
           - 适合随机环境
             Good for stochastic environments
        
        2. 替换迹 Replacing:
           - 重复访问时迹重置为1
             Trace resets to 1 on revisits  
           - 最大值为1
             Maximum value is 1
           - 适合确定性环境
             Good for deterministic environments
        
        3. Dutch迹:
           - 介于累积和替换之间
             Between accumulating and replacing
           - 考虑学习率α
             Considers learning rate α
           - 理论性质更好
             Better theoretical properties
        """)
        
        return fig


# ================================================================================
# 主函数：演示资格迹基础
# Main Function: Demonstrate Eligibility Trace Foundations
# ================================================================================

def demonstrate_eligibility_traces():
    """
    演示资格迹基础概念
    Demonstrate eligibility trace foundations
    """
    print("\n" + "="*80)
    print("第6.1节：资格迹基础")
    print("Section 6.1: Eligibility Trace Foundations")
    print("="*80)
    
    # 1. 基本资格迹操作
    # 1. Basic eligibility trace operations
    print("\n" + "="*60)
    print("1. 基本资格迹操作")
    print("1. Basic Eligibility Trace Operations")
    print("="*60)
    
    trace = EligibilityTrace(gamma=0.9, lambda_=0.8)
    
    # 模拟状态序列
    # Simulate state sequence
    state_sequence = ['A', 'B', 'C', 'B', 'D']
    
    print(f"\n状态访问序列: {' → '.join(state_sequence)}")
    print(f"State visit sequence: {' → '.join(state_sequence)}")
    
    for step, state in enumerate(state_sequence):
        trace.update(state)
        print(f"\nStep {step+1}: Visit {state}")
        print(f"  Active traces: {dict((s, f'{v:.3f}') for s, v in trace.traces.items())}")
    
    trace.analyze_traces()
    
    # 2. λ-回报演示
    # 2. Lambda-return demonstration
    print("\n" + "="*60)
    print("2. λ-回报计算")
    print("2. Lambda-Return Computation")
    print("="*60)
    
    LambdaReturn.demonstrate_lambda_effect()
    
    # 3. 前向-后向等价性
    # 3. Forward-backward equivalence
    print("\n" + "="*60)
    print("3. 前向与后向视角等价性")
    print("3. Forward-Backward View Equivalence")
    print("="*60)
    
    ForwardBackwardEquivalence.demonstrate_equivalence()
    
    # 4. 不同迹类型比较
    # 4. Different trace types comparison
    print("\n" + "="*60)
    print("4. 不同迹类型可视化")
    print("4. Visualization of Different Trace Types")
    print("="*60)
    
    fig1 = TraceVisualizer.visualize_trace_types()
    
    # 5. 迹演化可视化
    # 5. Trace evolution visualization
    print("\n" + "="*60)
    print("5. 资格迹动态演化")
    print("5. Dynamic Evolution of Eligibility Traces")
    print("="*60)
    
    # 创建示例数据
    # Create example data
    trace_history = {
        'A': [1.0, 0.72, 0.52, 0.37, 0.27, 1.19, 0.86, 0.62, 0.45, 0.32],
        'B': [0.0, 1.0, 0.72, 0.52, 0.37, 0.27, 1.19, 0.86, 0.62, 0.45],
        'C': [0.0, 0.0, 1.0, 0.72, 0.52, 0.37, 0.27, 1.19, 0.86, 0.62]
    }
    td_errors = [0.5, -0.2, 0.8, -0.1, 0.3, 0.6, -0.4, 0.2, 0.1, -0.3]
    
    fig2 = TraceVisualizer.plot_trace_evolution(trace_history, td_errors)
    
    print("\n" + "="*80)
    print("资格迹基础演示完成！")
    print("Eligibility Trace Foundations Demo Complete!")
    print("="*80)
    
    print("""
    关键要点 Key Takeaways:
    ======================
    
    1. 资格迹是信用分配的记忆机制
       Eligibility traces are memory mechanism for credit assignment
       
    2. λ控制TD和MC之间的权衡
       λ controls tradeoff between TD and MC
       
    3. 前向视角理论优雅，后向视角计算高效
       Forward view is theoretically elegant, backward view is computationally efficient
       
    4. 不同迹类型适合不同问题
       Different trace types suit different problems
       
    5. TD(λ)统一了整个TD家族
       TD(λ) unifies the entire TD family
    """)
    
    plt.show()


if __name__ == "__main__":
    demonstrate_eligibility_traces()