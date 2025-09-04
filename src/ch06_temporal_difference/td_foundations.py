"""
================================================================================
第5章：时序差分学习 - 强化学习的核心
Chapter 5: Temporal-Difference Learning - The Core of RL
================================================================================

TD学习是强化学习最重要的思想！
TD learning is the most important idea in RL!

TD = Monte Carlo + Dynamic Programming
- 像MC：不需要模型，从经验学习
  Like MC: Model-free, learn from experience
- 像DP：自举（bootstrap），不需要完整回合
  Like DP: Bootstrap, no need for complete episodes

核心创新 Core Innovation:
使用估计值更新估计值！
Use estimates to update estimates!

TD误差（TD Error）:
δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)

这个简单的公式革命性地改变了强化学习！
This simple formula revolutionized RL!

为什么TD如此重要？
Why is TD so important?
1. 在线学习（每步都能学习）
   Online learning (learn at every step)
2. 不需要完整回合
   No need for complete episodes
3. 低方差（比MC）
   Lower variance (than MC)
4. 收敛保证（在某些条件下）
   Convergence guarantees (under conditions)

TD方法家族 TD Method Family:
- TD(0): 一步TD，最基本
  One-step TD, most basic
- SARSA: On-policy TD控制
  On-policy TD control
- Q-learning: Off-policy TD控制（最著名！）
  Off-policy TD control (most famous!)
- Expected SARSA: SARSA的改进
  Improvement of SARSA
- n-step TD: 多步TD，介于TD和MC之间
  Multi-step TD, between TD and MC
- TD(λ): 资格迹，统一所有方法
  Eligibility traces, unifies all methods

历史意义 Historical Significance:
Q-learning (Watkins, 1989) 是深度强化学习的基础
Q-learning is the foundation of Deep RL
- DQN = Q-learning + Deep Neural Networks
- 开启了现代AI革命！
  Started the modern AI revolution!

本章将深入理解TD的数学原理和实现细节
This chapter deeply understands TD's mathematical principles and implementation details
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import logging
import matplotlib.pyplot as plt
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

# 设置日志
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第5.1节：TD学习的数学基础
# Section 5.1: Mathematical Foundation of TD Learning
# ================================================================================

class TDTheory:
    """
    TD学习理论
    TD Learning Theory
    
    深入理解TD的数学原理
    Deep understanding of TD mathematical principles
    
    核心思想：增量式贝尔曼方程
    Core idea: Incremental Bellman equation
    
    贝尔曼方程：
    Bellman equation:
    V^π(s) = E_π[R_{t+1} + γV^π(S_{t+1}) | S_t = s]
    
    TD更新（将期望变为采样）：
    TD update (expectation to sampling):
    V(S_t) ← V(S_t) + α[R_{t+1} + γV(S_{t+1}) - V(S_t)]
                           ↑________________↑
                              TD target    预测值
                                          prediction
    
    TD误差/优势 TD error/advantage:
    δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)
    
    这是：
    This is:
    - 预测误差 Prediction error
    - 时序差分 Temporal difference  
    - Bellman误差的样本 Sample of Bellman error
    - 优势函数的估计 Estimate of advantage function
    
    为什么叫"时序差分"？
    Why called "Temporal Difference"?
    因为它是两个时刻价值估计的差！
    Because it's the difference between value estimates at two time points!
    
    V(S_{t+1})在t+1时刻，V(S_t)在t时刻
    V(S_{t+1}) at time t+1, V(S_t) at time t
    """
    
    @staticmethod
    def explain_td_vs_mc_vs_dp():
        """
        详解TD vs MC vs DP
        Detailed explanation of TD vs MC vs DP
        
        三种方法的本质区别
        Essential differences of three methods
        """
        print("\n" + "="*80)
        print("TD vs MC vs DP 深度对比")
        print("TD vs MC vs DP Deep Comparison")
        print("="*80)
        
        print("""
        📊 三种方法的更新公式
        Update Formulas of Three Methods
        =================================
        
        1. 动态规划 Dynamic Programming (DP):
        ------------------------------------------
        V(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γV(s')]
        
        特点 Characteristics:
        - 需要完整模型 p(s',r|s,a)
          Needs complete model
        - 全宽度更新（考虑所有可能）
          Full-width update (consider all possibilities)
        - 精确但计算量大
          Exact but computationally expensive
        
        2. 蒙特卡洛 Monte Carlo (MC):
        ------------------------------------------
        V(S_t) ← V(S_t) + α[G_t - V(S_t)]
        
        其中 where: G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...
        
        特点 Characteristics:
        - 不需要模型
          Model-free
        - 使用真实回报G_t
          Uses actual return G_t
        - 必须等到回合结束
          Must wait until episode ends
        - 无偏但高方差
          Unbiased but high variance
        
        3. 时序差分 Temporal Difference (TD):
        ------------------------------------------
        V(S_t) ← V(S_t) + α[R_{t+1} + γV(S_{t+1}) - V(S_t)]
        
        TD target: R_{t+1} + γV(S_{t+1})
        
        特点 Characteristics:
        - 不需要模型
          Model-free
        - 使用估计值V(S_{t+1})
          Uses estimate V(S_{t+1})
        - 每步都可以学习
          Can learn at every step
        - 有偏但低方差
          Biased but low variance
        
        🎯 关键洞察 Key Insights
        ========================
        
        1. Bootstrap（自举）:
        -------------------
        DP: ✓ (使用V(s')更新V(s))
        MC: ✗ (使用真实G_t)
        TD: ✓ (使用V(S_{t+1}))
        
        Bootstrap = 用估计更新估计
        Bootstrap = Update estimate with estimate
        
        2. Sampling（采样）:
        -------------------
        DP: ✗ (考虑所有转移)
        MC: ✓ (采样完整轨迹)
        TD: ✓ (采样单步转移)
        
        Sampling = 用样本代替期望
        Sampling = Replace expectation with samples
        
        3. 更新时机 Update Timing:
        --------------------------
        DP: 任意（通常sweep所有状态）
             Arbitrary (usually sweep all states)
        MC: 回合结束
             Episode end
        TD: 每一步
             Every step
        
        4. 偏差-方差权衡 Bias-Variance Tradeoff:
        ----------------------------------------
        方差 Variance: MC > TD > DP
        偏差 Bias:     DP = 0, TD > 0 (初期), MC = 0
        
        MC高方差因为G_t包含整条轨迹的随机性
        MC high variance because G_t contains randomness of entire trajectory
        
        TD低方差因为只有一步随机性
        TD low variance because only one step randomness
        
        TD有偏因为使用有偏的估计V(S_{t+1})
        TD biased because uses biased estimate V(S_{t+1})
        
        5. 收敛性 Convergence:
        ----------------------
        DP: 总是收敛到V^π
             Always converges to V^π
        MC: 收敛到V^π（足够探索）
             Converges to V^π (sufficient exploration)  
        TD: 收敛到V^π（线性近似+递减步长）
             Converges to V^π (linear approx + decreasing stepsize)
        
        6. 效率 Efficiency:
        -------------------
        数据效率 Data efficiency: TD > MC
        (TD每步学习，MC等回合结束)
        (TD learns every step, MC waits for episode end)
        
        计算效率 Computational: TD ≈ MC >> DP
        (DP需要遍历所有状态)
        (DP needs to sweep all states)
        
        内存效率 Memory: TD ≈ MC > DP
        (DP需要存储完整模型)
        (DP needs to store complete model)
        
        🔑 TD的优势 TD's Advantages
        ============================
        
        1. 在线学习 Online Learning:
           可以在与环境交互时立即学习
           Can learn immediately during interaction
           
        2. 不完整回合 Incomplete Episodes:
           可以学习持续性任务
           Can learn continuing tasks
           
        3. 低方差 Low Variance:
           比MC更稳定
           More stable than MC
           
        4. 计算简单 Simple Computation:
           只需要简单的增量更新
           Only needs simple incremental update
           
        5. 生物学合理性 Biological Plausibility:
           类似多巴胺神经元的预测误差信号
           Similar to dopamine neuron prediction error signal
        """)
    
    @staticmethod
    def demonstrate_td_convergence():
        """
        演示TD收敛性
        Demonstrate TD convergence
        
        展示TD如何收敛到真实价值
        Show how TD converges to true values
        """
        print("\n" + "="*80)
        print("TD收敛性演示")
        print("TD Convergence Demonstration")
        print("="*80)
        
        # 创建简单马尔可夫链
        # Create simple Markov chain
        print("\n简单马尔可夫奖励过程 Simple Markov Reward Process:")
        print("""
        A → B → C → D → E → [终止]
        0   0   0   0   1
        
        只有到达E才有奖励+1
        Only reward +1 at E
        
        真实价值（γ=1）True values (γ=1):
        V(A)=0.5, V(B)=0.5, V(C)=0.5, V(D)=0.5, V(E)=1.0
        
        因为从任何状态，50%概率向右，50%概率终止
        Because from any state, 50% right, 50% terminate
        """)
        
        # 模拟TD学习
        # Simulate TD learning
        states = ['A', 'B', 'C', 'D', 'E']
        true_values = {'A': 0.5, 'B': 0.5, 'C': 0.5, 'D': 0.5, 'E': 1.0}
        
        # TD(0)学习
        # TD(0) learning
        V = {s: 0.0 for s in states}  # 初始化为0
        alpha = 0.1  # 学习率
        gamma = 1.0  # 无折扣
        
        episodes = 100
        np.random.seed(42)
        
        # 记录学习过程
        # Record learning process
        history = {s: [] for s in states}
        
        print("\n开始TD(0)学习...")
        print("Starting TD(0) learning...")
        
        for episode in range(episodes):
            # 生成轨迹（简化：总是从A开始）
            # Generate trajectory (simplified: always start from A)
            trajectory = ['A']
            current = 'A'
            
            while current != 'E':
                if np.random.random() < 0.5:
                    # 终止
                    # Terminate
                    break
                else:
                    # 向右移动
                    # Move right
                    idx = states.index(current)
                    if idx < len(states) - 1:
                        current = states[idx + 1]
                        trajectory.append(current)
            
            # TD更新（沿轨迹）
            # TD update (along trajectory)
            for i in range(len(trajectory) - 1):
                s = trajectory[i]
                s_next = trajectory[i + 1]
                
                # 奖励（只有到E才有）
                # Reward (only at E)
                r = 1.0 if s_next == 'E' else 0.0
                
                # TD误差
                # TD error
                td_error = r + gamma * V[s_next] - V[s]
                
                # 更新
                # Update
                V[s] += alpha * td_error
            
            # 记录
            # Record
            for s in states:
                history[s].append(V[s])
            
            if episode % 20 == 0:
                print(f"Episode {episode}: ", end="")
                for s in states[:3]:  # 显示前3个状态
                    print(f"V({s})={V[s]:.3f} ", end="")
                print()
        
        # 可视化收敛过程
        # Visualize convergence process
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, s in enumerate(states):
            ax = axes[i]
            
            # 学习曲线
            # Learning curve
            ax.plot(history[s], 'b-', alpha=0.7, label='TD estimate')
            ax.axhline(y=true_values[s], color='r', linestyle='--', 
                      label=f'True value = {true_values[s]}')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Value Estimate')
            ax.set_title(f'State {s}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 最后一个子图：收敛误差
        # Last subplot: Convergence error
        ax = axes[5]
        errors = []
        for ep in range(episodes):
            error = sum((history[s][ep] - true_values[s])**2 for s in states)
            errors.append(np.sqrt(error / len(states)))  # RMSE
        
        ax.plot(errors, 'g-', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('RMSE')
        ax.set_title('Convergence Error')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('TD(0) Convergence to True Values', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        print("\n最终估计 vs 真实价值:")
        print("Final estimates vs true values:")
        print("-" * 40)
        for s in states:
            error = abs(V[s] - true_values[s])
            print(f"{s}: TD={V[s]:.3f}, True={true_values[s]:.3f}, Error={error:.3f}")
        
        return fig


# ================================================================================  
# 第5.2节：TD误差和优势函数
# Section 5.2: TD Error and Advantage Function
# ================================================================================

@dataclass
class TDError:
    """
    TD误差分析
    TD Error Analysis
    
    TD误差δ是强化学习最重要的信号！
    TD error δ is the most important signal in RL!
    
    数学定义 Mathematical Definition:
    δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)
    
    多种解释 Multiple Interpretations:
    
    1. 预测误差 Prediction Error:
       实际发生的 vs 预期的
       What happened vs What expected
       
    2. 优势 Advantage:
       A^π(s,a) ≈ δ when following π
       这个动作比平均好多少
       How much better this action than average
       
    3. 学习信号 Learning Signal:
       正δ → 提高V(S_t)
       Positive δ → Increase V(S_t)
       负δ → 降低V(S_t)  
       Negative δ → Decrease V(S_t)
       
    4. 神经科学 Neuroscience:
       多巴胺神经元编码TD误差！
       Dopamine neurons encode TD error!
       奖励预测误差假说(Schultz et al., 1997)
       Reward prediction error hypothesis
       
    TD误差的性质 Properties of TD Error:
    
    1. 期望为0（收敛后）：
       E[δ_t | S_t=s] = 0 when V = V^π
       
    2. 与优势函数的关系：
       E[δ_t | S_t=s, A_t=a] = Q^π(s,a) - V^π(s) = A^π(s,a)
       
    3. 贝尔曼残差：
       δ是贝尔曼方程误差的无偏估计
       δ is unbiased estimate of Bellman equation error
    """
    
    # TD误差值
    # TD error value
    value: float
    
    # 时间步
    # Time step
    timestep: int
    
    # 相关状态
    # Related states
    state: State
    next_state: Optional[State] = None
    
    # 奖励和价值
    # Reward and values
    reward: float = 0.0
    state_value: float = 0.0
    next_state_value: float = 0.0
    
    # 其他信息
    # Other info
    info: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        return f"TDError(δ={self.value:.3f}, t={self.timestep})"
    
    def analyze(self):
        """
        分析TD误差
        Analyze TD error
        
        提供诊断信息
        Provide diagnostic information
        """
        print(f"\nTD误差分析 TD Error Analysis:")
        print(f"时间步 Timestep: {self.timestep}")
        print(f"TD误差值 TD Error Value: {self.value:.3f}")
        
        if self.value > 0:
            print("  → 正误差：实际比预期好")
            print("     Positive: Better than expected")
            print("  → 应该增加V(S_t)")
            print("     Should increase V(S_t)")
        elif self.value < 0:
            print("  → 负误差：实际比预期差")
            print("     Negative: Worse than expected")  
            print("  → 应该减少V(S_t)")
            print("     Should decrease V(S_t)")
        else:
            print("  → 零误差：完美预测")
            print("     Zero: Perfect prediction")
        
        print(f"\n分解 Decomposition:")
        print(f"  R_{{{self.timestep+1}}} = {self.reward:.3f}")
        print(f"  γV(S_{{{self.timestep+1}}}) = {self.next_state_value:.3f}")
        print(f"  V(S_{{{self.timestep}}}) = {self.state_value:.3f}")
        print(f"  δ = {self.reward:.3f} + {self.next_state_value:.3f} - {self.state_value:.3f}")
        print(f"    = {self.value:.3f}")


class TDErrorAnalyzer:
    """
    TD误差分析器
    TD Error Analyzer
    
    收集和分析TD误差模式
    Collect and analyze TD error patterns
    
    用于：
    Used for:
    1. 调试算法
       Debug algorithms
    2. 监控收敛
       Monitor convergence  
    3. 发现问题
       Discover issues
    """
    
    def __init__(self, window_size: int = 100):
        """
        初始化分析器
        Initialize analyzer
        
        Args:
            window_size: 滑动窗口大小
                        Sliding window size
        """
        self.window_size = window_size
        
        # TD误差历史
        # TD error history
        self.errors: List[TDError] = []
        
        # 滑动窗口
        # Sliding window
        self.recent_errors = deque(maxlen=window_size)
        
        # 统计
        # Statistics
        self.total_errors = 0
        self.sum_errors = 0.0
        self.sum_squared_errors = 0.0
        
        logger.info(f"初始化TD误差分析器，窗口大小={window_size}")
    
    def add_error(self, td_error: TDError):
        """
        添加TD误差
        Add TD error
        """
        self.errors.append(td_error)
        self.recent_errors.append(td_error.value)
        
        self.total_errors += 1
        self.sum_errors += td_error.value
        self.sum_squared_errors += td_error.value ** 2
    
    def get_statistics(self) -> Dict[str, float]:
        """
        获取统计信息
        Get statistics
        
        Returns:
            统计字典
            Statistics dict
        """
        if self.total_errors == 0:
            return {}
        
        mean = self.sum_errors / self.total_errors
        variance = self.sum_squared_errors / self.total_errors - mean ** 2
        std = np.sqrt(variance) if variance > 0 else 0
        
        # 最近的统计
        # Recent statistics
        if self.recent_errors:
            recent_mean = np.mean(self.recent_errors)
            recent_std = np.std(self.recent_errors)
            recent_abs_mean = np.mean(np.abs(self.recent_errors))
        else:
            recent_mean = recent_std = recent_abs_mean = 0
        
        return {
            'total_errors': self.total_errors,
            'mean': mean,
            'std': std,
            'recent_mean': recent_mean,
            'recent_std': recent_std,
            'recent_abs_mean': recent_abs_mean,
            'convergence_metric': recent_abs_mean  # 越小越收敛
        }
    
    def plot_analysis(self):
        """
        绘制分析图
        Plot analysis
        
        可视化TD误差模式
        Visualize TD error patterns
        """
        if not self.errors:
            print("没有TD误差数据")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 图1：TD误差时间序列
        # Plot 1: TD error time series
        ax1 = axes[0, 0]
        error_values = [e.value for e in self.errors]
        ax1.plot(error_values, 'b-', alpha=0.5, linewidth=0.5)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('TD Error')
        ax1.set_title('TD Error Over Time')
        ax1.grid(True, alpha=0.3)
        
        # 添加移动平均
        # Add moving average
        if len(error_values) > 20:
            window = min(50, len(error_values) // 10)
            moving_avg = np.convolve(error_values, 
                                     np.ones(window)/window, 
                                     mode='valid')
            ax1.plot(range(window-1, len(error_values)), 
                    moving_avg, 'r-', linewidth=2, 
                    label=f'Moving Avg (w={window})')
            ax1.legend()
        
        # 图2：TD误差分布
        # Plot 2: TD error distribution  
        ax2 = axes[0, 1]
        ax2.hist(error_values, bins=50, density=True, 
                alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        
        # 拟合正态分布
        # Fit normal distribution
        mean, std = np.mean(error_values), np.std(error_values)
        x = np.linspace(min(error_values), max(error_values), 100)
        ax2.plot(x, stats.norm.pdf(x, mean, std), 'r-', linewidth=2,
                label=f'Normal(μ={mean:.3f}, σ={std:.3f})')
        ax2.set_xlabel('TD Error')
        ax2.set_ylabel('Density')
        ax2.set_title('TD Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 图3：绝对TD误差（收敛指标）
        # Plot 3: Absolute TD error (convergence metric)
        ax3 = axes[1, 0]
        abs_errors = np.abs(error_values)
        
        # 计算滑动平均
        # Compute sliding average
        window = min(100, len(abs_errors) // 10) if len(abs_errors) > 10 else 1
        abs_moving_avg = np.convolve(abs_errors,
                                     np.ones(window)/window,
                                     mode='valid')
        
        ax3.plot(range(window-1, len(abs_errors)), 
                abs_moving_avg, 'g-', linewidth=2)
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Mean Absolute TD Error')
        ax3.set_title('Convergence Metric (|δ|)')
        ax3.grid(True, alpha=0.3)
        
        # 图4：TD误差自相关
        # Plot 4: TD error autocorrelation
        ax4 = axes[1, 1]
        if len(error_values) > 50:
            from scipy.signal import correlate
            # 计算自相关
            # Compute autocorrelation
            autocorr = correlate(error_values[:1000], error_values[:1000], mode='same')
            autocorr = autocorr / np.max(autocorr)  # 归一化
            center = len(autocorr) // 2
            lags = 50
            ax4.plot(range(-lags, lags+1), 
                    autocorr[center-lags:center+lags+1], 'b-')
            ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax4.set_xlabel('Lag')
            ax4.set_ylabel('Autocorrelation')
            ax4.set_title('TD Error Autocorrelation')
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle('TD Error Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # 打印统计摘要
        # Print statistical summary
        stats_dict = self.get_statistics()
        print("\nTD误差统计摘要 TD Error Statistics Summary:")
        print("-" * 40)
        for key, value in stats_dict.items():
            print(f"{key}: {value:.4f}")
        
        return fig


# ================================================================================
# 第5.3节：TD(0)算法实现
# Section 5.3: TD(0) Algorithm Implementation
# ================================================================================

class TD0:
    """
    TD(0)算法 - 最基本的TD方法
    TD(0) Algorithm - Most basic TD method
    
    也叫一步TD (One-step TD)
    Also called one-step TD
    
    更新规则 Update rule:
    V(S_t) ← V(S_t) + α[R_{t+1} + γV(S_{t+1}) - V(S_t)]
    
    算法步骤 Algorithm steps:
    1. 初始化V(s)任意，V(terminal)=0
       Initialize V(s) arbitrarily, V(terminal)=0
    2. 重复每个回合：
       Repeat for each episode:
       a. 初始化S
          Initialize S
       b. 重复每步：
          Repeat for each step:
          - 选择并执行动作A（根据策略π）
            Choose and execute action A (according to π)
          - 观察R和S'
            Observe R and S'
          - V(S) ← V(S) + α[R + γV(S') - V(S)]
          - S ← S'
       直到S是终止状态
       Until S is terminal
    
    特性 Properties:
    1. 在线 Online: 每步更新
                   Update every step
    2. 增量 Incremental: 不存储历史
                        No history storage
    3. 模型无关 Model-free: 不需要P和R
                           No need for P and R
    4. 自举 Bootstrapping: 用V(S')估计
                          Use V(S') estimate
    
    收敛条件 Convergence conditions:
    1. 策略π固定
       Policy π fixed
    2. 步长满足Robbins-Monro条件：
       Step size satisfies Robbins-Monro:
       Σα_t = ∞, Σα_t² < ∞
    3. 所有状态被无限访问
       All states visited infinitely
    
    则V收敛到V^π (概率1)
    Then V converges to V^π (with probability 1)
    """
    
    def __init__(self, 
                 env: MDPEnvironment,
                 gamma: float = 1.0,
                 alpha: Union[float, Callable] = 0.1):
        """
        初始化TD(0)
        Initialize TD(0)
        
        Args:
            env: 环境
                Environment
            gamma: 折扣因子
                  Discount factor
            alpha: 学习率（固定或函数）
                  Learning rate (fixed or function)
        """
        self.env = env
        self.gamma = gamma
        
        # 学习率（可以是常数或递减函数）
        # Learning rate (can be constant or decreasing function)
        if callable(alpha):
            self.alpha_func = alpha
        else:
            self.alpha_func = lambda t: alpha
        
        # 价值函数
        # Value function
        self.V = StateValueFunction(env.state_space, initial_value=0.0)
        
        # TD误差分析器
        # TD error analyzer
        self.td_analyzer = TDErrorAnalyzer()
        
        # 统计
        # Statistics
        self.episode_count = 0
        self.step_count = 0
        self.episode_returns = []
        
        logger.info(f"初始化TD(0): γ={gamma}, α={alpha}")
    
    def learn_episode(self, policy: Policy) -> float:
        """
        学习一个回合
        Learn one episode
        
        Args:
            policy: 要评估的策略
                   Policy to evaluate
        
        Returns:
            回合回报
            Episode return
        """
        state = self.env.reset()
        episode_return = 0.0
        episode_steps = 0
        
        while True:
            # 选择动作
            # Select action
            action = policy.select_action(state)
            
            # 执行动作
            # Execute action
            next_state, reward, done, info = self.env.step(action)
            
            # TD更新
            # TD update
            if not state.is_terminal:
                # 获取当前学习率
                # Get current learning rate
                alpha = self.alpha_func(self.step_count)
                
                # 计算TD误差
                # Compute TD error
                v_current = self.V.get_value(state)
                v_next = self.V.get_value(next_state) if not done else 0.0
                td_error = reward + self.gamma * v_next - v_current
                
                # 更新价值函数
                # Update value function
                new_value = v_current + alpha * td_error
                self.V.set_value(state, new_value)
                
                # 记录TD误差
                # Record TD error
                td_err_obj = TDError(
                    value=td_error,
                    timestep=self.step_count,
                    state=state,
                    next_state=next_state,
                    reward=reward,
                    state_value=v_current,
                    next_state_value=v_next
                )
                self.td_analyzer.add_error(td_err_obj)
            
            # 累积回报
            # Accumulate return
            episode_return += reward * (self.gamma ** episode_steps)
            
            # 更新计数
            # Update counts
            self.step_count += 1
            episode_steps += 1
            
            # 转移到下一状态
            # Transition to next state
            state = next_state
            
            if done:
                break
        
        self.episode_count += 1
        self.episode_returns.append(episode_return)
        
        return episode_return
    
    def learn(self, 
             policy: Policy,
             n_episodes: int = 1000,
             verbose: bool = True) -> StateValueFunction:
        """
        学习价值函数
        Learn value function
        
        Args:
            policy: 策略
                   Policy
            n_episodes: 回合数
                       Number of episodes
            verbose: 是否输出进度
                    Whether to output progress
        
        Returns:
            学习的价值函数
            Learned value function
        """
        if verbose:
            print(f"\n开始TD(0)学习: {n_episodes}回合")
            print(f"Starting TD(0) learning: {n_episodes} episodes")
        
        for episode in range(n_episodes):
            episode_return = self.learn_episode(policy)
            
            if verbose and (episode + 1) % max(1, n_episodes // 10) == 0:
                stats = self.td_analyzer.get_statistics()
                avg_return = np.mean(self.episode_returns[-100:]) if self.episode_returns else 0
                
                print(f"Episode {episode + 1}/{n_episodes}: "
                      f"Return={episode_return:.2f}, "
                      f"Avg Return={avg_return:.2f}, "
                      f"TD Error={stats.get('recent_mean', 0):.4f}")
        
        if verbose:
            print(f"\n学习完成!")
            print(f"Learning complete!")
            
            # 显示最终统计
            # Show final statistics
            stats = self.td_analyzer.get_statistics()
            print(f"最终TD误差: {stats.get('recent_abs_mean', 0):.4f}")
            print(f"Final TD error: {stats.get('recent_abs_mean', 0):.4f}")
        
        return self.V
    
    def compare_with_mc(self, mc_values: StateValueFunction):
        """
        与MC比较
        Compare with MC
        
        展示TD vs MC的差异
        Show differences between TD and MC
        
        Args:
            mc_values: MC估计的价值函数
                      MC estimated value function
        """
        print("\n" + "="*60)
        print("TD(0) vs Monte Carlo 比较")
        print("TD(0) vs Monte Carlo Comparison")
        print("="*60)
        
        # 计算差异
        # Compute differences
        differences = []
        
        for state in self.env.state_space:
            if not state.is_terminal:
                td_value = self.V.get_value(state)
                mc_value = mc_values.get_value(state)
                diff = abs(td_value - mc_value)
                differences.append((state.id, td_value, mc_value, diff))
        
        # 按差异排序
        # Sort by difference
        differences.sort(key=lambda x: x[3], reverse=True)
        
        print("\n价值估计比较（前10个差异最大的状态）：")
        print("Value estimate comparison (top 10 largest differences):")
        print("-" * 60)
        print(f"{'State':<15} {'TD(0)':<10} {'MC':<10} {'|Diff|':<10}")
        print("-" * 60)
        
        for state_id, td_val, mc_val, diff in differences[:10]:
            print(f"{str(state_id):<15} {td_val:<10.3f} {mc_val:<10.3f} {diff:<10.3f}")
        
        # 统计分析
        # Statistical analysis
        all_diffs = [d[3] for d in differences]
        
        print("\n统计分析 Statistical Analysis:")
        print("-" * 40)
        print(f"平均绝对差异 Mean Absolute Difference: {np.mean(all_diffs):.4f}")
        print(f"最大差异 Max Difference: {np.max(all_diffs):.4f}")
        print(f"差异标准差 Std of Differences: {np.std(all_diffs):.4f}")
        
        # 分析原因
        # Analyze reasons
        print("\n差异原因分析 Difference Analysis:")
        print("-" * 40)
        print("""
        TD(0)和MC的差异来源于：
        Differences between TD(0) and MC come from:
        
        1. Bootstrap vs Full Return:
           TD: 使用V(S')的估计
               Uses estimate of V(S')
           MC: 使用真实的G_t
               Uses actual G_t
        
        2. Bias vs Variance:
           TD: 初期有偏，但方差小
               Initially biased, but low variance
           MC: 无偏，但方差大
               Unbiased, but high variance
        
        3. Update Frequency:
           TD: 每步更新，信息传播快
               Update every step, fast propagation
           MC: 回合结束更新，信息传播慢
               Update at episode end, slow propagation
        
        4. Convergence Path:
           不同的路径收敛到相同的V^π
           Different paths converge to same V^π
        """)


# ================================================================================
# 主函数：演示TD基础
# Main Function: Demonstrate TD Foundations
# ================================================================================

def demonstrate_td_foundations():
    """
    演示TD学习基础
    Demonstrate TD learning foundations
    """
    print("\n" + "="*80)
    print("第5章：时序差分学习 - 基础理论")
    print("Chapter 5: Temporal-Difference Learning - Foundations")
    print("="*80)
    
    # 1. TD vs MC vs DP对比
    # 1. TD vs MC vs DP comparison
    TDTheory.explain_td_vs_mc_vs_dp()
    
    # 2. TD收敛性演示
    # 2. TD convergence demonstration
    fig1 = TDTheory.demonstrate_td_convergence()
    
    # 3. 在GridWorld上测试TD(0)
    # 3. Test TD(0) on GridWorld
    print("\n" + "="*80)
    print("TD(0)在GridWorld上的实验")
    print("TD(0) Experiment on GridWorld")
    print("="*80)
    
    from src.ch03_finite_mdp.gridworld import GridWorld
    from src.ch03_finite_mdp.policies_and_values import UniformRandomPolicy
    
    # 创建环境
    # Create environment
    env = GridWorld(rows=4, cols=4, start_pos=(0,0), goal_pos=(3,3))
    print(f"创建4×4 GridWorld")
    
    # 创建随机策略
    # Create random policy
    policy = UniformRandomPolicy(env.action_space)
    
    # TD(0)学习
    # TD(0) learning
    td0 = TD0(env, gamma=0.9, alpha=0.1)
    V_td = td0.learn(policy, n_episodes=1000, verbose=True)
    
    # 分析TD误差
    # Analyze TD errors
    fig2 = td0.td_analyzer.plot_analysis()
    
    # 显示学习的价值函数
    # Show learned value function
    print("\n学习的状态价值（部分）：")
    print("Learned state values (partial):")
    for i in range(min(5, len(env.state_space))):
        state = env.state_space[i]
        if not state.is_terminal:
            value = V_td.get_value(state)
            print(f"  V({state.id}) = {value:.3f}")
    
    print("\n" + "="*80)
    print("TD基础演示完成！")
    print("TD Foundation Demo Complete!")
    print("="*80)
    
    plt.show()


# ================================================================================
# 执行主函数
# Execute Main Function  
# ================================================================================

if __name__ == "__main__":
    demonstrate_td_foundations()