"""
================================================================================
第1章：多臂赌博机问题 - 探索与利用的核心矛盾
Chapter 1: Multi-Armed Bandits - The Core Dilemma of Exploration vs Exploitation
================================================================================

本文件对应教材：Sutton & Barto《强化学习导论》第二版 - 第1章
This file corresponds to: Sutton & Barto "RL: An Introduction" 2nd Ed - Chapter 1

学习目标 Learning Objectives:
1. 理解多臂赌博机问题的本质
2. 掌握探索与利用的权衡
3. 实现各种行动选择算法
4. 理解不同算法的优缺点和适用场景

================================================================================
第1.1节：什么是多臂赌博机问题？
Section 1.1: What is the Multi-Armed Bandit Problem?
================================================================================

想象你在赌场面对一排老虎机（单臂强盗）：
- 每台机器的赢钱概率不同（但你不知道）
- 你有有限的游戏次数
- 目标：最大化总收益

这就是多臂赌博机问题的本质！

Imagine you're in a casino facing a row of slot machines:
- Each machine has different winning probabilities (but you don't know)
- You have limited plays
- Goal: Maximize total winnings

This is the essence of the multi-armed bandit problem!

现实应用 Real-world Applications:
1. 广告投放：选择哪个广告获得最多点击
2. 医疗试验：选择哪种治疗方案最有效
3. 推荐系统：推荐哪些内容用户最喜欢
4. A/B测试：选择哪个版本的产品更好
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置绘图风格
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


# ================================================================================
# 第1.2节：多臂赌博机问题的数学定义
# Section 1.2: Mathematical Definition of Multi-Armed Bandit Problem
# ================================================================================

class BanditProblemDefinition:
    """
    多臂赌博机问题的形式化定义
    Formal Definition of Multi-Armed Bandit Problem
    
    这不是完整的强化学习问题，因为：
    1. 只有一个状态（没有状态转移）
    2. 动作不影响后续状态
    3. 每个动作的奖励分布是独立的
    
    This is not a full RL problem because:
    1. Only one state (no state transitions)
    2. Actions don't affect subsequent states
    3. Reward distributions are independent for each action
    
    但它是理解探索vs利用的完美起点！
    But it's the perfect starting point for understanding exploration vs exploitation!
    """
    
    @staticmethod
    def explain_problem():
        """
        详细解释多臂赌博机问题
        Detailed explanation of the multi-armed bandit problem
        """
        print("\n" + "="*80)
        print("多臂赌博机问题定义")
        print("Multi-Armed Bandit Problem Definition")
        print("="*80)
        
        print("""
        数学定义 Mathematical Definition:
        ---------------------------------
        
        设有k个动作（臂）：A = {a₁, a₂, ..., aₖ}
        Given k actions (arms): A = {a₁, a₂, ..., aₖ}
        
        每个动作有未知的期望奖励：
        Each action has unknown expected reward:
        q*(a) = E[R_t | A_t = a]
        
        目标：最大化累积奖励
        Goal: Maximize cumulative reward
        Σ_{t=1}^T R_t
        
        核心挑战 Core Challenge:
        ------------------------
        
        探索 Exploration:
        - 尝试不同的动作以了解它们的价值
        - Try different actions to learn their values
        - 短期损失，长期收益
        - Short-term loss, long-term gain
        
        利用 Exploitation:
        - 选择当前认为最好的动作
        - Choose the currently best-known action
        - 短期收益，可能错过更好的选择
        - Short-term gain, might miss better options
        
        遗憾 Regret:
        -----------
        
        遗憾是选择次优动作造成的损失：
        Regret is the loss from choosing suboptimal actions:
        
        L_T = T·max_a q*(a) - Σ_{t=1}^T q*(A_t)
        
        其中：
        - T: 总时间步数
        - max_a q*(a): 最优动作的价值
        - q*(A_t): 实际选择动作的价值
        
        好的算法应该使遗憾亚线性增长：L_T = o(T)
        Good algorithms should have sublinear regret: L_T = o(T)
        """)
        
        # 可视化探索vs利用
        print("\n探索vs利用的权衡 Exploration vs Exploitation Trade-off:")
        print("-" * 60)
        
        # 创建示例场景
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 左图：纯利用
        ax1 = axes[0]
        steps = np.arange(100)
        pure_exploit = np.cumsum(np.random.normal(0.5, 0.2, 100))  # 次优臂
        ax1.plot(steps, pure_exploit, 'b-', label='Pure Exploitation')
        ax1.plot(steps, steps * 0.8, 'r--', label='Optimal (if known)')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Cumulative Reward')
        ax1.set_title('纯利用策略 Pure Exploitation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右图：探索+利用
        ax2 = axes[1]
        explore_exploit = []
        current_sum = 0
        for t in range(100):
            if t < 20:  # 前20步探索
                reward = np.random.normal(0.3, 0.2)  # 探索期间较低奖励
            else:  # 之后利用
                reward = np.random.normal(0.8, 0.2)  # 发现了更好的臂
            current_sum += reward
            explore_exploit.append(current_sum)
        
        ax2.plot(steps, explore_exploit, 'g-', label='Explore then Exploit')
        ax2.plot(steps, steps * 0.8, 'r--', label='Optimal (if known)')
        ax2.axvline(x=20, color='orange', linestyle=':', label='Switch Point')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Cumulative Reward')
        ax2.set_title('探索后利用策略 Explore then Exploit')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# ================================================================================
# 第1.3节：多臂赌博机环境实现
# Section 1.3: Multi-Armed Bandit Environment Implementation
# ================================================================================

class MultiArmedBandit:
    """
    多臂赌博机环境
    Multi-Armed Bandit Environment
    
    这是最基础的强化学习环境，特点是：
    1. 单状态：不存在状态转移
    2. 即时奖励：动作立即产生奖励
    3. 独立性：每次拉杆相互独立
    
    This is the most basic RL environment, characterized by:
    1. Single state: No state transitions
    2. Immediate rewards: Actions produce immediate rewards
    3. Independence: Each pull is independent
    """
    
    def __init__(self, k: int = 10, 
                 stationary: bool = True,
                 reward_type: str = 'gaussian',
                 seed: Optional[int] = None):
        """
        初始化k臂赌博机
        Initialize k-armed bandit
        
        Args:
            k: 臂的数量 Number of arms
            stationary: 是否平稳（奖励分布不变）Whether stationary
            reward_type: 奖励类型 ('gaussian', 'bernoulli', 'uniform')
            seed: 随机种子 Random seed
        
        深入理解：
        - 平稳问题：奖励分布不随时间变化，适合长期学习
        - 非平稳问题：奖励分布会变化，需要持续探索
        
        Deep Understanding:
        - Stationary: Reward distributions don't change, suitable for long-term learning
        - Non-stationary: Distributions change, requiring continuous exploration
        """
        self.k = k  # 臂的数量 Number of arms
        self.stationary = stationary  # 是否平稳 Whether stationary
        self.reward_type = reward_type  # 奖励类型 Reward type
        
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
        
        # 初始化真实的动作价值（期望奖励）
        # Initialize true action values (expected rewards)
        if reward_type == 'gaussian':
            # 高斯分布：q*(a) ~ N(0, 1)
            # Gaussian: q*(a) ~ N(0, 1)
            self.q_true = np.random.randn(k)
            self.optimal_action = np.argmax(self.q_true)
            logger.info(f"初始化{k}臂高斯赌博机")
            logger.info(f"真实价值: {self.q_true}")
            logger.info(f"最优动作: {self.optimal_action} (价值={self.q_true[self.optimal_action]:.3f})")
            
        elif reward_type == 'bernoulli':
            # 伯努利分布：q*(a) ~ Uniform(0, 1)
            # Bernoulli: q*(a) ~ Uniform(0, 1)
            self.q_true = np.random.uniform(0, 1, k)
            self.optimal_action = np.argmax(self.q_true)
            logger.info(f"初始化{k}臂伯努利赌博机")
            
        elif reward_type == 'uniform':
            # 均匀分布：奖励在[q*(a)-0.5, q*(a)+0.5]
            # Uniform: rewards in [q*(a)-0.5, q*(a)+0.5]
            self.q_true = np.random.uniform(-3, 3, k)
            self.optimal_action = np.argmax(self.q_true)
            logger.info(f"初始化{k}臂均匀分布赌博机")
        
        # 记录统计信息
        self.action_counts = np.zeros(k)  # 每个臂被拉的次数
        self.total_reward = 0  # 总奖励
        self.step_count = 0  # 总步数
        
        # 非平稳问题的漂移参数
        self.drift_std = 0.01  # 每步漂移的标准差
        
    def step(self, action: int) -> float:
        """
        执行动作（拉臂）并返回奖励
        Execute action (pull arm) and return reward
        
        Args:
            action: 选择的臂 Selected arm
            
        Returns:
            reward: 获得的奖励 Obtained reward
            
        数学原理：
        奖励是从以q*(a)为中心的分布中采样：
        R ~ Distribution(q*(a))
        
        Mathematical Principle:
        Reward is sampled from distribution centered at q*(a):
        R ~ Distribution(q*(a))
        """
        assert 0 <= action < self.k, f"无效动作: {action}"
        
        # 更新统计
        self.action_counts[action] += 1
        self.step_count += 1
        
        # 生成奖励
        if self.reward_type == 'gaussian':
            # R ~ N(q*(a), 1)
            reward = np.random.randn() + self.q_true[action]
            
        elif self.reward_type == 'bernoulli':
            # R ~ Bernoulli(q*(a))
            reward = float(np.random.random() < self.q_true[action])
            
        elif self.reward_type == 'uniform':
            # R ~ Uniform[q*(a)-0.5, q*(a)+0.5]
            reward = np.random.uniform(
                self.q_true[action] - 0.5,
                self.q_true[action] + 0.5
            )
        
        self.total_reward += reward
        
        # 非平稳问题：随机游走
        # Non-stationary: random walk
        if not self.stationary:
            self.q_true += np.random.randn(self.k) * self.drift_std
            self.optimal_action = np.argmax(self.q_true)
        
        return reward
    
    def reset(self):
        """
        重置环境
        Reset environment
        """
        self.action_counts = np.zeros(self.k)
        self.total_reward = 0
        self.step_count = 0
        
        # 如果是非平稳问题，重新初始化真实价值
        if not self.stationary:
            if self.reward_type == 'gaussian':
                self.q_true = np.random.randn(self.k)
            elif self.reward_type == 'bernoulli':
                self.q_true = np.random.uniform(0, 1, self.k)
            elif self.reward_type == 'uniform':
                self.q_true = np.random.uniform(-3, 3, self.k)
            
            self.optimal_action = np.argmax(self.q_true)
    
    def get_optimal_action(self) -> int:
        """获取最优动作 Get optimal action"""
        return self.optimal_action
    
    def get_true_values(self) -> np.ndarray:
        """获取真实价值 Get true values"""
        return self.q_true.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取环境统计信息
        Get environment statistics
        """
        return {
            'action_counts': self.action_counts.copy(),
            'action_proportions': self.action_counts / max(1, self.step_count),
            'total_reward': self.total_reward,
            'average_reward': self.total_reward / max(1, self.step_count),
            'optimal_action_proportion': self.action_counts[self.optimal_action] / max(1, self.step_count),
            'step_count': self.step_count
        }
    
    def visualize_bandit(self):
        """
        可视化赌博机的真实价值分布
        Visualize true value distribution of the bandit
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 左图：真实价值
        ax1 = axes[0]
        colors = ['green' if i == self.optimal_action else 'blue' for i in range(self.k)]
        bars = ax1.bar(range(self.k), self.q_true, color=colors)
        ax1.set_xlabel('Action / 动作')
        ax1.set_ylabel('True Value q*(a) / 真实价值')
        ax1.set_title('True Action Values / 真实动作价值')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.grid(True, alpha=0.3)
        
        # 标记最优动作
        ax1.scatter(self.optimal_action, self.q_true[self.optimal_action], 
                   color='red', s=100, marker='*', zorder=5,
                   label=f'Optimal (a*={self.optimal_action})')
        ax1.legend()
        
        # 右图：拉动次数分布
        ax2 = axes[1]
        if self.step_count > 0:
            proportions = self.action_counts / self.step_count
            bars = ax2.bar(range(self.k), proportions, color=colors)
            ax2.set_xlabel('Action / 动作')
            ax2.set_ylabel('Selection Proportion / 选择比例')
            ax2.set_title(f'Action Selection Distribution (n={self.step_count}) / 动作选择分布')
            ax2.set_ylim([0, 1])
            
            # 添加最优动作比例文本
            optimal_prop = self.action_counts[self.optimal_action] / self.step_count
            ax2.text(0.5, 0.95, f'Optimal Action Proportion: {optimal_prop:.2%}',
                    transform=ax2.transAxes, ha='center', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax2.text(0.5, 0.5, 'No actions taken yet / 还未采取动作',
                    transform=ax2.transAxes, ha='center', va='center')
        
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# ================================================================================
# 第1.4节：动作价值估计方法
# Section 1.4: Action-Value Estimation Methods
# ================================================================================

class ActionValueEstimation:
    """
    动作价值估计的核心方法
    Core Methods for Action-Value Estimation
    
    这是所有赌博机算法的基础：
    如何从观察到的奖励估计每个动作的真实价值？
    
    This is the foundation of all bandit algorithms:
    How to estimate true value of each action from observed rewards?
    """
    
    @staticmethod
    def explain_estimation_methods():
        """
        解释不同的价值估计方法
        Explain different value estimation methods
        """
        print("\n" + "="*80)
        print("动作价值估计方法")
        print("Action-Value Estimation Methods")
        print("="*80)
        
        print("""
        1. 样本平均法 Sample-Average Method
        ------------------------------------
        
        最直观的估计方法：
        The most intuitive estimation method:
        
        Q_t(a) = (R₁ + R₂ + ... + R_{N_t(a)}) / N_t(a)
        
        其中 Where:
        - Q_t(a): 时刻t对动作a的价值估计 / Value estimate of action a at time t
        - N_t(a): 到时刻t动作a被选择的次数 / Number of times a selected until t
        - R_i: 第i次选择动作a获得的奖励 / Reward from i-th selection of a
        
        根据大数定律 By Law of Large Numbers:
        Q_t(a) → q*(a) as N_t(a) → ∞
        
        2. 增量更新公式 Incremental Update Rule
        ----------------------------------------
        
        避免存储所有历史奖励：
        Avoid storing all historical rewards:
        
        Q_{n+1} = Q_n + (1/n)[R_n - Q_n]
        
        一般形式 General form:
        NewEstimate = OldEstimate + StepSize[Target - OldEstimate]
        新估计 = 旧估计 + 步长[目标 - 旧估计]
        
        这个公式是强化学习的核心模式！
        This formula is the core pattern in RL!
        
        3. 指数衰减加权平均 Exponentially Weighted Average
        -------------------------------------------------
        
        对于非平稳问题，使用固定步长：
        For non-stationary problems, use fixed step size:
        
        Q_{n+1} = Q_n + α[R_n - Q_n]
        
        展开后 Expanding:
        Q_{n+1} = (1-α)ⁿQ₁ + Σᵢ₌₁ⁿ α(1-α)^{n-i}R_i
        
        特点 Properties:
        - 最近的奖励权重更大 / Recent rewards have higher weight
        - 权重指数衰减 / Weights decay exponentially
        - 适合跟踪变化 / Suitable for tracking changes
        
        4. 初始值的影响 Effect of Initial Values
        ----------------------------------------
        
        Q₁(a) 的选择很重要：
        Choice of Q₁(a) is important:
        
        - 乐观初始值 Optimistic: Q₁(a) = 5 (当q*(a) ∈ [-1,1])
          → 鼓励探索 Encourages exploration
          
        - 悲观初始值 Pessimistic: Q₁(a) = -5
          → 减少探索 Reduces exploration
          
        - 现实初始值 Realistic: Q₁(a) = 0
          → 平衡的行为 Balanced behavior
        """)
    
    @staticmethod
    def incremental_update(old_estimate: float, 
                          new_sample: float, 
                          step_size: float) -> float:
        """
        增量更新规则
        Incremental update rule
        
        这是强化学习中最重要的更新模式
        This is the most important update pattern in RL
        
        Args:
            old_estimate: 旧的估计值 Q_n
            new_sample: 新的样本（奖励）R_n
            step_size: 步长 α 或 1/n
            
        Returns:
            new_estimate: 新的估计值 Q_{n+1}
        """
        error = new_sample - old_estimate  # 预测误差 Prediction error
        new_estimate = old_estimate + step_size * error
        return new_estimate
    
    @staticmethod
    def compute_step_size(n: int, 
                          mode: str = 'sample_average',
                          alpha: float = 0.1) -> float:
        """
        计算步长
        Compute step size
        
        Args:
            n: 当前是第n次更新
            mode: 'sample_average' 或 'constant'
            alpha: 固定步长值（用于constant模式）
            
        Returns:
            步长值 Step size value
        """
        if mode == 'sample_average':
            # 样本平均：步长 = 1/n
            return 1.0 / n if n > 0 else 1.0
        elif mode == 'constant':
            # 固定步长：适合非平稳问题
            return alpha
        else:
            raise ValueError(f"未知模式: {mode}")
    
    @staticmethod
    def demonstrate_convergence():
        """
        演示价值估计的收敛过程
        Demonstrate convergence of value estimation
        """
        print("\n演示：价值估计收敛 Demonstration: Value Estimation Convergence")
        print("-" * 60)
        
        # 模拟参数
        true_value = 1.5  # 真实价值
        n_samples = 1000  # 样本数
        noise_std = 1.0  # 噪声标准差
        
        # 生成奖励样本
        rewards = np.random.normal(true_value, noise_std, n_samples)
        
        # 不同方法的估计
        estimates_sample_avg = []
        estimates_constant_01 = []
        estimates_constant_001 = []
        
        Q_sa = 0  # 样本平均估计
        Q_c01 = 0  # 固定步长0.1估计
        Q_c001 = 0  # 固定步长0.01估计
        
        for n, r in enumerate(rewards, 1):
            # 样本平均
            Q_sa = ActionValueEstimation.incremental_update(Q_sa, r, 1/n)
            estimates_sample_avg.append(Q_sa)
            
            # 固定步长0.1
            Q_c01 = ActionValueEstimation.incremental_update(Q_c01, r, 0.1)
            estimates_constant_01.append(Q_c01)
            
            # 固定步长0.01
            Q_c001 = ActionValueEstimation.incremental_update(Q_c001, r, 0.01)
            estimates_constant_001.append(Q_c001)
        
        # 绘图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        steps = np.arange(1, n_samples + 1)
        ax.plot(steps, estimates_sample_avg, 'b-', label='Sample Average (1/n)', alpha=0.7)
        ax.plot(steps, estimates_constant_01, 'g-', label='Constant α=0.1', alpha=0.7)
        ax.plot(steps, estimates_constant_001, 'r-', label='Constant α=0.01', alpha=0.7)
        ax.axhline(y=true_value, color='black', linestyle='--', label='True Value')
        
        ax.set_xlabel('Steps / 步数')
        ax.set_ylabel('Value Estimate / 价值估计')
        ax.set_title('Convergence of Different Estimation Methods / 不同估计方法的收敛')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加说明文本
        ax.text(0.02, 0.98, 
               f'样本平均：收敛到真值，但速度变慢\n'
               f'α=0.1：快速响应，但有振荡\n'
               f'α=0.01：平滑但响应慢',
               transform=ax.transAxes,
               fontsize=9,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig


# ================================================================================
# 第1.5节：基础赌博机智能体
# Section 1.5: Basic Bandit Agent
# ================================================================================

class BaseBanditAgent(ABC):
    """
    赌博机智能体基类
    Base class for bandit agents
    
    所有赌博机算法都需要：
    1. 维护价值估计 Q(a)
    2. 选择动作（探索vs利用）
    3. 更新价值估计
    
    All bandit algorithms need to:
    1. Maintain value estimates Q(a)
    2. Select actions (exploration vs exploitation)
    3. Update value estimates
    """
    
    def __init__(self, k: int, 
                 initial_value: float = 0.0,
                 step_size_mode: str = 'sample_average',
                 alpha: float = 0.1):
        """
        初始化智能体
        Initialize agent
        
        Args:
            k: 动作数量 Number of actions
            initial_value: 初始价值估计 Initial value estimates
            step_size_mode: 步长模式 'sample_average' or 'constant'
            alpha: 固定步长值 Fixed step size
        """
        self.k = k
        self.initial_value = initial_value
        self.step_size_mode = step_size_mode
        self.alpha = alpha
        
        # 初始化价值估计和计数
        self.reset()
        
        logger.info(f"初始化{self.__class__.__name__}: k={k}, "
                   f"initial_value={initial_value}, "
                   f"step_size_mode={step_size_mode}")
    
    def reset(self):
        """
        重置智能体状态
        Reset agent state
        """
        # 价值估计 Q(a)
        self.Q = np.ones(self.k) * self.initial_value
        
        # 动作计数 N(a)
        self.N = np.zeros(self.k)
        
        # 历史记录
        self.action_history = []
        self.reward_history = []
        self.Q_history = [self.Q.copy()]
    
    @abstractmethod
    def select_action(self) -> int:
        """
        选择动作（需要子类实现）
        Select action (to be implemented by subclasses)
        """
        pass
    
    def update(self, action: int, reward: float):
        """
        更新价值估计
        Update value estimate
        
        使用增量更新规则：
        Q(a) ← Q(a) + α[R - Q(a)]
        
        Args:
            action: 执行的动作
            reward: 获得的奖励
        """
        # 更新计数
        self.N[action] += 1
        
        # 计算步长
        if self.step_size_mode == 'sample_average':
            step_size = 1.0 / self.N[action]
        else:  # constant
            step_size = self.alpha
        
        # 增量更新
        self.Q[action] += step_size * (reward - self.Q[action])
        
        # 记录历史
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.Q_history.append(self.Q.copy())
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取智能体统计信息
        Get agent statistics
        """
        if len(self.reward_history) == 0:
            return {
                'average_reward': 0,
                'total_reward': 0,
                'best_action': np.argmax(self.Q),
                'Q_values': self.Q.copy(),
                'action_counts': self.N.copy()
            }
        
        return {
            'average_reward': np.mean(self.reward_history),
            'total_reward': np.sum(self.reward_history),
            'best_action': np.argmax(self.Q),
            'Q_values': self.Q.copy(),
            'action_counts': self.N.copy(),
            'action_proportions': self.N / len(self.action_history)
        }
    
    def run_episode(self, env: MultiArmedBandit, n_steps: int) -> Dict[str, List]:
        """
        运行一个回合
        Run one episode
        
        Args:
            env: 赌博机环境
            n_steps: 步数
            
        Returns:
            包含奖励、动作、遗憾等信息的字典
        """
        # 重置
        env.reset()
        self.reset()
        
        # 记录
        rewards = []
        actions = []
        optimal_actions = []
        regrets = []
        
        # 运行
        for step in range(n_steps):
            # 选择动作
            action = self.select_action()
            
            # 执行动作
            reward = env.step(action)
            
            # 更新
            self.update(action, reward)
            
            # 记录
            rewards.append(reward)
            actions.append(action)
            optimal_actions.append(action == env.get_optimal_action())
            
            # 计算遗憾（累积）
            optimal_value = env.get_true_values()[env.get_optimal_action()]
            actual_value = env.get_true_values()[action]
            regret = optimal_value - actual_value
            if step == 0:
                regrets.append(regret)
            else:
                regrets.append(regrets[-1] + regret)
        
        return {
            'rewards': rewards,
            'actions': actions,
            'optimal_actions': optimal_actions,
            'regrets': regrets,
            'Q_history': self.Q_history
        }


# ================================================================================
# 主函数：演示基础概念
# Main Function: Demonstrate Basic Concepts
# ================================================================================

def demonstrate_chapter1_basics():
    """
    演示第1章基础概念
    Demonstrate Chapter 1 basic concepts
    """
    print("\n" + "="*80)
    print("第1章：多臂赌博机 - 基础概念演示")
    print("Chapter 1: Multi-Armed Bandits - Basic Concepts Demo")
    print("="*80)
    
    # 1. 问题定义
    BanditProblemDefinition.explain_problem()
    fig1 = BanditProblemDefinition.explain_problem()
    
    # 2. 环境演示
    print("\n" + "="*80)
    print("多臂赌博机环境演示")
    print("Multi-Armed Bandit Environment Demo")
    print("="*80)
    
    # 创建10臂赌博机
    bandit = MultiArmedBandit(k=10, stationary=True, seed=42)
    
    print(f"\n创建了10臂赌博机")
    print(f"最优动作: {bandit.get_optimal_action()}")
    print(f"最优价值: {bandit.get_true_values()[bandit.get_optimal_action()]:.3f}")
    
    # 随机拉几次
    print("\n随机拉动10次：")
    for i in range(10):
        action = np.random.randint(10)
        reward = bandit.step(action)
        print(f"  动作{action} -> 奖励{reward:.3f}")
    
    # 可视化
    fig2 = bandit.visualize_bandit()
    
    # 3. 价值估计方法
    ActionValueEstimation.explain_estimation_methods()
    fig3 = ActionValueEstimation.demonstrate_convergence()
    
    print("\n" + "="*80)
    print("基础概念演示完成！")
    print("Basic Concepts Demo Complete!")
    print("\n接下来我们将实现具体的算法：")
    print("1. ε-贪婪算法")
    print("2. UCB（置信上界）算法")
    print("3. 梯度赌博机算法")
    print("="*80)
    
    return [fig1, fig2, fig3]


if __name__ == "__main__":
    """
    运行第1章基础概念演示
    Run Chapter 1 basic concepts demo
    """
    figures = demonstrate_chapter1_basics()
    plt.show()