"""
================================================================================
第1.8节：梯度赌博机算法 - 基于偏好的学习
Section 1.8: Gradient Bandit Algorithm - Preference-Based Learning
================================================================================

梯度赌博机算法不估计动作价值，而是学习动作偏好
Gradient bandit doesn't estimate action values, but learns action preferences

核心思想 Core Idea:
使用softmax策略，通过随机梯度上升优化期望奖励
Use softmax policy and optimize expected reward via stochastic gradient ascent

这个算法展示了策略梯度方法的雏形！
This algorithm shows the prototype of policy gradient methods!
"""

import numpy as np
from typing import Optional, Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging

from .bandit_introduction import BaseBanditAgent, MultiArmedBandit

# 设置日志
logger = logging.getLogger(__name__)


# ================================================================================
# 第1.8.1节：梯度赌博机原理
# Section 1.8.1: Gradient Bandit Principle
# ================================================================================

class GradientBanditPrinciple:
    """
    梯度赌博机算法原理
    Gradient Bandit Algorithm Principle
    
    这是策略梯度方法的最简单形式
    This is the simplest form of policy gradient methods
    """
    
    @staticmethod
    def explain_principle():
        """
        详解梯度赌博机原理
        Detailed Explanation of Gradient Bandit Principle
        """
        print("\n" + "="*80)
        print("梯度赌博机算法原理")
        print("Gradient Bandit Algorithm Principle")
        print("="*80)
        
        print("""
        1. 核心概念：偏好而非价值
        Core Concept: Preferences, Not Values
        ----------------------------------------
        
        不同于之前的算法，梯度赌博机：
        Unlike previous algorithms, gradient bandit:
        
        • 不估计动作价值 Q(a)
          Doesn't estimate action values Q(a)
        • 维护动作偏好 H(a) ∈ ℝ
          Maintains action preferences H(a) ∈ ℝ
        • 使用softmax将偏好转换为概率
          Uses softmax to convert preferences to probabilities
        
        动作选择概率 Action selection probability:
        
        π_t(a) = P(A_t = a) = exp(H_t(a)) / Σ_b exp(H_t(b))
        
        这就是softmax策略！
        This is the softmax policy!
        
        2. 梯度上升优化
        Gradient Ascent Optimization
        -----------------------------
        
        目标：最大化期望奖励
        Goal: Maximize expected reward
        
        J = E[R_t] = Σ_a π_t(a)·q*(a)
        
        使用随机梯度上升：
        Using stochastic gradient ascent:
        
        H_{t+1}(a) = H_t(a) + α·∂E[R_t]/∂H_t(a)
        
        梯度推导（REINFORCE算法的雏形）：
        Gradient derivation (prototype of REINFORCE):
        
        ∂E[R_t]/∂H_t(a) = E[(R_t - baseline)·(𝟙_{A_t=a} - π_t(a))]
        
        其中 Where:
        - R_t: 时刻t的奖励 Reward at time t
        - baseline: 基线（通常是平均奖励）Baseline (usually average reward)
        - 𝟙_{A_t=a}: 指示函数 Indicator function
        - π_t(a): 动作a的概率 Probability of action a
        
        3. 更新规则
        Update Rule
        -----------
        
        对于选中的动作 A_t：
        For selected action A_t:
        
        H_{t+1}(A_t) = H_t(A_t) + α(R_t - R̄_t)(1 - π_t(A_t))
        
        对于其他动作 a ≠ A_t：
        For other actions a ≠ A_t:
        
        H_{t+1}(a) = H_t(a) - α(R_t - R̄_t)π_t(a)
        
        其中R̄_t是基线（平均奖励）
        Where R̄_t is the baseline (average reward)
        
        4. 直观理解
        Intuitive Understanding
        -----------------------
        
        • 如果奖励 > 基线：
          If reward > baseline:
          - 增加选中动作的偏好
            Increase preference for selected action
          - 减少其他动作的偏好
            Decrease preference for other actions
            
        • 如果奖励 < 基线：
          If reward < baseline:
          - 减少选中动作的偏好
            Decrease preference for selected action
          - 增加其他动作的偏好
            Increase preference for other actions
        
        5. 为什么使用基线？
        Why Use a Baseline?
        -------------------
        
        基线减少方差，加速学习：
        Baseline reduces variance and speeds up learning:
        
        Var[gradient] with baseline < Var[gradient] without baseline
        
        最优基线：E[R_t²]/E[R_t]
        Optimal baseline: E[R_t²]/E[R_t]
        
        实践中使用移动平均：
        In practice, use moving average:
        R̄_{t+1} = R̄_t + β(R_t - R̄_t)
        """)
        
        # 可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 左图：偏好到概率的转换（softmax）
        ax1 = axes[0]
        H_values = np.linspace(-3, 3, 100)
        for temp in [0.5, 1.0, 2.0]:
            # 模拟3个动作的情况
            H = np.array([H_values, np.zeros_like(H_values), -np.ones_like(H_values)])
            exp_H = np.exp(H / temp)
            probs = exp_H[0] / np.sum(exp_H, axis=0)
            ax1.plot(H_values, probs, label=f'τ={temp}', alpha=0.8)
        
        ax1.set_xlabel('Preference H(a) / 偏好')
        ax1.set_ylabel('Probability π(a) / 概率')
        ax1.set_title('Softmax: Preference to Probability / Softmax：偏好到概率')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # 中图：梯度更新方向
        ax2 = axes[1]
        rewards = np.linspace(-2, 2, 100)
        baseline = 0
        gradient_selected = (rewards - baseline) * (1 - 0.3)  # π(a)=0.3
        gradient_others = -(rewards - baseline) * 0.3
        
        ax2.plot(rewards, gradient_selected, 'b-', label='Selected Action', linewidth=2)
        ax2.plot(rewards, gradient_others, 'r-', label='Other Actions', linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.axvline(x=baseline, color='green', linestyle='--', label='Baseline')
        ax2.fill_between(rewards[rewards > baseline], 0, gradient_selected[rewards > baseline],
                         alpha=0.3, color='blue', label='Increase preference')
        ax2.fill_between(rewards[rewards < baseline], 0, gradient_selected[rewards < baseline],
                         alpha=0.3, color='red', label='Decrease preference')
        
        ax2.set_xlabel('Reward R_t / 奖励')
        ax2.set_ylabel('Gradient ∇H / 梯度')
        ax2.set_title('Gradient Direction / 梯度方向')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 右图：基线的作用
        ax3 = axes[2]
        n_samples = 1000
        rewards_no_baseline = np.random.normal(1, 2, n_samples)
        rewards_with_baseline = rewards_no_baseline - np.mean(rewards_no_baseline)
        
        # 计算梯度的方差
        grad_no_baseline = rewards_no_baseline * (1 - 0.3)
        grad_with_baseline = rewards_with_baseline * (1 - 0.3)
        
        data = [grad_no_baseline, grad_with_baseline]
        labels = ['No Baseline', 'With Baseline']
        
        bp = ax3.boxplot(data, labels=labels, patch_artist=True)
        colors = ['lightcoral', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax3.set_ylabel('Gradient Value / 梯度值')
        ax3.set_title('Baseline Reduces Variance / 基线减少方差')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 添加方差文本
        var_no_baseline = np.var(grad_no_baseline)
        var_with_baseline = np.var(grad_with_baseline)
        ax3.text(0.5, 0.95, f'Var(no baseline) = {var_no_baseline:.2f}\n'
                           f'Var(with baseline) = {var_with_baseline:.2f}\n'
                           f'Reduction = {(1-var_with_baseline/var_no_baseline)*100:.1f}%',
                transform=ax3.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig


# ================================================================================
# 第1.8.2节：梯度赌博机实现
# Section 1.8.2: Gradient Bandit Implementation
# ================================================================================

class GradientBanditAgent(BaseBanditAgent):
    """
    梯度赌博机智能体
    Gradient Bandit Agent
    
    使用softmax策略和梯度上升
    Uses softmax policy and gradient ascent
    """
    
    def __init__(self, k: int = None,
                 n_arms: int = None,
                 alpha: float = 0.1,
                 use_baseline: bool = True,
                 baseline_alpha: float = 0.1,
                 temperature: float = 1.0,
                 **kwargs):
        """
        初始化梯度赌博机
        Initialize gradient bandit
        
        Args:
            k: 动作数量 Number of actions
            alpha: 学习率 Learning rate
            use_baseline: 是否使用基线 Whether to use baseline
            baseline_alpha: 基线更新率 Baseline update rate
            temperature: Softmax温度参数 Softmax temperature
            **kwargs: 传递给父类的参数
        
        深入理解 Deep Understanding:
        
        1. 偏好初始化：
           Preference initialization:
           - H(a) = 0: 所有动作概率相等 All actions equally likely
           - H(a) ~ N(0,σ): 打破对称性 Break symmetry
        
        2. 温度参数：
           Temperature parameter:
           - τ < 1: 更确定的选择 More deterministic
           - τ = 1: 标准softmax Standard softmax
           - τ > 1: 更随机的选择 More random
        
        3. 基线选择：
           Baseline choice:
           - 0: 简单但次优 Simple but suboptimal
           - 平均奖励: 常用选择 Common choice
           - 加权平均: 更精确 More accurate
        """
        # 处理n_arms参数（向后兼容）
        if n_arms is not None:
            k = n_arms
        if k is None:
            raise ValueError("必须提供k或n_arms参数")
            
        # 注意：梯度赌博机不使用Q值，所以传递特殊参数给父类
        super().__init__(k, initial_value=0.0, **kwargs)
        
        self.alpha = alpha
        self.use_baseline = use_baseline
        self.baseline_alpha = baseline_alpha
        self.temperature = temperature
        
        # 动作偏好（不是价值！）
        # Action preferences (not values!)
        self.H = np.zeros(k)
        
        # 基线（平均奖励）
        # Baseline (average reward)
        self.baseline = 0.0
        
        # 动作概率
        # Action probabilities
        self.pi = np.ones(k) / k
        
        # 统计
        self.total_steps = 0
        
        logger.info(f"初始化梯度赌博机: k={k}, α={alpha}, "
                   f"baseline={use_baseline}, τ={temperature}")
    
    def _compute_softmax(self) -> np.ndarray:
        """
        计算softmax概率
        Compute softmax probabilities
        
        使用数值稳定的实现
        Use numerically stable implementation
        
        Returns:
            动作概率分布 Action probability distribution
        """
        # 数值稳定性：减去最大值
        # Numerical stability: subtract maximum
        H_stable = self.H - np.max(self.H)
        
        # 应用温度参数
        # Apply temperature parameter
        exp_H = np.exp(H_stable / self.temperature)
        
        # 归一化得到概率
        # Normalize to get probabilities
        self.pi = exp_H / np.sum(exp_H)
        
        return self.pi
    
    def select_action(self) -> int:
        """
        使用softmax策略选择动作
        Select action using softmax policy
        
        Returns:
            选择的动作 Selected action
        """
        # 计算动作概率
        # Compute action probabilities
        probabilities = self._compute_softmax()
        
        # 按概率选择动作
        # Select action according to probabilities
        action = np.random.choice(self.k, p=probabilities)
        
        self.total_steps += 1
        
        logger.debug(f"Step {self.total_steps}: "
                    f"Selected action {action} with prob {probabilities[action]:.3f}")
        
        return action
    
    def update(self, action: int, reward: float):
        """
        使用梯度上升更新偏好
        Update preferences using gradient ascent
        
        这是REINFORCE算法的简化版本！
        This is a simplified version of REINFORCE!
        
        Args:
            action: 执行的动作 Action taken
            reward: 获得的奖励 Reward received
        """
        # 更新基线（如果使用）
        # Update baseline (if used)
        if self.use_baseline:
            # 指数移动平均
            # Exponential moving average
            self.baseline += self.baseline_alpha * (reward - self.baseline)
            advantage = reward - self.baseline
        else:
            advantage = reward
        
        # 梯度上升更新
        # Gradient ascent update
        
        # 对于选中的动作
        # For selected action
        self.H[action] += self.alpha * advantage * (1 - self.pi[action])
        
        # 对于其他动作
        # For other actions
        for a in range(self.k):
            if a != action:
                self.H[a] -= self.alpha * advantage * self.pi[a]
        
        # 记录（父类方法）
        # Record (parent class method)
        self.action_history.append(action)
        self.reward_history.append(reward)
        
        # 注意：梯度赌博机不更新Q值
        # Note: Gradient bandit doesn't update Q values
        self.N[action] += 1
        
        logger.debug(f"Updated H[{action}]: advantage={advantage:.3f}, "
                    f"new H={self.H[action]:.3f}")
    
    def reset(self):
        """
        重置智能体
        Reset agent
        """
        super().reset()
        self.H = np.zeros(self.k)
        self.baseline = 0.0
        self.pi = np.ones(self.k) / self.k
        self.total_steps = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        Get statistics
        """
        stats = super().get_statistics()
        stats.update({
            'preferences': self.H.copy(),
            'probabilities': self.pi.copy(),
            'baseline': self.baseline,
            'temperature': self.temperature,
            'entropy': -np.sum(self.pi * np.log(self.pi + 1e-10))  # 策略熵
        })
        return stats


# ================================================================================
# 第1.8.3节：改进的梯度赌博机变体
# Section 1.8.3: Improved Gradient Bandit Variants
# ================================================================================

class NaturalGradientBandit(GradientBanditAgent):
    """
    自然梯度赌博机
    Natural Gradient Bandit
    
    使用Fisher信息矩阵修正梯度方向
    Use Fisher information matrix to correct gradient direction
    """
    
    def __init__(self, k: int, **kwargs):
        """初始化自然梯度赌博机"""
        super().__init__(k, **kwargs)
        
        # Fisher信息矩阵的估计
        self.fisher_matrix = np.eye(k) * 0.01  # 初始化为小的对角矩阵
        self.fisher_alpha = 0.01  # Fisher矩阵更新率
    
    def update(self, action: int, reward: float):
        """
        自然梯度更新
        Natural gradient update
        
        自然梯度 = Fisher^{-1} × 普通梯度
        Natural gradient = Fisher^{-1} × ordinary gradient
        """
        # 计算普通梯度
        if self.use_baseline:
            self.baseline += self.baseline_alpha * (reward - self.baseline)
            advantage = reward - self.baseline
        else:
            advantage = reward
        
        # 构建梯度向量
        gradient = np.zeros(self.k)
        gradient[action] = advantage * (1 - self.pi[action])
        for a in range(self.k):
            if a != action:
                gradient[a] = -advantage * self.pi[a]
        
        # 更新Fisher信息矩阵（近似）
        # 对于softmax策略：F = diag(π) - ππ^T
        self.fisher_matrix = (1 - self.fisher_alpha) * self.fisher_matrix + \
                           self.fisher_alpha * (np.diag(self.pi) - np.outer(self.pi, self.pi))
        
        # 添加正则化避免奇异
        self.fisher_matrix += np.eye(self.k) * 0.01
        
        # 计算自然梯度
        try:
            natural_gradient = np.linalg.solve(self.fisher_matrix, gradient)
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用普通梯度
            natural_gradient = gradient
        
        # 更新偏好
        self.H += self.alpha * natural_gradient
        
        # 更新其他统计
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.N[action] += 1


class AdaptiveGradientBandit(GradientBanditAgent):
    """
    自适应梯度赌博机
    Adaptive Gradient Bandit
    
    自动调整学习率和温度参数
    Automatically adjust learning rate and temperature
    """
    
    def __init__(self, k: int, **kwargs):
        """初始化自适应梯度赌博机"""
        super().__init__(k, **kwargs)
        
        # 自适应参数
        self.alpha_initial = self.alpha
        self.temperature_initial = self.temperature
        
        # 性能跟踪
        self.performance_window = []
        self.window_size = 100
    
    def update(self, action: int, reward: float):
        """
        自适应更新
        Adaptive update
        """
        # 跟踪性能
        self.performance_window.append(reward)
        if len(self.performance_window) > self.window_size:
            self.performance_window.pop(0)
        
        # 根据性能调整参数
        if len(self.performance_window) >= self.window_size:
            # 计算性能趋势
            first_half = np.mean(self.performance_window[:self.window_size//2])
            second_half = np.mean(self.performance_window[self.window_size//2:])
            improvement = second_half - first_half
            
            # 调整学习率
            if improvement > 0:
                # 性能提升，可以增加学习率
                self.alpha = min(1.0, self.alpha * 1.01)
            else:
                # 性能下降，减小学习率
                self.alpha = max(0.001, self.alpha * 0.99)
            
            # 调整温度（探索程度）
            # 计算策略熵
            entropy = -np.sum(self.pi * np.log(self.pi + 1e-10))
            target_entropy = np.log(self.k) * 0.5  # 目标熵为最大熵的一半
            
            if entropy < target_entropy:
                # 熵太小，增加温度（更多探索）
                self.temperature = min(5.0, self.temperature * 1.01)
            else:
                # 熵太大，减小温度（更多利用）
                self.temperature = max(0.1, self.temperature * 0.99)
        
        # 执行标准更新
        super().update(action, reward)


class EntropyRegularizedGradientBandit(GradientBanditAgent):
    """
    熵正则化梯度赌博机
    Entropy-Regularized Gradient Bandit
    
    在目标中加入熵正则项，鼓励探索
    Add entropy regularization to encourage exploration
    
    这是软演员-评论家(SAC)算法的雏形！
    This is a prototype of Soft Actor-Critic (SAC)!
    """
    
    def __init__(self, k: int,
                 entropy_coef: float = 0.01,
                 **kwargs):
        """
        初始化熵正则化梯度赌博机
        
        Args:
            entropy_coef: 熵系数 Entropy coefficient
        """
        super().__init__(k, **kwargs)
        self.entropy_coef = entropy_coef
    
    def update(self, action: int, reward: float):
        """
        带熵正则的更新
        Update with entropy regularization
        
        目标：J = E[R] + β·H(π)
        Objective: J = E[R] + β·H(π)
        
        其中H(π)是策略熵
        Where H(π) is policy entropy
        """
        # 更新基线
        if self.use_baseline:
            self.baseline += self.baseline_alpha * (reward - self.baseline)
            advantage = reward - self.baseline
        else:
            advantage = reward
        
        # 标准梯度
        gradient = np.zeros(self.k)
        gradient[action] = advantage * (1 - self.pi[action])
        for a in range(self.k):
            if a != action:
                gradient[a] = -advantage * self.pi[a]
        
        # 熵梯度：∇H(π) = -∇Σ_a π(a)log π(a)
        # Entropy gradient
        entropy_gradient = np.zeros(self.k)
        for a in range(self.k):
            if self.pi[a] > 0:
                entropy_gradient[a] = -self.pi[a] * (np.log(self.pi[a]) + 1) * (
                    (1 if a == action else 0) - self.pi[a]
                )
        
        # 组合梯度
        # Combined gradient
        total_gradient = gradient + self.entropy_coef * entropy_gradient
        
        # 更新偏好
        self.H += self.alpha * total_gradient
        
        # 记录
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.N[action] += 1


# ================================================================================
# 第1.8.4节：梯度赌博机理论分析
# Section 1.8.4: Gradient Bandit Theoretical Analysis
# ================================================================================

class GradientBanditAnalysis:
    """
    梯度赌博机理论分析
    Gradient Bandit Theoretical Analysis
    """
    
    @staticmethod
    def convergence_analysis():
        """
        收敛性分析
        Convergence Analysis
        """
        print("\n" + "="*80)
        print("梯度赌博机收敛性分析")
        print("Gradient Bandit Convergence Analysis")
        print("="*80)
        
        print("""
        1. 梯度的无偏性
        Unbiasedness of Gradient
        ------------------------
        
        定理：梯度赌博机的更新是期望奖励的无偏梯度估计
        Theorem: Gradient bandit update is unbiased gradient estimate
        
        证明 Proof:
        E[∂J/∂H(a)] = E[(R_t - b)(𝟙_{A_t=a} - π_t(a))]
                     = Σ_a' π_t(a')[q*(a') - b][𝟙_{a'=a} - π_t(a)]
                     = π_t(a)[q*(a) - b][1 - π_t(a)] - π_t(a)Σ_{a'≠a} π_t(a')[q*(a') - b]
                     = π_t(a)[q*(a) - E_π[q*]]  (当b = E_π[q*]时)
                     = ∂E_π[q*]/∂H(a)
        
        这证明了算法确实在优化期望奖励！
        This proves the algorithm is optimizing expected reward!
        
        2. 收敛条件
        Convergence Conditions
        ----------------------
        
        Robbins-Monro条件：
        Σ_t α_t = ∞ 且 Σ_t α_t² < ∞
        
        例如 For example:
        - α_t = c/t: 满足，保证收敛 Satisfies, guarantees convergence
        - α_t = c: 不满足第二个条件 Doesn't satisfy second condition
        
        3. 收敛速度
        Convergence Rate
        ----------------
        
        在适当条件下：
        Under appropriate conditions:
        
        E[||π_t - π*||²] = O(1/t^β)
        
        其中β ∈ (0.5, 1]取决于问题结构
        Where β ∈ (0.5, 1] depends on problem structure
        
        4. 基线的作用
        Role of Baseline
        ----------------
        
        方差减少 Variance Reduction:
        
        Var[gradient with baseline] / Var[gradient without] ≈ 1 - ρ²
        
        其中ρ是奖励与最优基线的相关系数
        Where ρ is correlation between reward and optimal baseline
        
        最优基线 Optimal Baseline:
        b* = E[R_t² · π_t(A_t)] / E[π_t(A_t)] = E[R_t²] (当策略均匀时)
        
        5. 与策略梯度的联系
        Connection to Policy Gradient
        ------------------------------
        
        梯度赌博机是REINFORCE算法的特例：
        Gradient bandit is special case of REINFORCE:
        
        ∇J(θ) = E_π[(R - b)∇log π(a|θ)]
        
        对于softmax策略：
        For softmax policy:
        ∇log π(a) = ∇H(a) - E_π[∇H]
                  = e_a - π
        
        这是策略梯度方法的起点！
        This is the starting point of policy gradient methods!
        """)
    
    @staticmethod
    def demonstrate_convergence():
        """
        演示收敛过程
        Demonstrate Convergence Process
        """
        print("\n演示：梯度赌博机收敛")
        print("Demo: Gradient Bandit Convergence")
        print("-" * 60)
        
        # 创建简单的2臂赌博机
        k = 2
        true_values = np.array([0.3, 0.7])  # 真实价值
        
        # 不同配置的梯度赌博机
        configs = [
            ('With Baseline', True, 0.1),
            ('No Baseline', False, 0.1),
            ('Large α', True, 0.5),
            ('Small α', True, 0.01),
        ]
        
        n_steps = 2000
        n_runs = 100
        
        results = {}
        
        for name, use_baseline, alpha in configs:
            all_probs = []
            
            for run in range(n_runs):
                agent = GradientBanditAgent(k=k, alpha=alpha, use_baseline=use_baseline)
                probs_history = []
                
                for step in range(n_steps):
                    # 选择动作
                    agent._compute_softmax()
                    action = np.random.choice(k, p=agent.pi)
                    
                    # 获得奖励（伯努利）
                    reward = float(np.random.random() < true_values[action])
                    
                    # 更新
                    agent.update(action, reward)
                    
                    # 记录概率
                    probs_history.append(agent.pi[1])  # 记录选择更好臂的概率
                
                all_probs.append(probs_history)
            
            results[name] = np.mean(all_probs, axis=0)
        
        # 绘制结果
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 左图：收敛过程
        ax1 = axes[0]
        for name, probs in results.items():
            ax1.plot(probs, label=name, alpha=0.8)
        
        ax1.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, 
                   label='Optimal π(a*)')
        ax1.set_xlabel('Steps / 步数')
        ax1.set_ylabel('P(choosing better arm) / 选择更好臂的概率')
        ax1.set_title('Convergence to Optimal Policy / 收敛到最优策略')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # 右图：学习速度比较
        ax2 = axes[1]
        convergence_steps = {}
        threshold = 0.65  # 认为收敛的阈值
        
        for name, probs in results.items():
            # 找到首次超过阈值的步数
            converged = np.where(probs > threshold)[0]
            if len(converged) > 0:
                convergence_steps[name] = converged[0]
            else:
                convergence_steps[name] = n_steps
        
        names = list(convergence_steps.keys())
        steps = list(convergence_steps.values())
        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
        
        bars = ax2.bar(range(len(names)), steps, color=colors, alpha=0.7)
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.set_ylabel('Steps to Converge / 收敛步数')
        ax2.set_title(f'Convergence Speed (threshold={threshold}) / 收敛速度')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, step in zip(bars, steps):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{step}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        print("\n关键发现 Key Findings:")
        print("1. 基线显著加速收敛")
        print("2. 学习率影响收敛速度和稳定性")
        print("3. 大学习率快但不稳定")
        print("4. 小学习率稳定但慢")
        
        return fig


# ================================================================================
# 第1.8.5节：综合比较实验
# Section 1.8.5: Comprehensive Comparison Experiment
# ================================================================================

def compare_all_algorithms():
    """
    比较所有赌博机算法
    Compare all bandit algorithms
    """
    print("\n" + "="*80)
    print("综合算法比较")
    print("Comprehensive Algorithm Comparison")
    print("="*80)
    
    # 导入其他算法
    from .epsilon_greedy import EpsilonGreedyAgent
    from .ucb_algorithm import UCBAgent
    
    # 配置
    k = 10
    n_runs = 100
    n_steps = 1000
    
    # 所有算法
    algorithms = [
        ('ε-Greedy (ε=0.1)', EpsilonGreedyAgent(k=k, epsilon=0.1)),
        ('UCB (c=2)', UCBAgent(k=k, c=2.0)),
        ('Gradient (baseline)', GradientBanditAgent(k=k, alpha=0.1, use_baseline=True)),
        ('Gradient (no baseline)', GradientBanditAgent(k=k, alpha=0.1, use_baseline=False)),
        ('Entropy-Regularized', EntropyRegularizedGradientBandit(k=k, alpha=0.1)),
    ]
    
    # 运行实验
    results = {name: {'rewards': [], 'optimal': [], 'regrets': []} 
              for name, _ in algorithms}
    
    print("运行综合实验...")
    for run in tqdm(range(n_runs), desc="Runs"):
        env = MultiArmedBandit(k=k, seed=run)
        
        for name, agent in algorithms:
            agent.reset()
            episode_data = agent.run_episode(env, n_steps)
            
            results[name]['rewards'].append(episode_data['rewards'])
            results[name]['optimal'].append(episode_data['optimal_actions'])
            results[name]['regrets'].append(episode_data['regrets'])
            
            env.reset()
    
    # 绘制结果
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 平均奖励
    ax1 = axes[0, 0]
    for name in results:
        mean_rewards = np.mean(results[name]['rewards'], axis=0)
        ax1.plot(mean_rewards, label=name, alpha=0.8)
    ax1.set_xlabel('Steps / 步数')
    ax1.set_ylabel('Average Reward / 平均奖励')
    ax1.set_title('Average Reward Comparison / 平均奖励比较')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. 最优动作比例
    ax2 = axes[0, 1]
    for name in results:
        optimal_rate = np.mean(results[name]['optimal'], axis=0) * 100
        ax2.plot(optimal_rate, label=name, alpha=0.8)
    ax2.set_xlabel('Steps / 步数')
    ax2.set_ylabel('Optimal Action % / 最优动作百分比')
    ax2.set_title('Optimal Action Selection / 最优动作选择')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])
    
    # 3. 累积遗憾（对数尺度）
    ax3 = axes[1, 0]
    for name in results:
        mean_regrets = np.mean(results[name]['regrets'], axis=0)
        ax3.plot(mean_regrets, label=name, alpha=0.8)
    ax3.set_xlabel('Steps (log scale) / 步数（对数尺度）')
    ax3.set_ylabel('Cumulative Regret / 累积遗憾')
    ax3.set_title('Cumulative Regret (Log Scale) / 累积遗憾（对数尺度）')
    ax3.set_xscale('log')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. 最终性能总结
    ax4 = axes[1, 1]
    
    # 计算最终指标
    final_metrics = {}
    for name in results:
        final_reward = np.mean([np.mean(r[-100:]) for r in results[name]['rewards']])
        final_optimal = np.mean([np.mean(o[-100:]) for o in results[name]['optimal']]) * 100
        final_regret = np.mean([r[-1] for r in results[name]['regrets']])
        final_metrics[name] = {
            'reward': final_reward,
            'optimal': final_optimal,
            'regret': final_regret
        }
    
    # 创建雷达图
    categories = ['Reward\n(normalized)', 'Optimal %\n(normalized)', 'Low Regret\n(normalized)']
    
    # 归一化指标（0-1）
    max_reward = max(m['reward'] for m in final_metrics.values())
    min_reward = min(m['reward'] for m in final_metrics.values())
    max_regret = max(m['regret'] for m in final_metrics.values())
    min_regret = min(m['regret'] for m in final_metrics.values())
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    ax4 = plt.subplot(2, 2, 4, projection='polar')
    
    for i, (name, metrics) in enumerate(final_metrics.items()):
        # 归一化值
        norm_reward = (metrics['reward'] - min_reward) / (max_reward - min_reward) if max_reward > min_reward else 0.5
        norm_optimal = metrics['optimal'] / 100
        norm_regret = 1 - (metrics['regret'] - min_regret) / (max_regret - min_regret) if max_regret > min_regret else 0.5
        
        values = [norm_reward, norm_optimal, norm_regret]
        values += values[:1]  # 闭合
        
        ax4.plot(angles, values, 'o-', linewidth=2, label=name, alpha=0.7)
        ax4.fill(angles, values, alpha=0.1)
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories)
    ax4.set_ylim(0, 1)
    ax4.set_title('Overall Performance / 整体性能', y=1.08)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax4.grid(True)
    
    plt.tight_layout()
    
    # 打印总结
    print("\n算法性能总结 Algorithm Performance Summary:")
    print("=" * 80)
    print(f"{'Algorithm':25s} {'Final Reward':>15s} {'Optimal %':>15s} {'Total Regret':>15s}")
    print("-" * 80)
    for name, metrics in final_metrics.items():
        print(f"{name:25s} {metrics['reward']:>15.3f} {metrics['optimal']:>14.1f}% {metrics['regret']:>15.1f}")
    
    print("\n关键结论 Key Conclusions:")
    print("-" * 60)
    print("1. UCB: 最佳理论保证，稳定性能")
    print("2. ε-Greedy: 简单有效，易于实现")
    print("3. Gradient Bandit: 适合需要随机策略的场景")
    print("4. 基线对梯度方法至关重要")
    print("5. 熵正则化有助于持续探索")
    
    return fig


# ================================================================================
# 主函数
# Main Function
# ================================================================================

def main():
    """
    运行梯度赌博机完整演示
    Run complete gradient bandit demo
    """
    print("\n" + "="*80)
    print("第1.8节：梯度赌博机算法")
    print("Section 1.8: Gradient Bandit Algorithm")
    print("="*80)
    
    # 1. 原理解释
    fig1 = GradientBanditPrinciple.explain_principle()
    
    # 2. 理论分析
    GradientBanditAnalysis.convergence_analysis()
    fig2 = GradientBanditAnalysis.demonstrate_convergence()
    
    # 3. 综合比较
    fig3 = compare_all_algorithms()
    
    print("\n" + "="*80)
    print("梯度赌博机演示完成！")
    print("Gradient Bandit Demo Complete!")
    print("\n这标志着第1章的结束！")
    print("This marks the end of Chapter 1!")
    print("="*80)
    
    plt.show()
    
    return [fig1, fig2, fig3]


if __name__ == "__main__":
    main()