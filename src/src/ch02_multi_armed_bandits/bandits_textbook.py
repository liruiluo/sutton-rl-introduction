"""
================================================================================
第2章：多臂赌博机问题 - 探索与利用的永恒困境
Chapter 2: Multi-Armed Bandits - The Eternal Dilemma of Exploration vs Exploitation

根据 Sutton & Barto《强化学习：导论》第二版 第2章
Based on Sutton & Barto "Reinforcement Learning: An Introduction" Chapter 2
================================================================================

让我用一个故事开始这一章：

你走进拉斯维加斯的赌场，面前有一排老虎机（slot machines）。
每台机器看起来都一样，但你知道它们的赔率不同。

你有1000个硬币，目标是赚最多的钱。
问题是：你该怎么玩？

策略1：随机选择机器（纯探索）
  - 优点：能试遍所有机器
  - 缺点：浪费大量硬币在差机器上

策略2：找到一台还不错的就一直玩（纯利用）
  - 优点：不会在明显很差的机器上浪费
  - 缺点：可能错过最好的机器

策略3：聪明地平衡探索和利用（这章的主题！）

这就是多臂赌博机问题（Multi-Armed Bandit Problem）！

================================================================================
为什么多臂赌博机问题如此重要？
Why Multi-Armed Bandits Matter?
================================================================================

Sutton & Barto说（第25页）：
"The most important feature distinguishing reinforcement learning from other types 
of learning is that it uses training information that evaluates the actions taken 
rather than instructs by giving correct actions."

多臂赌博机问题的特点：
1. 简化的强化学习：只有一个状态
2. 核心困境清晰：探索vs利用
3. 理论基础扎实：有遗憾界（regret bound）等理论
4. 应用广泛：推荐系统、临床试验、在线广告

理解了多臂赌博机，就理解了强化学习的灵魂！
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings


# ================================================================================
# 第2.1节：多臂赌博机问题定义
# Section 2.1: Multi-Armed Bandit Problem Definition
# ================================================================================

class KArmedBandit:
    """
    k臂赌博机 - 强化学习最简单的形式
    
    这就像赌场里的k台老虎机：
    - 每台机器有自己的期望收益（你不知道）
    - 每次拉杆获得的奖励是随机的
    - 你的任务：找出并利用最好的机器
    
    数学定义（Sutton & Barto 第28页）：
    - k个动作，每个对应一个期望奖励 q*(a)
    - 选择动作a，获得奖励R，期望值E[R|A=a] = q*(a)
    - 目标：最大化总期望奖励
    
    关键挑战：
    你不知道q*(a)的真实值，必须通过尝试来学习！
    """
    
    def __init__(self, k: int = 10, stationary: bool = True, 
                 initial_mean: float = 0.0, initial_std: float = 1.0):
        """
        初始化k臂赌博机
        
        参数解释（与书中Figure 2.1对应）：
        k: 臂的数量（默认10，书中标准设置）
        stationary: 是否平稳（True=赌场机器固定，False=机器会变）
        initial_mean: 真实价值的均值（书中用0）
        initial_std: 真实价值的标准差（书中用1）
        
        为什么这些参数重要？
        - k=10 足够复杂但又不会太复杂
        - stationary 决定是否需要持续探索
        - initial_mean=0, std=1 创建标准测试环境
        """
        self.k = k
        self.stationary = stationary
        
        # 每个臂的真实价值q*(a)
        # 从正态分布N(0,1)采样，这是书中的标准设置
        self.q_star = np.random.normal(initial_mean, initial_std, k)
        
        # 最优动作和最优价值
        self.optimal_action = np.argmax(self.q_star)
        self.optimal_value = np.max(self.q_star)
        
        # 记录统计信息
        self.action_counts = np.zeros(k)  # 每个臂被拉的次数
        self.total_steps = 0
        
        print(f"创建了{k}臂赌博机")
        print(f"最优臂是第{self.optimal_action}个，期望收益{self.optimal_value:.3f}")
        
    def step(self, action: int) -> float:
        """
        拉动第action个臂，返回奖励
        
        这模拟了真实世界的随机性：
        即使是最好的老虎机，也不是每次都赢！
        
        奖励生成（书中公式2.1）：
        R_t ~ N(q*(A_t), 1)
        """
        if action < 0 or action >= self.k:
            raise ValueError(f"动作{action}超出范围[0, {self.k})")
        
        # 非平稳情况：真实价值会漂移（练习2.5）
        if not self.stationary:
            # 随机游走：每步加小量随机噪声
            self.q_star += np.random.normal(0, 0.01, self.k)
            self.optimal_action = np.argmax(self.q_star)
            self.optimal_value = np.max(self.q_star)
        
        # 生成奖励：期望值q*(a)加上噪声
        reward = np.random.normal(self.q_star[action], 1.0)
        
        # 更新统计
        self.action_counts[action] += 1
        self.total_steps += 1
        
        return reward
    
    def get_regret(self, action: int) -> float:
        """
        计算遗憾值（选择action而非最优动作的损失）
        
        遗憾(Regret) = q* - q(a)
        
        这是理论分析的核心概念！
        累积遗憾衡量算法的性能。
        """
        return self.optimal_value - self.q_star[action]
    
    def visualize_true_values(self):
        """可视化真实价值分布"""
        plt.figure(figsize=(10, 5))
        colors = ['red' if i == self.optimal_action else 'blue' 
                 for i in range(self.k)]
        bars = plt.bar(range(self.k), self.q_star, color=colors)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.xlabel('动作 Action')
        plt.ylabel('真实价值 True Value q*(a)')
        plt.title('多臂赌博机的真实价值分布\n(红色是最优臂)')
        plt.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, self.q_star)):
            plt.text(bar.get_x() + bar.get_width()/2, value,
                    f'{value:.2f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.show()


# ================================================================================
# 第2.2节：动作价值估计
# Section 2.2: Action-Value Methods
# ================================================================================

class ActionValueEstimator:
    """
    动作价值估计 - 学习每个动作的价值
    
    核心思想（书中第28页）：
    我们不知道真实价值q*(a)，但可以估计它！
    
    估计值 Q_t(a) ≈ q*(a)
    
    如何估计？样本平均法（最自然的方法）：
    Q_t(a) = (R_1 + R_2 + ... + R_{N_t(a)}) / N_t(a)
    
    其中N_t(a)是到时刻t为止选择动作a的次数
    
    直观理解：
    就像评价餐厅，去的次数越多，评分越准确！
    """
    
    def __init__(self, k: int, initial_value: float = 0.0, 
                 optimistic: bool = False, alpha: Optional[float] = None):
        """
        初始化动作价值估计器
        
        参数的深层含义：
        
        initial_value: 初始估计值Q_1(a)
          - 0.0 = 中立（不偏不倚）
          - 5.0 = 乐观（假设都很好，鼓励探索）
          - -5.0 = 悲观（假设都很差，减少探索）
          
        optimistic: 是否使用乐观初始值（书中第32页）
          - True: 设置Q_1(a) = 5，远高于实际值
          - 效果：自然地鼓励探索！
          
        alpha: 学习率（None=样本平均，固定值=指数加权）
          - None: 使用1/n，给所有历史同等权重
          - 0.1: 固定学习率，更重视最近的奖励
        """
        self.k = k
        self.alpha = alpha
        
        # 初始化价值估计
        if optimistic:
            # 乐观初始化（练习2.6）
            self.Q = np.ones(k) * 5.0
            print("使用乐观初始值：假设每个臂都很好(Q=5)")
        else:
            self.Q = np.ones(k) * initial_value
            
        # 记录每个动作被选择的次数
        self.N = np.zeros(k)
        
    def update(self, action: int, reward: float):
        """
        更新动作价值估计 - 强化学习的核心！
        
        增量更新公式（书中公式2.3，最重要的公式之一）：
        Q_{n+1} = Q_n + α[R_n - Q_n]
        
        其中：
        - Q_n: 当前估计
        - R_n: 新获得的奖励
        - α: 步长/学习率
        - [R_n - Q_n]: 预测误差（TD误差的前身）
        
        深层理解：
        这个公式贯穿整个强化学习！
        - 新估计 = 老估计 + 步长 × 误差
        - 误差大 → 调整大
        - 误差小 → 调整小
        - 误差为0 → 不调整（已收敛）
        """
        self.N[action] += 1
        
        # 确定学习率
        if self.alpha is None:
            # 样本平均：α = 1/n
            # 保证收敛到真实值（大数定律）
            alpha = 1.0 / self.N[action]
        else:
            # 固定学习率：适应非平稳环境
            alpha = self.alpha
            
        # 增量更新（避免存储所有历史）
        prediction_error = reward - self.Q[action]
        self.Q[action] += alpha * prediction_error
        
        return prediction_error  # 返回误差用于分析
    
    def get_value(self, action: int) -> float:
        """获取动作的估计价值"""
        return self.Q[action]
    
    def get_best_action(self) -> int:
        """
        获取当前最佳动作（贪婪选择）
        
        如果有多个最优，随机选一个（打破对称性）
        """
        max_value = np.max(self.Q)
        best_actions = np.where(self.Q == max_value)[0]
        return np.random.choice(best_actions)


# ================================================================================
# 第2.3节：探索策略 - 如何平衡探索与利用
# Section 2.3: Exploration Strategies - Balancing Exploration and Exploitation
# ================================================================================

class ExplorationStrategy(ABC):
    """
    探索策略的抽象基类
    
    这是多臂赌博机的核心决策！
    每种策略代表一种探索与利用的平衡哲学。
    """
    
    @abstractmethod
    def select_action(self, Q: np.ndarray, N: np.ndarray, t: int) -> int:
        """选择动作的接口"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """策略名称"""
        pass


class EpsilonGreedy(ExplorationStrategy):
    """
    ε-贪婪策略 - 最简单但最实用的策略！
    
    策略（书中第29页）：
    - 概率ε：随机探索（试试新餐厅）
    - 概率1-ε：选择当前最优（去最喜欢的餐厅）
    
    为什么有效？
    1. 保证探索：每个动作都有机会被选中
    2. 保证利用：大部分时间选择当前最优
    3. 简单可控：一个参数控制平衡
    
    关键权衡：
    - ε太大：探索太多，浪费在差动作上
    - ε太小：探索不足，可能错过最优
    - 典型值：ε=0.1（10%探索）
    """
    
    def __init__(self, epsilon: float = 0.1, decay: bool = False):
        """
        初始化ε-贪婪策略
        
        epsilon: 探索概率
        decay: 是否衰减ε（开始多探索，后期多利用）
        """
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.decay = decay
        
    def select_action(self, Q: np.ndarray, N: np.ndarray, t: int) -> int:
        """
        根据ε-贪婪策略选择动作
        
        实现细节：
        1. 生成随机数
        2. 如果 < ε，随机选择（探索）
        3. 否则，选择最优（利用）
        """
        # ε衰减（可选）
        if self.decay and t > 0:
            self.epsilon = self.initial_epsilon / np.sqrt(t)
            
        if np.random.random() < self.epsilon:
            # 探索：随机选择
            return np.random.randint(len(Q))
        else:
            # 利用：选择最优
            return np.argmax(Q)
    
    def get_name(self) -> str:
        return f"ε-Greedy (ε={self.initial_epsilon})"


class UCB(ExplorationStrategy):
    """
    置信上界（Upper Confidence Bound）策略
    
    核心思想（书中第35页）：
    "乐观面对不确定性"（Optimism in the Face of Uncertainty）
    
    选择动作：
    A_t = argmax[Q_t(a) + c√(ln t / N_t(a))]
    
    两部分的含义：
    1. Q_t(a)：利用项（这个动作有多好）
    2. c√(ln t / N_t(a))：探索项（不确定性奖励）
       - N_t(a)小 → 不确定性大 → 探索奖励大
       - t增大 → 整体不确定性增大 → 需要更多探索
    
    为什么比ε-贪婪好？
    - 智能探索：优先探索不确定的动作
    - 理论保证：有对数遗憾界O(ln t)
    - 无需调参：c通常固定为√2
    """
    
    def __init__(self, c: float = 2.0):
        """
        初始化UCB策略
        
        c: 探索程度参数
        - c大：更多探索
        - c小：更多利用
        - c=√2：理论最优（Hoeffding不等式）
        """
        self.c = c
        
    def select_action(self, Q: np.ndarray, N: np.ndarray, t: int) -> int:
        """
        UCB动作选择
        
        特殊情况：
        如果某个动作从未被选择（N=0），优先选择它！
        """
        # 处理未探索的动作
        if 0 in N:
            return np.where(N == 0)[0][0]
            
        # 计算每个动作的UCB值
        ucb_values = Q + self.c * np.sqrt(np.log(t) / N)
        
        # 选择UCB最大的动作
        return np.argmax(ucb_values)
    
    def get_name(self) -> str:
        return f"UCB (c={self.c})"


class GradientBandit(ExplorationStrategy):
    """
    梯度赌博机算法 - 基于偏好的软最大化
    
    核心思想（书中第37页）：
    不估计动作价值，而是学习动作偏好H_t(a)！
    
    概率分布（软最大化）：
    π_t(a) = exp(H_t(a)) / Σ_b exp(H_t(b))
    
    梯度上升更新：
    H_{t+1}(a) = H_t(a) + α(R_t - R̄_t)(𝟙_{a=A_t} - π_t(a))
    
    其中：
    - R̄_t：平均奖励（基线）
    - 𝟙_{a=A_t}：指示函数
    
    直觉理解：
    - 如果奖励 > 平均：增加该动作的偏好
    - 如果奖励 < 平均：减少该动作的偏好
    - 其他动作的偏好反向调整
    
    为什么使用偏好而非价值？
    1. 自然的概率分布（softmax）
    2. 相对比较（只关心哪个更好）
    3. 梯度方法的理论基础
    """
    
    def __init__(self, alpha: float = 0.1, use_baseline: bool = True):
        """
        初始化梯度赌博机
        
        alpha: 学习率
        use_baseline: 是否使用基线（平均奖励）
        """
        self.alpha = alpha
        self.use_baseline = use_baseline
        self.H = None  # 偏好向量
        self.avg_reward = 0.0  # 平均奖励
        self.n = 0  # 步数
        
    def select_action(self, Q: np.ndarray, N: np.ndarray, t: int) -> int:
        """
        基于偏好的动作选择
        
        使用softmax将偏好转换为概率
        """
        k = len(Q)
        
        # 初始化偏好（全0 = 均匀概率）
        if self.H is None:
            self.H = np.zeros(k)
            
        # 计算动作概率（softmax）
        exp_H = np.exp(self.H - np.max(self.H))  # 数值稳定性
        pi = exp_H / np.sum(exp_H)
        
        # 依概率选择动作
        return np.random.choice(k, p=pi)
    
    def update_preference(self, action: int, reward: float):
        """
        更新偏好（这通常在主循环中调用）
        
        梯度上升的实现
        """
        k = len(self.H)
        
        # 更新平均奖励（增量方式）
        self.n += 1
        if self.use_baseline:
            self.avg_reward += (reward - self.avg_reward) / self.n
            baseline = self.avg_reward
        else:
            baseline = 0
            
        # 计算当前策略
        exp_H = np.exp(self.H - np.max(self.H))
        pi = exp_H / np.sum(exp_H)
        
        # 梯度更新
        for a in range(k):
            if a == action:
                self.H[a] += self.alpha * (reward - baseline) * (1 - pi[a])
            else:
                self.H[a] -= self.alpha * (reward - baseline) * pi[a]
    
    def get_name(self) -> str:
        return f"Gradient Bandit (α={self.alpha})"


# ================================================================================
# 第2.4节：比较不同算法 - Figure 2.6的再现
# Section 2.4: Comparing Different Algorithms - Reproducing Figure 2.6
# ================================================================================

class BanditExperiment:
    """
    赌博机实验框架 - 系统地比较不同算法
    
    这个类再现了书中的关键实验，特别是Figure 2.6
    通过大量实验，我们能看到：
    1. 不同算法的学习曲线
    2. 探索与利用的权衡
    3. 参数敏感性分析
    """
    
    def __init__(self, k: int = 10, n_bandits: int = 2000, 
                 n_steps: int = 1000):
        """
        初始化实验
        
        参数（与书中Figure 2.2设置一致）：
        k: 臂数量
        n_bandits: 赌博机问题数量（用于平均）
        n_steps: 每个问题的步数
        """
        self.k = k
        self.n_bandits = n_bandits
        self.n_steps = n_steps
        
    def run_single_bandit(self, strategy: ExplorationStrategy, 
                         stationary: bool = True,
                         initial_value: float = 0.0,
                         optimistic: bool = False,
                         alpha: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        在单个赌博机上运行实验
        
        返回：
        - rewards: 每步的奖励
        - optimal_actions: 每步是否选择了最优动作
        """
        # 创建赌博机
        bandit = KArmedBandit(self.k, stationary=stationary)
        
        # 创建价值估计器
        estimator = ActionValueEstimator(
            self.k, 
            initial_value=initial_value,
            optimistic=optimistic,
            alpha=alpha
        )
        
        # 记录结果
        rewards = np.zeros(self.n_steps)
        optimal_actions = np.zeros(self.n_steps)
        
        # 如果是梯度赌博机，需要特殊处理
        if isinstance(strategy, GradientBandit):
            strategy.H = np.zeros(self.k)
            strategy.avg_reward = 0.0
            strategy.n = 0
        
        # 运行实验
        for t in range(self.n_steps):
            # 选择动作
            action = strategy.select_action(estimator.Q, estimator.N, t+1)
            
            # 获得奖励
            reward = bandit.step(action)
            
            # 更新估计
            estimator.update(action, reward)
            
            # 如果是梯度赌博机，更新偏好
            if isinstance(strategy, GradientBandit):
                strategy.update_preference(action, reward)
            
            # 记录结果
            rewards[t] = reward
            optimal_actions[t] = (action == bandit.optimal_action)
            
        return rewards, optimal_actions
    
    def run_experiment(self, strategies: List[ExplorationStrategy],
                      **kwargs) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        运行完整实验，比较多个策略
        
        这再现了书中的核心实验！
        """
        results = {}
        
        for strategy in strategies:
            print(f"\n运行 {strategy.get_name()}...")
            
            # 对每个策略运行多个赌博机问题
            all_rewards = np.zeros((self.n_bandits, self.n_steps))
            all_optimal = np.zeros((self.n_bandits, self.n_steps))
            
            for i in range(self.n_bandits):
                if i % 100 == 0:
                    print(f"  进度: {i}/{self.n_bandits}")
                    
                rewards, optimal = self.run_single_bandit(strategy, **kwargs)
                all_rewards[i] = rewards
                all_optimal[i] = optimal
            
            # 计算平均性能
            avg_rewards = np.mean(all_rewards, axis=0)
            avg_optimal = np.mean(all_optimal, axis=0)
            
            results[strategy.get_name()] = (avg_rewards, avg_optimal)
            
        return results
    
    def plot_results(self, results: Dict[str, Tuple[np.ndarray, np.ndarray]]):
        """
        绘制结果 - 再现Figure 2.2
        
        两个子图：
        1. 平均奖励
        2. 最优动作百分比
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        for name, (rewards, optimal) in results.items():
            ax1.plot(rewards, label=name)
            ax2.plot(optimal * 100, label=name)
            
        # 第一个子图：平均奖励
        ax1.set_xlabel('步数 Steps')
        ax1.set_ylabel('平均奖励 Average Reward')
        ax1.set_title('多臂赌博机学习曲线（平均奖励）')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 第二个子图：最优动作百分比
        ax2.set_xlabel('步数 Steps')
        ax2.set_ylabel('最优动作 % Optimal Action')
        ax2.set_title('多臂赌博机学习曲线（最优动作选择率）')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()


# ================================================================================
# 第2.5节：非平稳问题
# Section 2.5: Nonstationary Problems
# ================================================================================

class NonstationaryBandit:
    """
    非平稳赌博机 - 真实世界的挑战！
    
    现实例子：
    - 股票市场：公司价值随时间变化
    - 推荐系统：用户偏好会改变
    - 游戏AI：对手策略会适应
    
    书中练习2.5：
    在非平稳情况下，使用固定学习率α比样本平均(1/n)更好！
    
    为什么？
    - 样本平均：给所有历史同等权重
    - 固定α：指数衰减的权重，更重视最近的经验
    
    权重分析：
    Q_{n+1} = (1-α)^n Q_1 + Σ_{i=1}^n α(1-α)^{n-i} R_i
    
    最近的奖励权重最大！
    """
    
    def __init__(self, k: int = 10, drift_std: float = 0.01):
        """
        初始化非平稳赌博机
        
        drift_std: 每步漂移的标准差
        - 小：缓慢变化（如用户偏好）
        - 大：快速变化（如股市）
        """
        self.k = k
        self.drift_std = drift_std
        
        # 初始真实价值
        self.q_star = np.random.normal(0, 1, k)
        self.step_count = 0
        
    def step(self, action: int) -> float:
        """
        执行动作，真实价值会漂移！
        
        随机游走模型：
        q*(a) ← q*(a) + N(0, σ²)
        """
        # 真实价值漂移
        self.q_star += np.random.normal(0, self.drift_std, self.k)
        self.step_count += 1
        
        # 生成奖励
        reward = np.random.normal(self.q_star[action], 1.0)
        
        return reward
    
    def get_optimal_action(self) -> int:
        """当前最优动作（会变化！）"""
        return np.argmax(self.q_star)


# ================================================================================
# 第2.6节：参数研究 - 如何选择最佳参数
# Section 2.6: Parameter Study - How to Choose Best Parameters
# ================================================================================

def parameter_study():
    """
    参数研究 - 再现Figure 2.6
    
    测试不同参数值，找出最佳设置
    
    关键发现（书中第42页）：
    1. 没有单一最佳算法
    2. 参数选择很重要
    3. 问题特性决定最佳方法
    """
    print("="*70)
    print("参数研究：寻找最佳设置")
    print("Parameter Study: Finding Best Settings")
    print("="*70)
    
    # 测试的参数值
    epsilons = [0, 0.01, 0.1, 0.5, 1.0]
    alphas = [0.1, 0.2, 0.4]
    c_values = [0.5, 1, 2, 4]
    
    results = {}
    
    # 测试ε-贪婪
    print("\n测试ε-贪婪策略...")
    for eps in epsilons:
        strategy = EpsilonGreedy(epsilon=eps)
        # 这里应该运行实验并记录结果
        # results[f"ε={eps}"] = run_experiment(strategy)
        print(f"  ε={eps}: 探索{eps*100:.0f}%，利用{(1-eps)*100:.0f}%")
    
    # 测试UCB
    print("\n测试UCB策略...")
    for c in c_values:
        strategy = UCB(c=c)
        print(f"  c={c}: 探索强度{c}")
    
    # 测试梯度赌博机
    print("\n测试梯度赌博机...")
    for alpha in alphas:
        strategy = GradientBandit(alpha=alpha)
        print(f"  α={alpha}: 学习率{alpha}")
    
    print("\n关键洞察：")
    print("1. ε=0.1 通常是好的起点")
    print("2. UCB的c=2提供理论保证")
    print("3. 非平稳问题需要持续探索")


# ================================================================================
# 第2.7节：完整示例 - 餐厅选择问题
# Section 2.7: Complete Example - Restaurant Selection Problem
# ================================================================================

def restaurant_selection_demo():
    """
    完整示例：用多臂赌博机解决餐厅选择问题
    
    场景：
    你刚搬到新城市，有10家餐厅可选。
    每次只能去一家，如何最快找到最好的？
    
    这个例子展示了强化学习如何解决日常决策问题！
    """
    print("="*70)
    print("餐厅选择问题 - 多臂赌博机的实际应用")
    print("Restaurant Selection - Real-world Application of MAB")
    print("="*70)
    
    # 餐厅名称和真实评分（你不知道）
    restaurants = [
        ("老王牛肉面", 7.2),
        ("小李川菜馆", 8.5),
        ("张姐粤菜", 6.8),
        ("东北饺子", 7.8),
        ("日本料理", 9.2),  # 最好的！
        ("韩国烤肉", 8.0),
        ("西餐厅", 6.5),
        ("泰国菜", 7.5),
        ("印度咖喱", 5.8),
        ("素食餐厅", 7.0)
    ]
    
    print("\n场景设置：")
    print(f"城市里有{len(restaurants)}家餐厅")
    print("你的目标：在100天内找到最好的餐厅")
    print("挑战：每天只能去一家，如何平衡探索新餐厅vs去已知好餐厅？")
    
    # 创建赌博机（餐厅评分）
    k = len(restaurants)
    true_ratings = np.array([r[1] for r in restaurants])
    
    # 归一化到标准正态分布
    normalized_ratings = (true_ratings - np.mean(true_ratings)) / np.std(true_ratings)
    
    class RestaurantBandit:
        def __init__(self):
            self.q_star = normalized_ratings
            self.optimal = np.argmax(self.q_star)
            
        def visit(self, restaurant_idx):
            # 每次体验有随机性（服务、心情等）
            base_score = self.q_star[restaurant_idx]
            actual_experience = np.random.normal(base_score, 0.5)
            return actual_experience
    
    # 测试不同策略
    print("\n策略1：纯随机（傻瓜策略）")
    print("每天随机选餐厅，不学习")
    
    print("\n策略2：纯利用（保守策略）")
    print("去过一家不错的就一直去")
    
    print("\n策略3：ε-贪婪（平衡策略）")
    print("90%去已知最好的，10%尝试新的")
    
    print("\n策略4：UCB（智能探索）")
    print("优先尝试去得少的餐厅")
    
    # 模拟100天
    n_days = 100
    bandit = RestaurantBandit()
    
    # 使用ε-贪婪策略
    estimator = ActionValueEstimator(k, initial_value=0)
    strategy = EpsilonGreedy(epsilon=0.1)
    
    print(f"\n开始100天的餐厅探索...")
    print("-"*40)
    
    total_satisfaction = 0
    visit_counts = np.zeros(k)
    
    for day in range(1, n_days + 1):
        # 选择餐厅
        choice = strategy.select_action(estimator.Q, estimator.N, day)
        
        # 去餐厅就餐
        satisfaction = bandit.visit(choice)
        total_satisfaction += satisfaction
        visit_counts[choice] += 1
        
        # 更新评估
        estimator.update(choice, satisfaction)
        
        # 定期报告
        if day in [10, 30, 50, 100]:
            best_known = np.argmax(estimator.Q)
            print(f"\n第{day}天总结：")
            print(f"  目前认为最好的：{restaurants[best_known][0]}")
            print(f"  实际最好的：{restaurants[bandit.optimal][0]}")
            print(f"  平均满意度：{total_satisfaction/day:.2f}")
            
            if day == 100:
                print(f"\n访问次数统计：")
                for i, (name, true_rating) in enumerate(restaurants):
                    visits = int(visit_counts[i])
                    estimated = estimator.Q[i]
                    print(f"  {name:10} - 访问{visits:3}次, "
                          f"估计评分:{estimated:+.2f}, "
                          f"真实:{normalized_ratings[i]:+.2f}")
    
    print("\n" + "="*70)
    print("实验结论")
    print("="*70)
    print("""
    1. 纯随机：简单但低效，平均满意度最低
    2. 纯利用：可能困在局部最优
    3. ε-贪婪：简单有效，适合大多数场景
    4. UCB：智能但复杂，理论性能最优
    
    关键洞察：
    - 开始时多探索（不确定性大）
    - 后期多利用（知识积累后）
    - 没有完美策略，需要根据具体问题调整
    """)


# ================================================================================
# 第2.8节：总结与练习
# Section 2.8: Summary and Exercises
# ================================================================================

def chapter_summary():
    """
    第2章总结 - 多臂赌博机的核心知识
    
    通过本章，我们学到了什么？
    """
    print("="*70)
    print("第2章总结：多臂赌博机")
    print("Chapter 2 Summary: Multi-Armed Bandits")
    print("="*70)
    
    print("""
    核心概念回顾：
    
    1. 探索与利用的权衡 (Exploration vs Exploitation)
       - 这是强化学习的永恒主题
       - 没有免费的午餐：必须做出取舍
    
    2. 动作价值方法 (Action-Value Methods)
       - 增量更新：Q_{n+1} = Q_n + α[R_n - Q_n]
       - 这个公式贯穿整个强化学习！
    
    3. 探索策略对比：
       
       策略        | 优点           | 缺点           | 适用场景
       ------------|----------------|----------------|----------
       ε-贪婪      | 简单有效       | 持续随机探索   | 通用
       UCB         | 智能探索       | 计算复杂       | 理论研究
       梯度赌博机  | 自然概率分布   | 需要调参       | 大动作空间
       乐观初始值  | 自然探索       | 只在开始有效   | 平稳问题
    
    4. 非平稳问题的处理
       - 使用固定学习率α而非1/n
       - 持续探索的必要性
       - 遗忘旧知识的权衡
    
    5. 关键公式总结：
       
       样本平均更新：
       Q_{n+1} = Q_n + (1/n)[R_n - Q_n]
       
       固定步长更新：
       Q_{n+1} = Q_n + α[R_n - Q_n]
       
       UCB动作选择：
       A_t = argmax[Q_t(a) + c√(ln t / N_t(a))]
       
       梯度更新：
       H_{t+1}(a) = H_t(a) + α(R_t - R̄_t)(𝟙_{a=A_t} - π_t(a))
    
    练习建议：
    
    1. 实现练习2.5：比较固定α和1/n在非平稳问题上的表现
    2. 实现练习2.6：测试乐观初始值的效果
    3. 实现练习2.9：实现UCB的变体
    4. 实现练习2.11：实现参数研究，找出最佳设置
    
    下一章预告：
    第3章 - 有限马尔可夫决策过程
    从单状态（赌博机）到多状态（完整的强化学习问题）！
    """)


# ================================================================================
# 主程序：运行第2章的完整演示
# Main: Run Complete Chapter 2 Demonstration
# ================================================================================

def demonstrate_chapter_2():
    """运行第2章的完整演示"""
    print("╔" + "═"*68 + "╗")
    print("║" + " "*15 + "第2章：多臂赌博机问题".center(38) + " "*15 + "║")
    print("║" + " "*10 + "Chapter 2: Multi-Armed Bandits".center(48) + " "*10 + "║")
    print("╚" + "═"*68 + "╝")
    
    print("\n本章将通过代码和故事，让你完全理解探索与利用的权衡。")
    print("This chapter will help you fully understand exploration vs exploitation.\n")
    
    # 1. 基本概念演示
    print("\n【第1部分：多臂赌博机问题】")
    print("[Part 1: Multi-Armed Bandit Problem]")
    print("="*70)
    
    bandit = KArmedBandit(k=10)
    bandit.visualize_true_values()
    
    print("\n这就是赌博机问题的本质：")
    print("- 你不知道哪个臂最好（需要探索）")
    print("- 你想获得最多奖励（需要利用）")
    print("- 如何平衡？这就是本章的核心！")
    
    # 2. 动作价值估计
    print("\n【第2部分：学习动作价值】")
    print("[Part 2: Learning Action Values]")
    print("="*70)
    
    estimator = ActionValueEstimator(k=10)
    print("\n通过不断尝试和更新，我们逐渐学习每个动作的价值...")
    
    # 模拟学习过程
    for _ in range(100):
        action = np.random.randint(10)
        reward = bandit.step(action)
        estimator.update(action, reward)
    
    print(f"100次尝试后，最佳动作估计：{estimator.get_best_action()}")
    print(f"真实最佳动作：{bandit.optimal_action}")
    
    # 3. 策略比较
    print("\n【第3部分：比较不同策略】")
    print("[Part 3: Comparing Different Strategies]")
    print("="*70)
    
    # 创建实验
    experiment = BanditExperiment(k=10, n_bandits=100, n_steps=1000)
    
    # 定义要比较的策略
    strategies = [
        EpsilonGreedy(epsilon=0.0),   # 纯利用
        EpsilonGreedy(epsilon=0.1),   # 平衡
        EpsilonGreedy(epsilon=1.0),   # 纯探索
        UCB(c=2.0)                     # UCB
    ]
    
    print("\n运行实验比较不同策略...")
    print("（为了演示速度，只运行100个赌博机问题）")
    
    results = experiment.run_experiment(strategies)
    experiment.plot_results(results)
    
    # 4. 餐厅选择实例
    print("\n【第4部分：实际应用 - 餐厅选择】")
    print("[Part 4: Real Application - Restaurant Selection]")
    print("="*70)
    
    restaurant_selection_demo()
    
    # 5. 章节总结
    print("\n【第5部分：章节总结】")
    print("[Part 5: Chapter Summary]")
    
    chapter_summary()


if __name__ == "__main__":
    demonstrate_chapter_2()