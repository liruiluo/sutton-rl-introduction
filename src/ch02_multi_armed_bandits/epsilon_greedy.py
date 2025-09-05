"""
================================================================================
第2.2节：ε-贪婪算法 - 探索与利用的第一课
Section 2.2: ε-Greedy Algorithm - Your First Lesson in Exploration vs Exploitation
================================================================================

想象一下这个场景：
你刚搬到一个新城市，面前有10家餐厅。你该怎么找到最好的那家？

策略1：随便进一家，如果还不错就一直去这家（纯利用 Pure Exploitation）
  问题：可能错过更好的餐厅！第一家是7分，但其实有一家是10分的

策略2：每天随机选一家不同的（纯探索 Pure Exploration）  
  问题：明明知道哪家好，却还在浪费时间尝试差的餐厅

策略3：大部分时候去已知最好的，偶尔(比如10%的时候)尝试新的（ε-贪婪）
  完美！这就是ε-贪婪的智慧

================================================================================
数学原理深入讲解 Mathematical Principles in Detail
================================================================================

让我们一步步理解ε-贪婪算法：

1. 价值估计 Value Estimation
   -------------------------
   我们为每个动作a维护一个价值估计Q(a)，表示选择这个动作的平均奖励。
   就像给每家餐厅打分一样。
   
   Q(a) = 该动作获得的总奖励 / 选择该动作的次数
   
   例子：
   - 餐厅A：去了3次，评分[7, 8, 6]，Q(A) = (7+8+6)/3 = 7.0
   - 餐厅B：去了2次，评分[9, 8]，Q(B) = (9+8)/2 = 8.5
   - 餐厅C：还没去过，Q(C) = 0（或乐观初始值）

2. 动作选择策略 Action Selection Strategy
   ---------------------------------------
   每次选择时，我们抛一枚有偏的硬币：
   
   if random() < ε:  # 发生概率为ε（比如0.1，即10%）
       选择 = 随机选一个动作（探索！去尝试可能更好的）
   else:  # 发生概率为1-ε（比如0.9，即90%）
       选择 = 当前Q值最高的动作（利用！去已知最好的）
   
   这可以写成概率分布：
   P(选择动作a) = {
       (1-ε) + ε/k,  如果a是当前最优动作（既可能因贪婪选中，也可能随机选中）
       ε/k,          如果a不是当前最优动作（只能随机选中）
   }
   
   其中k是动作总数。让我们算个具体例子：
   - 10个动作，ε=0.1
   - 动作5当前最优
   - P(选择动作5) = 0.9 + 0.1/10 = 0.91（91%概率）
   - P(选择动作3) = 0.1/10 = 0.01（1%概率）

3. 为什么这个策略有效？ Why Does This Work?
   -----------------------------------------
   关键洞察：在无限时间内，每个动作都会被尝试无限次！
   
   证明：设动作a被选择的次数为N(a)，时间步数为t
   - 每一步选择动作a的概率至少为ε/k > 0
   - 根据大数定律：N(a)/t → 至少ε/k 当t→∞
   - 因此 N(a) → ∞ 当t→∞
   
   这保证了：
   1. 我们最终会发现真正的最优动作（探索的价值）
   2. 大部分时间我们在利用当前最优（利用的价值）

4. ε的选择艺术 The Art of Choosing ε
   -----------------------------------
   ε太小（如0.01）：
   - 优点：浪费在次优动作上的时间少
   - 缺点：可能很久才能发现真正的最优动作
   - 适用：当你比较确定当前最优就是真最优时
   
   ε太大（如0.5）：
   - 优点：快速探索所有选项
   - 缺点：即使知道最优也在50%的时间选择次优
   - 适用：环境快速变化，需要持续探索
   
   ε衰减策略：
   - 开始时ε大（如0.5）：还不了解环境，多探索
   - 逐渐减小（如每步×0.999）：越来越确定，多利用
   - 最小值（如0.01）：始终保持一点探索，应对变化

================================================================================
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging

from .bandit_introduction import BaseBanditAgent, MultiArmedBandit

# 设置日志
logger = logging.getLogger(__name__)


# ================================================================================
# 第2.2.1节：ε-贪婪智能体实现
# Section 2.2.1: ε-Greedy Agent Implementation
# ================================================================================

class EpsilonGreedyAgent(BaseBanditAgent):
    """
    ε-贪婪智能体 - 你的第一个强化学习算法
    ε-Greedy Agent - Your First Reinforcement Learning Algorithm
    
    这个类实现了最基础但极其重要的ε-贪婪算法。
    让我们通过代码理解每个设计决策背后的深意。
    """
    
    def __init__(self, 
                 k: int = None,           # 动作数量（多少个臂/选项）
                 n_arms: int = None,      # 向后兼容的参数名
                 epsilon: float = 0.1,    # 探索概率（0.1意味着10%时间探索）
                 epsilon_decay: Optional[float] = None,  # ε衰减率（如0.999）
                 epsilon_min: float = 0.01,  # 最小ε值（永远保持一点探索）
                 **kwargs):  # 其他参数传给父类
        """
        初始化ε-贪婪智能体 - 设置探索策略
        
        参数详解（每个都很重要！）：
        ------------------------
        
        k 或 n_arms: 动作数量
            想象你面前有k个按钮，每个按下后给你不同的奖励。
            你的任务是找出哪个按钮平均奖励最高。
            典型值：赌场老虎机10个臂，围棋361个位置
        
        epsilon: 探索率（核心参数！）
            这决定了你的"冒险精神"：
            - epsilon = 0.0：永不冒险，只选已知最好的（可能错过更好的）
            - epsilon = 0.1：10%概率冒险尝试（经典选择）
            - epsilon = 1.0：完全随机，不利用已有知识（像没有学习能力）
            
            如何选择epsilon？考虑这些因素：
            1. 环境稳定性：稳定环境用小ε，变化环境用大ε
            2. 试错成本：成本高用小ε（谨慎），成本低用大ε（大胆）
            3. 时间限制：时间少用小ε（快速收敛），时间多可以用大ε（充分探索）
        
        epsilon_decay: 衰减率（让算法越来越"成熟"）
            每一步后：epsilon = epsilon * epsilon_decay
            
            例子：epsilon_decay = 0.999
            - 第1步：ε = 0.1
            - 第100步：ε = 0.1 × 0.999^100 ≈ 0.0905
            - 第1000步：ε = 0.1 × 0.999^1000 ≈ 0.0368
            - 第5000步：ε = 0.1 × 0.999^5000 ≈ 0.0007
            
            物理意义：就像人类学习，开始时多尝试（学生时代），
                     later多应用（工作后），但永远保持一点好奇心
        
        epsilon_min: 最小探索率（永远不要完全停止探索！）
            为什么需要？
            1. 环境可能变化（昨天最好的餐厅今天可能倒闭）
            2. 初始估计可能有误（采样不足导致的误判）
            3. 防止算法"傲慢"（总觉得自己已经知道最优解）
            
            典型值：0.01（1%）或 0.001（0.1%）
        """
        # 步骤1：参数验证和处理
        # ----------------------
        # 向后兼容：有些旧代码用n_arms，新代码用k
        if n_arms is not None:
            k = n_arms
        if k is None:
            raise ValueError("必须提供k或n_arms参数 - 我需要知道有多少个选择！")
        
        # 步骤2：存储ε相关参数
        # --------------------
        # 为什么要保存initial值？因为可能需要重置智能体重新学习
        self.epsilon_initial = epsilon  # 保存初始值，reset时用
        self.epsilon = epsilon          # 当前值，会随时间变化
        self.epsilon_decay = epsilon_decay  # 衰减率
        self.epsilon_min = epsilon_min      # 最小值，不能比这更小
        
        # 步骤3：记录历史（用于分析学习过程）
        # ---------------------------------
        self.epsilon_history = [epsilon]  # 记录ε的变化历程
        self.exploration_count = 0        # 探索了多少次
        self.exploitation_count = 0       # 利用了多少次
        
        # 步骤4：调用父类初始化
        # --------------------
        # 父类会初始化Q值表、计数器等基础结构
        super().__init__(k, **kwargs)
        
        logger.info(f"初始化ε-贪婪智能体: ε={epsilon}, decay={epsilon_decay}, min={epsilon_min}")
    
    def select_action(self) -> int:
        """
        核心方法：根据ε-贪婪策略选择动作
        
        这个方法被调用时的内心独白：
        1. "让我看看现在的探索率是多少..." (self.epsilon)
        2. "掷个骰子看看这次是探索还是利用..." (random < epsilon?)
        3. "如果探索，随机选一个试试看"
        4. "如果利用，选目前认为最好的"
        5. "记录这次的选择，方便后续分析"
        
        返回：选中的动作索引（0到k-1之间）
        """
        # 步骤1：这次是探索还是利用？
        # -------------------------
        # np.random.random()生成[0,1)之间的随机数
        # 如果 < epsilon，就探索（发生概率正好是epsilon）
        if np.random.random() < self.epsilon:
            # 探索分支：完全随机选择
            # ----------------------
            # 给每个动作平等的机会，不管其历史表现如何
            # 这就像闭着眼睛随机指一个
            action = np.random.choice(self.k)
            self.exploration_count += 1
            
            # 偷偷告诉你：其实这个随机选择可能碰巧选中最优的！
            # 这种情况下既是探索也达到了利用的效果，运气不错
            
        else:
            # 利用分支：选择当前最优
            # --------------------
            # 基于历史经验，选择平均奖励最高的动作
            
            # self.Q 是我们的"记忆"，存储每个动作的平均奖励估计
            # 形状：[k,]，每个元素是对应动作的价值估计
            action = np.argmax(self.Q)
            self.exploitation_count += 1
            
            # 细节处理：如果有多个最大值怎么办？
            # np.argmax默认返回第一个最大值的索引
            # 更公平的做法是随机选择一个最大值（下面注释的代码）
            #
            # max_q = np.max(self.Q)
            # max_actions = np.where(self.Q == max_q)[0]
            # action = np.random.choice(max_actions)
            
        # 步骤2：更新epsilon（如果设置了衰减）
        # ----------------------------------
        if self.epsilon_decay is not None:
            # 指数衰减：ε(t+1) = ε(t) × decay
            # 为什么是指数衰减而不是线性衰减？
            # 1. 指数衰减在早期下降快，后期下降慢，符合学习规律
            # 2. 永远不会真正达到0（理论上），保持探索可能性
            # 3. 数学上优雅，分析方便
            
            old_epsilon = self.epsilon
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            
            # 记录衰减（用于可视化学习过程）
            if old_epsilon != self.epsilon:
                self.epsilon_history.append(self.epsilon)
        
        return action
    
    def update(self, action: int, reward: float) -> None:
        """
        更新动作价值估计 - 学习的核心
        
        这里发生了什么？
        1. 我们刚执行了动作action，获得了奖励reward
        2. 现在要更新我们对这个动作价值的估计
        3. 使用增量更新公式（后面会详细解释）
        
        参数：
            action: 刚执行的动作
            reward: 获得的奖励
        """
        # 调用父类的更新方法
        # 父类实现了标准的增量更新公式：
        # Q(a) = Q(a) + α[R - Q(a)]
        # 其中α是学习率，R是新奖励，Q(a)是旧估计
        super().update(action, reward)
        
        # 可以在这里添加ε-贪婪特定的逻辑
        # 比如根据学习进度自适应调整ε
        
    def reset(self) -> None:
        """
        重置智能体 - 忘记所有学习，重新开始
        
        什么时候需要重置？
        1. 环境变化了（赌场换了新老虎机）
        2. 想要重新训练（对比不同初始化）
        3. 开始新的实验回合
        """
        # 重置ε到初始值
        self.epsilon = self.epsilon_initial
        self.epsilon_history = [self.epsilon]
        
        # 重置统计
        self.exploration_count = 0
        self.exploitation_count = 0
        
        # 调用父类重置（重置Q值、计数器等）
        super().reset()
        
        logger.info(f"重置ε-贪婪智能体，ε恢复到{self.epsilon_initial}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取详细的学习统计 - 了解算法表现
        
        这些统计帮助我们理解：
        1. 算法是否在正确学习？
        2. 探索-利用平衡是否合适？
        3. 是否需要调整参数？
        """
        stats = super().get_statistics()  # 获取基础统计
        
        # 添加ε-贪婪特定的统计
        stats.update({
            'current_epsilon': self.epsilon,
            'exploration_count': self.exploration_count,
            'exploitation_count': self.exploitation_count,
            'exploration_ratio': self.exploration_count / max(1, self.exploration_count + self.exploitation_count),
            'epsilon_history': self.epsilon_history,
        })
        
        # 计算额外的分析指标
        if len(self.rewards) > 0:
            # 后期表现（最后100步）vs 早期表现（前100步）
            recent_rewards = self.rewards[-100:] if len(self.rewards) > 100 else self.rewards
            early_rewards = self.rewards[:100] if len(self.rewards) > 100 else self.rewards
            
            stats['improvement'] = np.mean(recent_rewards) - np.mean(early_rewards)
            stats['stability'] = np.std(recent_rewards)  # 方差越小越稳定
        
        return stats


# ================================================================================
# 第2.2.2节：ε-贪婪算法分析器
# Section 2.2.2: ε-Greedy Algorithm Analyzer
# ================================================================================

class EpsilonGreedyAnalyzer:
    """
    深入分析ε-贪婪算法的表现
    
    这个类帮助我们理解：
    1. 不同ε值的影响
    2. 衰减策略的效果
    3. 最优参数选择
    """
    
    def __init__(self):
        """初始化分析器"""
        self.results = {}
        
    def compare_epsilon_values(self, 
                              epsilons: List[float] = [0.0, 0.01, 0.1, 0.3, 0.5, 1.0],
                              n_steps: int = 1000,
                              n_runs: int = 100) -> Dict:
        """
        实验：比较不同ε值的效果
        
        科学问题：ε如何影响学习效果？
        
        实验设计：
        1. 用相同的赌博机环境
        2. 测试不同的ε值
        3. 每个配置重复多次（消除随机性）
        4. 记录和分析结果
        
        预期结果：
        - ε=0：快速收敛但可能卡在次优
        - ε=0.1：良好的平衡
        - ε=1：学习慢但最终可能找到最优
        """
        print("="*80)
        print("实验：ε值对学习效果的影响")
        print("Experiment: Impact of ε on Learning Performance")
        print("="*80)
        
        for epsilon in epsilons:
            print(f"\n测试 ε = {epsilon}")
            
            rewards_over_time = []
            optimal_action_freq = []
            
            for run in tqdm(range(n_runs), desc=f"ε={epsilon}"):
                # 创建环境和智能体
                bandit = MultiArmedBandit(k=10)
                agent = EpsilonGreedyAgent(k=10, epsilon=epsilon)
                
                # 运行实验
                rewards = []
                optimal_actions = []
                
                for step in range(n_steps):
                    action = agent.select_action()
                    reward = bandit.pull_arm(action)
                    agent.update(action, reward)
                    
                    rewards.append(reward)
                    optimal_actions.append(action == bandit.optimal_action)
                
                rewards_over_time.append(rewards)
                optimal_action_freq.append(optimal_actions)
            
            # 分析结果
            avg_rewards = np.mean(rewards_over_time, axis=0)
            avg_optimal = np.mean(optimal_action_freq, axis=0)
            
            self.results[epsilon] = {
                'average_rewards': avg_rewards,
                'optimal_action_frequency': avg_optimal,
                'final_performance': np.mean(avg_rewards[-100:]),
                'convergence_speed': self._find_convergence_point(avg_rewards),
                'exploration_efficiency': np.mean(avg_optimal)
            }
            
            print(f"  最终性能: {self.results[epsilon]['final_performance']:.3f}")
            print(f"  最优动作频率: {self.results[epsilon]['exploration_efficiency']:.1%}")
        
        self._plot_comparison()
        return self.results
    
    def _find_convergence_point(self, rewards: np.ndarray, threshold: float = 0.95) -> int:
        """
        找到收敛点 - 性能达到最终水平95%的时间
        
        这告诉我们算法多快能学到不错的策略
        """
        final_performance = np.mean(rewards[-100:])
        target = final_performance * threshold
        
        # 使用滑动窗口找到稳定达到目标的点
        window_size = 50
        for i in range(len(rewards) - window_size):
            if np.mean(rewards[i:i+window_size]) >= target:
                return i
        
        return len(rewards)  # 未收敛
    
    def _plot_comparison(self):
        """
        可视化不同ε值的学习曲线
        
        图表解读：
        - X轴：时间步
        - Y轴：平均奖励
        - 不同曲线：不同ε值
        
        观察要点：
        1. 哪条线上升最快？（学习速度）
        2. 哪条线最终最高？（最终性能）
        3. 哪条线最稳定？（方差小）
        """
        plt.figure(figsize=(15, 6))
        
        # 子图1：学习曲线
        plt.subplot(1, 3, 1)
        for epsilon, data in self.results.items():
            plt.plot(data['average_rewards'], label=f'ε={epsilon}', alpha=0.7)
        plt.xlabel('时间步 Steps')
        plt.ylabel('平均奖励 Average Reward')
        plt.title('学习曲线 Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2：最优动作频率
        plt.subplot(1, 3, 2)
        for epsilon, data in self.results.items():
            plt.plot(data['optimal_action_frequency'], label=f'ε={epsilon}', alpha=0.7)
        plt.xlabel('时间步 Steps')
        plt.ylabel('最优动作频率 Optimal Action Frequency')
        plt.title('探索效率 Exploration Efficiency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图3：最终性能对比
        plt.subplot(1, 3, 3)
        epsilons = list(self.results.keys())
        performances = [data['final_performance'] for data in self.results.values()]
        plt.bar(range(len(epsilons)), performances, tick_label=[f'{e}' for e in epsilons])
        plt.xlabel('ε值 Epsilon Value')
        plt.ylabel('最终性能 Final Performance')
        plt.title('性能对比 Performance Comparison')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()


# ================================================================================
# 第2.2.3节：高级ε策略
# Section 2.2.3: Advanced ε Strategies
# ================================================================================

class AdaptiveEpsilonGreedy(EpsilonGreedyAgent):
    """
    自适应ε-贪婪：根据学习进度动态调整探索率
    
    核心思想：
    - 当不确定时（Q值接近）→ 多探索
    - 当很确定时（Q值差距大）→ 多利用
    
    这模拟了人类决策：
    - 选餐厅：如果几家评分接近(8.1, 8.0, 7.9)，不妨多试试
    - 选餐厅：如果差距明显(9.5, 6.0, 5.0)，就去最好的那家
    """
    
    def __init__(self, k: int, confidence_threshold: float = 0.5, **kwargs):
        """
        初始化自适应ε-贪婪
        
        confidence_threshold: 置信度阈值
            Q值标准差超过此值时认为"很确定"
        """
        super().__init__(k, **kwargs)
        self.confidence_threshold = confidence_threshold
        self.adaptive_epsilon_history = []
        
    def select_action(self) -> int:
        """
        根据Q值分布自适应调整ε
        
        自适应策略：
        1. 计算Q值的标准差（衡量确定性）
        2. 标准差大 = 很确定谁好谁坏 = 减少探索
        3. 标准差小 = 不太确定 = 增加探索
        """
        # 计算当前的不确定性
        q_std = np.std(self.Q)  # Q值的标准差
        q_range = np.max(self.Q) - np.min(self.Q)  # Q值的范围
        
        # 根据不确定性调整ε
        if q_range > 0:
            # 确定性 = 标准差 / 范围（归一化到[0,1]）
            certainty = min(1.0, q_std / q_range)
            
            # ε与不确定性成反比
            # 很确定时(certainty→1)，ε→epsilon_min
            # 不确定时(certainty→0)，ε→epsilon_initial
            adaptive_epsilon = self.epsilon_min + \
                             (self.epsilon_initial - self.epsilon_min) * (1 - certainty)
        else:
            # 所有Q值相同，完全不确定，使用初始ε
            adaptive_epsilon = self.epsilon_initial
        
        # 临时使用自适应ε
        original_epsilon = self.epsilon
        self.epsilon = adaptive_epsilon
        self.adaptive_epsilon_history.append(adaptive_epsilon)
        
        # 选择动作
        action = super().select_action()
        
        # 恢复原始ε（如果有衰减策略）
        if self.epsilon_decay is not None:
            self.epsilon = original_epsilon
            
        return action


# ================================================================================
# 第2.2.4节：实践示例 - 用ε-贪婪解决10臂赌博机
# Section 2.2.4: Practical Example - Solving 10-Armed Bandit with ε-Greedy
# ================================================================================

def demonstrate_epsilon_greedy():
    """
    完整演示：如何用ε-贪婪算法解决多臂赌博机问题
    
    这个例子会：
    1. 创建一个10臂赌博机
    2. 用不同配置的ε-贪婪算法学习
    3. 展示和解释结果
    """
    print("="*80)
    print("ε-贪婪算法完整演示")
    print("Complete ε-Greedy Algorithm Demonstration")
    print("="*80)
    
    # 步骤1：创建问题环境
    print("\n步骤1：创建10臂赌博机环境")
    print("Step 1: Creating 10-armed bandit environment")
    print("-"*40)
    
    np.random.seed(42)  # 固定随机种子，保证可重复性
    bandit = MultiArmedBandit(k=10)
    
    print(f"赌博机已创建:")
    print(f"  · 臂数: {bandit.k}")
    print(f"  · 最优臂: 臂{bandit.optimal_action}")
    print(f"  · 真实期望奖励: {bandit.means}")
    print(f"  · 最优期望奖励: {bandit.means[bandit.optimal_action]:.3f}")
    
    # 步骤2：创建并训练不同的智能体
    print("\n步骤2：训练不同配置的ε-贪婪智能体")
    print("Step 2: Training different ε-greedy agents")
    print("-"*40)
    
    configs = [
        {"name": "纯贪婪 Pure Greedy", "epsilon": 0.0},
        {"name": "保守探索 Conservative", "epsilon": 0.01},
        {"name": "平衡策略 Balanced", "epsilon": 0.1},
        {"name": "激进探索 Aggressive", "epsilon": 0.3},
        {"name": "衰减策略 Decaying", "epsilon": 0.5, "epsilon_decay": 0.995},
    ]
    
    n_steps = 1000
    results = {}
    
    for config in configs:
        print(f"\n训练: {config['name']}")
        
        # 创建智能体
        agent = EpsilonGreedyAgent(k=10, **{k: v for k, v in config.items() if k != 'name'})
        
        # 训练过程
        rewards = []
        optimal_actions = []
        
        for step in range(n_steps):
            # 选择动作
            action = agent.select_action()
            
            # 获得奖励
            reward = bandit.pull_arm(action)
            
            # 更新智能体
            agent.update(action, reward)
            
            # 记录数据
            rewards.append(reward)
            optimal_actions.append(action == bandit.optimal_action)
            
            # 定期报告进度
            if (step + 1) % 200 == 0:
                recent_reward = np.mean(rewards[-100:])
                recent_optimal = np.mean(optimal_actions[-100:])
                print(f"  步骤{step+1}: 平均奖励={recent_reward:.3f}, "
                      f"最优动作率={recent_optimal:.1%}, ε={agent.epsilon:.3f}")
        
        # 保存结果
        results[config['name']] = {
            'agent': agent,
            'rewards': rewards,
            'optimal_actions': optimal_actions,
            'final_Q': agent.Q.copy(),
            'statistics': agent.get_statistics()
        }
    
    # 步骤3：分析和比较结果
    print("\n步骤3：结果分析")
    print("Step 3: Results Analysis")
    print("-"*40)
    
    print("\n最终性能对比:")
    print("策略名称              | 平均奖励 | 最优动作率 | 探索次数")
    print("-"*60)
    
    for name, data in results.items():
        final_reward = np.mean(data['rewards'][-100:])
        final_optimal = np.mean(data['optimal_actions'][-100:])
        stats = data['statistics']
        
        print(f"{name:20} | {final_reward:8.3f} | {final_optimal:10.1%} | "
              f"{stats.get('exploration_count', 'N/A'):8}")
    
    # 步骤4：可视化学习过程
    print("\n步骤4：可视化学习过程")
    print("Step 4: Visualizing Learning Process")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 图1：累积平均奖励
    ax = axes[0, 0]
    for name, data in results.items():
        cumulative_avg = np.cumsum(data['rewards']) / (np.arange(len(data['rewards'])) + 1)
        ax.plot(cumulative_avg, label=name, alpha=0.7)
    ax.set_xlabel('步骤 Steps')
    ax.set_ylabel('累积平均奖励 Cumulative Average Reward')
    ax.set_title('学习效率对比 Learning Efficiency Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 图2：最优动作选择频率
    ax = axes[0, 1]
    window = 50
    for name, data in results.items():
        optimal_freq = np.convolve(data['optimal_actions'], 
                                   np.ones(window)/window, 
                                   mode='valid')
        ax.plot(optimal_freq, label=name, alpha=0.7)
    ax.set_xlabel('步骤 Steps')
    ax.set_ylabel(f'最优动作率({window}步窗口) Optimal Action Rate')
    ax.set_title('探索效率 Exploration Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 图3：Q值收敛情况（平衡策略）
    ax = axes[1, 0]
    agent = results['平衡策略 Balanced']['agent']
    for arm in range(10):
        color = 'red' if arm == bandit.optimal_action else 'blue'
        alpha = 1.0 if arm == bandit.optimal_action else 0.3
        label = f'臂{arm}(最优)' if arm == bandit.optimal_action else None
        ax.bar(arm, agent.Q[arm], color=color, alpha=alpha, label=label)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('臂编号 Arm Number')
    ax.set_ylabel('Q值估计 Q-value Estimate')
    ax.set_title('最终Q值（平衡策略） Final Q-values (Balanced)')
    if label:
        ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 图4：ε衰减曲线（如果有）
    ax = axes[1, 1]
    for name, data in results.items():
        if 'epsilon_history' in data['statistics'] and len(data['statistics']['epsilon_history']) > 1:
            ax.plot(data['statistics']['epsilon_history'], label=name)
    ax.set_xlabel('更新次数 Updates')
    ax.set_ylabel('ε值 Epsilon Value')
    ax.set_title('探索率变化 Exploration Rate Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 步骤5：深入分析最佳策略
    print("\n步骤5：深入分析 - 为什么平衡策略表现最好？")
    print("Step 5: Deep Analysis - Why Balanced Strategy Works Best?")
    print("-"*40)
    
    balanced = results['平衡策略 Balanced']
    print(f"\n平衡策略(ε=0.1)的优势:")
    print(f"1. 探索充分: 尝试了所有臂至少{min(balanced['agent'].action_counts)}次")
    print(f"2. 利用高效: {balanced['statistics']['exploitation_count']/n_steps:.1%}的时间在利用")
    print(f"3. 快速收敛: 在第{_find_convergence(balanced['rewards'])}步达到稳定性能")
    print(f"4. 稳定性好: 后期奖励标准差仅{np.std(balanced['rewards'][-100:]):.3f}")
    
    # 对比纯贪婪的问题
    greedy = results['纯贪婪 Pure Greedy']
    if greedy['agent'].action_counts[bandit.optimal_action] == 0:
        print(f"\n纯贪婪(ε=0)的问题:")
        print(f"  · 从未尝试最优臂{bandit.optimal_action}！")
        print(f"  · 被次优臂困住了")
        print(f"  · 这就是'探索不足'的代价")
    
    print("\n关键洞察 Key Insights:")
    print("1. 纯贪婪可能永远找不到最优解（局部最优陷阱）")
    print("2. 过度探索浪费时间在已知的差选项上")
    print("3. 适度的ε(如0.1)提供最佳平衡")
    print("4. 衰减策略结合了早期探索和后期利用的优势")
    print("5. 没有普适最优的ε，需要根据具体问题调整")


def _find_convergence(rewards: List[float], window: int = 50, threshold: float = 0.95) -> int:
    """辅助函数：找到算法收敛的时间点"""
    if len(rewards) < window:
        return len(rewards)
    
    final_performance = np.mean(rewards[-window:])
    target = final_performance * threshold
    
    for i in range(len(rewards) - window):
        if np.mean(rewards[i:i+window]) >= target:
            return i
    
    return len(rewards)


# ================================================================================
# 第2.2.5节：带衰减的ε-贪婪（向后兼容）
# Section 2.2.5: Decaying ε-Greedy (Backward Compatibility)
# ================================================================================

class DecayingEpsilonGreedy(EpsilonGreedyAgent):
    """
    带衰减的ε-贪婪智能体（向后兼容类）
    
    注意：这个类主要为了向后兼容。
    现在基础的EpsilonGreedyAgent已经支持衰减功能。
    """
    
    def __init__(self, k: int, 
                 initial_epsilon: float = 0.5,
                 decay_rate: float = 0.995,
                 min_epsilon: float = 0.01,
                 **kwargs):
        """
        初始化带衰减的ε-贪婪智能体
        
        参数：
            k: 动作数量
            initial_epsilon: 初始探索率
            decay_rate: 衰减率
            min_epsilon: 最小探索率
        """
        super().__init__(
            k=k,
            epsilon=initial_epsilon,
            epsilon_decay=decay_rate,
            epsilon_min=min_epsilon,
            **kwargs
        )


# ================================================================================
# 第2.2.6节：ε-贪婪分析（向后兼容）
# Section 2.2.6: ε-Greedy Analysis (Backward Compatibility)
# ================================================================================

# 为了向后兼容，创建别名
EpsilonGreedyAnalysis = EpsilonGreedyAnalyzer


def compare_epsilon_greedy_variants(n_steps: int = 1000,
                                   n_runs: int = 20) -> Dict:
    """
    比较不同ε-贪婪变体的性能（向后兼容函数）
    
    参数：
        n_steps: 每次运行的步数
        n_runs: 运行次数
    
    返回：
        包含比较结果的字典
    """
    print("="*80)
    print("比较ε-贪婪算法变体")
    print("Comparing ε-Greedy Variants")
    print("="*80)
    
    # 创建分析器并运行比较
    analyzer = EpsilonGreedyAnalyzer()
    
    # 定义要比较的变体
    epsilons = [0.0, 0.01, 0.1, 0.3]
    results = analyzer.compare_epsilon_values(
        epsilons=epsilons,
        n_steps=n_steps,
        n_runs=n_runs
    )
    
    # 额外测试衰减策略
    print("\n测试衰减策略...")
    bandit = MultiArmedBandit(k=10)
    
    # 创建衰减版本
    decaying_agent = DecayingEpsilonGreedy(
        k=10,
        initial_epsilon=0.5,
        decay_rate=0.995,
        min_epsilon=0.01
    )
    
    # 运行测试
    rewards = []
    for step in range(n_steps):
        action = decaying_agent.select_action()
        reward = bandit.pull_arm(action)
        decaying_agent.update(action, reward)
        rewards.append(reward)
    
    results['decaying'] = {
        'rewards': rewards,
        'final_performance': np.mean(rewards[-100:]),
        'epsilon_history': decaying_agent.epsilon_history
    }
    
    print(f"衰减策略最终性能: {results['decaying']['final_performance']:.3f}")
    
    return results


# ================================================================================
# 第2.2.7节：理论深度 - ε-贪婪的数学分析
# Section 2.2.7: Theoretical Depth - Mathematical Analysis of ε-Greedy
# ================================================================================

class EpsilonGreedyTheory:
    """
    ε-贪婪算法的理论分析
    
    深入探讨：
    1. 遗憾界 (Regret Bounds)
    2. 收敛性证明
    3. 最优ε的选择
    """
    
    @staticmethod
    def calculate_regret_bound(n_steps: int, n_arms: int, epsilon: float, 
                              reward_range: float = 1.0) -> float:
        """
        计算ε-贪婪的理论遗憾上界
        
        遗憾(Regret)是什么？
        - 定义：最优策略的累积奖励 - 实际获得的累积奖励
        - 意义：衡量因为不知道哪个是最优而造成的损失
        - 目标：让遗憾增长尽可能慢
        
        ε-贪婪的遗憾界：
        R(T) ≤ ε·T·Δ + (1-ε)·K·(1/ε)·log(T)·Δ
        
        其中：
        - T: 时间步数
        - K: 臂的数量
        - Δ: 最优臂与次优臂的期望奖励差
        - ε: 探索率
        
        这个界告诉我们：
        1. 第一项ε·T·Δ：因为随机探索造成的线性遗憾
        2. 第二项：因为需要学习造成的对数遗憾
        """
        # 简化假设：所有次优臂的差距都是reward_range/2
        delta = reward_range / 2
        
        # ε-贪婪的遗憾界（简化版）
        exploration_regret = epsilon * n_steps * delta
        learning_regret = (n_arms - 1) * np.log(n_steps) * delta / epsilon
        
        total_regret = exploration_regret + learning_regret
        
        print(f"遗憾界分析 (T={n_steps}, K={n_arms}, ε={epsilon}):")
        print(f"  探索遗憾: {exploration_regret:.1f}")
        print(f"  学习遗憾: {learning_regret:.1f}")
        print(f"  总遗憾界: {total_regret:.1f}")
        print(f"  平均每步遗憾: {total_regret/n_steps:.4f}")
        
        return total_regret
    
    @staticmethod
    def find_optimal_epsilon(n_steps: int, n_arms: int) -> float:
        """
        理论最优ε的计算
        
        对于固定时间T，最优ε约为：
        ε* ≈ sqrt(K·log(T) / T)
        
        这个公式的直觉：
        - T越大，ε越小（时间多就可以少探索）
        - K越大，ε越大（选项多需要多探索）
        """
        optimal_epsilon = np.sqrt(n_arms * np.log(n_steps) / n_steps)
        optimal_epsilon = min(1.0, optimal_epsilon)  # 确保ε≤1
        
        print(f"理论最优ε (T={n_steps}, K={n_arms}):")
        print(f"  ε* = {optimal_epsilon:.4f}")
        
        # 计算使用最优ε的遗憾界
        regret = EpsilonGreedyTheory.calculate_regret_bound(n_steps, n_arms, optimal_epsilon)
        
        return optimal_epsilon


# ================================================================================
# 主程序入口
# Main Program Entry
# ================================================================================

if __name__ == "__main__":
    print("╔" + "═"*78 + "╗")
    print("║" + " "*20 + "ε-贪婪算法深度教学".center(38) + " "*20 + "║")
    print("║" + " "*15 + "ε-Greedy Algorithm Deep Tutorial".center(48) + " "*15 + "║")
    print("╚" + "═"*78 + "╝")
    print("\n欢迎来到强化学习的第一课！")
    print("Welcome to your first lesson in Reinforcement Learning!")
    print("\n我们将深入理解ε-贪婪算法的每个细节。")
    print("We will deeply understand every detail of the ε-greedy algorithm.")
    print("="*80)
    
    # 1. 基础演示
    print("\n第一部分：基础演示")
    print("Part 1: Basic Demonstration")
    demonstrate_epsilon_greedy()
    
    # 2. 对比实验
    print("\n第二部分：ε值对比实验")
    print("Part 2: Epsilon Comparison Experiment")
    analyzer = EpsilonGreedyAnalyzer()
    analyzer.compare_epsilon_values()
    
    # 3. 理论分析
    print("\n第三部分：理论分析")
    print("Part 3: Theoretical Analysis")
    theory = EpsilonGreedyTheory()
    
    # 不同场景的最优ε
    scenarios = [
        (100, 10, "短期学习 Short-term"),
        (1000, 10, "中期学习 Medium-term"),
        (10000, 10, "长期学习 Long-term"),
        (1000, 100, "多选项 Many options"),
    ]
    
    print("\n不同场景的最优ε:")
    print("Optimal ε for different scenarios:")
    print("-"*40)
    for n_steps, n_arms, desc in scenarios:
        print(f"\n{desc}:")
        theory.find_optimal_epsilon(n_steps, n_arms)
    
    print("\n"*2)
    print("="*80)
    print("课程总结 Lesson Summary")
    print("="*80)
    print("""
    ε-贪婪算法的核心要点：
    
    1. 本质：用一个参数ε控制探索与利用的平衡
    2. 优点：简单、直观、有效、容易实现
    3. 缺点：不区分动作优劣，平等对待所有非最优动作
    4. 改进：衰减ε、自适应ε、基于不确定性的探索
    5. 应用：A/B测试、推荐系统、游戏AI、资源分配
    
    记住：没有免费的午餐！
    - 不探索，可能错过最优
    - 探索太多，浪费在次优
    - ε-贪婪给出了一个简单的平衡方案
    
    下一课：UCB算法 - 更智能的探索策略
    Next: UCB Algorithm - Smarter Exploration Strategy
    """)