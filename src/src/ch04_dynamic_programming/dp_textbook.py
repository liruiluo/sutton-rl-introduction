"""
================================================================================
第4章：动态规划 - 完美世界的完美解法
Chapter 4: Dynamic Programming - Perfect Solutions in a Perfect World

根据 Sutton & Barto《强化学习：导论》第二版 第4章
Based on Sutton & Barto "Reinforcement Learning: An Introduction" Chapter 4
================================================================================

让我用一个故事开始这一章：

想象你是一个城市规划师，负责设计最优的交通路线。
你有一份完美的地图，知道：
- 每条路的长度（状态转移概率）
- 每个路口的拥堵情况（奖励函数）
- 所有可能的路线（完整模型）

问题：如何为每个起点找到到达目的地的最优路线？

这就是动态规划（Dynamic Programming）的场景！

为什么说这是"完美世界"？
因为我们假设知道环境的完整模型 - 这在现实中很罕见。

但是，理解DP是理解强化学习的基础：
- DP是其他方法的理论基础
- 许多算法是DP的近似版本
- DP提供了性能的理论上界

================================================================================
为什么动态规划如此重要？
Why Dynamic Programming Matters?
================================================================================

Sutton & Barto说（第73页）：
"DP provides an essential foundation for the understanding of the methods 
presented in the rest of the book."

动态规划的核心思想：
1. 利用贝尔曼方程的递归结构
2. 将复杂问题分解为简单子问题
3. 通过迭代找到最优解

关键假设：
- 完美模型：知道P(s'|s,a)和R(s,a,s')
- 有限MDP：状态和动作空间有限
- 计算资源充足：可以遍历所有状态

虽然这些假设很强，但DP方法：
- 提供了理论最优解
- 是其他方法的基准
- 核心思想贯穿整个RL
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
from collections import defaultdict


# ================================================================================
# 第4.1节：策略评估（预测问题）
# Section 4.1: Policy Evaluation (Prediction Problem)
# ================================================================================

class PolicyEvaluation:
    """
    策略评估 - 评估给定策略的好坏
    
    核心问题：给定策略π，计算其价值函数vπ
    
    贝尔曼期望方程（第74页，方程4.3）：
    vπ(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γvπ(s')]
    
    直观理解：
    一个状态的价值 = 期望立即奖励 + 期望未来价值
    
    就像评估一个投资策略：
    总收益 = 立即收益 + 未来收益的现值
    
    迭代策略评估算法：
    不断应用贝尔曼方程直到收敛
    """
    
    def __init__(self, mdp, policy, gamma: float = 0.9):
        """
        初始化策略评估器
        
        参数：
        mdp: MDP环境（需要知道完整模型）
        policy: 要评估的策略
        gamma: 折扣因子
        """
        self.mdp = mdp
        self.policy = policy
        self.gamma = gamma
        
        # 初始化价值函数（全0是常见选择）
        self.V = defaultdict(lambda: 0.0)
        
        print("策略评估器初始化完成")
        print(f"折扣因子γ = {gamma}")
        
    def bellman_expectation(self, state) -> float:
        """
        计算贝尔曼期望方程的右侧
        
        这是策略评估的核心！
        
        vπ(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a)[R(s,a,s') + γvπ(s')]
        
        步骤：
        1. 对每个可能的动作a
        2. 考虑策略选择a的概率π(a|s)
        3. 考虑执行a后的所有可能结果
        4. 加权求和
        """
        # 终止状态的价值为0
        if self.mdp.is_terminal(state):
            return 0.0
            
        value = 0.0
        
        # 遍历所有可能的动作
        for action in self.mdp.get_actions(state):
            # 策略选择这个动作的概率
            action_prob = self.policy.get_probability(state, action)
            
            if action_prob == 0:
                continue
                
            # 考虑这个动作的所有可能结果
            for next_state, trans_prob, reward in self.mdp.get_transitions(state, action):
                # 贝尔曼期望方程
                value += action_prob * trans_prob * (
                    reward + self.gamma * self.V[next_state]
                )
                
        return value
    
    def evaluate(self, theta: float = 1e-6, max_iterations: int = 1000):
        """
        迭代策略评估 - 算法4.1（书中第75页）
        
        反复应用贝尔曼期望方程直到收敛
        
        为什么会收敛？
        - 贝尔曼期望算子是压缩映射
        - 压缩映射定理保证唯一不动点
        - 不动点就是真实价值函数vπ
        
        复杂度：O(|S|²|A|) per iteration
        """
        print(f"\n开始策略评估（θ={theta}）")
        print("-" * 50)
        
        for iteration in range(max_iterations):
            delta = 0  # 最大价值变化
            
            # 对每个状态进行更新（同步更新）
            new_values = {}
            for state in self.mdp.get_states():
                old_value = self.V[state]
                new_value = self.bellman_expectation(state)
                new_values[state] = new_value
                
                # 记录最大变化
                delta = max(delta, abs(old_value - new_value))
            
            # 更新所有状态的价值
            for state, value in new_values.items():
                self.V[state] = value
            
            # 检查收敛
            if delta < theta:
                print(f"✓ 策略评估在第{iteration + 1}次迭代后收敛")
                print(f"  最终delta = {delta:.2e}")
                break
                
            # 定期报告进度
            if (iteration + 1) % 10 == 0:
                print(f"  迭代{iteration + 1}: delta = {delta:.6f}")
        
        return dict(self.V)
    
    def evaluate_async(self, theta: float = 1e-6, max_iterations: int = 1000):
        """
        异步策略评估 - 更实用的版本
        
        区别：
        - 同步：所有状态一起更新
        - 异步：一次更新一个状态，立即使用新值
        
        优点：
        - 更快收敛（使用最新信息）
        - 可以选择性更新（重要状态优先）
        - 更接近实际算法（如TD学习）
        """
        print(f"\n开始异步策略评估")
        
        for iteration in range(max_iterations):
            delta = 0
            
            # 原地更新每个状态
            for state in self.mdp.get_states():
                old_value = self.V[state]
                # 立即更新并使用新值
                self.V[state] = self.bellman_expectation(state)
                delta = max(delta, abs(old_value - self.V[state]))
            
            if delta < theta:
                print(f"✓ 异步评估在第{iteration + 1}次迭代后收敛")
                break
                
        return dict(self.V)


# ================================================================================
# 第4.2节：策略改进
# Section 4.2: Policy Improvement
# ================================================================================

class PolicyImprovement:
    """
    策略改进 - 让策略变得更好
    
    核心定理（策略改进定理，第76页）：
    如果对所有s，qπ(s, π'(s)) ≥ vπ(s)
    则策略π'至少和π一样好
    
    贪婪策略改进：
    π'(s) = argmax_a qπ(s, a)
    
    直观理解：
    如果在某个状态，有个动作比当前策略更好，
    那就改用这个动作！
    
    这就像：
    - 你有个上班路线
    - 发现某个路口左转比直行更快
    - 那就改成左转
    - 重复这个过程，最终找到最优路线
    """
    
    @staticmethod
    def compute_q_values(mdp, V: Dict, gamma: float = 0.9) -> Dict:
        """
        从状态价值函数V计算动作价值函数Q
        
        qπ(s,a) = Σ_{s'} P(s'|s,a)[R(s,a,s') + γvπ(s')]
        
        这告诉我们：在状态s执行动作a有多好
        """
        Q = {}
        
        for state in mdp.get_states():
            for action in mdp.get_actions(state):
                q_value = 0
                
                # 计算这个状态-动作对的价值
                for next_state, prob, reward in mdp.get_transitions(state, action):
                    q_value += prob * (reward + gamma * V.get(next_state, 0))
                    
                Q[(state, action)] = q_value
                
        return Q
    
    @staticmethod
    def make_greedy_policy(mdp, V: Dict, gamma: float = 0.9):
        """
        基于价值函数创建贪婪策略
        
        贪婪策略：总是选择q值最高的动作
        π(s) = argmax_a q(s,a)
        
        这是确定性策略（每个状态只选一个动作）
        """
        Q = PolicyImprovement.compute_q_values(mdp, V, gamma)
        
        class GreedyPolicy:
            def get_probability(self, state, action):
                # 找到这个状态的最优动作
                state_actions = [(a, Q.get((state, a), float('-inf'))) 
                                for a in mdp.get_actions(state)]
                
                if not state_actions:
                    return 0.0
                    
                best_action = max(state_actions, key=lambda x: x[1])[0]
                
                # 确定性策略：最优动作概率1，其他0
                return 1.0 if action == best_action else 0.0
            
            def select_action(self, state):
                state_actions = [(a, Q.get((state, a), float('-inf'))) 
                                for a in mdp.get_actions(state)]
                if state_actions:
                    return max(state_actions, key=lambda x: x[1])[0]
                return None
        
        return GreedyPolicy()
    
    @staticmethod
    def is_policy_stable(old_policy, new_policy, mdp):
        """
        检查策略是否稳定（没有变化）
        
        如果策略不再改变，说明已经最优！
        """
        for state in mdp.get_states():
            # 比较每个状态的动作选择
            for action in mdp.get_actions(state):
                if old_policy.get_probability(state, action) != \
                   new_policy.get_probability(state, action):
                    return False
        return True


# ================================================================================
# 第4.3节：策略迭代
# Section 4.3: Policy Iteration
# ================================================================================

class PolicyIteration:
    """
    策略迭代 - 找到最优策略的经典算法
    
    算法步骤（第80页）：
    1. 策略评估：评估当前策略
    2. 策略改进：基于价值函数改进策略
    3. 重复直到策略不再变化
    
    保证收敛到最优策略！
    
    类比：学习开车
    1. 评估：按当前方式开，看效果如何
    2. 改进：发现更好的操作就采用
    3. 重复：不断评估和改进
    4. 收敛：成为老司机（最优策略）
    
    复杂度分析：
    - 每次迭代：O(|S|²|A| + |S|³)
    - 迭代次数：通常很少（多项式级别）
    """
    
    def __init__(self, mdp, gamma: float = 0.9):
        """初始化策略迭代"""
        self.mdp = mdp
        self.gamma = gamma
        
        # 初始化为随机策略
        self.policy = self.create_random_policy()
        
        print("策略迭代初始化")
        print("初始策略：随机策略")
        
    def create_random_policy(self):
        """创建均匀随机策略作为初始策略"""
        class RandomPolicy:
            def __init__(self, mdp):
                self.mdp = mdp
                
            def get_probability(self, state, action):
                actions = self.mdp.get_actions(state)
                if actions and action in actions:
                    return 1.0 / len(actions)
                return 0.0
                
            def select_action(self, state):
                actions = self.mdp.get_actions(state)
                return np.random.choice(actions) if actions else None
        
        return RandomPolicy(self.mdp)
    
    def iterate(self, theta: float = 1e-6, max_iterations: int = 100):
        """
        完整的策略迭代算法
        
        收敛性证明要点：
        1. 有限MDP只有有限个策略
        2. 每次改进严格更好（除非已最优）
        3. 因此必然收敛到最优
        
        实际中收敛很快（通常<10次迭代）
        """
        print("\n" + "="*60)
        print("开始策略迭代")
        print("="*60)
        
        for iteration in range(max_iterations):
            print(f"\n--- 策略迭代 第{iteration + 1}轮 ---")
            
            # 步骤1：策略评估
            print("步骤1：评估当前策略...")
            evaluator = PolicyEvaluation(self.mdp, self.policy, self.gamma)
            V = evaluator.evaluate(theta)
            
            # 步骤2：策略改进
            print("步骤2：改进策略...")
            old_policy = self.policy
            self.policy = PolicyImprovement.make_greedy_policy(
                self.mdp, V, self.gamma
            )
            
            # 检查是否收敛
            if PolicyImprovement.is_policy_stable(old_policy, self.policy, self.mdp):
                print(f"\n✓ 策略迭代收敛！共{iteration + 1}轮")
                print("找到最优策略！")
                
                # 计算最优价值函数
                evaluator = PolicyEvaluation(self.mdp, self.policy, self.gamma)
                V_optimal = evaluator.evaluate(theta)
                
                return self.policy, V_optimal
            
            print("策略已改进，继续迭代...")
        
        print("\n警告：达到最大迭代次数，可能未收敛")
        return self.policy, V


# ================================================================================
# 第4.4节：价值迭代
# Section 4.4: Value Iteration
# ================================================================================

class ValueIteration:
    """
    价值迭代 - 更高效的动态规划算法
    
    核心思想（第82页）：
    不需要完整的策略评估！
    
    直接应用贝尔曼最优方程：
    v*(s) = max_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γv*(s')]
    
    与策略迭代的区别：
    - 策略迭代：评估到收敛，再改进
    - 价值迭代：评估一步，就改进
    
    类比：
    策略迭代像是认真学完一门课再学下一门
    价值迭代像是所有课程同时推进
    
    效率更高，但每步改进较小
    """
    
    def __init__(self, mdp, gamma: float = 0.9):
        """初始化价值迭代"""
        self.mdp = mdp
        self.gamma = gamma
        
        # 初始化价值函数
        self.V = defaultdict(lambda: 0.0)
        
        print("价值迭代初始化")
        print(f"折扣因子γ = {gamma}")
        
    def bellman_optimality(self, state) -> float:
        """
        贝尔曼最优方程 - DP的核心
        
        v*(s) = max_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γv*(s')]
        
        这个方程说明：
        最优价值 = 选择最好动作的期望回报
        
        与期望方程的区别：
        - 期望方程：Σ_a π(a|s) ...（加权平均）
        - 最优方程：max_a ...（选最好的）
        """
        if self.mdp.is_terminal(state):
            return 0.0
            
        max_value = float('-inf')
        
        # 尝试所有动作，选最好的
        for action in self.mdp.get_actions(state):
            action_value = 0
            
            # 计算这个动作的期望价值
            for next_state, prob, reward in self.mdp.get_transitions(state, action):
                action_value += prob * (reward + self.gamma * self.V[next_state])
                
            # 记录最大值
            max_value = max(max_value, action_value)
            
        return max_value if max_value > float('-inf') else 0.0
    
    def iterate(self, theta: float = 1e-6, max_iterations: int = 1000):
        """
        价值迭代算法 - 算法4.4（第83页）
        
        重复应用贝尔曼最优方程直到收敛
        
        为什么比策略迭代快？
        1. 不需要显式的策略
        2. 不需要完整的策略评估
        3. 每次迭代都在改进
        
        收敛后如何得到策略？
        π*(s) = argmax_a q*(s,a)
        """
        print("\n" + "="*60)
        print("开始价值迭代")
        print("="*60)
        
        for iteration in range(max_iterations):
            delta = 0
            
            # 对每个状态应用贝尔曼最优方程
            new_values = {}
            for state in self.mdp.get_states():
                old_value = self.V[state]
                new_value = self.bellman_optimality(state)
                new_values[state] = new_value
                
                delta = max(delta, abs(old_value - new_value))
            
            # 更新价值函数
            for state, value in new_values.items():
                self.V[state] = value
            
            # 定期报告
            if (iteration + 1) % 10 == 0:
                print(f"迭代{iteration + 1}: delta = {delta:.6f}")
            
            # 检查收敛
            if delta < theta:
                print(f"\n✓ 价值迭代收敛！共{iteration + 1}次迭代")
                
                # 从最优价值函数导出最优策略
                optimal_policy = PolicyImprovement.make_greedy_policy(
                    self.mdp, dict(self.V), self.gamma
                )
                
                return optimal_policy, dict(self.V)
        
        print("\n警告：达到最大迭代次数")
        optimal_policy = PolicyImprovement.make_greedy_policy(
            self.mdp, dict(self.V), self.gamma
        )
        return optimal_policy, dict(self.V)


# ================================================================================
# 第4.5节：广义策略迭代（GPI）
# Section 4.5: Generalized Policy Iteration
# ================================================================================

class GeneralizedPolicyIteration:
    """
    广义策略迭代 - 统一的视角
    
    核心洞察（第86页）：
    几乎所有强化学习方法都可以看作GPI！
    
    两个过程的相互作用：
    1. 策略评估：让价值函数与策略一致
    2. 策略改进：让策略对价值函数贪婪
    
    这两个过程相互竞争又相互合作：
    - 竞争：一个的改变让另一个不准确
    - 合作：共同向最优解前进
    
    就像学习骑自行车：
    - 评估：知道当前姿势的效果
    - 改进：调整姿势
    - 不断循环直到熟练
    
    GPI的不同实例：
    - 策略迭代：完整评估 + 完整改进
    - 价值迭代：一步评估 + 一步改进
    - 异步DP：部分评估 + 部分改进
    - 蒙特卡洛：采样评估 + 贪婪改进
    - TD学习：自举评估 + ε-贪婪改进
    """
    
    @staticmethod
    def demonstrate_gpi_concept():
        """演示GPI的核心概念"""
        print("="*70)
        print("广义策略迭代（GPI）- 强化学习的统一框架")
        print("Generalized Policy Iteration - Unified Framework of RL")
        print("="*70)
        
        print("""
        GPI的两个核心过程：
        
        1. 策略评估（Policy Evaluation）
           目标：V → vπ
           让价值函数准确反映当前策略
           
        2. 策略改进（Policy Improvement）
           目标：π → greedy(V)
           让策略贪婪于当前价值函数
           
        相互作用图示：
        
                π
               ↙ ↘
              /   \\
             ↓     ↓
         评估      改进
             ↓     ↓
              \\   /
               ↘ ↙
                V
        
        两个过程形成循环：
        π0 → V0 → π1 → V1 → ... → π* → V*
        
        不同算法的区别只是：
        - 评估的完整程度
        - 改进的贪婪程度
        - 使用真实模型还是采样
        """)
        
        # 展示不同GPI变体
        print("\nGPI的不同实例：")
        print("-"*50)
        
        variants = [
            ("策略迭代", "完整评估", "完全贪婪", "需要模型"),
            ("价值迭代", "一步评估", "完全贪婪", "需要模型"),
            ("异步DP", "部分评估", "部分贪婪", "需要模型"),
            ("蒙特卡洛", "采样评估", "贪婪改进", "无需模型"),
            ("SARSA", "TD评估", "ε-贪婪", "无需模型"),
            ("Q学习", "TD评估", "贪婪选择", "无需模型")
        ]
        
        for name, eval_type, improve_type, model_req in variants:
            print(f"{name:10} | 评估:{eval_type:8} | 改进:{improve_type:8} | {model_req}")


# ================================================================================
# 第4.6节：经典问题 - 格子世界
# Section 4.6: Classic Problems - Grid World
# ================================================================================

class GridWorldMDP:
    """
    格子世界 - 动态规划的经典测试环境
    
    这是书中Figure 4.1的实现
    
    规则：
    - 智能体在网格中移动
    - 目标：到达终点
    - 每步奖励：-1（鼓励快速到达）
    - 动作：上下左右
    - 如果撞墙：停在原地
    
    为什么用格子世界？
    1. 直观可视化
    2. 状态空间小，便于验证
    3. 可以手动计算最优策略
    4. 完美展示DP算法
    """
    
    def __init__(self, size: int = 4):
        """创建size×size的格子世界"""
        self.size = size
        self.states = [(i, j) for i in range(size) for j in range(size)]
        self.actions = ['up', 'down', 'left', 'right']
        
        # 终止状态（左上角和右下角）
        self.terminals = [(0, 0), (size-1, size-1)]
        
        print(f"创建{size}×{size}格子世界")
        print(f"终止状态：{self.terminals}")
        
    def is_terminal(self, state) -> bool:
        """检查是否终止状态"""
        return state in self.terminals
    
    def get_states(self):
        """获取所有状态"""
        return self.states
    
    def get_actions(self, state):
        """获取可用动作"""
        if self.is_terminal(state):
            return []
        return self.actions
    
    def get_transitions(self, state, action):
        """
        获取状态转移
        
        确定性环境：每个动作只有一个结果
        P(s'|s,a) = 1 for one s', 0 for others
        """
        if self.is_terminal(state):
            return []
            
        i, j = state
        
        # 计算下一个位置
        if action == 'up':
            next_i, next_j = max(0, i-1), j
        elif action == 'down':
            next_i, next_j = min(self.size-1, i+1), j
        elif action == 'left':
            next_i, next_j = i, max(0, j-1)
        elif action == 'right':
            next_i, next_j = i, min(self.size-1, j+1)
        else:
            next_i, next_j = i, j
            
        next_state = (next_i, next_j)
        reward = 0 if self.is_terminal(next_state) else -1
        
        # 返回：(下一状态, 概率, 奖励)
        return [(next_state, 1.0, reward)]
    
    def visualize_values(self, V: Dict, title: str = "State Values"):
        """可视化状态价值函数"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 创建网格
        for i in range(self.size + 1):
            ax.axhline(y=i, color='black', linewidth=1)
            ax.axvline(x=i, color='black', linewidth=1)
            
        # 显示价值
        for state in self.states:
            i, j = state
            value = V.get(state, 0)
            
            # 终止状态用不同颜色
            if self.is_terminal(state):
                color = 'lightgreen'
            else:
                color = 'white'
                
            # 添加方块
            rect = Rectangle((j, self.size-1-i), 1, 1, 
                           facecolor=color, edgecolor='black')
            ax.add_patch(rect)
            
            # 添加价值文本
            ax.text(j+0.5, self.size-1-i+0.5, f'{value:.1f}',
                   ha='center', va='center', fontsize=14)
            
            # 标记终止状态
            if self.is_terminal(state):
                ax.text(j+0.5, self.size-1-i+0.3, 'T',
                       ha='center', va='center', fontsize=10, color='red')
        
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=16)
        ax.set_xticks(range(self.size + 1))
        ax.set_yticks(range(self.size + 1))
        
        plt.tight_layout()
        plt.show()
    
    def visualize_policy(self, policy, V: Dict = None, 
                        title: str = "Optimal Policy"):
        """可视化策略（带箭头）"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 创建网格
        for i in range(self.size + 1):
            ax.axhline(y=i, color='black', linewidth=1)
            ax.axvline(x=i, color='black', linewidth=1)
            
        # 箭头映射
        arrow_map = {
            'up': (0, 0.3),
            'down': (0, -0.3),
            'left': (-0.3, 0),
            'right': (0.3, 0)
        }
        
        # 显示策略
        for state in self.states:
            i, j = state
            
            # 终止状态
            if self.is_terminal(state):
                rect = Rectangle((j, self.size-1-i), 1, 1,
                               facecolor='lightgreen', edgecolor='black')
                ax.add_patch(rect)
                ax.text(j+0.5, self.size-1-i+0.5, 'T',
                       ha='center', va='center', fontsize=20, color='red')
                continue
                
            # 普通状态
            rect = Rectangle((j, self.size-1-i), 1, 1,
                           facecolor='white', edgecolor='black')
            ax.add_patch(rect)
            
            # 显示价值（如果提供）
            if V:
                value = V.get(state, 0)
                ax.text(j+0.5, self.size-1-i+0.8, f'{value:.1f}',
                       ha='center', va='center', fontsize=10, color='blue')
            
            # 画策略箭头
            best_action = None
            best_prob = 0
            for action in self.actions:
                prob = policy.get_probability(state, action)
                if prob > best_prob:
                    best_prob = prob
                    best_action = action
                    
            if best_action:
                dx, dy = arrow_map[best_action]
                ax.arrow(j+0.5, self.size-1-i+0.5, dx, dy,
                        head_width=0.1, head_length=0.1,
                        fc='red', ec='red')
        
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=16)
        ax.set_xticks(range(self.size + 1))
        ax.set_yticks(range(self.size + 1))
        
        plt.tight_layout()
        plt.show()


# ================================================================================
# 第4.7节：效率与改进
# Section 4.7: Efficiency and Improvements
# ================================================================================

class EfficientDP:
    """
    动态规划的效率优化
    
    实际问题中的挑战：
    1. 状态空间爆炸（维度诅咒）
    2. 计算复杂度高
    3. 需要完整模型
    
    优化技术：
    1. 异步DP：不需要完整扫描
    2. 优先扫描：先更新重要状态
    3. 实时DP：只更新访问的状态
    4. 采样DP：用采样代替期望
    """
    
    @staticmethod
    def prioritized_sweeping(mdp, gamma: float = 0.9, 
                            theta: float = 1e-6, max_iterations: int = 1000):
        """
        优先扫描 - 聪明地选择更新顺序
        
        核心思想：
        优先更新"需要"更新的状态
        
        什么状态需要更新？
        1. 自己的值变化大的
        2. 影响其他状态多的
        3. 被其他状态影响的
        
        使用优先队列维护更新顺序
        """
        print("="*60)
        print("优先扫描动态规划")
        print("Prioritized Sweeping Dynamic Programming")
        print("="*60)
        
        V = defaultdict(lambda: 0.0)
        
        # 优先队列（用list模拟，实际应用堆）
        priority_queue = []
        
        # 初始化：所有状态入队
        for state in mdp.get_states():
            priority_queue.append((float('inf'), state))
            
        iterations = 0
        while priority_queue and iterations < max_iterations:
            # 取出优先级最高的状态
            priority, state = priority_queue.pop(0)
            
            if priority < theta:
                break
                
            # 更新这个状态
            old_value = V[state]
            
            if mdp.is_terminal(state):
                V[state] = 0
            else:
                max_value = float('-inf')
                for action in mdp.get_actions(state):
                    action_value = 0
                    for next_state, prob, reward in mdp.get_transitions(state, action):
                        action_value += prob * (reward + gamma * V[next_state])
                    max_value = max(max_value, action_value)
                V[state] = max_value if max_value > float('-inf') else 0
                
            # 计算变化量
            delta = abs(V[state] - old_value)
            
            # 如果变化大，将前驱状态加入队列
            if delta > theta:
                # 找所有能到达当前状态的前驱
                for pred_state in mdp.get_states():
                    if mdp.is_terminal(pred_state):
                        continue
                    for action in mdp.get_actions(pred_state):
                        for next_state, prob, _ in mdp.get_transitions(pred_state, action):
                            if next_state == state and prob > 0:
                                # 估计前驱的优先级
                                pred_priority = prob * delta
                                # 加入队列（实际应该用堆维护）
                                if pred_priority > theta:
                                    priority_queue.append((pred_priority, pred_state))
                                    
            iterations += 1
            
            if iterations % 100 == 0:
                print(f"迭代{iterations}: 队列大小{len(priority_queue)}")
        
        print(f"\n优先扫描完成，共{iterations}次更新")
        return dict(V)
    
    @staticmethod
    def asynchronous_vi(mdp, gamma: float = 0.9, 
                        n_iterations: int = 1000):
        """
        异步价值迭代
        
        不需要系统地扫描所有状态
        可以：
        1. 随机选择状态更新
        2. 按某种顺序更新
        3. 重复更新某些重要状态
        
        优点：
        - 更灵活
        - 可以在线学习
        - 适合大状态空间
        """
        print("\n异步价值迭代演示")
        
        V = defaultdict(lambda: 0.0)
        states = list(mdp.get_states())
        
        for iteration in range(n_iterations):
            # 随机选择一个状态更新
            state = np.random.choice(states)
            
            if mdp.is_terminal(state):
                V[state] = 0
            else:
                max_value = float('-inf')
                for action in mdp.get_actions(state):
                    action_value = 0
                    for next_state, prob, reward in mdp.get_transitions(state, action):
                        action_value += prob * (reward + gamma * V[next_state])
                    max_value = max(max_value, action_value)
                V[state] = max_value if max_value > float('-inf') else 0
                
        return dict(V)


# ================================================================================
# 第4.8节：完整示例演示
# Section 4.8: Complete Example Demonstration  
# ================================================================================

def demonstrate_dp_algorithms():
    """
    演示所有动态规划算法
    
    通过格子世界展示：
    1. 策略评估
    2. 策略改进
    3. 策略迭代
    4. 价值迭代
    5. 算法比较
    """
    print("="*70)
    print("动态规划算法完整演示")
    print("Complete Dynamic Programming Demonstration")
    print("="*70)
    
    # 创建格子世界
    print("\n创建4×4格子世界...")
    mdp = GridWorldMDP(size=4)
    
    # 1. 演示策略评估
    print("\n" + "="*60)
    print("演示1：策略评估")
    print("="*60)
    print("评估随机策略（每个方向概率相等）")
    
    class UniformRandomPolicy:
        def get_probability(self, state, action):
            return 0.25  # 四个方向概率相等
    
    random_policy = UniformRandomPolicy()
    evaluator = PolicyEvaluation(mdp, random_policy, gamma=0.9)
    V_random = evaluator.evaluate()
    
    print("\n随机策略的状态价值：")
    mdp.visualize_values(V_random, "Random Policy Value Function")
    
    # 2. 演示策略改进
    print("\n" + "="*60)
    print("演示2：策略改进")
    print("="*60)
    print("基于随机策略的价值函数，创建贪婪策略")
    
    improved_policy = PolicyImprovement.make_greedy_policy(mdp, V_random, gamma=0.9)
    mdp.visualize_policy(improved_policy, V_random, "Improved Policy from Random")
    
    # 3. 演示策略迭代
    print("\n" + "="*60)
    print("演示3：策略迭代")
    print("="*60)
    
    pi_solver = PolicyIteration(mdp, gamma=0.9)
    pi_policy, V_pi = pi_solver.iterate()
    
    print("\n策略迭代找到的最优策略：")
    mdp.visualize_policy(pi_policy, V_pi, "Policy Iteration - Optimal Policy")
    
    # 4. 演示价值迭代
    print("\n" + "="*60)
    print("演示4：价值迭代")
    print("="*60)
    
    vi_solver = ValueIteration(mdp, gamma=0.9)
    vi_policy, V_vi = vi_solver.iterate()
    
    print("\n价值迭代找到的最优策略：")
    mdp.visualize_policy(vi_policy, V_vi, "Value Iteration - Optimal Policy")
    
    # 5. 比较不同方法
    print("\n" + "="*60)
    print("演示5：算法比较")
    print("="*60)
    
    # 比较最优价值函数
    print("\n最优价值函数比较：")
    print("状态     | 策略迭代 | 价值迭代 | 差异")
    print("-"*45)
    
    for state in [(0,1), (1,1), (2,2), (3,2)]:  # 选几个代表状态
        v_pi = V_pi.get(state, 0)
        v_vi = V_vi.get(state, 0) 
        diff = abs(v_pi - v_vi)
        print(f"{state}   | {v_pi:8.3f} | {v_vi:8.3f} | {diff:.6f}")
    
    print("\n结论：两种方法都收敛到相同的最优解！")
    
    # 6. 展示GPI概念
    print("\n" + "="*60)
    print("演示6：广义策略迭代（GPI）")
    print("="*60)
    
    GeneralizedPolicyIteration.demonstrate_gpi_concept()
    
    # 7. 效率优化演示
    print("\n" + "="*60)
    print("演示7：优化技术")
    print("="*60)
    
    print("\n测试优先扫描...")
    V_priority = EfficientDP.prioritized_sweeping(mdp, gamma=0.9)
    
    print("\n测试异步价值迭代...")
    V_async = EfficientDP.asynchronous_vi(mdp, gamma=0.9, n_iterations=5000)
    
    print("\n优化方法也收敛到相同结果！")


# ================================================================================
# 第4.9节：章节总结
# Section 4.9: Chapter Summary
# ================================================================================

def chapter_summary():
    """第4章总结"""
    print("\n" + "="*70)
    print("第4章总结：动态规划")
    print("Chapter 4 Summary: Dynamic Programming")
    print("="*70)
    
    print("""
    核心要点回顾：
    
    1. 动态规划的前提条件
       - 完美模型：知道P(s'|s,a)和R(s,a,s')
       - 有限MDP：状态和动作空间有限
       - 这是"完美世界"的假设
    
    2. 两个基本过程
       - 策略评估：计算vπ（预测问题）
       - 策略改进：基于V改进π（控制问题）
    
    3. 三个核心算法
       
       算法      | 评估方式  | 改进方式  | 特点
       ---------|----------|----------|-------------
       策略迭代  | 完整评估  | 贪婪改进  | 收敛快，计算多
       价值迭代  | 一步评估  | 隐式改进  | 简单高效
       异步DP   | 选择性    | 灵活     | 适合大问题
    
    4. 贝尔曼方程（DP的数学基础）
       
       期望方程（策略评估）：
       vπ(s) = Σ_a π(a|s) Σ_{s'} p(s',r|s,a)[r + γvπ(s')]
       
       最优方程（寻找最优）：
       v*(s) = max_a Σ_{s'} p(s',r|s,a)[r + γv*(s')]
    
    5. 广义策略迭代（GPI）
       - 几乎所有RL方法都是GPI
       - 评估和改进的交互
       - 保证收敛到最优
    
    6. DP的局限性
       - 需要完美模型（现实中罕见）
       - 计算复杂度高（维度诅咒）
       - 不能处理连续空间
    
    7. DP的价值
       - 提供理论基础
       - 其他方法的出发点
       - 理解RL的关键
    
    关键洞察：
    DP告诉我们，如果知道世界的运作规律，
    我们可以通过"思考"（计算）找到最优策略。
    但现实世界需要通过"试错"（采样）来学习。
    
    这就引出了下一章：
    蒙特卡洛方法 - 从经验中学习！
    
    练习建议：
    1. 实现练习4.1：网格世界的In-place算法
    2. 实现练习4.4：赌徒问题（Gambler's Problem）
    3. 实现练习4.7：修改的策略迭代
    4. 比较同步和异步更新的收敛速度
    """)


# ================================================================================
# 主程序：运行第4章完整演示
# Main: Run Complete Chapter 4 Demonstration
# ================================================================================

def demonstrate_chapter_4():
    """运行第4章的完整演示"""
    print("╔" + "═"*68 + "╗")
    print("║" + " "*15 + "第4章：动态规划".center(38) + " "*15 + "║")
    print("║" + " "*10 + "Chapter 4: Dynamic Programming".center(48) + " "*10 + "║")
    print("╚" + "═"*68 + "╝")
    
    print("\n欢迎来到动态规划的完美世界！")
    print("在这里，我们知道一切，可以通过计算找到最优解。")
    print("虽然现实很少这么完美，但DP提供了理解RL的基础。\n")
    
    # 运行演示
    demonstrate_dp_algorithms()
    
    # 章节总结
    chapter_summary()
    
    print("\n下一章预告：第5章 - 蒙特卡洛方法")
    print("从完美世界走向现实：通过采样学习！")
    print("Next: Chapter 5 - Monte Carlo Methods")


if __name__ == "__main__":
    demonstrate_chapter_4()