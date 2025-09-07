"""
================================================================================
第6章：时序差分学习 - 边做边学的智慧
Chapter 6: Temporal-Difference Learning - Learning While Doing

根据 Sutton & Barto《强化学习：导论》第二版 第6章
Based on Sutton & Barto "Reinforcement Learning: An Introduction" Chapter 6
================================================================================

让我用一个故事开始这一章：

想象你是一个出租车司机，刚到一个新城市工作。

用DP方法学习：
你需要一张完整地图，在家计算每条路线的时间。
问题：你没有地图！

用MC方法学习：
每天下班后，回顾今天所有行程，总结经验。
问题：要等一整天才能学到东西！

用TD方法学习（边做边学）：
每过一个路口，立即更新这段路的时间估计。
不需要地图，不需要等到终点，实时学习！

这就是TD学习的精髓：结合了DP的自举和MC的采样！

================================================================================
为什么TD学习是强化学习的核心？
Why TD Learning is Central to RL?
================================================================================

Sutton & Barto说（第119页）：
"If one had to identify one idea as central and novel to reinforcement learning, 
it would undoubtedly be temporal-difference learning."

TD的革命性在于：
1. 不需要模型（像MC）
2. 不需要等到结束（像DP）
3. 在线学习，实时更新
4. 低方差，快速收敛

TD是现代深度强化学习的基础：
- Q-learning → DQN
- SARSA → A3C
- TD(λ) → TD3
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import defaultdict
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arrow
import matplotlib.patches as mpatches


# ================================================================================
# 第6.1节：TD预测
# Section 6.1: TD Prediction
# ================================================================================

class TDPrediction:
    """
    时序差分预测 - TD学习的基础
    
    核心方程（第120页，最重要的方程之一）：
    V(St) ← V(St) + α[Rt+1 + γV(St+1) - V(St)]
    
    分解这个方程：
    - V(St): 当前状态的价值估计
    - Rt+1 + γV(St+1): TD目标（自举估计）
    - Rt+1 + γV(St+1) - V(St): TD误差δt
    - α: 学习率
    
    为什么叫"时序差分"？
    - 时序：使用不同时间的估计
    - 差分：V(St+1) - V(St)的差异
    
    TD结合了MC和DP的优点：
    - 像MC：从真实经验学习，不需要模型
    - 像DP：使用自举，不需要等到结束
    """
    
    def __init__(self, gamma: float = 0.9, alpha: float = 0.1):
        """
        初始化TD预测
        
        gamma: 折扣因子
        alpha: 学习率（步长）
        """
        self.gamma = gamma
        self.alpha = alpha
        
        # 价值函数
        self.V = defaultdict(lambda: 0.0)
        
        # 记录TD误差（用于分析）
        self.td_errors = []
        
        print("TD预测初始化")
        print(f"参数：γ={gamma}, α={alpha}")
        
    def td0_update(self, state, reward, next_state):
        """
        TD(0)更新 - 单步TD
        
        这是最基础的TD算法！
        每一步都学习，不需要等到回合结束。
        
        TD目标：Rt+1 + γV(St+1)
        - 用下一状态的估计代替未来的真实回报
        - 这就是"自举"（bootstrapping）
        
        类比学开车：
        - MC：开完全程才知道要多久
        - TD：每过一个路口就更新时间估计
        """
        # TD目标
        td_target = reward + self.gamma * self.V[next_state]
        
        # TD误差（预测误差）
        td_error = td_target - self.V[state]
        
        # 更新价值
        self.V[state] += self.alpha * td_error
        
        # 记录误差
        self.td_errors.append(td_error)
        
        return td_error
    
    def episode(self, trajectory: List[Tuple]):
        """
        处理一个回合
        
        trajectory: [(s0,a0,r1), (s1,a1,r2), ...]
        
        注意：TD可以在线更新，不需要等到回合结束！
        """
        total_error = 0
        
        for t in range(len(trajectory)):
            state, action, reward = trajectory[t]
            
            # 获取下一状态
            if t + 1 < len(trajectory):
                next_state = trajectory[t + 1][0]
            else:
                next_state = None  # 终止状态
                
            # TD更新
            if next_state is not None:
                error = self.td0_update(state, reward, next_state)
                total_error += abs(error)
            else:
                # 终止状态的处理
                td_target = reward  # 没有未来了
                td_error = td_target - self.V[state]
                self.V[state] += self.alpha * td_error
                total_error += abs(td_error)
                
        return total_error / len(trajectory)
    
    def batch_td0(self, episodes: List[List[Tuple]], epochs: int = 1):
        """
        批量TD(0)学习
        
        可以多次遍历经验（类似经验回放）
        """
        print(f"\n批量TD(0)学习（{len(episodes)}个回合，{epochs}轮）")
        
        for epoch in range(epochs):
            total_error = 0
            
            for episode in episodes:
                error = self.episode(episode)
                total_error += error
                
            avg_error = total_error / len(episodes)
            
            if (epoch + 1) % 10 == 0:
                print(f"  轮{epoch + 1}: 平均TD误差={avg_error:.4f}")
                
        return dict(self.V)


# ================================================================================
# 第6.2节：TD的优势
# Section 6.2: Advantages of TD
# ================================================================================

class TDAdvantages:
    """
    展示TD学习的优势
    
    通过对比实验展示TD相对于MC和DP的优点
    """
    
    @staticmethod
    def compare_convergence_speed():
        """
        比较TD和MC的收敛速度
        
        关键发现（第124页）：
        TD通常比MC收敛更快！
        
        原因：
        1. TD方差更低（只有一步的随机性）
        2. TD可以在线学习
        3. TD利用了马尔可夫性质
        """
        print("="*60)
        print("TD vs MC 收敛速度比较")
        print("="*60)
        
        # 创建简单的马尔可夫链
        print("\n创建随机游走环境...")
        print("A -- B -- C -- D -- E")
        print("奖励：A=0, E=1, 其他=0")
        
        # 真实价值（可以计算出来）
        true_values = {
            'A': 1/6,
            'B': 2/6,
            'C': 3/6,
            'D': 4/6,
            'E': 5/6
        }
        
        print("\n真实价值：")
        for state, value in true_values.items():
            print(f"  V({state}) = {value:.3f}")
        
        # 生成经验
        def generate_random_walk():
            """生成随机游走轨迹"""
            states = ['A', 'B', 'C', 'D', 'E']
            trajectory = []
            state_idx = 2  # 从C开始
            
            while True:
                state = states[state_idx]
                
                # 随机选择方向
                if np.random.random() < 0.5:
                    next_idx = state_idx - 1
                else:
                    next_idx = state_idx + 1
                    
                # 检查终止
                if next_idx < 0:
                    trajectory.append((state, 'left', 0))
                    break
                elif next_idx >= len(states):
                    trajectory.append((state, 'right', 1))
                    break
                else:
                    trajectory.append((state, 'move', 0))
                    state_idx = next_idx
                    
            return trajectory
        
        # 生成100个回合
        episodes = [generate_random_walk() for _ in range(100)]
        
        # TD学习
        print("\n\nTD(0)学习：")
        td = TDPrediction(gamma=1.0, alpha=0.1)
        td_values = td.batch_td0(episodes, epochs=10)
        
        print("\nTD学习结果：")
        for state in ['A', 'B', 'C', 'D', 'E']:
            td_v = td_values.get(state, 0)
            true_v = true_values[state]
            error = abs(td_v - true_v)
            print(f"  V({state}): TD={td_v:.3f}, 真实={true_v:.3f}, 误差={error:.3f}")
        
        # MC学习（对比）
        print("\n\nMC学习（对比）：")
        mc_values = defaultdict(lambda: 0.0)
        mc_counts = defaultdict(lambda: 0)
        
        for episode in episodes:
            # 计算回报
            G = 0
            for t in range(len(episode) - 1, -1, -1):
                state, _, reward = episode[t]
                G = reward + G  # γ=1
                
                # MC更新
                mc_counts[state] += 1
                mc_values[state] += (G - mc_values[state]) / mc_counts[state]
        
        print("\nMC学习结果：")
        for state in ['A', 'B', 'C', 'D', 'E']:
            mc_v = mc_values[state]
            true_v = true_values[state]
            error = abs(mc_v - true_v)
            print(f"  V({state}): MC={mc_v:.3f}, 真实={true_v:.3f}, 误差={error:.3f}")
        
        print("\n结论：TD通常收敛更快，误差更小！")
    
    @staticmethod
    def demonstrate_online_learning():
        """
        演示TD的在线学习能力
        
        TD可以在每一步后立即学习，
        不需要等到回合结束！
        """
        print("\n" + "="*60)
        print("TD在线学习演示")
        print("="*60)
        
        print("\n场景：出租车司机实时学习路况")
        print("每经过一个路口，立即更新时间估计")
        
        # 模拟路径
        path = [
            ("起点", "路口A", 5, "预计5分钟，实际5分钟"),
            ("路口A", "路口B", 8, "预计10分钟，实际8分钟"),
            ("路口B", "路口C", 12, "预计10分钟，实际12分钟"),
            ("路口C", "终点", 7, "预计10分钟，实际7分钟")
        ]
        
        td = TDPrediction(gamma=0.9, alpha=0.3)
        
        # 初始估计
        for from_loc, to_loc, _, _ in path:
            td.V[from_loc] = 10  # 初始估计都是10分钟
        
        print("\n初始时间估计（都是10分钟）")
        
        print("\n开始行程（实时更新）：")
        print("-"*40)
        
        for i, (from_loc, to_loc, actual_time, desc) in enumerate(path):
            print(f"\n第{i+1}段：{from_loc} → {to_loc}")
            print(f"  {desc}")
            
            # TD更新
            old_estimate = td.V[from_loc]
            td_target = actual_time + td.gamma * td.V[to_loc]
            td_error = td_target - old_estimate
            td.V[from_loc] += td.alpha * td_error
            
            print(f"  更新：{old_estimate:.1f} → {td.V[from_loc]:.1f}")
            print(f"  TD误差：{td_error:.2f}")
        
        print("\n学习后的时间估计：")
        for from_loc, _, _, _ in path:
            print(f"  {from_loc}: {td.V[from_loc]:.1f}分钟")
        
        print("\n关键：每一步都在学习，不需要等到终点！")


# ================================================================================
# 第6.3节：SARSA - 在策略TD控制
# Section 6.3: SARSA - On-Policy TD Control
# ================================================================================

class SARSA:
    """
    SARSA - State-Action-Reward-State-Action
    
    第一个完整的TD控制算法！
    
    核心方程（第130页）：
    Q(St,At) ← Q(St,At) + α[Rt+1 + γQ(St+1,At+1) - Q(St,At)]
    
    为什么叫SARSA？
    因为用到了五个元素：(S,A,R,S',A')
    
    SARSA的特点：
    - 在策略（on-policy）：评估和改进同一个策略
    - 保守：学习实际执行的动作
    - 安全：避免危险的探索
    
    类比：学开车
    SARSA像是谨慎的学生：
    - 学习实际开的路线
    - 不会想象"如果我开快点会怎样"
    - 更安全但可能更慢
    """
    
    def __init__(self, n_actions: int, gamma: float = 0.9, 
                 alpha: float = 0.1, epsilon: float = 0.1):
        """
        初始化SARSA
        
        n_actions: 动作数量
        gamma: 折扣因子
        alpha: 学习率
        epsilon: 探索率（ε-贪婪）
        """
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        
        # Q函数
        self.Q = defaultdict(lambda: np.zeros(n_actions))
        
        # 统计
        self.episode_rewards = []
        self.episode_lengths = []
        
        print("SARSA初始化")
        print(f"参数：γ={gamma}, α={alpha}, ε={epsilon}")
    
    def select_action(self, state) -> int:
        """
        ε-贪婪动作选择
        
        这是SARSA的策略！
        """
        if np.random.random() < self.epsilon:
            # 探索：随机选择
            return np.random.randint(self.n_actions)
        else:
            # 利用：选择最优
            return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state, next_action):
        """
        SARSA更新
        
        关键：使用实际要执行的next_action
        而不是最优动作（这是与Q-learning的区别）
        """
        # TD目标
        td_target = reward + self.gamma * self.Q[next_state][next_action]
        
        # TD误差
        td_error = td_target - self.Q[state][action]
        
        # 更新Q值
        self.Q[state][action] += self.alpha * td_error
        
        return td_error
    
    def episode(self, env, max_steps: int = 1000):
        """
        运行一个回合
        
        完整的SARSA算法流程
        """
        state = env.reset()
        action = self.select_action(state)
        
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            # 执行动作
            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                # 终止状态更新
                td_target = reward
                td_error = td_target - self.Q[state][action]
                self.Q[state][action] += self.alpha * td_error
                break
            
            # 选择下一个动作（SARSA的关键）
            next_action = self.select_action(next_state)
            
            # SARSA更新
            self.update(state, action, reward, next_state, next_action)
            
            # 转移
            state = next_state
            action = next_action
        
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(steps)
        
        return total_reward
    
    def train(self, env, n_episodes: int = 1000):
        """
        训练SARSA
        """
        print(f"\n训练SARSA（{n_episodes}回合）...")
        
        for episode in range(n_episodes):
            reward = self.episode(env)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"  回合{episode + 1}: 平均奖励={avg_reward:.2f}")
        
        return self.Q


# ================================================================================
# 第6.4节：Q-Learning - 离策略TD控制
# Section 6.4: Q-Learning - Off-Policy TD Control
# ================================================================================

class QLearning:
    """
    Q-Learning - 最著名的强化学习算法！
    
    核心方程（第131页）：
    Q(St,At) ← Q(St,At) + α[Rt+1 + γ max_a Q(St+1,a) - Q(St,At)]
    
    Q-Learning的革命性：
    1. 离策略：学习最优策略，同时用其他策略探索
    2. 直接学习Q*：不需要策略改进步骤
    3. 收敛保证：在一定条件下收敛到最优
    
    与SARSA的关键区别：
    - SARSA：Q(S',A') - 用实际要执行的动作
    - Q-Learning：max Q(S',a) - 用最优动作
    
    类比：
    SARSA像谨慎的学生，Q-Learning像大胆的探索者
    - SARSA：学习实际走的路
    - Q-Learning：想象最优的路
    
    这是DQN的基础！
    """
    
    def __init__(self, n_actions: int, gamma: float = 0.9,
                 alpha: float = 0.1, epsilon: float = 0.1):
        """初始化Q-Learning"""
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        
        # Q函数
        self.Q = defaultdict(lambda: np.zeros(n_actions))
        
        # 统计
        self.episode_rewards = []
        
        print("Q-Learning初始化")
        print(f"参数：γ={gamma}, α={alpha}, ε={epsilon}")
    
    def select_action(self, state) -> int:
        """ε-贪婪动作选择"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state):
        """
        Q-Learning更新
        
        关键：使用max Q(S',a)
        这是"乐观"的更新，总是假设未来会选择最优动作
        """
        # Q-Learning的TD目标（与SARSA的区别）
        td_target = reward + self.gamma * np.max(self.Q[next_state])
        
        # TD误差
        td_error = td_target - self.Q[state][action]
        
        # 更新Q值
        self.Q[state][action] += self.alpha * td_error
        
        return td_error
    
    def episode(self, env, max_steps: int = 1000):
        """运行一个回合"""
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # 选择动作（探索）
            action = self.select_action(state)
            
            # 执行动作
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            if done:
                # 终止状态
                td_target = reward
                td_error = td_target - self.Q[state][action]
                self.Q[state][action] += self.alpha * td_error
                break
            
            # Q-Learning更新（关键：不需要next_action）
            self.update(state, action, reward, next_state)
            
            # 转移
            state = next_state
        
        self.episode_rewards.append(total_reward)
        return total_reward
    
    def train(self, env, n_episodes: int = 1000):
        """训练Q-Learning"""
        print(f"\n训练Q-Learning（{n_episodes}回合）...")
        
        for episode in range(n_episodes):
            reward = self.episode(env)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"  回合{episode + 1}: 平均奖励={avg_reward:.2f}")
        
        return self.Q


# ================================================================================
# 第6.5节：期望SARSA
# Section 6.5: Expected SARSA
# ================================================================================

class ExpectedSARSA:
    """
    期望SARSA - SARSA和Q-Learning的统一
    
    核心方程（第133页）：
    Q(St,At) ← Q(St,At) + α[Rt+1 + γ Σ_a π(a|St+1)Q(St+1,a) - Q(St,At)]
    
    关键洞察：
    - SARSA：使用采样的next_action
    - Q-Learning：使用最优动作（贪婪）
    - Expected SARSA：使用期望（加权平均）
    
    优点：
    1. 消除了SARSA的采样方差
    2. 比Q-Learning更稳定
    3. 可以处理随机策略
    
    实际上：
    - 当π是贪婪策略时，退化为Q-Learning
    - 当π是ε-贪婪时，性能通常最好
    """
    
    def __init__(self, n_actions: int, gamma: float = 0.9,
                 alpha: float = 0.1, epsilon: float = 0.1):
        """初始化Expected SARSA"""
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        
        self.Q = defaultdict(lambda: np.zeros(n_actions))
        
        print("Expected SARSA初始化")
        print(f"参数：γ={gamma}, α={alpha}, ε={epsilon}")
    
    def get_expected_value(self, state):
        """
        计算期望价值
        
        Σ_a π(a|s)Q(s,a)
        
        这是Expected SARSA的核心！
        """
        q_values = self.Q[state]
        
        # ε-贪婪策略的期望
        best_action = np.argmax(q_values)
        expected = 0.0
        
        for action in range(self.n_actions):
            if action == best_action:
                # 最优动作的概率
                prob = (1 - self.epsilon) + self.epsilon / self.n_actions
            else:
                # 非最优动作的概率
                prob = self.epsilon / self.n_actions
                
            expected += prob * q_values[action]
            
        return expected
    
    def update(self, state, action, reward, next_state):
        """Expected SARSA更新"""
        # 期望TD目标
        expected_value = self.get_expected_value(next_state)
        td_target = reward + self.gamma * expected_value
        
        # TD误差
        td_error = td_target - self.Q[state][action]
        
        # 更新
        self.Q[state][action] += self.alpha * td_error
        
        return td_error


# ================================================================================
# 第6.6节：双Q学习
# Section 6.6: Double Q-Learning
# ================================================================================

class DoubleQLearning:
    """
    双Q学习 - 解决最大化偏差问题
    
    问题（第134页）：
    Q-Learning有正向偏差！
    max操作会高估动作价值。
    
    原因：
    max E[X] ≤ E[max X]
    
    解决方案：
    使用两个Q函数
    - 一个选择动作
    - 另一个评估价值
    
    这消除了正向偏差！
    
    实际例子：
    想象评估餐厅
    - 单Q：总是选看起来最好的（可能是运气好）
    - 双Q：一个人选餐厅，另一个人评分（更客观）
    """
    
    def __init__(self, n_actions: int, gamma: float = 0.9,
                 alpha: float = 0.1, epsilon: float = 0.1):
        """初始化Double Q-Learning"""
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        
        # 两个Q函数
        self.Q1 = defaultdict(lambda: np.zeros(n_actions))
        self.Q2 = defaultdict(lambda: np.zeros(n_actions))
        
        print("Double Q-Learning初始化")
        print("使用两个Q函数消除最大化偏差")
    
    def select_action(self, state) -> int:
        """
        动作选择
        
        使用两个Q函数的平均
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            # 使用平均Q值
            q_sum = self.Q1[state] + self.Q2[state]
            return np.argmax(q_sum)
    
    def update(self, state, action, reward, next_state):
        """
        Double Q-Learning更新
        
        随机选择更新Q1或Q2
        """
        if np.random.random() < 0.5:
            # 更新Q1，用Q2评估
            best_action = np.argmax(self.Q1[next_state])
            td_target = reward + self.gamma * self.Q2[next_state][best_action]
            td_error = td_target - self.Q1[state][action]
            self.Q1[state][action] += self.alpha * td_error
        else:
            # 更新Q2，用Q1评估
            best_action = np.argmax(self.Q2[next_state])
            td_target = reward + self.gamma * self.Q1[next_state][best_action]
            td_error = td_target - self.Q2[state][action]
            self.Q2[state][action] += self.alpha * td_error
            
        return td_error


# ================================================================================
# 第6.7节：经典环境 - 悬崖行走
# Section 6.7: Classic Environment - Cliff Walking
# ================================================================================

class CliffWalkingEnv:
    """
    悬崖行走环境 - Example 6.6（第132页）
    
    展示SARSA和Q-Learning的区别！
    
    环境描述：
    - 网格世界，底部是悬崖
    - 掉下悬崖：-100奖励，回到起点
    - 每步：-1奖励
    - 目标：从起点到终点
    
    关键洞察：
    - SARSA学习安全路径（绕开悬崖）
    - Q-Learning学习最优路径（贴着悬崖）
    - 但Q-Learning在训练时可能掉下悬崖！
    
    这完美展示了在策略vs离策略的区别！
    """
    
    def __init__(self, width: int = 12, height: int = 4):
        """初始化悬崖行走环境"""
        self.width = width
        self.height = height
        
        # 起点和终点
        self.start = (0, 0)
        self.goal = (width - 1, 0)
        
        # 悬崖位置
        self.cliff = [(x, 0) for x in range(1, width - 1)]
        
        # 动作
        self.actions = {
            0: (-1, 0),  # 左
            1: (1, 0),   # 右
            2: (0, -1),  # 上
            3: (0, 1)    # 下
        }
        
        self.reset()
        
    def reset(self):
        """重置到起点"""
        self.position = self.start
        return self.position
    
    def step(self, action: int):
        """
        执行动作
        
        返回：(下一状态, 奖励, 是否结束)
        """
        # 计算新位置
        dx, dy = self.actions[action]
        new_x = self.position[0] + dx
        new_y = self.position[1] + dy
        
        # 边界检查
        new_x = max(0, min(new_x, self.width - 1))
        new_y = max(0, min(new_y, self.height - 1))
        
        new_position = (new_x, new_y)
        
        # 检查是否掉下悬崖
        if new_position in self.cliff:
            # 掉下悬崖！
            reward = -100
            new_position = self.start  # 回到起点
            done = False
        elif new_position == self.goal:
            # 到达目标！
            reward = 0
            done = True
        else:
            # 普通移动
            reward = -1
            done = False
            
        self.position = new_position
        return new_position, reward, done
    
    def render(self, Q_values=None, path=None):
        """
        渲染环境
        
        显示悬崖、路径和Q值
        """
        print("\n悬崖行走环境：")
        print("S: 起点, G: 终点, C: 悬崖, .: 安全")
        
        for y in range(self.height - 1, -1, -1):
            row = ""
            for x in range(self.width):
                pos = (x, y)
                
                if pos == self.start:
                    row += "S "
                elif pos == self.goal:
                    row += "G "
                elif pos in self.cliff:
                    row += "C "
                elif path and pos in path:
                    row += "* "  # 路径
                else:
                    row += ". "
                    
            print(row)
    
    @staticmethod
    def compare_algorithms():
        """
        比较SARSA和Q-Learning在悬崖行走中的表现
        
        这是书中的经典实验！
        """
        print("="*60)
        print("悬崖行走：SARSA vs Q-Learning")
        print("="*60)
        
        env = CliffWalkingEnv()
        env.render()
        
        # 训练SARSA
        print("\n训练SARSA...")
        sarsa = SARSA(n_actions=4, epsilon=0.1)
        
        for episode in range(500):
            state = env.reset()
            action = sarsa.select_action(state)
            
            while True:
                next_state, reward, done = env.step(action)
                
                if done:
                    sarsa.Q[state][action] += sarsa.alpha * (reward - sarsa.Q[state][action])
                    break
                    
                next_action = sarsa.select_action(next_state)
                sarsa.update(state, action, reward, next_state, next_action)
                
                state = next_state
                action = next_action
        
        print("SARSA训练完成")
        
        # 训练Q-Learning
        print("\n训练Q-Learning...")
        qlearning = QLearning(n_actions=4, epsilon=0.1)
        
        for episode in range(500):
            state = env.reset()
            
            while True:
                action = qlearning.select_action(state)
                next_state, reward, done = env.step(action)
                
                if done:
                    qlearning.Q[state][action] += qlearning.alpha * (reward - qlearning.Q[state][action])
                    break
                    
                qlearning.update(state, action, reward, next_state)
                state = next_state
        
        print("Q-Learning训练完成")
        
        # 提取路径
        def extract_path(Q, env):
            """从Q值提取路径"""
            path = []
            state = env.reset()
            
            for _ in range(100):  # 最多100步
                path.append(state)
                
                if state == env.goal:
                    break
                    
                # 贪婪选择
                action = np.argmax(Q[state])
                state, _, done = env.step(action)
                
                if done:
                    break
                    
            return path
        
        sarsa_path = extract_path(sarsa.Q, env)
        qlearning_path = extract_path(qlearning.Q, env)
        
        print("\nSARSA学到的路径（安全路径）：")
        env.render(path=sarsa_path)
        
        print("\nQ-Learning学到的路径（最优但危险）：")
        env.render(path=qlearning_path)
        
        print("\n关键发现：")
        print("- SARSA：学习安全路径，绕开悬崖")
        print("- Q-Learning：学习最短路径，贴着悬崖")
        print("- SARSA更保守，Q-Learning更激进")


# ================================================================================
# 第6.8节：统一视角
# Section 6.8: Unified View
# ================================================================================

class TDMethods:
    """
    TD方法的统一视角
    
    展示不同TD方法的关系和权衡
    """
    
    @staticmethod
    def unified_framework():
        """
        TD方法的统一框架
        
        所有TD方法都遵循：
        Q(s,a) ← Q(s,a) + α[目标 - Q(s,a)]
        
        区别在于"目标"的计算：
        """
        print("="*60)
        print("TD方法统一框架")
        print("="*60)
        
        print("""
        所有TD控制方法的统一形式：
        Q(s,a) ← Q(s,a) + α[目标 - Q(s,a)]
        
        不同方法的目标：
        
        1. SARSA（在策略）
           目标 = R + γQ(S',A')
           - A'是实际要执行的动作
           - 学习执行策略的价值
           - 更安全，更保守
        
        2. Q-Learning（离策略）
           目标 = R + γ max_a Q(S',a)
           - 使用最优动作
           - 学习最优策略
           - 更激进，可能不稳定
        
        3. Expected SARSA（中间）
           目标 = R + γ Σ_a π(a|S')Q(S',a)
           - 使用期望价值
           - 消除采样方差
           - 通常性能最好
        
        4. Double Q-Learning（无偏）
           目标 = R + γ Q_2(S', argmax_a Q_1(S',a))
           - 用两个Q函数
           - 消除最大化偏差
           - 更准确的估计
        
        选择哪种方法？
        - 需要安全：SARSA
        - 追求最优：Q-Learning
        - 要稳定：Expected SARSA
        - 怕高估：Double Q-Learning
        """)
    
    @staticmethod
    def convergence_properties():
        """
        收敛性质比较
        """
        print("\n" + "="*60)
        print("TD方法的收敛性质")
        print("="*60)
        
        print("""
        收敛条件（Robbins-Monro条件）：
        1. Σ α_t = ∞（学习率和发散）
        2. Σ α_t² < ∞（学习率平方和收敛）
        
        各方法的收敛性：
        
        方法            | 表格型 | 函数近似 | 条件
        ---------------|--------|---------|-------------
        TD(0)预测      | ✓      | ✓*      | 线性近似
        SARSA          | ✓      | ✗       | 可能震荡
        Q-Learning     | ✓      | ✗       | 可能发散
        Expected SARSA | ✓      | ✓*      | 更稳定
        
        ✓ = 保证收敛
        ✓* = 条件收敛
        ✗ = 可能不收敛
        
        关键洞察：
        离策略+函数近似+自举 = "致命三角"
        这是深度RL的主要挑战！
        """)


# ================================================================================
# 第6.9节：章节总结与实践
# Section 6.9: Chapter Summary and Practice
# ================================================================================

def chapter_summary():
    """第6章总结"""
    print("\n" + "="*70)
    print("第6章总结：时序差分学习")
    print("Chapter 6 Summary: Temporal-Difference Learning")
    print("="*70)
    
    print("""
    核心要点回顾：
    
    1. TD学习的革命性
       - 结合MC的无模型和DP的自举
       - 在线、增量式学习
       - 现代深度RL的基础
    
    2. TD(0)预测
       V(S) ← V(S) + α[R + γV(S') - V(S)]
       - TD误差：δ = R + γV(S') - V(S)
       - 自举：用估计更新估计
    
    3. TD控制算法家族
       
       算法         | 策略  | 更新目标           | 特点
       ------------|-------|-------------------|----------
       SARSA       | 在    | Q(S',A')          | 保守安全
       Q-Learning  | 离    | max Q(S',a)       | 激进最优
       Expected    | 在    | E[Q(S',a)]        | 稳定
       Double Q    | 离    | 分离选择和评估     | 无偏
    
    4. TD vs MC vs DP
       
       方面      | DP    | MC    | TD
       ---------|-------|-------|-------
       模型      | 需要  | 不需要 | 不需要
       自举      | 是    | 否    | 是
       在线      | 否    | 否    | 是
       回合      | 不需要 | 需要  | 不需要
       方差      | 无    | 高    | 中
       偏差      | 无    | 无    | 有
    
    5. 悬崖行走的启示
       - SARSA学安全路径（在策略）
       - Q-Learning学最短路径（离策略）
       - 探索时的行为影响学习结果！
    
    6. TD误差的重要性
       - 是奖励预测误差（RPE）
       - 对应大脑多巴胺信号
       - 驱动所有学习
    
    关键洞察：
    TD学习让我们能够"边做边学"，
    不需要完美模型，不需要等到结束，
    这使得RL能够解决现实世界的问题！
    
    TD是强化学习皇冠上的明珠。
    理解了TD，就理解了现代RL的精髓。
    
    下一章预告：
    n步TD方法 - 统一MC和TD的视角！
    
    练习建议：
    1. 实现练习6.9：风格子世界的比较
    2. 实现练习6.13：双期望SARSA
    3. 比较不同α值的影响
    4. 在更复杂环境测试各算法
    """)


# ================================================================================
# 主程序：运行第6章完整演示
# Main: Run Complete Chapter 6 Demonstration
# ================================================================================

def demonstrate_chapter_6():
    """运行第6章的完整演示"""
    print("╔" + "═"*68 + "╗")
    print("║" + " "*15 + "第6章：时序差分学习".center(38) + " "*15 + "║")
    print("║" + " "*10 + "Chapter 6: Temporal-Difference Learning".center(48) + " "*10 + "║")
    print("╚" + "═"*68 + "╝")
    
    print("\n欢迎来到TD学习的世界！")
    print("这里，我们将学习如何'边做边学'。")
    print("不需要模型，不需要等待，实时更新！\n")
    
    # 1. TD预测演示
    print("\n【第1部分：TD预测基础】")
    print("[Part 1: TD Prediction Basics]")
    print("="*70)
    
    # 展示TD和MC的收敛速度比较
    TDAdvantages.compare_convergence_speed()
    
    # 展示在线学习
    TDAdvantages.demonstrate_online_learning()
    
    # 2. 悬崖行走比较
    print("\n【第2部分：SARSA vs Q-Learning】")
    print("[Part 2: SARSA vs Q-Learning]")
    print("="*70)
    
    CliffWalkingEnv.compare_algorithms()
    
    # 3. 统一框架
    print("\n【第3部分：TD方法的统一视角】")
    print("[Part 3: Unified View of TD Methods]")
    print("="*70)
    
    TDMethods.unified_framework()
    TDMethods.convergence_properties()
    
    # 4. 章节总结
    print("\n【第4部分：章节总结】")
    print("[Part 4: Chapter Summary]")
    
    chapter_summary()
    
    print("\n时序差分学习的智慧：")
    print("过去已逝，未来未至，")
    print("我们只能在当下学习。")
    print("TD让我们用当下的经验，")
    print("同时改进对过去的理解和对未来的预期。")
    print("\n这就是'边做边学'的艺术！")


if __name__ == "__main__":
    demonstrate_chapter_6()