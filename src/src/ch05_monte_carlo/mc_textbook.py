"""
================================================================================
第5章：蒙特卡洛方法 - 从经验中学习的赌场智慧
Chapter 5: Monte Carlo Methods - Learning from Experience in the Casino

根据 Sutton & Barto《强化学习：导论》第二版 第5章
Based on Sutton & Barto "Reinforcement Learning: An Introduction" Chapter 5
================================================================================

让我从一个真实的赌场故事开始：

1960年代，数学家爱德华·索普（Edward Thorp）用卡片计数法
在拉斯维加斯的21点赌桌上大获成功。

他是怎么做到的？
1. 观察大量的牌局（收集经验）
2. 统计各种情况下的胜率（估计价值）
3. 找出最佳策略（什么时候要牌，什么时候停牌）
4. 根据牌面调整策略（策略改进）

这就是蒙特卡洛方法的精髓！

关键区别：
- 动态规划：知道赌场规则，在家里计算最优策略
- 蒙特卡洛：不知道确切规则，通过玩牌来学习

为什么叫"蒙特卡洛"？
这个名字来自摩纳哥的蒙特卡洛赌场。
二战时期，科学家用随机采样方法模拟中子运动，
代号就用了这个著名赌场的名字。

================================================================================
蒙特卡洛方法的核心思想
Core Ideas of Monte Carlo Methods
================================================================================

Sutton & Barto（第91页）：
"Monte Carlo methods require only experience—sample sequences of states, 
actions, and rewards from actual or simulated interaction with an environment."

MC方法的特点：
1. 无需模型：只需要经验（采样）
2. 完整序列：等到回合结束才更新
3. 无偏估计：收敛到真实值
4. 高方差：需要大量采样

什么时候用MC？
- 不知道环境模型
- 回合制任务（有终止）
- 想要无偏估计
- 可以大量采样
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass
import matplotlib.pyplot as plt
from enum import Enum


# ================================================================================
# 第5.1节：蒙特卡洛预测（策略评估）
# Section 5.1: Monte Carlo Prediction
# ================================================================================

class MonteCarloPrediction:
    """
    蒙特卡洛预测 - 通过采样评估策略
    
    核心思想：
    价值 = 平均回报
    
    数学原理（第92页）：
    vπ(s) = Eπ[Gt | St = s]
    
    其中Gt是从t时刻开始的回报：
    Gt = Rt+1 + γRt+2 + γ²Rt+3 + ... 
    
    根据大数定律，采样平均收敛到期望值！
    
    就像评估一个投资策略：
    - 不是计算理论收益（DP）
    - 而是看历史表现（MC）
    - 回测足够多次，就知道策略好坏
    """
    
    def __init__(self, gamma: float = 1.0):
        """
        初始化MC预测
        
        gamma: 折扣因子
        注意：MC方法可以用γ=1（无折扣），因为是回合制任务
        """
        self.gamma = gamma
        
        # 价值估计
        self.V = defaultdict(lambda: 0.0)
        
        # 访问计数（用于平均）
        self.returns_sum = defaultdict(lambda: 0.0)
        self.returns_count = defaultdict(lambda: 0)
        
        print("蒙特卡洛预测初始化")
        print(f"折扣因子γ = {gamma}")
        
    def first_visit_prediction(self, episodes: List[List[Tuple]]) -> Dict:
        """
        首次访问MC预测 - 算法5.1（第92页）
        
        对每个状态，只考虑每个回合中的第一次访问
        
        为什么要区分首次访问？
        - 避免同一回合中的相关性
        - 理论性质更好（无偏）
        - 收敛性有保证
        
        例子：在21点中
        - 一局牌中可能多次出现"手牌15点"的状态
        - 首次访问：只用第一次的结果
        - 避免重复计数导致的偏差
        """
        print("\n执行首次访问MC预测...")
        print(f"处理{len(episodes)}个回合的经验")
        
        for episode_num, episode in enumerate(episodes):
            # episode是一个轨迹：[(s0,a0,r1), (s1,a1,r2), ...]
            
            # 计算每个时刻的回报G
            G = 0
            returns = []
            
            # 从后向前计算回报（更高效）
            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                G = reward + self.gamma * G  # 递推计算回报
                returns.insert(0, (state, G))
            
            # 记录已访问的状态（用于首次访问检查）
            visited_states = set()
            
            # 更新价值估计
            for state, G in returns:
                if state not in visited_states:
                    # 首次访问这个状态
                    visited_states.add(state)
                    self.returns_sum[state] += G
                    self.returns_count[state] += 1
                    
                    # 更新价值（采样平均）
                    self.V[state] = self.returns_sum[state] / self.returns_count[state]
            
            # 定期报告
            if (episode_num + 1) % 100 == 0:
                print(f"  处理了{episode_num + 1}个回合")
                
        print(f"✓ MC预测完成")
        print(f"  评估了{len(self.V)}个状态")
        
        return dict(self.V)
    
    def every_visit_prediction(self, episodes: List[List[Tuple]]) -> Dict:
        """
        每次访问MC预测
        
        考虑每个回合中状态的所有访问
        
        优点：
        - 更多数据，收敛可能更快
        - 实现更简单
        
        缺点：
        - 同一回合的访问相关
        - 理论性质稍弱（但实践中常用）
        
        21点例子：
        如果一局中两次到达"手牌15点"
        两次的结果都用于估计
        """
        print("\n执行每次访问MC预测...")
        
        for episode in episodes:
            G = 0
            
            # 从后向前处理
            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                G = reward + self.gamma * G
                
                # 每次访问都更新（不检查是否首次）
                self.returns_sum[state] += G
                self.returns_count[state] += 1
                self.V[state] = self.returns_sum[state] / self.returns_count[state]
                
        return dict(self.V)
    
    def incremental_update(self, episodes: List[List[Tuple]], 
                          alpha: Optional[float] = None) -> Dict:
        """
        增量式MC预测
        
        使用增量更新公式：
        V(St) ← V(St) + α[Gt - V(St)]
        
        这是TD(0)的前身！
        
        alpha: 学习率
        - None: 使用1/n（样本平均）
        - 固定值: 适合非平稳问题
        
        优点：
        - 不需要存储所有回报
        - 可以处理非平稳问题
        - 为TD学习铺路
        """
        print("\n执行增量式MC预测...")
        
        N = defaultdict(int)  # 访问次数
        
        for episode in episodes:
            G = 0
            returns = []
            
            # 计算回报
            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                G = reward + self.gamma * G
                returns.insert(0, (state, G))
            
            # 增量更新
            visited = set()
            for state, G in returns:
                if state not in visited:  # 首次访问
                    visited.add(state)
                    N[state] += 1
                    
                    # 确定学习率
                    if alpha is None:
                        a = 1.0 / N[state]  # 样本平均
                    else:
                        a = alpha  # 固定学习率
                    
                    # 增量更新
                    self.V[state] += a * (G - self.V[state])
                    
        return dict(self.V)


# ================================================================================
# 第5.2节：21点游戏环境
# Section 5.2: Blackjack Environment
# ================================================================================

class BlackjackEnv:
    """
    21点（Blackjack）- MC方法的经典测试环境
    
    为什么21点适合展示MC方法？
    1. 回合短（适合MC）
    2. 规则清晰但策略复杂
    3. 真实赌场游戏（实用）
    4. 状态空间适中（容易分析）
    
    游戏规则（简化版）：
    - 目标：手牌点数接近21但不超过
    - 动作：要牌(hit)或停牌(stick)
    - A可以算1或11（软牌/硬牌）
    - 庄家规则固定：<17必须要牌
    
    这是书中的Example 5.1（第93页）
    """
    
    def __init__(self):
        """初始化21点环境"""
        # 牌面值
        # A=1(或11), 2-9=面值, 10,J,Q,K=10
        self.card_values = list(range(1, 10)) + [10, 10, 10, 10]
        
        self.reset()
        
    def reset(self) -> Tuple:
        """
        开始新游戏
        
        初始状态：
        - 玩家拿两张牌
        - 庄家一张明牌，一张暗牌
        - 返回：(玩家点数, 庄家明牌, 是否软牌)
        """
        # 发牌
        self.player_cards = [self.draw_card(), self.draw_card()]
        self.dealer_cards = [self.draw_card(), self.draw_card()]
        
        # 初始状态
        player_sum, usable_ace = self.get_hand_value(self.player_cards)
        dealer_showing = self.dealer_cards[0]
        
        return (player_sum, dealer_showing, usable_ace)
    
    def draw_card(self) -> int:
        """
        抽一张牌
        
        简化：假设无限副牌（牌不会用完）
        实际赌场：6-8副牌，这就是卡片计数的基础
        """
        return np.random.choice(self.card_values)
    
    def get_hand_value(self, cards: List[int]) -> Tuple[int, bool]:
        """
        计算手牌价值
        
        关键：A的处理
        - 如果A算11不爆牌，就算11（软牌）
        - 否则算1（硬牌）
        
        返回：(点数, 是否有可用的A)
        """
        total = sum(cards)
        has_ace = 1 in cards
        
        # 如果有A且算11不爆牌
        if has_ace and total + 10 <= 21:
            return total + 10, True
        else:
            return total, False
    
    def step(self, state: Tuple, action: str) -> Tuple[Tuple, float, bool]:
        """
        执行动作
        
        action: 'hit'(要牌) 或 'stick'(停牌)
        返回: (下一状态, 奖励, 是否结束)
        """
        player_sum, dealer_showing, usable_ace = state
        
        if action == 'hit':
            # 玩家要牌
            card = self.draw_card()
            player_sum += card
            
            # 处理A
            if card == 1 and player_sum + 10 <= 21:
                player_sum += 10
                usable_ace = True
            elif player_sum > 21 and usable_ace:
                # 软牌变硬牌
                player_sum -= 10
                usable_ace = False
            
            # 检查是否爆牌
            if player_sum > 21:
                return (player_sum, dealer_showing, usable_ace), -1, True
            else:
                return (player_sum, dealer_showing, usable_ace), 0, False
                
        else:  # stick
            # 玩家停牌，庄家开始
            dealer_sum, dealer_ace = self.get_hand_value(self.dealer_cards)
            
            # 庄家策略：<17必须要牌
            while dealer_sum < 17:
                card = self.draw_card()
                dealer_sum += card
                
                if card == 1 and dealer_sum + 10 <= 21:
                    dealer_sum += 10
                    dealer_ace = True
                elif dealer_sum > 21 and dealer_ace:
                    dealer_sum -= 10
                    dealer_ace = False
            
            # 判定输赢
            if dealer_sum > 21:  # 庄家爆牌
                reward = 1
            elif dealer_sum > player_sum:  # 庄家赢
                reward = -1
            elif dealer_sum < player_sum:  # 玩家赢
                reward = 1
            else:  # 平局
                reward = 0
                
            return (player_sum, dealer_showing, usable_ace), reward, True
    
    def generate_episode(self, policy) -> List[Tuple]:
        """
        生成一个完整回合
        
        用给定策略玩一局21点
        返回：[(state, action, reward), ...]
        """
        episode = []
        state = self.reset()
        
        # 玩到游戏结束
        done = False
        while not done:
            action = policy(state)
            next_state, reward, done = self.step(state, action)
            episode.append((state, action, reward))
            state = next_state
            
        return episode
    
    @staticmethod
    def visualize_policy(policy_table: Dict, title: str = "Blackjack Policy"):
        """
        可视化21点策略
        
        两个热图：
        1. 无可用A（硬牌）
        2. 有可用A（软牌）
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 准备数据
        player_sums = range(12, 22)
        dealer_showings = range(1, 11)
        
        # 硬牌策略
        hard_policy = np.zeros((10, 10))
        for i, p in enumerate(player_sums):
            for j, d in enumerate(dealer_showings):
                state = (p, d, False)
                # 1=要牌, 0=停牌
                hard_policy[i, j] = 1 if policy_table.get(state, 0) == 'hit' else 0
        
        im1 = ax1.imshow(hard_policy, cmap='RdYlGn', aspect='auto')
        ax1.set_xticks(range(10))
        ax1.set_xticklabels(dealer_showings)
        ax1.set_yticks(range(10))
        ax1.set_yticklabels(player_sums)
        ax1.set_xlabel('庄家明牌 Dealer Showing')
        ax1.set_ylabel('玩家点数 Player Sum')
        ax1.set_title('硬牌策略 Hard Hand Policy')
        
        # 添加网格
        for i in range(10):
            for j in range(10):
                text = 'H' if hard_policy[i, j] == 1 else 'S'
                ax1.text(j, i, text, ha='center', va='center')
        
        # 软牌策略
        soft_policy = np.zeros((10, 10))
        for i, p in enumerate(player_sums):
            for j, d in enumerate(dealer_showings):
                state = (p, d, True)
                soft_policy[i, j] = 1 if policy_table.get(state, 0) == 'hit' else 0
        
        im2 = ax2.imshow(soft_policy, cmap='RdYlGn', aspect='auto')
        ax2.set_xticks(range(10))
        ax2.set_xticklabels(dealer_showings)
        ax2.set_yticks(range(10))
        ax2.set_yticklabels(player_sums)
        ax2.set_xlabel('庄家明牌 Dealer Showing')
        ax2.set_ylabel('玩家点数 Player Sum')
        ax2.set_title('软牌策略 Soft Hand Policy')
        
        for i in range(10):
            for j in range(10):
                text = 'H' if soft_policy[i, j] == 1 else 'S'
                ax2.text(j, i, text, ha='center', va='center')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()


# ================================================================================
# 第5.3节：蒙特卡洛控制
# Section 5.3: Monte Carlo Control
# ================================================================================

class MonteCarloControl:
    """
    蒙特卡洛控制 - 寻找最优策略
    
    核心思想：广义策略迭代（GPI）
    1. 评估：MC预测估计Q值
    2. 改进：对Q值贪婪
    
    挑战：探索问题
    - 如果策略确定，某些动作永远不会尝试
    - 无法知道未尝试动作的价值
    - 可能错过最优策略
    
    解决方案（第99页）：
    1. 探索性起始（Exploring Starts）
    2. ε-soft策略
    3. 离策略方法（Off-policy）
    """
    
    def __init__(self, env, gamma: float = 1.0, epsilon: float = 0.1):
        """
        初始化MC控制
        
        env: 环境
        gamma: 折扣因子
        epsilon: 探索率（ε-贪婪）
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Q值估计
        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))
        
        # 访问计数
        self.returns_sum = defaultdict(lambda: 0.0)
        self.returns_count = defaultdict(lambda: 0)
        
        print("蒙特卡洛控制初始化")
        print(f"γ={gamma}, ε={epsilon}")
    
    def monte_carlo_es(self, n_episodes: int = 10000):
        """
        带探索性起始的MC控制 - 算法5.3（第99页）
        
        探索性起始（ES）：
        每个回合随机选择初始状态和动作
        保证所有(s,a)对都被访问
        
        这就像在赌场：
        - 不总是按常规开局
        - 偶尔尝试"疯狂"的打法
        - 可能发现意外的好策略
        
        局限：
        - 现实中难以实现（不能随意设置初始状态）
        - 但理论上保证收敛到最优
        """
        print(f"\n执行MC-ES控制（{n_episodes}回合）...")
        
        # 所有可能的状态-动作对（用于探索性起始）
        states = []
        for player in range(12, 22):
            for dealer in range(1, 11):
                for ace in [True, False]:
                    states.append((player, dealer, ace))
        actions = ['hit', 'stick']
        
        for episode_num in range(n_episodes):
            # 探索性起始：随机初始状态和动作
            init_state = states[np.random.choice(len(states))]
            init_action = np.random.choice(actions)
            
            # 生成回合
            episode = [(init_state, init_action, 0)]  # 初始无奖励
            
            state = init_state
            action = init_action
            done = False
            
            # 第一步
            next_state, reward, done = self.env.step(state, action)
            episode[-1] = (state, action, reward)
            
            # 继续回合（用贪婪策略）
            while not done:
                state = next_state
                # 贪婪选择
                if self.Q[state]['hit'] > self.Q[state]['stick']:
                    action = 'hit'
                else:
                    action = 'stick'
                    
                next_state, reward, done = self.env.step(state, action)
                episode.append((state, action, reward))
            
            # 更新Q值（首次访问）
            G = 0
            visited = set()
            
            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                G = reward + self.gamma * G
                
                state_action = (state, action)
                if state_action not in visited:
                    visited.add(state_action)
                    self.returns_sum[state_action] += G
                    self.returns_count[state_action] += 1
                    self.Q[state][action] = (
                        self.returns_sum[state_action] / 
                        self.returns_count[state_action]
                    )
            
            # 定期报告
            if (episode_num + 1) % 1000 == 0:
                print(f"  完成{episode_num + 1}回合")
        
        print("✓ MC-ES控制完成")
        return self.extract_policy()
    
    def on_policy_control(self, n_episodes: int = 10000):
        """
        在策略MC控制（ε-soft）- 算法5.4（第101页）
        
        ε-贪婪策略保证探索：
        - 概率ε：随机动作
        - 概率1-ε：贪婪动作
        
        收敛到ε-贪婪策略下的最优（不是真正最优）
        
        赌场类比：
        - 大部分时间用"最佳"策略
        - 偶尔故意"乱打"试试
        - 持续学习和调整
        """
        print(f"\n执行在策略MC控制（ε={self.epsilon}）...")
        
        for episode_num in range(n_episodes):
            # 生成回合
            episode = []
            state = self.env.reset()
            
            done = False
            while not done:
                # ε-贪婪动作选择
                if np.random.random() < self.epsilon:
                    action = np.random.choice(['hit', 'stick'])
                else:
                    # 贪婪选择
                    if self.Q[state]['hit'] > self.Q[state]['stick']:
                        action = 'hit'
                    else:
                        action = 'stick'
                
                next_state, reward, done = self.env.step(state, action)
                episode.append((state, action, reward))
                state = next_state
            
            # 更新Q值
            G = 0
            visited = set()
            
            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                G = reward + self.gamma * G
                
                state_action = (state, action)
                if state_action not in visited:
                    visited.add(state_action)
                    self.returns_sum[state_action] += G
                    self.returns_count[state_action] += 1
                    self.Q[state][action] = (
                        self.returns_sum[state_action] / 
                        self.returns_count[state_action]
                    )
            
            if (episode_num + 1) % 1000 == 0:
                print(f"  完成{episode_num + 1}回合")
        
        return self.extract_policy()
    
    def extract_policy(self) -> Dict:
        """
        从Q值提取策略
        
        贪婪策略：π(s) = argmax_a Q(s,a)
        """
        policy = {}
        
        for state in self.Q:
            if self.Q[state]['hit'] > self.Q[state]['stick']:
                policy[state] = 'hit'
            else:
                policy[state] = 'stick'
                
        return policy


# ================================================================================
# 第5.4节：重要性采样与离策略方法
# Section 5.4: Importance Sampling and Off-Policy Methods
# ================================================================================

class OffPolicyMC:
    """
    离策略蒙特卡洛 - 从其他策略的经验中学习
    
    核心概念（第103页）：
    - 行为策略b(a|s)：生成数据的策略（探索）
    - 目标策略π(a|s)：要评估/优化的策略（利用）
    
    重要性采样（Importance Sampling）：
    调整不同分布下的样本权重
    
    ρ = ∏ π(At|St) / b(At|St)
    
    类比：
    你想知道职业牌手的策略有多好（目标策略）
    但只能观察业余玩家的牌局（行为策略）
    通过调整权重，从业余数据推断职业策略的价值
    
    优点：
    - 可以从任意策略学习
    - 行为策略可以充分探索
    - 目标策略可以是确定性的
    
    缺点：
    - 高方差（权重可能很大）
    - 需要b(a|s) > 0当π(a|s) > 0
    """
    
    def __init__(self, env, gamma: float = 1.0):
        """初始化离策略MC"""
        self.env = env
        self.gamma = gamma
        
        # 累积权重和回报（用于加权平均）
        self.C = defaultdict(lambda: 0.0)  # 累积权重
        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))
        
        print("离策略蒙特卡洛初始化")
        
    def ordinary_importance_sampling(self, episodes: List, 
                                    target_policy, behavior_policy) -> Dict:
        """
        普通重要性采样 - 方程5.4（第104页）
        
        V(s) = Σ ρ(τ) G(τ) / N
        
        简单但高方差
        """
        print("\n执行普通重要性采样...")
        
        returns_sum = defaultdict(lambda: 0.0)
        returns_count = defaultdict(lambda: 0)
        
        for episode in episodes:
            G = 0
            W = 1  # 重要性采样比率
            
            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                G = reward + self.gamma * G
                
                # 计算重要性权重
                # W = W * π(a|s) / b(a|s)
                pi_prob = target_policy.get_probability(state, action)
                b_prob = behavior_policy.get_probability(state, action)
                
                if b_prob == 0:
                    break  # 不可能发生的轨迹
                    
                W *= pi_prob / b_prob
                
                returns_sum[state] += W * G
                returns_count[state] += 1
                
                if W == 0:
                    break  # 后续权重都是0
        
        # 计算价值
        V = {}
        for state in returns_sum:
            if returns_count[state] > 0:
                V[state] = returns_sum[state] / returns_count[state]
            else:
                V[state] = 0
                
        return V
    
    def weighted_importance_sampling(self, episodes: List,
                                   target_policy, behavior_policy) -> Dict:
        """
        加权重要性采样 - 方程5.5（第104页）
        
        V(s) = Σ W(τ) G(τ) / Σ W(τ)
        
        偏差但低方差
        通常表现更好
        """
        print("\n执行加权重要性采样...")
        
        weighted_returns = defaultdict(lambda: 0.0)
        weights_sum = defaultdict(lambda: 0.0)
        
        for episode in episodes:
            G = 0
            W = 1
            
            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                G = reward + self.gamma * G
                
                # 重要性权重
                pi_prob = target_policy.get_probability(state, action)
                b_prob = behavior_policy.get_probability(state, action)
                
                if b_prob == 0:
                    break
                    
                W *= pi_prob / b_prob
                
                weighted_returns[state] += W * G
                weights_sum[state] += W
                
                if W == 0:
                    break
        
        # 加权平均
        V = {}
        for state in weighted_returns:
            if weights_sum[state] > 0:
                V[state] = weighted_returns[state] / weights_sum[state]
            else:
                V[state] = 0
                
        return V
    
    def off_policy_control(self, n_episodes: int = 10000, epsilon: float = 0.1):
        """
        离策略MC控制 - 算法5.6（第111页）
        
        使用加权重要性采样的增量实现
        
        关键：
        - 行为策略：ε-soft（保证探索）
        - 目标策略：贪婪（追求最优）
        - 通过重要性采样连接两者
        
        这就像：
        - 让新手随意尝试（收集数据）
        - 分析哪些动作导致好结果
        - 提炼出专家策略
        """
        print(f"\n执行离策略MC控制...")
        
        # 行为策略（ε-贪婪）
        def behavior_policy(state):
            if np.random.random() < epsilon:
                return np.random.choice(['hit', 'stick'])
            else:
                if self.Q[state]['hit'] > self.Q[state]['stick']:
                    return 'hit'
                else:
                    return 'stick'
        
        for episode_num in range(n_episodes):
            # 用行为策略生成回合
            episode = []
            state = self.env.reset()
            
            done = False
            while not done:
                action = behavior_policy(state)
                next_state, reward, done = self.env.step(state, action)
                episode.append((state, action, reward))
                state = next_state
            
            # 离策略学习
            G = 0
            W = 1
            
            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                G = reward + self.gamma * G
                
                # 更新Q值
                self.C[state, action] += W
                self.Q[state][action] += (W / self.C[state, action]) * \
                                         (G - self.Q[state][action])
                
                # 如果动作不是贪婪的，终止（目标策略是贪婪的）
                if self.Q[state]['hit'] > self.Q[state]['stick']:
                    best_action = 'hit'
                else:
                    best_action = 'stick'
                    
                if action != best_action:
                    break  # 目标策略不会选择这个动作
                    
                # 更新权重
                # W = W * 1 / b(a|s)（目标策略是确定的，π=1）
                if action == best_action:
                    b_prob = (1 - epsilon) + epsilon / 2
                else:
                    b_prob = epsilon / 2
                    
                W = W / b_prob
            
            if (episode_num + 1) % 1000 == 0:
                print(f"  完成{episode_num + 1}回合")
        
        # 提取贪婪策略
        policy = {}
        for state in self.Q:
            if self.Q[state]['hit'] > self.Q[state]['stick']:
                policy[state] = 'hit'
            else:
                policy[state] = 'stick'
                
        return policy


# ================================================================================
# 第5.5节：完整的21点求解
# Section 5.5: Complete Blackjack Solution
# ================================================================================

def solve_blackjack():
    """
    用MC方法完整求解21点
    
    展示：
    1. 策略评估
    2. 策略改进
    3. 最优策略
    4. 与已知策略比较
    """
    print("="*70)
    print("21点完整求解 - 蒙特卡洛方法")
    print("Complete Blackjack Solution with Monte Carlo")
    print("="*70)
    
    env = BlackjackEnv()
    
    # 1. 评估固定策略
    print("\n步骤1：评估简单策略（20点以下要牌）")
    print("-"*50)
    
    def simple_policy(state):
        player_sum, dealer_showing, usable_ace = state
        return 'hit' if player_sum < 20 else 'stick'
    
    # 生成经验
    episodes = []
    for _ in range(10000):
        episodes.append(env.generate_episode(simple_policy))
    
    # MC预测
    mc_pred = MonteCarloPrediction(gamma=1.0)
    V_simple = mc_pred.first_visit_prediction(episodes)
    
    print(f"评估了{len(V_simple)}个状态")
    
    # 显示一些状态的价值
    sample_states = [
        ((20, 5, False), "玩家20点,庄家5,硬牌"),
        ((18, 10, False), "玩家18点,庄家10,硬牌"),
        ((12, 2, False), "玩家12点,庄家2,硬牌"),
        ((19, 1, True), "玩家19点,庄家A,软牌")
    ]
    
    print("\n样本状态价值：")
    for state, desc in sample_states:
        value = V_simple.get(state, 0)
        print(f"  {desc}: {value:.3f}")
    
    # 2. 寻找最优策略
    print("\n步骤2：寻找最优策略")
    print("-"*50)
    
    mc_control = MonteCarloControl(env, gamma=1.0, epsilon=0.1)
    optimal_policy = mc_control.on_policy_control(n_episodes=50000)
    
    print(f"学习到{len(optimal_policy)}个状态的策略")
    
    # 3. 可视化最优策略
    print("\n步骤3：可视化最优策略")
    BlackjackEnv.visualize_policy(optimal_policy, "MC学习的最优策略")
    
    # 4. 与专家策略比较
    print("\n步骤4：与已知策略比较")
    print("-"*50)
    
    def expert_policy(state):
        """
        基本策略（赌场公开的）
        简化版本
        """
        player_sum, dealer_showing, usable_ace = state
        
        if usable_ace:
            # 软牌策略
            if player_sum >= 19:
                return 'stick'
            elif player_sum == 18:
                return 'stick' if dealer_showing not in [9, 10, 1] else 'hit'
            else:
                return 'hit'
        else:
            # 硬牌策略
            if player_sum >= 17:
                return 'stick'
            elif player_sum >= 13:
                return 'stick' if dealer_showing <= 6 else 'hit'
            elif player_sum == 12:
                return 'stick' if 4 <= dealer_showing <= 6 else 'hit'
            else:
                return 'hit'
    
    # 比较策略差异
    differences = 0
    total = 0
    
    for state in optimal_policy:
        learned_action = optimal_policy[state]
        expert_action = expert_policy(state)
        
        if learned_action != expert_action:
            differences += 1
        total += 1
    
    agreement = (total - differences) / total * 100
    print(f"MC策略与专家策略的一致率：{agreement:.1f}%")
    print(f"差异状态数：{differences}/{total}")
    
    # 5. 演示一局游戏
    print("\n步骤5：演示学习到的策略")
    print("-"*50)
    
    state = env.reset()
    print(f"初始状态：玩家{state[0]}点，庄家{state[1]}，{'软牌' if state[2] else '硬牌'}")
    
    done = False
    while not done:
        action = optimal_policy.get(state, 'stick')
        print(f"  选择：{action}")
        
        next_state, reward, done = env.step(state, action)
        
        if done:
            if reward > 0:
                print("  结果：赢了！")
            elif reward < 0:
                print("  结果：输了")
            else:
                print("  结果：平局")
        else:
            state = next_state
            print(f"  新状态：玩家{state[0]}点")


# ================================================================================
# 第5.6节：赛马问题（练习5.12）
# Section 5.6: Racetrack Problem (Exercise 5.12)
# ================================================================================

class RacetrackEnv:
    """
    赛车问题 - MC方法的经典练习
    
    问题描述（练习5.12，第111页）：
    - 在赛道上控制赛车
    - 动作：加速/减速（横向和纵向）
    - 目标：尽快到达终点
    - 挑战：速度控制和转弯
    
    这个问题展示：
    1. 连续状态的离散化
    2. 延迟奖励（只在终点给奖励）
    3. MC方法处理稀疏奖励的能力
    """
    
    def __init__(self, track_file: str = None):
        """初始化赛道"""
        if track_file:
            self.load_track(track_file)
        else:
            # 简单的默认赛道
            self.create_simple_track()
            
    def create_simple_track(self):
        """创建简单赛道用于演示"""
        # 1=赛道, 0=墙, S=起点, F=终点
        self.track = [
            "0000000000",
            "0111111110",
            "0111111110",
            "S111111110",
            "S111111110",
            "0111111FF0",
            "0111111FF0",
            "0000000000"
        ]
        
        self.height = len(self.track)
        self.width = len(self.track[0])
        
        # 找起点和终点
        self.start_positions = []
        self.finish_positions = []
        
        for i, row in enumerate(self.track):
            for j, cell in enumerate(row):
                if cell == 'S':
                    self.start_positions.append((i, j))
                elif cell == 'F':
                    self.finish_positions.append((i, j))
        
        print(f"创建{self.height}×{self.width}赛道")
        print(f"起点数：{len(self.start_positions)}")
        print(f"终点数：{len(self.finish_positions)}")
    
    def reset(self):
        """重置到起点"""
        pos = self.start_positions[np.random.randint(len(self.start_positions))]
        return (*pos, 0, 0)  # (x, y, vx, vy)
    
    def step(self, state, action):
        """
        执行动作
        
        state: (x, y, vx, vy) - 位置和速度
        action: (ax, ay) - 加速度（-1, 0, 1）
        """
        x, y, vx, vy = state
        ax, ay = action
        
        # 更新速度（限制最大速度）
        vx = np.clip(vx + ax, -5, 5)
        vy = np.clip(vy + ay, -5, 5)
        
        # 更新位置
        new_x = x + vx
        new_y = y + vy
        
        # 检查碰撞
        if not self.is_valid_position(new_x, new_y):
            # 撞墙，回到起点
            return self.reset(), -1, False
        
        # 检查是否到达终点
        if (new_x, new_y) in self.finish_positions:
            return (new_x, new_y, vx, vy), 0, True
        
        # 继续比赛
        return (new_x, new_y, vx, vy), -1, False
    
    def is_valid_position(self, x, y):
        """检查位置是否有效"""
        if x < 0 or x >= self.height or y < 0 or y >= self.width:
            return False
        return self.track[x][y] in ['1', 'S', 'F']


# ================================================================================
# 第5.7节：章节总结
# Section 5.7: Chapter Summary
# ================================================================================

def chapter_summary():
    """第5章总结"""
    print("\n" + "="*70)
    print("第5章总结：蒙特卡洛方法")
    print("Chapter 5 Summary: Monte Carlo Methods")
    print("="*70)
    
    print("""
    核心要点回顾：
    
    1. MC方法的本质
       - 从完整经验序列学习
       - 使用采样平均估计期望
       - 无需环境模型
       - 适合回合制任务
    
    2. MC预测（策略评估）
       - 首次访问 vs 每次访问
       - V(s) = average(Returns)
       - 大数定律保证收敛
    
    3. MC控制（寻找最优策略）
       
       方法         | 探索方式      | 特点
       ------------|--------------|------------------
       MC-ES       | 探索性起始    | 理论最优，实践困难
       ε-soft      | ε-贪婪       | 简单实用，次优
       离策略      | 重要性采样    | 灵活强大，高方差
    
    4. 重要性采样
       - 从行为策略学习目标策略
       - 普通 vs 加权（方差权衡）
       - 离策略学习的基础
    
    5. MC vs DP比较
       
       方面    | 动态规划  | 蒙特卡洛
       -------|----------|----------
       模型    | 需要     | 不需要
       更新    | 自举     | 采样
       任务    | 任意     | 回合制
       收敛    | 快       | 慢
       方差    | 低       | 高
    
    6. 21点案例启示
       - MC方法可以学到接近最优的策略
       - 不需要知道牌的概率分布
       - 通过经验自动发现模式
       - 类似人类学习过程
    
    7. 优缺点总结
       
       优点：
       ✓ 无需模型
       ✓ 概念简单
       ✓ 无偏估计
       ✓ 可以聚焦于感兴趣的状态
       
       缺点：
       ✗ 需要回合终止
       ✗ 高方差（需要大量采样）
       ✗ 不能在线学习（要等回合结束）
    
    关键洞察：
    MC方法展示了如何从纯粹的经验中学习，
    不需要了解世界的运作规律。
    这更接近人类和动物的学习方式！
    
    但MC有个限制：必须等到回合结束。
    如果能边走边学就好了...
    
    这就引出了下一章：
    时序差分学习 - MC和DP的完美结合！
    
    练习建议：
    1. 实现练习5.4：改进21点的探索
    2. 实现练习5.12：赛车问题
    3. 比较首次访问和每次访问MC
    4. 实验重要性采样的方差
    """)


# ================================================================================
# 主程序：运行第5章完整演示
# Main: Run Complete Chapter 5 Demonstration
# ================================================================================

def demonstrate_chapter_5():
    """运行第5章的完整演示"""
    print("╔" + "═"*68 + "╗")
    print("║" + " "*15 + "第5章：蒙特卡洛方法".center(38) + " "*15 + "║")
    print("║" + " "*10 + "Chapter 5: Monte Carlo Methods".center(48) + " "*10 + "║")
    print("╚" + "═"*68 + "╝")
    
    print("\n欢迎来到蒙特卡洛赌场！")
    print("在这里，我们通过玩牌来学习最优策略。")
    print("不需要知道赌场的秘密，只需要不断尝试！\n")
    
    # 1. 基本MC预测演示
    print("\n【第1部分：MC预测基础】")
    print("[Part 1: MC Prediction Basics]")
    print("="*70)
    
    # 创建简单环境演示
    print("\n创建21点环境...")
    env = BlackjackEnv()
    
    # 简单策略
    def always_stick_20(state):
        player_sum, _, _ = state
        return 'stick' if player_sum >= 20 else 'hit'
    
    print("生成100个回合的经验...")
    episodes = []
    for _ in range(100):
        episodes.append(env.generate_episode(always_stick_20))
    
    print(f"平均回合长度：{np.mean([len(ep) for ep in episodes]):.1f}")
    
    # MC预测
    mc = MonteCarloPrediction()
    V = mc.first_visit_prediction(episodes)
    
    print(f"\n一些状态的估计价值：")
    for state in list(V.keys())[:5]:
        print(f"  状态{state}: {V[state]:.3f}")
    
    # 2. 完整21点求解
    print("\n【第2部分：21点完整求解】")
    print("[Part 2: Complete Blackjack Solution]")
    print("="*70)
    
    solve_blackjack()
    
    # 3. 离策略方法演示
    print("\n【第3部分：离策略学习】")
    print("[Part 3: Off-Policy Learning]")
    print("="*70)
    
    print("\n离策略学习的意义：")
    print("- 行为策略：充分探索（如随机策略）")
    print("- 目标策略：追求最优（如贪婪策略）")
    print("- 重要性采样：连接二者")
    
    off_policy = OffPolicyMC(env)
    print("\n执行离策略控制（简化演示）...")
    # 这里可以添加更详细的离策略演示
    
    # 4. 章节总结
    print("\n【第4部分：章节总结】")
    print("[Part 4: Chapter Summary]")
    
    chapter_summary()
    
    print("\n蒙特卡洛方法的智慧：")
    print("不需要了解游戏规则，只需要玩足够多局，")
    print("就能发现最优策略。这就是经验的力量！")
    print("\n下一章预告：时序差分学习")
    print("如果MC是'先做完再学'，TD就是'边做边学'！")
    print("Next: Chapter 6 - Temporal Difference Learning")


if __name__ == "__main__":
    demonstrate_chapter_5()