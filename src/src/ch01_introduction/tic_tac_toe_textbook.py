"""
================================================================================
第1.4节：井字棋 - 通过完整例子理解强化学习
Section 1.4: Tic-Tac-Toe - Understanding RL Through a Complete Example

根据 Sutton & Barto《强化学习：导论》第二版 第1章
Based on Sutton & Barto "Reinforcement Learning: An Introduction" Chapter 1
================================================================================

让我们从一个小故事开始：

小明今年6岁，刚学会井字棋的规则。
第一次玩，他随机下棋，很快就输了。
第二次玩，他记住了上次输的位置，避开了那些走法。
第三次玩，他发现走中心格子似乎更容易赢。
...
第100次玩，他已经很难被打败了。

这个学习过程，就是强化学习的精髓！

没有人告诉小明"最优策略是什么"（不是监督学习）
他只是通过不断对弈，从输赢中学习（强化学习）

================================================================================
井字棋为什么是完美的RL入门例子？
Why Tic-Tac-Toe is Perfect for Learning RL?
================================================================================

Sutton & Barto选择井字棋作为第一个详细例子，因为：

1. 规则简单 Simple Rules
   - 3×3格子，两人轮流
   - 三个连成线就赢
   - 人人都会玩

2. 状态空间小 Small State Space  
   - 总共只有3^9 = 19683种可能状态
   - 考虑对称性后更少
   - 可以完全存储价值表

3. 延迟奖励明显 Clear Delayed Reward
   - 只有游戏结束才知道输赢
   - 需要学会评估中间状态
   - 完美展示时间信用分配问题

4. 可以自我对弈 Self-Play Possible
   - 不需要人类对手
   - 可以快速训练
   - 展示RL的自主学习能力
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
import matplotlib.pyplot as plt
from dataclasses import dataclass
import pickle
from collections import defaultdict


# ================================================================================
# 第1.4.1节：井字棋游戏环境
# Section 1.4.1: Tic-Tac-Toe Game Environment
# ================================================================================

class TicTacToeBoard:
    """
    井字棋棋盘 - 游戏环境
    
    这是强化学习中的"环境"(Environment)部分。
    它定义了：
    1. 状态空间（3×3的棋盘）
    2. 动作空间（可以下棋的位置）
    3. 游戏规则（如何判断胜负）
    4. 状态转移（下棋后棋盘如何变化）
    
    关键设计：
    - 用数字表示：1=X玩家, -1=O玩家, 0=空
    - 状态用元组表示，方便作为字典的键
    - 完整实现所有游戏逻辑
    """
    
    def __init__(self):
        """初始化空棋盘"""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # X先手
        
    def reset(self):
        """重置棋盘到初始状态"""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        return self.get_state()
    
    def get_state(self) -> tuple:
        """
        获取当前状态的唯一表示
        
        为什么用元组？
        1. 可以作为字典的键（不可变）
        2. 方便存储价值函数 V(s)
        3. 唯一标识一个棋盘状态
        """
        return tuple(self.board.flatten())
    
    def get_available_actions(self) -> List[int]:
        """
        获取所有合法动作
        
        返回所有空格子的位置（0-8的数字）
        这定义了当前状态下的动作空间 A(s)
        """
        return [i for i in range(9) if self.board.flat[i] == 0]
    
    def make_move(self, action: int) -> Tuple[int, bool]:
        """
        执行一个动作（下棋）
        
        参数:
            action: 位置索引 (0-8)
            
        返回:
            reward: 奖励信号 (+1赢, -1输, 0其他)
            done: 游戏是否结束
            
        这是环境的核心：执行动作并返回奖励！
        """
        if self.board.flat[action] != 0:
            raise ValueError(f"位置{action}已经有棋子了！")
        
        # 下棋
        row, col = action // 3, action % 3
        self.board[row, col] = self.current_player
        
        # 检查游戏结果
        winner = self._check_winner()
        if winner != 0:
            # 有人赢了
            reward = winner  # 赢家得+1，输家得-1
            return reward, True
        elif len(self.get_available_actions()) == 0:
            # 平局
            return 0, True
        else:
            # 游戏继续，切换玩家
            self.current_player = -self.current_player
            return 0, False
    
    def _check_winner(self) -> int:
        """
        检查是否有赢家
        
        返回: 1(X赢), -1(O赢), 0(未结束或平局)
        
        检查所有可能的获胜线路：
        - 3条横线
        - 3条竖线
        - 2条对角线
        """
        # 检查行
        for row in self.board:
            if abs(row.sum()) == 3:
                return row[0]
        
        # 检查列
        for col in self.board.T:
            if abs(col.sum()) == 3:
                return col[0]
        
        # 检查对角线
        diag1 = self.board.diagonal()
        if abs(diag1.sum()) == 3:
            return diag1[0]
        
        diag2 = np.fliplr(self.board).diagonal()
        if abs(diag2.sum()) == 3:
            return diag2[0]
        
        return 0
    
    def render(self):
        """
        可视化棋盘
        
        用符号显示：X, O, .（空）
        方便人类理解当前局面
        """
        symbols = {1: 'X', -1: 'O', 0: '.'}
        print("\n当前棋盘 Current Board:")
        print("  0 1 2")
        for i, row in enumerate(self.board):
            print(f"{i} " + " ".join(symbols[x] for x in row))
        print()
    
    def get_symmetries(self, state: tuple) -> List[tuple]:
        """
        获取状态的所有对称形式
        
        井字棋有8种对称：
        - 旋转90°、180°、270°（3种）
        - 水平翻转（1种）
        - 垂直翻转（1种）
        - 两条对角线翻转（2种）
        - 原始状态（1种）
        
        利用对称性可以大大减少需要学习的状态数！
        """
        board = np.array(state).reshape(3, 3)
        symmetries = []
        
        # 4种旋转
        for k in range(4):
            rotated = np.rot90(board, k)
            symmetries.append(tuple(rotated.flatten()))
        
        # 翻转后再旋转
        flipped = np.fliplr(board)
        for k in range(4):
            rotated = np.rot90(flipped, k)
            symmetries.append(tuple(rotated.flatten()))
        
        return list(set(symmetries))  # 去重


# ================================================================================
# 第1.4.2节：价值函数与时序差分学习
# Section 1.4.2: Value Function and Temporal Difference Learning
# ================================================================================

class ValueFunction:
    """
    状态价值函数 V(s)
    
    这是强化学习的核心概念！
    
    V(s) 表示：从状态s开始，采用当前策略，最终获胜的概率
    
    例如：
    - V(初始状态) ≈ 0.5（双方机会均等）
    - V(即将获胜的状态) ≈ 1.0（很可能赢）
    - V(即将失败的状态) ≈ 0.0（很可能输）
    
    关键思想：
    通过不断对弈，逐渐学习每个状态的真实价值！
    """
    
    def __init__(self, player: int = 1, initial_value: float = 0.5):
        """
        初始化价值函数
        
        参数:
            player: 玩家标识 (1或-1)
            initial_value: 初始价值估计（乐观？悲观？中立？）
            
        初始值的选择很重要：
        - 0.5 = 中立，不知道好坏
        - 1.0 = 乐观，假设都能赢（鼓励探索）
        - 0.0 = 悲观，假设都会输
        """
        self.values = {}  # 状态 -> 价值的映射
        self.player = player
        self.initial_value = initial_value
        
    def get_value(self, state: tuple) -> float:
        """
        获取状态的价值
        
        如果是新状态，返回初始值
        这实现了"乐观初始值"探索策略
        """
        if state not in self.values:
            self.values[state] = self.initial_value
        return self.values[state]
    
    def update_td(self, state: tuple, next_state: tuple, 
                  reward: float, alpha: float = 0.1):
        """
        时序差分(TD)更新 - 强化学习的核心算法！
        
        TD更新公式（书中最重要的公式之一）：
        V(S_t) ← V(S_t) + α[R_{t+1} + γV(S_{t+1}) - V(S_t)]
        
        其中：
        - α: 学习率（我们多相信新经验）
        - R_{t+1}: 立即奖励
        - γ: 折扣因子（未来的重要性）
        - V(S_{t+1}): 下一状态的价值估计
        - V(S_t): 当前状态的价值估计
        
        直觉理解：
        如果下一个状态比预期好 → 提高当前状态价值
        如果下一个状态比预期差 → 降低当前状态价值
        """
        current_value = self.get_value(state)
        next_value = self.get_value(next_state) if next_state else 0
        
        # TD误差 = (奖励 + 未来价值) - 当前估计
        td_error = (reward + next_value) - current_value
        
        # 更新价值
        self.values[state] = current_value + alpha * td_error
    
    def update_monte_carlo(self, episode_states: List[tuple], 
                           final_reward: float, alpha: float = 0.1):
        """
        蒙特卡洛更新 - 另一种学习方式
        
        与TD不同，MC等到游戏结束才更新
        根据最终结果更新整条路径上的所有状态
        
        这展示了RL中的两大学习范式：
        1. TD: 边走边学（在线学习）
        2. MC: 走完再学（离线学习）
        """
        for state in episode_states:
            current_value = self.get_value(state)
            # 向最终结果靠拢
            self.values[state] = current_value + alpha * (final_reward - current_value)


# ================================================================================
# 第1.4.3节：强化学习智能体
# Section 1.4.3: Reinforcement Learning Agent
# ================================================================================

class TicTacToeAgent:
    """
    井字棋强化学习智能体
    
    这个智能体展示了完整的强化学习循环：
    1. 观察状态 (Observe State)
    2. 选择动作 (Select Action) - 基于价值函数
    3. 执行动作 (Execute Action)
    4. 获得奖励 (Receive Reward)
    5. 学习更新 (Learn and Update)
    
    关键组件：
    - 价值函数：评估状态好坏
    - 探索策略：ε-贪婪
    - 学习算法：TD学习
    """
    
    def __init__(self, player: int = 1, epsilon: float = 0.1, 
                 alpha: float = 0.1, name: str = "RL_Agent"):
        """
        初始化智能体
        
        参数说明：
        player: 玩家标识 (1=X, -1=O)
        epsilon: 探索率（探索vs利用的平衡）
        alpha: 学习率（学习速度）
        name: 智能体名称
        
        这些超参数的选择艺术：
        - ε太小：可能错过更好的策略
        - ε太大：学习太慢，总在随机探索
        - α太小：学习太慢
        - α太大：不稳定，新经验覆盖旧知识
        """
        self.player = player
        self.epsilon = epsilon
        self.alpha = alpha
        self.name = name
        self.value_function = ValueFunction(player)
        
        # 记录学习历史
        self.state_history = []  # 一局游戏的状态序列
        self.win_history = []    # 胜率记录
        
    def select_action(self, board: TicTacToeBoard, training: bool = True) -> int:
        """
        选择动作 - ε-贪婪策略
        
        这是强化学习的核心决策过程！
        
        训练时：
        - 有ε概率随机探索（尝试新策略）
        - 有1-ε概率选择最优（利用已知知识）
        
        测试时：
        - 总是选择最优（展示学习成果）
        """
        available_actions = board.get_available_actions()
        
        if training and np.random.random() < self.epsilon:
            # 探索：随机选择
            # 就像小明偶尔会尝试新的下法
            return np.random.choice(available_actions)
        else:
            # 利用：选择价值最高的动作
            # 这需要"向前看一步"
            action_values = []
            
            for action in available_actions:
                # 想象：如果我下这一步，棋盘会变成什么样？
                future_board = self._imagine_move(board, action)
                future_state = future_board.get_state()
                
                # 这个未来状态的价值是多少？
                # 注意：对手的价值要取反！
                if board.current_player == self.player:
                    value = self.value_function.get_value(future_state)
                else:
                    value = 1 - self.value_function.get_value(future_state)
                
                action_values.append((action, value))
            
            # 选择价值最高的动作
            # 如果有多个最优，随机选一个
            max_value = max(v for _, v in action_values)
            best_actions = [a for a, v in action_values if v == max_value]
            return np.random.choice(best_actions)
    
    def _imagine_move(self, board: TicTacToeBoard, action: int) -> TicTacToeBoard:
        """
        想象下一步后的棋盘
        
        这是"前向思考"的能力
        不实际改变棋盘，只是想象结果
        """
        imaginary_board = TicTacToeBoard()
        imaginary_board.board = board.board.copy()
        imaginary_board.current_player = board.current_player
        imaginary_board.board.flat[action] = board.current_player
        return imaginary_board
    
    def start_episode(self):
        """开始新的一局游戏"""
        self.state_history = []
    
    def observe_and_act(self, board: TicTacToeBoard, training: bool = True) -> int:
        """
        观察状态并采取行动
        
        完整的智能体决策流程
        """
        # 记录当前状态
        state = board.get_state()
        self.state_history.append(state)
        
        # 选择并返回动作
        return self.select_action(board, training)
    
    def learn_from_episode(self, final_reward: float):
        """
        从一局游戏中学习
        
        游戏结束后，回顾整局游戏，更新价值函数
        
        关键思想：
        - 赢了：提高路径上所有状态的价值
        - 输了：降低路径上所有状态的价值
        - 平局：小幅调整
        
        这就是"延迟奖励"的学习！
        """
        if not self.state_history:
            return
        
        # TD学习：从后向前更新
        # 为什么从后向前？因为需要用到"下一状态"的价值
        for i in range(len(self.state_history) - 1, -1, -1):
            state = self.state_history[i]
            
            if i == len(self.state_history) - 1:
                # 最后一个状态，直接用最终奖励
                next_state = None
                reward = final_reward
            else:
                # 中间状态，奖励为0，但有下一状态价值
                next_state = self.state_history[i + 1]
                reward = 0
            
            # TD更新
            self.value_function.update_td(state, next_state, reward, self.alpha)
        
        # 清空历史，准备下一局
        self.state_history = []
    
    def save_knowledge(self, filepath: str):
        """
        保存学到的知识（价值函数）
        
        训练后的价值函数就是智能体的"经验"
        可以保存下来，下次直接使用
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.value_function.values, f)
        print(f"{self.name}的知识已保存到 {filepath}")
    
    def load_knowledge(self, filepath: str):
        """加载之前学到的知识"""
        with open(filepath, 'rb') as f:
            self.value_function.values = pickle.load(f)
        print(f"{self.name}已加载知识")


# ================================================================================
# 第1.4.4节：自我对弈训练
# Section 1.4.4: Self-Play Training
# ================================================================================

class SelfPlayTrainer:
    """
    自我对弈训练器
    
    强化学习的魅力之一：不需要人类对手！
    智能体可以通过自我对弈不断提升
    
    就像AlphaGo通过自我对弈成为世界冠军
    """
    
    def __init__(self, episodes: int = 10000):
        """
        初始化训练器
        
        episodes: 训练局数
        更多的训练 = 更强的智能体
        """
        self.episodes = episodes
        self.board = TicTacToeBoard()
        
    def train(self, agent1: TicTacToeAgent, agent2: TicTacToeAgent, 
              verbose: bool = True) -> Dict:
        """
        训练两个智能体
        
        通过大量对弈，两个智能体互相学习，共同进步
        这展示了"协同进化"的概念
        """
        results = {'agent1_wins': 0, 'agent2_wins': 0, 'draws': 0}
        win_rates = []
        
        for episode in range(self.episodes):
            # 开始新游戏
            self.board.reset()
            agent1.start_episode()
            agent2.start_episode()
            
            # 轮流下棋直到游戏结束
            current_agent = agent1
            other_agent = agent2
            
            while True:
                # 当前玩家行动
                action = current_agent.observe_and_act(self.board, training=True)
                reward, done = self.board.make_move(action)
                
                if done:
                    # 游戏结束，双方学习
                    if reward == agent1.player:
                        # agent1赢
                        agent1.learn_from_episode(1)
                        agent2.learn_from_episode(-1)
                        results['agent1_wins'] += 1
                    elif reward == agent2.player:
                        # agent2赢
                        agent1.learn_from_episode(-1)
                        agent2.learn_from_episode(1)
                        results['agent2_wins'] += 1
                    else:
                        # 平局
                        agent1.learn_from_episode(0)
                        agent2.learn_from_episode(0)
                        results['draws'] += 1
                    break
                
                # 交换玩家
                current_agent, other_agent = other_agent, current_agent
            
            # 定期报告进度
            if (episode + 1) % 1000 == 0:
                win_rate = results['agent1_wins'] / (episode + 1)
                win_rates.append(win_rate)
                
                if verbose:
                    print(f"训练进度 Episode {episode + 1}/{self.episodes}")
                    print(f"  Agent1胜率: {win_rate:.2%}")
                    print(f"  Agent2胜率: {results['agent2_wins']/(episode+1):.2%}")
                    print(f"  平局率: {results['draws']/(episode+1):.2%}")
                    print(f"  Agent1已学习{len(agent1.value_function.values)}个状态")
        
        return results, win_rates
    
    def demonstrate_learning(self):
        """
        演示学习过程
        
        展示智能体如何从随机玩家成长为高手
        """
        print("="*70)
        print("井字棋自我对弈学习演示")
        print("Tic-Tac-Toe Self-Play Learning Demonstration")
        print("="*70)
        
        # 创建两个智能体
        agent_x = TicTacToeAgent(player=1, epsilon=0.3, alpha=0.1, name="Agent_X")
        agent_o = TicTacToeAgent(player=-1, epsilon=0.3, alpha=0.1, name="Agent_O")
        
        print("\n初始状态：两个智能体都是新手")
        print("Initial: Both agents are beginners")
        print("-"*40)
        
        # 测试初始水平
        self._test_against_random(agent_x, n_games=100, verbose=True)
        
        # 开始训练
        print("\n开始自我对弈训练...")
        print("Starting self-play training...")
        print("-"*40)
        
        trainer = SelfPlayTrainer(episodes=5000)
        results, win_rates = trainer.train(agent_x, agent_o, verbose=True)
        
        print("\n训练完成！")
        print("Training complete!")
        print("-"*40)
        
        # 测试训练后水平
        print("\n训练后水平测试：")
        print("Post-training test:")
        self._test_against_random(agent_x, n_games=100, verbose=True)
        
        # 展示学习曲线
        self._plot_learning_curve(win_rates)
        
        return agent_x, agent_o
    
    def _test_against_random(self, agent: TicTacToeAgent, 
                             n_games: int = 100, verbose: bool = True):
        """测试智能体对战随机玩家"""
        wins = 0
        draws = 0
        
        for _ in range(n_games):
            self.board.reset()
            agent.start_episode()
            
            # 随机决定谁先手
            agent_first = np.random.choice([True, False])
            
            while True:
                if agent_first:
                    # 智能体先下
                    if self.board.current_player == agent.player:
                        action = agent.observe_and_act(self.board, training=False)
                    else:
                        # 随机玩家
                        action = np.random.choice(self.board.get_available_actions())
                else:
                    # 随机玩家先下
                    if self.board.current_player != agent.player:
                        action = np.random.choice(self.board.get_available_actions())
                    else:
                        action = agent.observe_and_act(self.board, training=False)
                
                reward, done = self.board.make_move(action)
                
                if done:
                    if reward == agent.player:
                        wins += 1
                    elif reward == 0:
                        draws += 1
                    break
        
        if verbose:
            print(f"{agent.name} vs 随机玩家 ({n_games}局):")
            print(f"  胜率: {wins/n_games:.1%}")
            print(f"  平局率: {draws/n_games:.1%}")
            print(f"  败率: {(n_games-wins-draws)/n_games:.1%}")
    
    def _plot_learning_curve(self, win_rates: List[float]):
        """绘制学习曲线"""
        plt.figure(figsize=(10, 6))
        episodes = [i * 1000 for i in range(1, len(win_rates) + 1)]
        plt.plot(episodes, win_rates, 'b-', linewidth=2)
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% baseline')
        plt.xlabel('训练局数 Training Episodes')
        plt.ylabel('Agent1 胜率 Win Rate')
        plt.title('井字棋自我对弈学习曲线\nTic-Tac-Toe Self-Play Learning Curve')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()


# ================================================================================
# 第1.4.5节：人机对战
# Section 1.4.5: Human vs AI
# ================================================================================

class HumanVsAI:
    """
    人机对战界面
    
    测试训练好的智能体，看看它学到了什么！
    """
    
    def __init__(self, agent: TicTacToeAgent):
        """初始化对战系统"""
        self.agent = agent
        self.board = TicTacToeBoard()
        
    def play(self):
        """开始人机对战"""
        print("\n" + "="*70)
        print("井字棋人机对战")
        print("Tic-Tac-Toe: Human vs AI")
        print("="*70)
        
        print("\n游戏说明:")
        print("输入位置编号(0-8)来下棋：")
        print("  0 1 2")
        print("  3 4 5")
        print("  6 7 8")
        
        # 随机决定谁先手
        human_first = np.random.choice([True, False])
        human_player = 1 if human_first else -1
        
        print(f"\n你是 {'X' if human_player == 1 else 'O'}")
        print(f"{'你' if human_first else 'AI'}先手")
        
        self.board.reset()
        self.agent.start_episode()
        
        while True:
            self.board.render()
            
            if self.board.current_player == human_player:
                # 人类回合
                while True:
                    try:
                        action = int(input("你的选择 (0-8): "))
                        if action in self.board.get_available_actions():
                            break
                        else:
                            print("该位置已有棋子，请重新选择")
                    except:
                        print("请输入0-8之间的数字")
            else:
                # AI回合
                print("AI思考中...")
                action = self.agent.observe_and_act(self.board, training=False)
                print(f"AI选择: {action}")
            
            reward, done = self.board.make_move(action)
            
            if done:
                self.board.render()
                if reward == human_player:
                    print("\n🎉 恭喜你赢了！")
                elif reward == -human_player:
                    print("\n😔 AI赢了！")
                else:
                    print("\n🤝 平局！")
                
                # 询问是否再来一局
                again = input("\n再来一局？(y/n): ").lower()
                if again == 'y':
                    self.board.reset()
                    self.agent.start_episode()
                    human_first = not human_first
                    human_player = 1 if human_first else -1
                    print(f"\n新游戏！你是 {'X' if human_player == 1 else 'O'}")
                    print(f"{'你' if human_first else 'AI'}先手")
                else:
                    break


# ================================================================================
# 实践：完整的井字棋强化学习
# Practice: Complete Tic-Tac-Toe Reinforcement Learning
# ================================================================================

def main():
    """主程序：展示完整的强化学习过程"""
    
    print("╔" + "═"*68 + "╗")
    print("║" + " "*15 + "第1章：井字棋强化学习完整实现".center(38) + " "*15 + "║")
    print("║" + " "*10 + "Chapter 1: Complete Tic-Tac-Toe RL Implementation".center(48) + " "*10 + "║")
    print("╚" + "═"*68 + "╝")
    
    print("\n欢迎来到井字棋强化学习世界！")
    print("Welcome to Tic-Tac-Toe Reinforcement Learning!")
    
    # 1. 演示学习过程
    print("\n【第1部分：自我对弈学习】")
    print("[Part 1: Self-Play Learning]")
    print("="*70)
    
    trainer = SelfPlayTrainer()
    agent_x, agent_o = trainer.demonstrate_learning()
    
    # 2. 分析学到的策略
    print("\n【第2部分：策略分析】")
    print("[Part 2: Strategy Analysis]")
    print("="*70)
    
    print("\n让我们看看智能体学到了什么策略：")
    print("Let's see what strategies the agent learned:")
    
    # 分析一些关键状态的价值
    board = TicTacToeBoard()
    
    # 初始状态
    print("\n1. 初始状态价值:")
    initial_state = board.get_state()
    print(f"   V(empty board) = {agent_x.value_function.get_value(initial_state):.3f}")
    print("   (应该接近0.5，表示双方机会均等)")
    
    # 中心格子
    board.board[1, 1] = 1
    center_state = board.get_state()
    print(f"\n2. 占据中心后的价值:")
    print(f"   V(X in center) = {agent_x.value_function.get_value(center_state):.3f}")
    print("   (应该>0.5，中心是好位置)")
    
    # 即将获胜
    board.reset()
    board.board[0, 0] = 1
    board.board[0, 1] = 1
    winning_state = board.get_state()
    print(f"\n3. 即将连线的价值:")
    print(f"   V(two X in row) = {agent_x.value_function.get_value(winning_state):.3f}")
    print("   (应该接近1.0，马上就赢了)")
    
    # 3. 人机对战
    print("\n【第3部分：人机对战】")
    print("[Part 3: Human vs AI]")
    print("="*70)
    
    play_vs_ai = input("\n想要挑战AI吗？(y/n): ").lower()
    if play_vs_ai == 'y':
        game = HumanVsAI(agent_x)
        game.play()
    
    # 4. 总结
    print("\n" + "="*70)
    print("学习总结 Learning Summary")
    print("="*70)
    print("""
    通过井字棋，我们学到了强化学习的核心概念：
    
    1. 价值函数 Value Function V(s)
       - 评估每个状态的好坏
       - 通过经验不断更新
    
    2. 时序差分学习 TD Learning
       - V(s) ← V(s) + α[R + V(s') - V(s)]
       - 从每一步中学习
    
    3. 探索vs利用 Exploration vs Exploitation
       - ε-贪婪策略平衡两者
       - 既要尝试新策略，也要用已知最好的
    
    4. 自我对弈 Self-Play
       - 不需要人类专家
       - 通过自我提升达到高水平
    
    这些概念将贯穿整个强化学习！
    These concepts will run through all of RL!
    
    下一步：学习更复杂的问题和算法
    Next: Learn more complex problems and algorithms
    """)
    
    # 保存训练好的智能体
    save = input("\n保存训练好的智能体吗？(y/n): ").lower()
    if save == 'y':
        agent_x.save_knowledge("tictactoe_agent_x.pkl")
        agent_o.save_knowledge("tictactoe_agent_o.pkl")
        print("智能体已保存！")


if __name__ == "__main__":
    main()