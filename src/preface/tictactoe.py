"""
Tic-Tac-Toe: Value Function and Temporal Difference Learning
井字棋：价值函数与时序差分学习

This implements the tic-tac-toe example from the preface, demonstrating:
1. State representation in RL
2. Value function approximation  
3. Temporal difference learning
4. Exploration vs exploitation

这实现了前言中的井字棋例子，演示了：
1. 强化学习中的状态表示
2. 价值函数近似
3. 时序差分学习
4. 探索与利用
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# Configure logging / 配置日志
logger = logging.getLogger(__name__)


class TicTacToeState:
    """
    Represents a state in tic-tac-toe
    表示井字棋的一个状态
    
    Board positions:
    棋盘位置：
    0 | 1 | 2
    ---------
    3 | 4 | 5
    ---------
    6 | 7 | 8
    
    Encoding:
    编码：
    0: empty / 空
    1: player 1 (X) / 玩家1 (X)
    -1: player 2 (O) / 玩家2 (O)
    """
    
    def __init__(self, board: Optional[np.ndarray] = None):
        """
        Initialize board state
        初始化棋盘状态
        
        Args:
            board: 3x3 array representing the board / 表示棋盘的3x3数组
        """
        if board is None:
            # Initialize empty board / 初始化空棋盘
            self.board = np.zeros((3, 3), dtype=np.int32)
        else:
            self.board = board.copy()
        
        # Calculate hash for efficient state lookup
        # 计算哈希值用于高效状态查找
        self._hash = None
        
    def get_hash(self) -> int:
        """
        Get unique hash for this state
        获取此状态的唯一哈希值
        
        Uses base-3 encoding: each position can be 0, 1, or 2
        使用三进制编码：每个位置可以是0、1或2
        
        Returns:
            Hash value / 哈希值
        """
        if self._hash is None:
            # Convert board to base-3 number
            # 将棋盘转换为三进制数
            flat_board = self.board.flatten() + 1  # Shift to 0,1,2
            self._hash = 0
            for i, val in enumerate(flat_board):
                self._hash += val * (3 ** i)
        return self._hash
    
    def is_terminal(self) -> Tuple[bool, Optional[int]]:
        """
        Check if state is terminal (game over)
        检查状态是否为终止状态（游戏结束）
        
        Returns:
            Tuple of (is_terminal, winner)
            元组：(是否终止, 获胜者)
            winner: 1 for player 1, -1 for player 2, 0 for draw, None if not terminal
            获胜者：1表示玩家1，-1表示玩家2，0表示平局，None表示未结束
        """
        # Check rows / 检查行
        for row in range(3):
            row_sum = np.sum(self.board[row, :])
            if row_sum == 3:  # Player 1 wins / 玩家1获胜
                return True, 1
            elif row_sum == -3:  # Player 2 wins / 玩家2获胜
                return True, -1
        
        # Check columns / 检查列  
        for col in range(3):
            col_sum = np.sum(self.board[:, col])
            if col_sum == 3:
                return True, 1
            elif col_sum == -3:
                return True, -1
        
        # Check diagonals / 检查对角线
        diag1_sum = self.board[0, 0] + self.board[1, 1] + self.board[2, 2]
        diag2_sum = self.board[0, 2] + self.board[1, 1] + self.board[2, 0]
        
        if diag1_sum == 3 or diag2_sum == 3:
            return True, 1
        elif diag1_sum == -3 or diag2_sum == -3:
            return True, -1
        
        # Check for draw (board full) / 检查平局（棋盘满）
        if np.all(self.board != 0):
            return True, 0
        
        # Game not over / 游戏未结束
        return False, None
    
    def get_available_actions(self) -> List[int]:
        """
        Get list of available actions (empty positions)
        获取可用动作列表（空位置）
        
        Returns:
            List of position indices (0-8) / 位置索引列表（0-8）
        """
        # Find empty positions / 查找空位置
        empty_positions = np.where(self.board.flatten() == 0)[0]
        return empty_positions.tolist()
    
    def take_action(self, action: int, player: int) -> 'TicTacToeState':
        """
        Take an action and return new state
        执行动作并返回新状态
        
        Args:
            action: Position to place marker (0-8) / 放置标记的位置（0-8）
            player: Player making move (1 or -1) / 进行移动的玩家（1或-1）
            
        Returns:
            New state after action / 动作后的新状态
        """
        assert action in self.get_available_actions(), f"Invalid action: {action}"
        assert player in [1, -1], f"Invalid player: {player}"
        
        # Create new state with action applied
        # 创建应用动作后的新状态
        new_board = self.board.copy()
        row, col = action // 3, action % 3
        new_board[row, col] = player
        
        return TicTacToeState(new_board)
    
    def display(self):
        """
        Display the board in a human-readable format
        以人类可读格式显示棋盘
        """
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        print("\n" + "=" * 13)
        for row in range(3):
            row_str = " | ".join([symbols[self.board[row, col]] for col in range(3)])
            print(f" {row_str} ")
            if row < 2:
                print("-" * 13)
        print("=" * 13 + "\n")
    
    def __repr__(self) -> str:
        return f"TicTacToeState(board=\n{self.board})"


class TicTacToePlayer:
    """
    Tic-tac-toe player using value function approximation
    使用价值函数近似的井字棋玩家
    
    This implements temporal difference learning for tic-tac-toe
    这实现了井字棋的时序差分学习
    """
    
    def __init__(self, player_id: int, epsilon: float = 0.1, 
                 alpha: float = 0.5, init_value: float = 0.5):
        """
        Initialize player
        初始化玩家
        
        Args:
            player_id: 1 for X, -1 for O / 1表示X，-1表示O
            epsilon: Exploration rate / 探索率
            alpha: Learning rate / 学习率
            init_value: Initial value for unknown states / 未知状态的初始值
        """
        self.player_id = player_id  # Player identifier / 玩家标识
        self.epsilon = epsilon  # ε for ε-greedy / ε-贪婪的ε
        self.alpha = alpha  # Learning rate α / 学习率α
        self.init_value = init_value  # Initial V(s) / 初始V(s)
        
        # State value function: V(s) for each state
        # 状态价值函数：每个状态的V(s)
        self.state_values = {}  # Dictionary mapping state_hash -> value
        
        # Track states visited in current episode
        # 跟踪当前回合访问的状态
        self.states_visited = []
        
        # Statistics / 统计
        self.wins = 0
        self.losses = 0
        self.draws = 0
        
        logger.info(f"Player {player_id} initialized: "
                   f"ε={epsilon}, α={alpha}, init_V={init_value}")
    
    def reset_episode(self):
        """
        Reset for new episode
        重置以开始新回合
        """
        self.states_visited = []
    
    def get_value(self, state: TicTacToeState) -> float:
        """
        Get value of a state, initializing if necessary
        获取状态的价值，必要时进行初始化
        
        Args:
            state: Game state / 游戏状态
            
        Returns:
            Value V(s) / 价值V(s)
        """
        state_hash = state.get_hash()
        
        if state_hash not in self.state_values:
            # Initialize value for new state
            # 为新状态初始化价值
            is_terminal, winner = state.is_terminal()
            
            if is_terminal:
                # Terminal states have fixed values
                # 终止状态有固定价值
                if winner == self.player_id:
                    self.state_values[state_hash] = 1.0  # Win / 赢
                elif winner == -self.player_id:
                    self.state_values[state_hash] = 0.0  # Loss / 输
                else:
                    self.state_values[state_hash] = 0.5  # Draw / 平
            else:
                # Non-terminal states start at init_value
                # 非终止状态从init_value开始
                self.state_values[state_hash] = self.init_value
        
        return self.state_values[state_hash]
    
    def select_action(self, state: TicTacToeState, 
                     training: bool = True) -> int:
        """
        Select action using ε-greedy policy
        使用ε-贪婪策略选择动作
        
        Args:
            state: Current state / 当前状态
            training: Whether in training mode / 是否在训练模式
            
        Returns:
            Selected action / 选择的动作
        """
        available_actions = state.get_available_actions()
        
        # Exploration vs exploitation / 探索与利用
        if training and np.random.random() < self.epsilon:
            # Explore: random action / 探索：随机动作
            action = np.random.choice(available_actions)
            logger.debug(f"Player {self.player_id} explores: action {action}")
        else:
            # Exploit: choose action with highest value
            # 利用：选择具有最高价值的动作
            action_values = []
            
            for action in available_actions:
                # Simulate taking action / 模拟执行动作
                next_state = state.take_action(action, self.player_id)
                value = self.get_value(next_state)
                action_values.append((action, value))
            
            # Select action with maximum value
            # 选择具有最大价值的动作
            action_values.sort(key=lambda x: x[1], reverse=True)
            action = action_values[0][0]
            
            logger.debug(f"Player {self.player_id} exploits: "
                        f"action {action} (V={action_values[0][1]:.3f})")
        
        return action
    
    def update_values(self, final_reward: float):
        """
        Update value function using temporal difference learning
        使用时序差分学习更新价值函数
        
        V(S_t) ← V(S_t) + α[V(S_{t+1}) - V(S_t)]
        
        For the final state:
        V(S_T) ← V(S_T) + α[R - V(S_T)]
        
        对于最终状态：
        V(S_T) ← V(S_T) + α[R - V(S_T)]
        
        Args:
            final_reward: Reward at end of episode / 回合结束时的奖励
        """
        # Update statistics / 更新统计
        if final_reward == 1.0:
            self.wins += 1
        elif final_reward == 0.0:
            self.losses += 1
        else:
            self.draws += 1
        
        # Temporal difference backup from end to start
        # 从结束到开始的时序差分备份
        for i in reversed(range(len(self.states_visited))):
            state_hash = self.states_visited[i]
            
            if i == len(self.states_visited) - 1:
                # Last state: use actual reward
                # 最后状态：使用实际奖励
                td_target = final_reward
            else:
                # Earlier states: use next state value
                # 早期状态：使用下一状态价值
                next_state_hash = self.states_visited[i + 1]
                td_target = self.state_values[next_state_hash]
            
            # TD update: V(s) ← V(s) + α[target - V(s)]
            # TD更新：V(s) ← V(s) + α[目标 - V(s)]
            current_value = self.state_values[state_hash]
            td_error = td_target - current_value
            self.state_values[state_hash] += self.alpha * td_error
            
            logger.debug(f"Updated V[{state_hash}]: "
                        f"{current_value:.3f} -> {self.state_values[state_hash]:.3f} "
                        f"(TD error: {td_error:.3f})")
    
    def save_policy(self, filepath: str):
        """
        Save learned value function to file
        将学习到的价值函数保存到文件
        
        Args:
            filepath: Path to save file / 保存文件路径
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.state_values, f)
        logger.info(f"Policy saved to {filepath} ({len(self.state_values)} states)")
    
    def load_policy(self, filepath: str):
        """
        Load value function from file
        从文件加载价值函数
        
        Args:
            filepath: Path to load file / 加载文件路径
        """
        with open(filepath, 'rb') as f:
            self.state_values = pickle.load(f)
        logger.info(f"Policy loaded from {filepath} ({len(self.state_values)} states)")


class TicTacToeEnvironment:
    """
    Tic-tac-toe game environment
    井字棋游戏环境
    
    Manages game flow and player interactions
    管理游戏流程和玩家交互
    """
    
    def __init__(self, player1: TicTacToePlayer, player2: TicTacToePlayer):
        """
        Initialize environment with two players
        使用两个玩家初始化环境
        
        Args:
            player1: First player (X) / 第一个玩家（X）
            player2: Second player (O) / 第二个玩家（O）
        """
        self.player1 = player1  # X player / X玩家
        self.player2 = player2  # O player / O玩家
        self.state = None
        self.current_player = None
        
    def reset(self) -> TicTacToeState:
        """
        Reset game to initial state
        重置游戏到初始状态
        
        Returns:
            Initial empty board / 初始空棋盘
        """
        self.state = TicTacToeState()
        self.current_player = 1  # Player 1 (X) starts / 玩家1（X）先手
        self.player1.reset_episode()
        self.player2.reset_episode()
        return self.state
    
    def play_episode(self, training: bool = True, verbose: bool = False) -> int:
        """
        Play one complete game episode
        进行一个完整的游戏回合
        
        Args:
            training: Whether players should learn / 玩家是否应该学习
            verbose: Whether to print game progress / 是否打印游戏进度
            
        Returns:
            Winner: 1 for player1, -1 for player2, 0 for draw
            获胜者：1表示玩家1，-1表示玩家2，0表示平局
        """
        # Initialize game / 初始化游戏
        self.reset()
        
        if verbose:
            print("\n" + "=" * 40)
            print("New Game / 新游戏")
            print("=" * 40)
            self.state.display()
        
        # Play until game ends / 玩到游戏结束
        while True:
            # Get current player / 获取当前玩家
            if self.current_player == 1:
                player = self.player1
            else:
                player = self.player2
            
            # Player selects action / 玩家选择动作
            action = player.select_action(self.state, training)
            
            # Record state for learning / 记录状态用于学习
            if training:
                player.states_visited.append(self.state.get_hash())
            
            # Execute action / 执行动作
            self.state = self.state.take_action(action, self.current_player)
            
            if verbose:
                print(f"\nPlayer {self.current_player} moves to position {action}:")
                self.state.display()
            
            # Check if game ended / 检查游戏是否结束
            is_terminal, winner = self.state.is_terminal()
            
            if is_terminal:
                # Game over - update values if training
                # 游戏结束 - 如果在训练则更新价值
                if training:
                    # Record final state / 记录最终状态
                    self.player1.states_visited.append(self.state.get_hash())
                    self.player2.states_visited.append(self.state.get_hash())
                    
                    # Calculate rewards / 计算奖励
                    if winner == 1:  # Player 1 wins / 玩家1赢
                        reward1, reward2 = 1.0, 0.0
                    elif winner == -1:  # Player 2 wins / 玩家2赢
                        reward1, reward2 = 0.0, 1.0
                    else:  # Draw / 平局
                        reward1, reward2 = 0.5, 0.5
                    
                    # Update value functions / 更新价值函数
                    self.player1.update_values(reward1)
                    self.player2.update_values(reward2)
                
                if verbose:
                    if winner == 1:
                        print("Player 1 (X) wins! / 玩家1（X）获胜！")
                    elif winner == -1:
                        print("Player 2 (O) wins! / 玩家2（O）获胜！")
                    else:
                        print("It's a draw! / 平局！")
                
                return winner
            
            # Switch players / 切换玩家
            self.current_player *= -1


def train_players(n_episodes: int = 10000, 
                  epsilon: float = 0.1,
                  alpha: float = 0.5) -> Tuple[TicTacToePlayer, TicTacToePlayer, List[float]]:
    """
    Train two tic-tac-toe players through self-play
    通过自我对弈训练两个井字棋玩家
    
    Args:
        n_episodes: Number of training episodes / 训练回合数
        epsilon: Exploration rate / 探索率
        alpha: Learning rate / 学习率
        
    Returns:
        Tuple of (player1, player2, win_rates)
        元组：(玩家1，玩家2，胜率)
    """
    # Initialize players / 初始化玩家
    player1 = TicTacToePlayer(player_id=1, epsilon=epsilon, alpha=alpha)
    player2 = TicTacToePlayer(player_id=-1, epsilon=epsilon, alpha=alpha)
    
    # Initialize environment / 初始化环境
    env = TicTacToeEnvironment(player1, player2)
    
    # Track statistics / 跟踪统计
    win_rates = []  # Player 1 win rate over time / 玩家1随时间的胜率
    
    logger.info(f"Training {n_episodes} episodes...")
    
    # Training loop / 训练循环
    for episode in tqdm(range(n_episodes), desc="Training"):
        # Play one episode / 进行一回合
        winner = env.play_episode(training=True, verbose=False)
        
        # Track win rate every 100 episodes / 每100回合跟踪胜率
        if (episode + 1) % 100 == 0:
            win_rate = player1.wins / (episode + 1)
            win_rates.append(win_rate)
            
            if (episode + 1) % 1000 == 0:
                logger.info(f"Episode {episode + 1}: "
                           f"P1 wins={player1.wins}, "
                           f"P2 wins={player2.wins}, "
                           f"Draws={player1.draws}")
    
    # Final statistics / 最终统计
    total_games = n_episodes
    logger.info("=" * 60)
    logger.info("Training Complete / 训练完成")
    logger.info("=" * 60)
    logger.info(f"Player 1 (X): Wins={player1.wins} ({100*player1.wins/total_games:.1f}%), "
               f"Losses={player1.losses} ({100*player1.losses/total_games:.1f}%), "
               f"Draws={player1.draws} ({100*player1.draws/total_games:.1f}%)")
    logger.info(f"Player 2 (O): Wins={player2.wins} ({100*player2.wins/total_games:.1f}%), "
               f"Losses={player2.losses} ({100*player2.losses/total_games:.1f}%), "
               f"Draws={player2.draws} ({100*player2.draws/total_games:.1f}%)")
    logger.info(f"States explored: P1={len(player1.state_values)}, P2={len(player2.state_values)}")
    
    return player1, player2, win_rates


def visualize_training_progress(win_rates: List[float]):
    """
    Visualize training progress
    可视化训练进度
    
    Args:
        win_rates: List of win rates over training / 训练过程中的胜率列表
    """
    plt.figure(figsize=(10, 6))
    
    # Plot win rate / 绘制胜率
    episodes = np.arange(1, len(win_rates) + 1) * 100
    plt.plot(episodes, win_rates, 'b-', linewidth=2, label='Player 1 Win Rate')
    
    # Add reference lines / 添加参考线
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% (Random)')
    plt.axhline(y=1/3, color='g', linestyle='--', alpha=0.5, label='33.3% (Equal skill)')
    
    # Labels and formatting / 标签和格式化
    plt.xlabel('Episodes / 回合数', fontsize=12)
    plt.ylabel('Win Rate / 胜率', fontsize=12)
    plt.title('Tic-Tac-Toe Training Progress / 井字棋训练进度', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # Add annotations / 添加注释
    plt.text(episodes[-1], win_rates[-1], f'{win_rates[-1]:.3f}',
            ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    return plt.gcf()


def demonstrate_value_function(player: TicTacToePlayer):
    """
    Demonstrate learned value function for specific board positions
    演示特定棋盘位置的学习价值函数
    
    Args:
        player: Trained player / 训练好的玩家
    """
    logger.info("=" * 60)
    logger.info("Value Function Examples / 价值函数示例")
    logger.info("=" * 60)
    
    # Example 1: Empty board / 示例1：空棋盘
    empty_board = TicTacToeState()
    value = player.get_value(empty_board)
    logger.info(f"\nEmpty board value / 空棋盘价值: {value:.3f}")
    empty_board.display()
    
    # Example 2: Near-win position for player 1
    # 示例2：玩家1接近获胜的位置
    near_win_board = np.array([
        [1, 1, 0],
        [-1, -1, 0],
        [0, 0, 0]
    ])
    near_win_state = TicTacToeState(near_win_board)
    value = player.get_value(near_win_state)
    logger.info(f"\nNear-win position value / 接近获胜位置价值: {value:.3f}")
    near_win_state.display()
    
    # Example 3: Defensive position
    # 示例3：防守位置
    defensive_board = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, 0]
    ])
    defensive_state = TicTacToeState(defensive_board)
    value = player.get_value(defensive_state)
    logger.info(f"\nDefensive position value / 防守位置价值: {value:.3f}")
    defensive_state.display()
    
    # Show best actions for each state / 显示每个状态的最佳动作
    for state, name in [(empty_board, "Empty"), 
                        (near_win_state, "Near-win"),
                        (defensive_state, "Defensive")]:
        logger.info(f"\nBest actions for {name} position / {name}位置的最佳动作:")
        
        available_actions = state.get_available_actions()
        action_values = []
        
        for action in available_actions:
            next_state = state.take_action(action, player.player_id)
            value = player.get_value(next_state)
            action_values.append((action, value))
        
        # Sort by value / 按价值排序
        action_values.sort(key=lambda x: x[1], reverse=True)
        
        for action, value in action_values[:3]:  # Top 3 actions / 前3个动作
            row, col = action // 3, action % 3
            logger.info(f"  Position ({row},{col}): V={value:.3f}")


def human_vs_ai_game(ai_player: TicTacToePlayer):
    """
    Play a game against the trained AI
    与训练好的AI对战
    
    Args:
        ai_player: Trained AI player / 训练好的AI玩家
    """
    print("\n" + "=" * 60)
    print("Human vs AI Tic-Tac-Toe / 人机井字棋对战")
    print("=" * 60)
    print("You are X, AI is O / 你是X，AI是O")
    print("Positions are numbered 0-8:")
    print("0 | 1 | 2")
    print("---------")
    print("3 | 4 | 5")
    print("---------")
    print("6 | 7 | 8")
    print("=" * 60)
    
    # Initialize game / 初始化游戏
    state = TicTacToeState()
    state.display()
    
    # Human plays as player 1 (X) / 人类玩家1（X）
    current_player = 1
    
    while True:
        if current_player == 1:
            # Human turn / 人类回合
            available = state.get_available_actions()
            print(f"\nAvailable positions / 可用位置: {available}")
            
            while True:
                try:
                    action = int(input("Enter position (0-8) / 输入位置（0-8）: "))
                    if action in available:
                        break
                    else:
                        print(f"Invalid position. Choose from {available}")
                except ValueError:
                    print("Please enter a number")
            
            state = state.take_action(action, current_player)
        else:
            # AI turn / AI回合
            print("\nAI is thinking / AI正在思考...")
            
            # AI selects best action (no exploration)
            # AI选择最佳动作（无探索）
            ai_player.epsilon = 0  # Disable exploration / 禁用探索
            
            # Find best action / 找到最佳动作
            available_actions = state.get_available_actions()
            action_values = []
            
            for action in available_actions:
                next_state = state.take_action(action, -1)  # AI is player -1
                value = ai_player.get_value(next_state)
                action_values.append((action, value))
            
            # Select best action / 选择最佳动作
            action_values.sort(key=lambda x: x[1], reverse=True)
            action = action_values[0][0]
            
            print(f"AI chooses position {action}")
            state = state.take_action(action, current_player)
        
        # Display board / 显示棋盘
        state.display()
        
        # Check if game over / 检查游戏是否结束
        is_terminal, winner = state.is_terminal()
        
        if is_terminal:
            if winner == 1:
                print("Congratulations! You win! / 恭喜！你赢了！")
            elif winner == -1:
                print("AI wins! Better luck next time! / AI赢了！下次好运！")
            else:
                print("It's a draw! / 平局！")
            break
        
        # Switch players / 切换玩家
        current_player *= -1
    
    # Ask to play again / 询问是否再玩
    play_again = input("\nPlay again? (y/n) / 再来一局？(y/n): ")
    if play_again.lower() == 'y':
        human_vs_ai_game(ai_player)