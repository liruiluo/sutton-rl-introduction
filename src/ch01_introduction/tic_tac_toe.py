"""
Tic-Tac-Toe Example from Chapter 1
第1章的井字游戏示例

A complete RL example showing value function learning
展示价值函数学习的完整RL示例
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import pickle


class TicTacToeGame:
    """
    Tic-Tac-Toe game environment
    井字游戏环境
    """
    
    def __init__(self):
        """
        Initialize the game
        初始化游戏
        """
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # 1 for X, -1 for O
        self.winner = None
        self.game_over = False
        
    def reset(self) -> np.ndarray:
        """
        Reset the game to initial state
        重置游戏到初始状态
        """
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.winner = None
        self.game_over = False
        return self.board.copy()
        
    def get_available_actions(self) -> List[Tuple[int, int]]:
        """
        Get all available positions
        获取所有可用位置
        """
        actions = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 0:
                    actions.append((i, j))
        return actions
        
    def make_move(self, position: Tuple[int, int]) -> Tuple[np.ndarray, float, bool]:
        """
        Make a move at the given position
        在给定位置下棋
        
        Returns:
            board: New board state
            reward: Reward for the move
            done: Whether game is over
        """
        i, j = position
        
        if self.board[i, j] != 0:
            raise ValueError(f"Position {position} is already occupied")
            
        self.board[i, j] = self.current_player
        
        # Check for winner
        # 检查获胜者
        if self._check_winner():
            self.winner = self.current_player
            self.game_over = True
            reward = 1.0 if self.current_player == 1 else -1.0
        elif len(self.get_available_actions()) == 0:
            # Draw
            # 平局
            self.game_over = True
            reward = 0.5
        else:
            reward = 0.0
            
        # Switch player
        # 切换玩家
        self.current_player = -self.current_player
        
        return self.board.copy(), reward, self.game_over
        
    def _check_winner(self) -> bool:
        """
        Check if current player has won
        检查当前玩家是否获胜
        """
        player = self.current_player
        
        # Check rows
        # 检查行
        for i in range(3):
            if all(self.board[i, j] == player for j in range(3)):
                return True
                
        # Check columns
        # 检查列
        for j in range(3):
            if all(self.board[i, j] == player for i in range(3)):
                return True
                
        # Check diagonals
        # 检查对角线
        if all(self.board[i, i] == player for i in range(3)):
            return True
        if all(self.board[i, 2-i] == player for i in range(3)):
            return True
            
        return False
        
    def get_state_hash(self) -> str:
        """
        Get unique hash for current board state
        获取当前棋盘状态的唯一哈希
        """
        return str(self.board.flatten())
        
    def render(self):
        """
        Display the board
        显示棋盘
        """
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        print("\n  0 1 2")
        for i in range(3):
            print(f"{i} ", end="")
            for j in range(3):
                print(symbols[self.board[i, j]], end="")
                if j < 2:
                    print("|", end="")
            print()
            if i < 2:
                print("  -----")


class TicTacToePlayer:
    """
    Base class for Tic-Tac-Toe players
    井字游戏玩家基类
    """
    
    def __init__(self, name: str, symbol: int):
        """
        Initialize player
        初始化玩家
        
        Args:
            name: Player name
            symbol: 1 for X, -1 for O
        """
        self.name = name
        self.symbol = symbol
        
    def choose_action(self, game: TicTacToeGame) -> Tuple[int, int]:
        """
        Choose an action given the game state
        给定游戏状态选择动作
        """
        raise NotImplementedError


class ValueFunctionPlayer(TicTacToePlayer):
    """
    Player that learns using value functions
    使用价值函数学习的玩家
    """
    
    def __init__(self, name: str, symbol: int, 
                 epsilon: float = 0.1, alpha: float = 0.3):
        """
        Initialize value function player
        初始化价值函数玩家
        
        Args:
            epsilon: Exploration rate
            alpha: Learning rate
        """
        super().__init__(name, symbol)
        self.epsilon = epsilon
        self.alpha = alpha
        self.values = {}  # State -> value mapping
        self.states_history = []  # States encountered in current game
        
    def get_value(self, state_hash: str) -> float:
        """
        Get value of a state
        获取状态的价值
        """
        if state_hash not in self.values:
            # Initialize with optimistic value
            # 用乐观值初始化
            self.values[state_hash] = 0.5
        return self.values[state_hash]
        
    def choose_action(self, game: TicTacToeGame) -> Tuple[int, int]:
        """
        Choose action using epsilon-greedy based on state values
        使用基于状态价值的ε-贪婪选择动作
        """
        available_actions = game.get_available_actions()
        
        if np.random.random() < self.epsilon:
            # Explore: random action
            # 探索：随机动作
            return available_actions[np.random.choice(len(available_actions))]
        else:
            # Exploit: choose action leading to best state
            # 利用：选择导向最佳状态的动作
            best_value = -float('inf')
            best_action = None
            
            for action in available_actions:
                # Simulate making the move
                # 模拟下棋
                next_board = game.board.copy()
                next_board[action[0], action[1]] = game.current_player
                next_hash = str(next_board.flatten())
                
                # Get value of resulting state
                # 获取结果状态的价值
                value = self.get_value(next_hash)
                
                # Adjust value based on perspective
                # 根据视角调整价值
                if game.current_player != self.symbol:
                    value = -value
                    
                if value > best_value:
                    best_value = value
                    best_action = action
                    
            return best_action
            
    def add_state(self, state_hash: str):
        """
        Add state to history
        将状态添加到历史
        """
        self.states_history.append(state_hash)
        
    def update_values(self, reward: float):
        """
        Update values using temporal difference learning
        使用时序差分学习更新价值
        """
        # Work backwards through states
        # 从后向前遍历状态
        for i in reversed(range(len(self.states_history))):
            state_hash = self.states_history[i]
            
            if i == len(self.states_history) - 1:
                # Last state gets the final reward
                # 最后状态获得最终奖励
                target = reward
            else:
                # Earlier states learn from next state value
                # 早期状态从下一状态价值学习
                next_hash = self.states_history[i + 1]
                target = self.get_value(next_hash)
                
            # TD update
            # TD更新
            old_value = self.get_value(state_hash)
            self.values[state_hash] = old_value + self.alpha * (target - old_value)
            
    def reset(self):
        """
        Reset for new game
        为新游戏重置
        """
        self.states_history = []
        
    def save_values(self, filename: str):
        """
        Save learned values to file
        保存学习到的价值到文件
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.values, f)
            
    def load_values(self, filename: str):
        """
        Load values from file
        从文件加载价值
        """
        with open(filename, 'rb') as f:
            self.values = pickle.load(f)


class RandomPlayer(TicTacToePlayer):
    """
    Player that chooses random actions
    选择随机动作的玩家
    """
    
    def choose_action(self, game: TicTacToeGame) -> Tuple[int, int]:
        """
        Choose random available action
        选择随机可用动作
        """
        available_actions = game.get_available_actions()
        return available_actions[np.random.choice(len(available_actions))]


def train_value_function_player(n_games: int = 10000) -> ValueFunctionPlayer:
    """
    Train a value function player through self-play
    通过自我对弈训练价值函数玩家
    
    Args:
        n_games: Number of training games
    
    Returns:
        Trained player
    """
    # Create two learning players
    # 创建两个学习玩家
    player1 = ValueFunctionPlayer("Player 1", 1, epsilon=0.1, alpha=0.3)
    player2 = ValueFunctionPlayer("Player 2", -1, epsilon=0.1, alpha=0.3)
    
    wins = {1: 0, -1: 0, 0: 0}  # X wins, O wins, draws
    
    print(f"Training through {n_games} games of self-play...")
    
    for game_num in range(n_games):
        game = TicTacToeGame()
        game.reset()
        
        player1.reset()
        player2.reset()
        
        current_player = player1
        other_player = player2
        
        while not game.game_over:
            # Record state
            # 记录状态
            state_hash = game.get_state_hash()
            current_player.add_state(state_hash)
            
            # Choose and make move
            # 选择并下棋
            action = current_player.choose_action(game)
            _, reward, done = game.make_move(action)
            
            # Switch players
            # 切换玩家
            current_player, other_player = other_player, current_player
            
        # Game over - update values
        # 游戏结束 - 更新价值
        if game.winner == 1:
            player1.update_values(1.0)
            player2.update_values(0.0)
            wins[1] += 1
        elif game.winner == -1:
            player1.update_values(0.0)
            player2.update_values(1.0)
            wins[-1] += 1
        else:
            player1.update_values(0.5)
            player2.update_values(0.5)
            wins[0] += 1
            
        # Print progress
        # 打印进度
        if (game_num + 1) % 1000 == 0:
            print(f"Game {game_num + 1}: X wins={wins[1]}, "
                  f"O wins={wins[-1]}, Draws={wins[0]}")
            
    return player1


def demonstrate_tic_tac_toe():
    """
    Demonstrate Tic-Tac-Toe with value function learning
    演示使用价值函数学习的井字游戏
    """
    print("\n" + "="*80)
    print("Tic-Tac-Toe with Value Function Learning")
    print("使用价值函数学习的井字游戏")
    print("="*80)
    
    # Train a player
    # 训练玩家
    print("\n1. Training Value Function Player 训练价值函数玩家")
    print("-" * 40)
    
    trained_player = train_value_function_player(n_games=5000)
    
    # Test against random player
    # 对抗随机玩家测试
    print("\n2. Testing Against Random Player 对抗随机玩家测试")
    print("-" * 40)
    
    trained_player.epsilon = 0.0  # No exploration during testing
    random_player = RandomPlayer("Random", -1)
    
    n_test_games = 100
    wins = {1: 0, -1: 0, 0: 0}
    
    for _ in range(n_test_games):
        game = TicTacToeGame()
        game.reset()
        
        while not game.game_over:
            if game.current_player == 1:
                action = trained_player.choose_action(game)
            else:
                action = random_player.choose_action(game)
                
            game.make_move(action)
            
        if game.winner == 1:
            wins[1] += 1
        elif game.winner == -1:
            wins[-1] += 1
        else:
            wins[0] += 1
            
    print(f"\nResults over {n_test_games} games:")
    print(f"Trained player wins: {wins[1]} ({100*wins[1]/n_test_games:.1f}%)")
    print(f"Random player wins: {wins[-1]} ({100*wins[-1]/n_test_games:.1f}%)")
    print(f"Draws: {wins[0]} ({100*wins[0]/n_test_games:.1f}%)")
    
    # Demonstrate a single game
    # 演示单个游戏
    print("\n3. Sample Game 示例游戏")
    print("-" * 40)
    
    game = TicTacToeGame()
    game.reset()
    
    print("\nStarting new game...")
    print("Trained Player (X) vs Random Player (O)")
    
    move_count = 0
    
    while not game.game_over:
        move_count += 1
        print(f"\n--- Move {move_count} ---")
        
        if game.current_player == 1:
            print("Trained Player's turn (X)")
            action = trained_player.choose_action(game)
        else:
            print("Random Player's turn (O)")
            action = random_player.choose_action(game)
            
        print(f"Chosen position: {action}")
        game.make_move(action)
        game.render()
        
    # Show result
    # 显示结果
    print("\n" + "="*40)
    if game.winner == 1:
        print("Trained Player (X) wins!")
    elif game.winner == -1:
        print("Random Player (O) wins!")
    else:
        print("It's a draw!")
        
    # Show some learned values
    # 显示一些学习到的价值
    print("\n4. Sample Learned Values 示例学习价值")
    print("-" * 40)
    
    # Empty board
    # 空棋盘
    empty_board = np.zeros((3, 3))
    empty_hash = str(empty_board.flatten())
    print(f"Empty board value: {trained_player.get_value(empty_hash):.3f}")
    
    # Center occupied by X
    # X占据中心
    center_x = np.zeros((3, 3))
    center_x[1, 1] = 1
    center_hash = str(center_x.flatten())
    print(f"X in center value: {trained_player.get_value(center_hash):.3f}")
    
    # Near win for X
    # X接近获胜
    near_win = np.array([[1, 1, 0], [0, -1, 0], [0, 0, 0]])
    near_win_hash = str(near_win.flatten())
    print(f"Near win for X value: {trained_player.get_value(near_win_hash):.3f}")
    
    print("\n" + "="*80)
    print("Tic-Tac-Toe Demonstration Complete!")
    print("井字游戏演示完成！")
    print("="*80)


if __name__ == "__main__":
    # Run the demonstration
    # 运行演示
    demonstrate_tic_tac_toe()