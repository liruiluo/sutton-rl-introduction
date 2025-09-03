"""
================================================================================
第4.5节：蒙特卡洛方法经典例子
Section 4.5: Classic Monte Carlo Examples
================================================================================

展示MC方法在经典问题上的应用。
Demonstrate MC methods on classic problems.

两个重要例子：
Two important examples:
1. 21点（Blackjack）
   - Sutton & Barto教材的经典例子
     Classic example from Sutton & Barto
   - 展示MC在部分可观测环境的应用
     Shows MC in partially observable environment
   - 不需要环境模型
     No environment model needed

2. 赛道问题（Racetrack）
   - 连续动作空间的离散化
     Discretization of continuous action space
   - 展示MC处理延迟奖励
     Shows MC handling delayed rewards
   - 探索的重要性
     Importance of exploration

这些例子展示了MC的实际威力！
These examples show the practical power of MC!
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
import seaborn as sns
from collections import defaultdict
import time

# 导入基础组件
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from ch02_mdp.mdp_framework import State, Action, MDPEnvironment
from ch02_mdp.policies_and_values import (
    Policy, StateValueFunction, ActionValueFunction,
    StochasticPolicy, DeterministicPolicy
)
from ch04_monte_carlo.mc_foundations import Episode, Experience
from ch04_monte_carlo.mc_control import OnPolicyMCControl, EpsilonGreedyPolicy

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第4.5.1节：21点游戏
# Section 4.5.1: Blackjack Game
# ================================================================================

class Blackjack(MDPEnvironment):
    """
    21点游戏环境
    Blackjack Game Environment
    
    经典的纸牌游戏，目标是牌面总和接近21但不超过。
    Classic card game, goal is to get cards summing close to 21 without going over.
    
    规则：
    Rules:
    1. 玩家和庄家各发两张牌
       Player and dealer each get two cards
    2. 玩家看到自己的牌和庄家的一张牌
       Player sees own cards and one dealer card
    3. 玩家可以要牌(hit)或停牌(stick)
       Player can hit or stick
    4. A可以算1或11（可用A）
       Ace can count as 1 or 11 (usable ace)
    5. 超过21点爆牌(bust)，立即输
       Going over 21 is bust, immediate loss
    6. 玩家停牌后，庄家按固定策略要牌（<17要牌）
       After player sticks, dealer plays fixed policy (hit if <17)
    
    状态空间（3维）：
    State space (3D):
    - 玩家当前总和: 12-21
      Player sum: 12-21
    - 庄家明牌: 1-10 (A算1)
      Dealer showing: 1-10 (A counts as 1)
    - 是否有可用A: True/False
      Usable ace: True/False
    
    动作空间：
    Action space:
    - 要牌 (hit): 0
    - 停牌 (stick): 1
    
    奖励：
    Rewards:
    - 赢: +1
      Win: +1
    - 输: -1
      Lose: -1
    - 平: 0
      Draw: 0
    
    为什么适合MC？
    Why suitable for MC?
    1. 回合短，容易采样
       Short episodes, easy to sample
    2. 不需要模型（牌的概率复杂）
       No model needed (card probabilities complex)
    3. 延迟奖励（只在回合结束）
       Delayed reward (only at episode end)
    """
    
    def __init__(self):
        """
        初始化21点环境
        Initialize Blackjack environment
        """
        # 先调用基类构造函数
        # Call base class constructor first
        super().__init__(name="Blackjack")
        
        # 然后创建状态空间（覆盖基类的空列表）
        # Then create state space (overwrite base class empty list)
        self.state_space = []
        
        # 所有可能的状态组合
        # All possible state combinations
        for player_sum in range(12, 22):  # 12-21
            for dealer_showing in range(1, 11):  # 1-10
                for usable_ace in [False, True]:
                    state = State(
                        id=f"p{player_sum}_d{dealer_showing}_{'ace' if usable_ace else 'no'}",
                        features={
                            'player_sum': player_sum,
                            'dealer_showing': dealer_showing,
                            'usable_ace': usable_ace
                        },
                        is_terminal=False
                    )
                    self.state_space.append(state)
        
        # 添加终止状态
        # Add terminal states
        terminal_state = State(
            id="terminal",
            features={},
            is_terminal=True
        )
        self.state_space.append(terminal_state)
        
        # 动作空间
        # Action space
        self.action_space = [
            Action("hit", "Hit - take another card"),
            Action("stick", "Stick - stop taking cards")
        ]
        
        # 当前游戏状态
        # Current game state
        self.player_cards = []
        self.dealer_cards = []
        self.current_state = None
        
        logger.info("初始化21点环境")
    
    def _draw_card(self) -> int:
        """
        抽一张牌
        Draw a card
        
        简化：无限副牌，每张牌概率相等
        Simplified: infinite deck, equal probability
        """
        card = np.random.randint(1, 14)  # 1-13
        return min(card, 10)  # J,Q,K都算10
    
    def _get_sum(self, cards: List[int]) -> Tuple[int, bool]:
        """
        计算手牌总和
        Calculate hand sum
        
        Returns:
            (总和, 是否有可用A)
            (sum, usable ace)
        """
        total = sum(cards)
        num_aces = cards.count(1)
        
        # 尝试将一个A当作11
        # Try to use one ace as 11
        usable_ace = False
        if num_aces > 0 and total + 10 <= 21:
            total += 10
            usable_ace = True
        
        return total, usable_ace
    
    def reset(self) -> State:
        """
        重置游戏
        Reset game
        
        发初始牌
        Deal initial cards
        """
        # 玩家两张牌
        # Player two cards
        self.player_cards = [self._draw_card(), self._draw_card()]
        
        # 庄家两张牌（一张明牌一张暗牌）
        # Dealer two cards (one showing, one hidden)
        self.dealer_cards = [self._draw_card(), self._draw_card()]
        
        # 计算初始状态
        # Calculate initial state
        player_sum, usable_ace = self._get_sum(self.player_cards)
        dealer_showing = self.dealer_cards[0]
        
        # 如果初始就爆牌或21点，需要特殊处理
        # Handle initial bust or blackjack
        if player_sum < 12:
            # 继续抽牌直到>=12
            # Keep drawing until >=12
            while player_sum < 12:
                self.player_cards.append(self._draw_card())
                player_sum, usable_ace = self._get_sum(self.player_cards)
        
        if player_sum > 21:
            # 初始爆牌（罕见）
            # Initial bust (rare)
            self.current_state = self.state_space[-1]  # terminal
        else:
            # 找到对应状态
            # Find corresponding state
            for state in self.state_space:
                if not state.is_terminal:
                    features = state.features
                    if (features.get('player_sum') == player_sum and
                        features.get('dealer_showing') == dealer_showing and
                        features.get('usable_ace') == usable_ace):
                        self.current_state = state
                        break
        
        return self.current_state
    
    def step(self, action: Action) -> Tuple[State, float, bool, Dict]:
        """
        执行动作
        Execute action
        
        Returns:
            (下一状态, 奖励, 是否结束, 信息)
            (next state, reward, done, info)
        """
        if self.current_state.is_terminal:
            return self.current_state, 0, True, {}
        
        player_sum = self.current_state.features['player_sum']
        dealer_showing = self.current_state.features['dealer_showing']
        usable_ace = self.current_state.features['usable_ace']
        
        if action.id == "hit":
            # 玩家要牌
            # Player hits
            self.player_cards.append(self._draw_card())
            player_sum, usable_ace = self._get_sum(self.player_cards)
            
            if player_sum > 21:
                # 爆牌，游戏结束
                # Bust, game over
                self.current_state = self.state_space[-1]  # terminal
                return self.current_state, -1, True, {'result': 'player_bust'}
            
            # 更新状态
            # Update state
            for state in self.state_space:
                if not state.is_terminal:
                    features = state.features
                    if (features.get('player_sum') == player_sum and
                        features.get('dealer_showing') == dealer_showing and
                        features.get('usable_ace') == usable_ace):
                        self.current_state = state
                        break
            
            return self.current_state, 0, False, {}
        
        else:  # stick
            # 玩家停牌，庄家开始
            # Player sticks, dealer plays
            
            # 庄家按固定策略玩
            # Dealer plays fixed policy
            dealer_sum, _ = self._get_sum(self.dealer_cards)
            
            while dealer_sum < 17:
                self.dealer_cards.append(self._draw_card())
                dealer_sum, _ = self._get_sum(self.dealer_cards)
            
            # 判断输赢
            # Determine outcome
            if dealer_sum > 21:
                # 庄家爆牌，玩家赢
                # Dealer bust, player wins
                reward = 1
                result = 'dealer_bust'
            elif dealer_sum > player_sum:
                # 庄家赢
                # Dealer wins
                reward = -1
                result = 'dealer_win'
            elif dealer_sum < player_sum:
                # 玩家赢
                # Player wins
                reward = 1
                result = 'player_win'
            else:
                # 平局
                # Draw
                reward = 0
                result = 'draw'
            
            self.current_state = self.state_space[-1]  # terminal
            return self.current_state, reward, True, {'result': result}


class BlackjackPolicy(Policy):
    """
    21点策略
    Blackjack Policy
    
    可以是阈值策略或学习的策略
    Can be threshold policy or learned policy
    """
    
    def __init__(self, threshold: int = 20, action_space: Optional[List[Action]] = None):
        """
        初始化策略
        Initialize policy
        
        Args:
            threshold: 停牌阈值
                      Stick threshold
            action_space: 动作空间
                        Action space
        """
        super().__init__()
        self.threshold = threshold
        self.Q = None  # 可以设置学习的Q函数
        self.action_space = action_space if action_space else [
            Action("hit", "Hit - take another card"),
            Action("stick", "Stick - stop taking cards")
        ]
    
    def select_action(self, state: State) -> Action:
        """
        选择动作
        Select action
        
        遵循基类Policy接口
        Follow base Policy interface
        """
        if state.is_terminal:
            return self.action_space[0]  # 任意
        
        if self.Q is not None:
            # 使用学习的Q函数
            # Use learned Q function
            q_values = [self.Q.get_value(state, a) for a in self.action_space]
            best_action_idx = np.argmax(q_values)
            return self.action_space[best_action_idx]
        else:
            # 使用简单阈值策略
            # Use simple threshold policy
            player_sum = state.features.get('player_sum', 0)
            
            if player_sum >= self.threshold:
                return self.action_space[1]  # stick
            else:
                return self.action_space[0]  # hit
    
    def get_action_probabilities(self, state: State, 
                                action_space: Optional[List[Action]] = None) -> Dict[Action, float]:
        """
        获取动作概率（确定性策略）
        Get action probabilities (deterministic policy)
        
        兼容新旧接口
        Compatible with both old and new interfaces
        """
        actions = action_space if action_space else self.action_space
        selected_action = self.select_action(state)
        probs = {a: 0.0 for a in actions}
        if selected_action in probs:
            probs[selected_action] = 1.0
        return probs


def visualize_blackjack_policy(Q: ActionValueFunction, usable_ace: bool = False):
    """
    可视化21点策略
    Visualize Blackjack policy
    
    展示在每个状态下的最优动作
    Show optimal action at each state
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 准备数据
    # Prepare data
    player_sums = range(12, 22)
    dealer_showings = range(1, 11)
    
    # 策略矩阵（1=stick, 0=hit）
    # Policy matrix (1=stick, 0=hit)
    policy_matrix = np.zeros((10, 10))
    
    # 价值矩阵
    # Value matrix
    value_matrix = np.zeros((10, 10))
    
    for i, player_sum in enumerate(player_sums):
        for j, dealer_showing in enumerate(dealer_showings):
            # 找到对应状态
            # Find corresponding state
            state_id = f"p{player_sum}_d{dealer_showing}_{'ace' if usable_ace else 'no'}"
            state = None
            
            # 简化：创建临时状态
            # Simplified: create temporary state
            state = State(
                id=state_id,
                features={
                    'player_sum': player_sum,
                    'dealer_showing': dealer_showing,
                    'usable_ace': usable_ace
                },
                is_terminal=False
            )
            
            # 获取动作价值
            # Get action values
            hit_action = Action("hit", "")
            stick_action = Action("stick", "")
            
            q_hit = Q.get_value(state, hit_action)
            q_stick = Q.get_value(state, stick_action)
            
            # 最优动作
            # Optimal action
            if q_stick >= q_hit:
                policy_matrix[i, j] = 1  # stick
                value_matrix[i, j] = q_stick
            else:
                policy_matrix[i, j] = 0  # hit
                value_matrix[i, j] = q_hit
    
    # 图1：策略
    # Plot 1: Policy
    im1 = ax1.imshow(policy_matrix, cmap='coolwarm', aspect='auto')
    ax1.set_xlabel('Dealer Showing')
    ax1.set_ylabel('Player Sum')
    ax1.set_title(f'Optimal Policy ({"Usable" if usable_ace else "No Usable"} Ace)')
    ax1.set_xticks(range(10))
    ax1.set_xticklabels(range(1, 11))
    ax1.set_yticks(range(10))
    ax1.set_yticklabels(range(12, 22))
    
    # 添加网格和标签
    # Add grid and labels
    for i in range(10):
        for j in range(10):
            action = 'S' if policy_matrix[i, j] == 1 else 'H'
            color = 'white' if policy_matrix[i, j] == 0 else 'black'
            ax1.text(j, i, action, ha='center', va='center',
                    color=color, fontweight='bold')
    
    # 图2：价值函数
    # Plot 2: Value function
    im2 = ax2.imshow(value_matrix, cmap='YlOrRd', aspect='auto')
    ax2.set_xlabel('Dealer Showing')
    ax2.set_ylabel('Player Sum')
    ax2.set_title(f'State Values ({"Usable" if usable_ace else "No Usable"} Ace)')
    ax2.set_xticks(range(10))
    ax2.set_xticklabels(range(1, 11))
    ax2.set_yticks(range(10))
    ax2.set_yticklabels(range(12, 22))
    
    # 添加数值
    # Add values
    for i in range(10):
        for j in range(10):
            value = value_matrix[i, j]
            color = 'white' if value < np.mean(value_matrix) else 'black'
            ax2.text(j, i, f'{value:.2f}', ha='center', va='center',
                    color=color, fontsize=8)
    
    # 颜色条
    # Colorbars
    plt.colorbar(im1, ax=ax1, label='Action (0=Hit, 1=Stick)')
    plt.colorbar(im2, ax=ax2, label='State Value')
    
    plt.suptitle('Blackjack Optimal Policy and Values', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


# ================================================================================
# 第4.5.2节：赛道问题
# Section 4.5.2: Racetrack Problem
# ================================================================================

class RaceTrack(MDPEnvironment):
    """
    赛道问题环境
    Racetrack Problem Environment
    
    赛车必须从起点到达终点，控制速度。
    Race car must reach finish line from start, controlling velocity.
    
    状态空间：
    State space:
    - 位置: (x, y)
    - 速度: (vx, vy)
    
    动作空间：
    Action space:
    - 加速度: (ax, ay) ∈ {-1, 0, 1} × {-1, 0, 1}
    
    动力学：
    Dynamics:
    - 新速度: v' = v + a (限制在[0, vmax])
      New velocity: v' = v + a (bounded to [0, vmax])
    - 新位置: p' = p + v'
      New position: p' = p + v'
    
    奖励：
    Rewards:
    - 每步: -1 (鼓励快速到达)
      Per step: -1 (encourage quick finish)
    - 撞墙: 回到起点
      Hit wall: return to start
    
    为什么适合MC？
    Why suitable for MC?
    1. 连续空间的离散化
       Discretization of continuous space
    2. 延迟奖励
       Delayed rewards
    3. 需要探索找到好路径
       Need exploration to find good path
    """
    
    def __init__(self, track_name: str = "simple"):
        """
        初始化赛道
        Initialize racetrack
        
        Args:
            track_name: 赛道名称
                       Track name
        """
        # 先调用基类构造函数
        # Call base class constructor first
        super().__init__(name=f"RaceTrack-{track_name}")
        
        # 创建赛道地图
        # Create track map
        self.track_map = self._create_track(track_name)
        self.height, self.width = self.track_map.shape
        
        # 找出起点和终点
        # Find start and finish lines
        self.start_positions = []
        self.finish_positions = []
        
        for i in range(self.height):
            for j in range(self.width):
                if self.track_map[i, j] == 2:  # 起点
                    self.start_positions.append((i, j))
                elif self.track_map[i, j] == 3:  # 终点
                    self.finish_positions.append((i, j))
        
        # 速度限制
        # Velocity limits
        self.max_velocity = 5
        
        # 创建状态和动作空间（覆盖基类的空列表）
        # Create state and action spaces (overwrite base class empty lists)
        self._create_spaces()
        
        # 当前状态
        # Current state
        self.position = None
        self.velocity = None
        self.current_state = None
        
        logger.info(f"初始化赛道环境: {track_name}")
    
    def _create_track(self, track_name: str) -> np.ndarray:
        """
        创建赛道地图
        Create track map
        
        0: 墙
        1: 赛道
        2: 起点
        3: 终点
        """
        if track_name == "simple":
            # 简单L形赛道
            # Simple L-shaped track
            track = np.zeros((10, 15), dtype=int)
            
            # 赛道
            # Track
            track[7:10, 1:10] = 1  # 横向部分
            track[2:10, 7:10] = 1  # 纵向部分
            
            # 起点（底部）
            # Start line (bottom)
            track[9, 1:3] = 2
            
            # 终点（顶部）
            # Finish line (top)
            track[2, 7:10] = 3
            
            return track
        
        elif track_name == "complex":
            # 复杂赛道
            # Complex track
            track = np.zeros((20, 30), dtype=int)
            
            # S形赛道
            # S-shaped track
            # ... (更复杂的设计)
            
            return track
        
        else:
            raise ValueError(f"Unknown track: {track_name}")
    
    def _create_spaces(self):
        """
        创建状态和动作空间
        Create state and action spaces
        """
        # 简化：只创建部分状态空间
        # Simplified: only create partial state space
        self.state_space = []
        
        # 添加一些代表性状态
        # Add some representative states
        for i in range(self.height):
            for j in range(self.width):
                if self.track_map[i, j] > 0:  # 赛道或起/终点
                    for vx in range(-self.max_velocity, self.max_velocity + 1):
                        for vy in range(-self.max_velocity, self.max_velocity + 1):
                            state = State(
                                id=f"p({i},{j})_v({vx},{vy})",
                                features={
                                    'x': i, 'y': j,
                                    'vx': vx, 'vy': vy
                                },
                                is_terminal=False
                            )
                            self.state_space.append(state)
        
        # 终止状态
        # Terminal state
        terminal_state = State(
            id="finish",
            features={},
            is_terminal=True
        )
        self.state_space.append(terminal_state)
        
        # 动作空间：9个加速度组合
        # Action space: 9 acceleration combinations
        self.action_space = []
        for ax in [-1, 0, 1]:
            for ay in [-1, 0, 1]:
                action = Action(
                    id=f"a({ax},{ay})",
                    name=f"Accelerate ({ax},{ay})"
                )
                action.ax = ax  # 存储加速度
                action.ay = ay
                self.action_space.append(action)
    
    def reset(self) -> State:
        """
        重置到起点
        Reset to start
        """
        # 随机选择起点
        # Random start position
        self.position = self.start_positions[np.random.randint(len(self.start_positions))]
        self.velocity = (0, 0)
        
        # 找到对应状态
        # Find corresponding state
        self.current_state = self._get_state(self.position, self.velocity)
        
        return self.current_state
    
    def _get_state(self, position: Tuple[int, int], 
                  velocity: Tuple[int, int]) -> State:
        """
        获取对应的状态对象
        Get corresponding state object
        """
        # 简化：创建新状态
        # Simplified: create new state
        state = State(
            id=f"p{position}_v{velocity}",
            features={
                'x': position[0], 'y': position[1],
                'vx': velocity[0], 'vy': velocity[1]
            },
            is_terminal=False
        )
        return state
    
    def step(self, action: Action) -> Tuple[State, float, bool, Dict]:
        """
        执行动作
        Execute action
        """
        # 获取加速度
        # Get acceleration
        ax = action.ax if hasattr(action, 'ax') else 0
        ay = action.ay if hasattr(action, 'ay') else 0
        
        # 更新速度
        # Update velocity
        new_vx = np.clip(self.velocity[0] + ax, -self.max_velocity, self.max_velocity)
        new_vy = np.clip(self.velocity[1] + ay, -self.max_velocity, self.max_velocity)
        
        # 至少要有一个方向的速度
        # At least one velocity component
        if new_vx == 0 and new_vy == 0:
            new_vx = 1
        
        # 更新位置（检查碰撞）
        # Update position (check collision)
        old_x, old_y = self.position
        new_x = old_x + new_vx
        new_y = old_y + new_vy
        
        # 检查路径上的碰撞
        # Check collision along path
        hit_wall = False
        
        # 简单线性插值检查
        # Simple linear interpolation check
        steps = max(abs(new_vx), abs(new_vy))
        for i in range(1, steps + 1):
            check_x = old_x + (new_x - old_x) * i // steps
            check_y = old_y + (new_y - old_y) * i // steps
            
            # 边界检查
            # Boundary check
            if (check_x < 0 or check_x >= self.height or
                check_y < 0 or check_y >= self.width or
                self.track_map[check_x, check_y] == 0):
                hit_wall = True
                break
            
            # 检查是否到达终点
            # Check if reached finish
            if self.track_map[check_x, check_y] == 3:
                self.current_state = self.state_space[-1]  # terminal
                return self.current_state, -1, True, {'finished': True}
        
        if hit_wall:
            # 撞墙，回到起点
            # Hit wall, return to start
            self.position = self.start_positions[np.random.randint(len(self.start_positions))]
            self.velocity = (0, 0)
            reward = -10  # 撞墙惩罚
        else:
            # 正常移动
            # Normal move
            self.position = (new_x, new_y)
            self.velocity = (new_vx, new_vy)
            reward = -1  # 时间成本
        
        self.current_state = self._get_state(self.position, self.velocity)
        
        return self.current_state, reward, False, {'hit_wall': hit_wall}
    
    def visualize_track(self, policy: Optional[Policy] = None,
                       trajectory: Optional[List[Tuple[int, int]]] = None):
        """
        可视化赛道和策略
        Visualize track and policy
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 绘制赛道
        # Draw track
        track_display = np.zeros_like(self.track_map, dtype=float)
        track_display[self.track_map == 0] = 0  # 墙 - 黑色
        track_display[self.track_map == 1] = 0.5  # 赛道 - 灰色
        track_display[self.track_map == 2] = 0.3  # 起点 - 深灰
        track_display[self.track_map == 3] = 1.0  # 终点 - 白色
        
        im = ax.imshow(track_display, cmap='gray', aspect='equal')
        
        # 标记起点和终点
        # Mark start and finish
        for x, y in self.start_positions:
            rect = Rectangle((y-0.5, x-0.5), 1, 1,
                           fill=False, edgecolor='green', linewidth=2)
            ax.add_patch(rect)
            ax.text(y, x, 'S', ha='center', va='center',
                   color='green', fontweight='bold')
        
        for x, y in self.finish_positions:
            rect = Rectangle((y-0.5, x-0.5), 1, 1,
                           fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(y, x, 'F', ha='center', va='center',
                   color='red', fontweight='bold')
        
        # 绘制轨迹
        # Draw trajectory
        if trajectory:
            traj_x = [pos[0] for pos in trajectory]
            traj_y = [pos[1] for pos in trajectory]
            ax.plot(traj_y, traj_x, 'b-', linewidth=2, alpha=0.7, label='Trajectory')
            ax.plot(traj_y[0], traj_x[0], 'go', markersize=10, label='Start')
            ax.plot(traj_y[-1], traj_x[-1], 'ro', markersize=10, label='End')
        
        # 绘制策略箭头（如果提供）
        # Draw policy arrows (if provided)
        if policy and hasattr(policy, 'Q'):
            # 在一些关键位置显示策略
            # Show policy at some key positions
            sample_positions = []
            for i in range(self.height):
                for j in range(self.width):
                    if self.track_map[i, j] == 1:  # 赛道上
                        if np.random.random() < 0.2:  # 20%采样率
                            sample_positions.append((i, j))
            
            for x, y in sample_positions:
                # 假设速度为0
                # Assume velocity is 0
                state = self._get_state((x, y), (0, 0))
                
                # 获取最优动作
                # Get optimal action
                q_values = [policy.Q.get_value(state, a) for a in self.action_space]
                best_action_idx = np.argmax(q_values)
                best_action = self.action_space[best_action_idx]
                
                if hasattr(best_action, 'ax'):
                    ax_val = best_action.ax
                    ay_val = best_action.ay
                    
                    # 绘制箭头
                    # Draw arrow
                    if ax_val != 0 or ay_val != 0:
                        ax.arrow(y, x, ay_val*0.3, ax_val*0.3,
                               head_width=0.1, head_length=0.05,
                               fc='blue', ec='blue', alpha=0.5)
        
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(self.height - 0.5, -0.5)
        ax.set_xlabel('Y')
        ax.set_ylabel('X')
        ax.set_title('Race Track')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig


# ================================================================================
# 第4.5.3节：MC例子运行器
# Section 4.5.3: MC Example Runner
# ================================================================================

class MCExampleRunner:
    """
    MC例子运行器
    MC Example Runner
    
    统一运行和分析经典MC例子
    Run and analyze classic MC examples uniformly
    """
    
    @staticmethod
    def run_blackjack_example(n_episodes: int = 100000):
        """
        运行21点例子
        Run Blackjack example
        
        展示MC控制学习最优策略
        Show MC control learning optimal policy
        """
        print("\n" + "="*80)
        print("21点游戏 - MC控制")
        print("Blackjack - MC Control")
        print("="*80)
        
        # 创建环境
        # Create environment
        env = Blackjack()
        
        print(f"\n环境信息:")
        print(f"  状态空间大小: {len(env.state_space)}")
        print(f"  动作空间: {[a.id for a in env.action_space]}")
        
        # 使用MC控制学习
        # Learn using MC control
        print(f"\n开始MC控制学习 ({n_episodes}回合)...")
        print(f"Starting MC control learning ({n_episodes} episodes)...")
        
        controller = OnPolicyMCControl(
            env,
            gamma=1.0,  # 无折扣
            epsilon=0.1,
            epsilon_decay=0.9999,
            epsilon_min=0.01,
            visit_type='first'
        )
        
        # 学习
        # Learn
        start_time = time.time()
        learned_policy = controller.learn(n_episodes, verbose=False)
        elapsed_time = time.time() - start_time
        
        print(f"\n学习完成:")
        print(f"  时间: {elapsed_time:.1f}秒")
        print(f"  最终ε: {controller.policy.epsilon:.4f}")
        print(f"  访问的(s,a)对: {len(controller.sa_visits)}")
        
        # 分析学习的策略
        # Analyze learned policy
        print(f"\n策略分析:")
        
        # 测试学习的策略
        # Test learned policy
        test_episodes = 10000
        wins = 0
        losses = 0
        draws = 0
        
        for _ in range(test_episodes):
            state = env.reset()
            done = False
            
            while not done:
                action = learned_policy.select_action(state)
                state, reward, done, info = env.step(action)
            
            if reward > 0:
                wins += 1
            elif reward < 0:
                losses += 1
            else:
                draws += 1
        
        print(f"\n测试结果 ({test_episodes}回合):")
        print(f"  胜率: {wins/test_episodes:.1%}")
        print(f"  败率: {losses/test_episodes:.1%}")
        print(f"  平局率: {draws/test_episodes:.1%}")
        
        # 与简单策略比较
        # Compare with simple policy
        print(f"\n与阈值策略(threshold=20)比较:")
        simple_policy = BlackjackPolicy(threshold=20)
        
        simple_wins = 0
        for _ in range(test_episodes):
            state = env.reset()
            done = False
            
            while not done:
                action = simple_policy.select_action(state)
                state, reward, done, info = env.step(action)
            
            if reward > 0:
                simple_wins += 1
        
        print(f"  简单策略胜率: {simple_wins/test_episodes:.1%}")
        print(f"  MC学习提升: {(wins-simple_wins)/test_episodes:.1%}")
        
        # 可视化策略
        # Visualize policy
        print(f"\n生成策略可视化...")
        
        # 为可用A和非可用A分别可视化
        # Visualize for usable and non-usable ace
        fig1 = visualize_blackjack_policy(controller.Q, usable_ace=False)
        fig2 = visualize_blackjack_policy(controller.Q, usable_ace=True)
        
        return controller, [fig1, fig2]
    
    @staticmethod
    def run_racetrack_example(n_episodes: int = 5000):
        """
        运行赛道例子
        Run Racetrack example
        
        展示MC在连续空间问题的应用
        Show MC application in continuous space problem
        """
        print("\n" + "="*80)
        print("赛道问题 - MC控制")
        print("Racetrack - MC Control")
        print("="*80)
        
        # 创建环境
        # Create environment
        env = RaceTrack(track_name="simple")
        
        print(f"\n环境信息:")
        print(f"  赛道大小: {env.height}×{env.width}")
        print(f"  起点数: {len(env.start_positions)}")
        print(f"  终点数: {len(env.finish_positions)}")
        print(f"  最大速度: {env.max_velocity}")
        print(f"  动作数: {len(env.action_space)}")
        
        # MC控制学习
        # MC control learning
        print(f"\n开始MC控制学习 ({n_episodes}回合)...")
        
        controller = OnPolicyMCControl(
            env,
            gamma=1.0,
            epsilon=0.2,
            epsilon_decay=0.995,
            epsilon_min=0.05,
            visit_type='first'
        )
        
        # 记录一些成功的轨迹
        # Record some successful trajectories
        successful_trajectories = []
        episode_lengths = []
        
        for episode_num in range(n_episodes):
            trajectory = []
            state = env.reset()
            trajectory.append(env.position)
            
            done = False
            steps = 0
            max_steps = 1000
            
            while not done and steps < max_steps:
                action = controller.policy.select_action(state)
                next_state, reward, done, info = env.step(action)
                
                trajectory.append(env.position)
                
                # 创建经验并更新
                # Create experience and update
                exp = Experience(state, action, reward, next_state, done)
                episode = Episode()
                episode.add_experience(exp)
                
                # 简化：只更新这一步
                # Simplified: only update this step
                if done and info.get('finished'):
                    controller.update_Q(episode)
                    successful_trajectories.append(trajectory)
                    episode_lengths.append(steps)
                
                state = next_state
                steps += 1
            
            # 衰减epsilon
            # Decay epsilon
            controller.policy.decay_epsilon()
            
            if (episode_num + 1) % 1000 == 0:
                if successful_trajectories:
                    avg_length = np.mean(episode_lengths[-100:]) if len(episode_lengths) > 100 else np.mean(episode_lengths)
                    print(f"  Episode {episode_num + 1}: "
                          f"成功率={len(successful_trajectories)/(episode_num+1):.1%}, "
                          f"平均长度={avg_length:.1f}")
        
        print(f"\n学习完成:")
        print(f"  总成功次数: {len(successful_trajectories)}")
        print(f"  成功率: {len(successful_trajectories)/n_episodes:.1%}")
        
        if successful_trajectories:
            print(f"  最短路径: {min(episode_lengths)}步")
            print(f"  平均路径: {np.mean(episode_lengths):.1f}步")
            
            # 可视化最佳轨迹
            # Visualize best trajectory
            best_idx = np.argmin(episode_lengths)
            best_trajectory = successful_trajectories[best_idx]
            
            print(f"\n生成赛道可视化...")
            fig = env.visualize_track(
                policy=controller.policy,
                trajectory=best_trajectory
            )
            
            return controller, [fig]
        else:
            print("  警告：没有成功到达终点的轨迹")
            return controller, []
    
    @staticmethod
    def analyze_exploration_importance():
        """
        分析探索的重要性
        Analyze importance of exploration
        
        比较不同ε值的效果
        Compare effects of different ε values
        """
        print("\n" + "="*80)
        print("探索的重要性分析")
        print("Analysis of Exploration Importance")
        print("="*80)
        
        # 在21点上测试不同的ε
        # Test different ε on Blackjack
        env = Blackjack()
        epsilons = [0.01, 0.05, 0.1, 0.2, 0.3]
        n_episodes = 10000
        
        results = {}
        
        for eps in epsilons:
            print(f"\n测试 ε={eps}...")
            
            controller = OnPolicyMCControl(
                env,
                gamma=1.0,
                epsilon=eps,
                epsilon_decay=1.0,  # 不衰减
                epsilon_min=eps,
                visit_type='first'
            )
            
            # 学习
            # Learn
            controller.learn(n_episodes, verbose=False)
            
            # 测试
            # Test
            test_episodes = 1000
            wins = 0
            
            for _ in range(test_episodes):
                state = env.reset()
                done = False
                
                # 测试时用贪婪策略
                # Use greedy policy for testing
                controller.policy.epsilon = 0
                
                while not done:
                    action = controller.policy.select_action(state)
                    state, reward, done, info = env.step(action)
                
                if reward > 0:
                    wins += 1
            
            # 恢复epsilon
            # Restore epsilon
            controller.policy.epsilon = eps
            
            win_rate = wins / test_episodes
            coverage = len(controller.sa_visits)
            
            results[eps] = {
                'win_rate': win_rate,
                'coverage': coverage,
                'visits': controller.sa_visits
            }
            
            print(f"  胜率: {win_rate:.1%}")
            print(f"  覆盖(s,a)对: {coverage}")
        
        # 可视化比较
        # Visualize comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 胜率vs ε
        # Win rate vs ε
        eps_list = list(results.keys())
        win_rates = [results[e]['win_rate'] for e in eps_list]
        
        ax1.plot(eps_list, win_rates, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Epsilon (ε)')
        ax1.set_ylabel('Win Rate')
        ax1.set_title('Performance vs Exploration Rate')
        ax1.grid(True, alpha=0.3)
        
        # 覆盖率vs ε
        # Coverage vs ε
        coverages = [results[e]['coverage'] for e in eps_list]
        
        ax2.plot(eps_list, coverages, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Epsilon (ε)')
        ax2.set_ylabel('State-Action Pairs Covered')
        ax2.set_title('Exploration Coverage vs ε')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Exploration-Exploitation Trade-off', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        print("\n" + "="*60)
        print("关键观察:")
        print("Key Observations:")
        print("="*60)
        print("""
        1. 低ε (0.01-0.05):
           - 覆盖率低，可能错过好策略
             Low coverage, may miss good policies
           - 收敛快但可能次优
             Fast convergence but possibly suboptimal
        
        2. 中ε (0.1-0.2):
           - 平衡的探索和利用
             Balanced exploration and exploitation
           - 通常获得最佳性能
             Usually achieves best performance
        
        3. 高ε (0.3+):
           - 高覆盖率但收敛慢
             High coverage but slow convergence
           - 过度探索影响性能
             Excessive exploration hurts performance
        
        4. 最优策略:
           - 开始高ε，逐渐衰减
             Start with high ε, gradually decay
           - 早期探索，后期利用
             Early exploration, later exploitation
        """)
        
        return fig


# ================================================================================
# 第4.5.4节：综合演示
# Section 4.5.4: Comprehensive Demo
# ================================================================================

def demonstrate_mc_examples():
    """
    综合演示MC经典例子
    Comprehensive demonstration of MC classic examples
    """
    print("\n" + "="*80)
    print("蒙特卡洛方法经典例子演示")
    print("Monte Carlo Classic Examples Demonstration")
    print("="*80)
    
    # 1. 21点例子
    # 1. Blackjack example
    print("\n" + "="*60)
    print("例子1：21点游戏")
    print("Example 1: Blackjack")
    print("="*60)
    
    blackjack_controller, blackjack_figs = MCExampleRunner.run_blackjack_example(
        n_episodes=50000  # 减少用于演示
    )
    
    # 2. 赛道例子
    # 2. Racetrack example
    print("\n" + "="*60)
    print("例子2：赛道问题")
    print("Example 2: Racetrack")
    print("="*60)
    
    racetrack_controller, racetrack_figs = MCExampleRunner.run_racetrack_example(
        n_episodes=2000  # 减少用于演示
    )
    
    # 3. 探索重要性分析
    # 3. Exploration importance analysis
    print("\n" + "="*60)
    print("分析：探索的重要性")
    print("Analysis: Importance of Exploration")
    print("="*60)
    
    exploration_fig = MCExampleRunner.analyze_exploration_importance()
    
    # 总结
    # Summary
    print("\n" + "="*80)
    print("MC方法总结")
    print("MC Methods Summary")
    print("="*80)
    print("""
    📚 关键学习点 Key Learning Points:
    =====================================
    
    1. MC的适用场景:
       When to use MC:
       - 回合式任务
         Episodic tasks
       - 不需要/没有模型
         No model needed/available
       - 可以模拟/采样
         Can simulate/sample
    
    2. MC的优势:
       MC Advantages:
       - 无模型学习
         Model-free learning
       - 处理随机性
         Handles stochasticity
       - 收敛到真实值
         Converges to true values
    
    3. MC的挑战:
       MC Challenges:
       - 高方差
         High variance
       - 需要完整回合
         Needs complete episodes
       - 探索-利用权衡
         Exploration-exploitation trade-off
    
    4. 21点例子展示:
       Blackjack shows:
       - 部分可观测也能学习
         Can learn with partial observability
       - 简单环境的最优策略
         Optimal policy for simple environment
       - MC控制的有效性
         Effectiveness of MC control
    
    5. 赛道例子展示:
       Racetrack shows:
       - 连续空间的处理
         Handling continuous space
       - 延迟奖励的学习
         Learning with delayed rewards
       - 探索的必要性
         Necessity of exploration
    
    6. 向TD方法的过渡:
       Transition to TD:
       - MC的高方差激发了TD
         MC's high variance motivated TD
       - TD = MC + DP的优点
         TD = MC + DP advantages
       - 下一章：TD学习
         Next chapter: TD learning
    """)
    print("="*80)
    
    # 显示所有图
    # Show all figures
    plt.show()


# ================================================================================
# 主函数
# Main Function
# ================================================================================

def main():
    """
    运行MC例子演示
    Run MC examples demo
    """
    demonstrate_mc_examples()


if __name__ == "__main__":
    main()