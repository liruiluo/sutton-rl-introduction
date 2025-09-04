"""
================================================================================
第3.5节：动态规划经典例子
Section 3.5: Classic Dynamic Programming Examples
================================================================================

这些经典例子展示了DP在不同类型问题上的应用。
These classic examples show DP applications on different types of problems.

三个重要例子：
Three important examples:
1. 网格世界 - 路径规划问题
   Grid World - Path planning problem
2. 赌徒问题 - 风险决策问题
   Gambler's Problem - Risk decision problem  
3. 杰克汽车租赁 - 资源分配问题
   Jack's Car Rental - Resource allocation problem

每个例子都展示了DP的不同方面：
Each example demonstrates different aspects of DP:
- 状态空间的设计
  State space design
- 动作空间的定义
  Action space definition
- 奖励函数的设置
  Reward function setup
- 最优策略的特征
  Characteristics of optimal policy
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
import seaborn as sns
from scipy.stats import poisson
from collections import defaultdict
import time

# 导入基础组件
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.ch03_finite_mdp.mdp_framework import (
    State, Action, MDPEnvironment,
    TransitionProbability, RewardFunction
)
from src.ch03_finite_mdp.policies_and_values import (
    Policy, StateValueFunction, ActionValueFunction,
    DeterministicPolicy
)
from src.ch03_finite_mdp.gridworld import GridWorld
from policy_iteration import PolicyIteration
from value_iteration import ValueIteration

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第3.5.1节：网格世界DP解决方案
# Section 3.5.1: Grid World DP Solution
# ================================================================================

class GridWorldDP:
    """
    网格世界的DP解决方案
    DP Solution for Grid World
    
    这是最直观的例子，展示了DP如何找到最短路径
    This is the most intuitive example, showing how DP finds shortest path
    
    问题特征：
    Problem characteristics:
    - 确定性转移（除了边界）
      Deterministic transitions (except boundaries)
    - 稀疏奖励（只在目标处）
      Sparse rewards (only at goal)
    - 明确的最优策略（最短路径）
      Clear optimal policy (shortest path)
    
    教学价值：
    Teaching value:
    - 可视化价值函数的传播
      Visualize value function propagation
    - 理解策略改进过程
      Understand policy improvement process
    - 观察不同γ的影响
      Observe effect of different γ
    """
    
    def __init__(self, rows: int = 5, cols: int = 5,
                 goal_reward: float = 1.0,
                 step_penalty: float = -0.01,
                 obstacles: Optional[Set[Tuple[int, int]]] = None):
        """
        初始化网格世界DP
        Initialize Grid World DP
        
        Args:
            rows, cols: 网格大小
                       Grid size
            goal_reward: 到达目标的奖励
                        Reward for reaching goal
            step_penalty: 每步的惩罚（鼓励找最短路）
                        Step penalty (encourages shortest path)
            obstacles: 障碍物位置
                      Obstacle positions
        
        设计考虑：
        Design considerations:
        - 小的步惩罚让智能体寻找最短路
          Small step penalty makes agent find shortest path
        - 障碍物增加问题复杂性
          Obstacles add problem complexity
        """
        # 创建基础网格世界环境
        self.env = GridWorld(
            rows=rows,
            cols=cols,
            start_pos=(0, 0),
            goal_pos=(rows-1, cols-1),
            obstacles=obstacles if obstacles else set()
        )
        
        self.rows = rows
        self.cols = cols
        
        logger.info(f"创建{rows}x{cols}网格世界DP")
    
    def solve_with_policy_iteration(self, gamma: float = 0.9,
                                   verbose: bool = True) -> Tuple[Policy, StateValueFunction]:
        """
        用策略迭代解决网格世界
        Solve Grid World with Policy Iteration
        
        展示策略迭代如何快速收敛
        Shows how policy iteration converges quickly
        """
        if verbose:
            print("\n使用策略迭代解决网格世界")
            print("Solving Grid World with Policy Iteration")
        
        pi = PolicyIteration(self.env, gamma=gamma)
        policy, V = pi.solve(verbose=verbose)
        
        return policy, V
    
    def solve_with_value_iteration(self, gamma: float = 0.9,
                                  verbose: bool = True) -> Tuple[Policy, StateValueFunction]:
        """
        用价值迭代解决网格世界
        Solve Grid World with Value Iteration
        
        展示价值如何从目标向外传播
        Shows how value propagates from goal outward
        """
        if verbose:
            print("\n使用价值迭代解决网格世界")
            print("Solving Grid World with Value Iteration")
        
        vi = ValueIteration(self.env, gamma=gamma)
        policy, V = vi.solve(verbose=verbose)
        
        return policy, V
    
    def visualize_solution(self, policy: Policy, V: StateValueFunction,
                          title: str = "Grid World Solution"):
        """
        可视化网格世界解决方案
        Visualize Grid World Solution
        
        展示最优策略和价值函数
        Show optimal policy and value function
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # ========== 左图：策略可视化 ==========
        ax1.set_title("Optimal Policy / 最优策略")
        ax1.set_xlim(-0.5, self.cols - 0.5)
        ax1.set_ylim(-0.5, self.rows - 0.5)
        ax1.set_aspect('equal')
        ax1.invert_yaxis()
        
        # 绘制网格
        for i in range(self.rows + 1):
            ax1.axhline(y=i - 0.5, color='gray', linewidth=0.5)
        for j in range(self.cols + 1):
            ax1.axvline(x=j - 0.5, color='gray', linewidth=0.5)
        
        # 动作箭头映射
        action_arrows = {
            'up': (0, -0.4),
            'down': (0, 0.4),
            'left': (-0.4, 0),
            'right': (0.4, 0)
        }
        
        # 绘制每个格子
        for i in range(self.rows):
            for j in range(self.cols):
                pos = (i, j)
                
                # 检查障碍物
                if pos in self.env.obstacles:
                    rect = Rectangle((j-0.5, i-0.5), 1, 1,
                                   facecolor='black', alpha=0.8)
                    ax1.add_patch(rect)
                    continue
                
                # 获取状态
                if pos in self.env.pos_to_state:
                    state = self.env.pos_to_state[pos]
                    
                    # 获取价值并着色
                    value = V.get_value(state)
                    
                    # 归一化价值用于着色
                    all_values = [V.get_value(s) for s in self.env.state_space]
                    norm_value = (value - min(all_values)) / (max(all_values) - min(all_values) + 1e-10)
                    color = plt.cm.YlOrRd(norm_value)
                    
                    rect = Rectangle((j-0.5, i-0.5), 1, 1,
                                   facecolor=color, alpha=0.3)
                    ax1.add_patch(rect)
                    
                    # 绘制策略箭头
                    if not state.is_terminal and isinstance(policy, DeterministicPolicy):
                        if state in policy.policy_map:
                            action = policy.policy_map[state]
                            if action.id in action_arrows:
                                dx, dy = action_arrows[action.id]
                                ax1.arrow(j, i, dx, dy, head_width=0.15,
                                        head_length=0.1, fc='blue', ec='blue')
                
                # 标记特殊位置
                if pos == self.env.start_pos:
                    ax1.text(j, i, 'S', ha='center', va='center',
                           fontweight='bold', color='green', fontsize=14)
                elif pos == self.env.goal_pos:
                    ax1.text(j, i, 'G', ha='center', va='center',
                           fontweight='bold', color='red', fontsize=14)
        
        ax1.set_xticks(range(self.cols))
        ax1.set_yticks(range(self.rows))
        ax1.grid(True, alpha=0.3)
        
        # ========== 右图：价值函数热力图 ==========
        ax2.set_title("Value Function / 价值函数")
        
        # 创建价值矩阵
        value_matrix = np.zeros((self.rows, self.cols))
        for i in range(self.rows):
            for j in range(self.cols):
                pos = (i, j)
                if pos in self.env.pos_to_state:
                    state = self.env.pos_to_state[pos]
                    value_matrix[i, j] = V.get_value(state)
                elif pos in self.env.obstacles:
                    value_matrix[i, j] = np.nan
        
        # 绘制热力图
        im = ax2.imshow(value_matrix, cmap='YlOrRd', aspect='equal')
        
        # 添加数值标签
        for i in range(self.rows):
            for j in range(self.cols):
                pos = (i, j)
                if pos not in self.env.obstacles and pos in self.env.pos_to_state:
                    value = value_matrix[i, j]
                    text_color = 'white' if value > np.nanmean(value_matrix) else 'black'
                    ax2.text(j, i, f'{value:.2f}', ha='center', va='center',
                           color=text_color, fontweight='bold')
        
        # 标记起点和终点
        ax2.plot(self.env.start_pos[1], self.env.start_pos[0], 'go', markersize=15, label='Start')
        ax2.plot(self.env.goal_pos[1], self.env.goal_pos[0], 'r*', markersize=20, label='Goal')
        
        ax2.set_xlim(-0.5, self.cols - 0.5)
        ax2.set_ylim(self.rows - 0.5, -0.5)
        ax2.set_xticks(range(self.cols))
        ax2.set_yticks(range(self.rows))
        ax2.legend()
        
        plt.colorbar(im, ax=ax2, label='State Value')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def demonstrate_gamma_effect():
        """
        演示折扣因子γ的影响
        Demonstrate Effect of Discount Factor γ
        
        展示不同γ如何影响最优策略
        Show how different γ affects optimal policy
        """
        print("\n" + "="*60)
        print("折扣因子γ的影响")
        print("Effect of Discount Factor γ")
        print("="*60)
        
        # 创建有陷阱的网格世界
        grid = GridWorldDP(
            rows=5, cols=5,
            goal_reward=10.0,
            step_penalty=-1.0,
            obstacles={(1, 1), (1, 2), (1, 3), (3, 1), (3, 2), (3, 3)}
        )
        
        gammas = [0.5, 0.9, 0.99]
        policies = []
        values = []
        
        for gamma in gammas:
            print(f"\n解决 γ={gamma}...")
            vi = ValueIteration(grid.env, gamma=gamma)
            policy, V = vi.solve(verbose=False)
            policies.append(policy)
            values.append(V)
            print(f"  收敛迭代: {vi.total_iterations}")
        
        # 可视化比较
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, (gamma, policy, V) in enumerate(zip(gammas, policies, values)):
            ax = axes[idx]
            ax.set_title(f'γ = {gamma}')
            
            # 绘制价值热力图
            value_matrix = np.zeros((grid.rows, grid.cols))
            for i in range(grid.rows):
                for j in range(grid.cols):
                    pos = (i, j)
                    if pos in grid.env.pos_to_state:
                        state = grid.env.pos_to_state[pos]
                        value_matrix[i, j] = V.get_value(state)
                    elif pos in grid.env.obstacles:
                        value_matrix[i, j] = np.nan
            
            im = ax.imshow(value_matrix, cmap='YlOrRd', aspect='equal')
            
            # 添加策略箭头
            action_arrows = {
                'up': (0, -0.3),
                'down': (0, 0.3),
                'left': (-0.3, 0),
                'right': (0.3, 0)
            }
            
            for i in range(grid.rows):
                for j in range(grid.cols):
                    pos = (i, j)
                    if pos not in grid.env.obstacles and pos in grid.env.pos_to_state:
                        state = grid.env.pos_to_state[pos]
                        if not state.is_terminal and isinstance(policy, DeterministicPolicy):
                            if state in policy.policy_map:
                                action = policy.policy_map[state]
                                if action.id in action_arrows:
                                    dx, dy = action_arrows[action.id]
                                    ax.arrow(j, i, dx, dy, head_width=0.15,
                                           head_length=0.1, fc='blue', ec='blue')
            
            ax.set_xlim(-0.5, grid.cols - 0.5)
            ax.set_ylim(grid.rows - 0.5, -0.5)
            ax.set_xticks(range(grid.cols))
            ax.set_yticks(range(grid.rows))
        
        plt.suptitle('Effect of γ on Optimal Policy', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        print("\n观察:")
        print("1. γ小：短视，只关心近期奖励")
        print("2. γ大：远视，考虑长期累积奖励")
        print("3. γ影响最优路径的选择")
        
        return fig


# ================================================================================
# 第3.5.2节：赌徒问题
# Section 3.5.2: Gambler's Problem
# ================================================================================

class GamblersProblem(MDPEnvironment):
    """
    赌徒问题
    Gambler's Problem
    
    一个赌徒有机会对硬币投掷的结果下注。
    A gambler has the opportunity to make bets on coin flips.
    
    问题设定：
    Problem setup:
    - 状态：当前资本 s ∈ {1, 2, ..., 99}
      States: Current capital s ∈ {1, 2, ..., 99}
    - 动作：下注金额 a ∈ {0, 1, ..., min(s, 100-s)}
      Actions: Bet amount a ∈ {0, 1, ..., min(s, 100-s)}
    - 转移：赢概率p_h，输概率1-p_h
      Transitions: Win with prob p_h, lose with prob 1-p_h
    - 奖励：达到100获得+1，否则0
      Rewards: +1 for reaching 100, 0 otherwise
    
    这个问题展示了：
    This problem demonstrates:
    - 风险与收益的权衡
      Risk-reward tradeoff
    - 非线性最优策略
      Non-linear optimal policy
    - 终止状态的处理
      Handling terminal states
    """
    
    def __init__(self, p_h: float = 0.4, goal: int = 100):
        """
        初始化赌徒问题
        Initialize Gambler's Problem
        
        Args:
            p_h: 硬币正面（赢）的概率
                Probability of heads (win)
            goal: 目标资本
                 Goal capital
        
        注意：p_h < 0.5使问题更有挑战性
        Note: p_h < 0.5 makes problem more challenging
        """
        self.p_h = p_h
        self.goal = goal
        
        # 创建状态空间（资本0和goal是终止状态）
        # Create state space (capital 0 and goal are terminal)
        self.state_space = []
        for capital in range(goal + 1):
            is_terminal = (capital == 0 or capital == goal)
            state = State(
                id=f"capital_{capital}",
                features={'capital': capital},
                is_terminal=is_terminal
            )
            self.state_space.append(state)
        
        # 创建动作空间（最大下注额）
        # Create action space (maximum bet)
        self.action_space = []
        for bet in range(goal):
            action = Action(
                id=f"bet_{bet}",
                name=f"Bet {bet}"
            )
            action.bet_amount = bet  # 将下注额作为属性存储
            self.action_space.append(action)
        
        # 设置转移概率和奖励
        self._setup_dynamics()
        
        super().__init__(
            name="Gambler's Problem",
            state_space=self.state_space,
            action_space=self.action_space,
            initial_state=self.state_space[1]  # 从资本1开始
        )
        
        logger.info(f"创建赌徒问题: p_h={p_h}, goal={goal}")
    
    def _setup_dynamics(self):
        """
        设置转移概率和奖励函数
        Setup transition probabilities and reward function
        """
        self.P = TransitionProbability()
        self.R = RewardFunction()
        
        for state in self.state_space:
            capital = state.features['capital']
            
            # 终止状态
            if state.is_terminal:
                continue
            
            # 对每个可能的下注
            for action in self.action_space:
                bet = action.bet_amount
                
                # 检查下注是否合法
                if bet > capital or bet > (self.goal - capital):
                    # 非法动作，保持原状态
                    self.P.set_probability(state, action, state, 0.0, 1.0)
                    self.R.set_reward(state, action, state, 0.0)
                elif bet == 0:
                    # 不下注，保持原状态
                    self.P.set_probability(state, action, state, 0.0, 1.0)
                    self.R.set_reward(state, action, state, 0.0)
                else:
                    # 赢的情况
                    new_capital_win = capital + bet
                    if new_capital_win >= self.goal:
                        # 达到目标
                        next_state_win = self.state_space[self.goal]
                        self.P.set_probability(state, action, next_state_win, 1.0, self.p_h)
                        self.R.set_reward(state, action, next_state_win, 1.0)
                    else:
                        # 未达到目标
                        next_state_win = self.state_space[new_capital_win]
                        self.P.set_probability(state, action, next_state_win, 0.0, self.p_h)
                        self.R.set_reward(state, action, next_state_win, 0.0)
                    
                    # 输的情况
                    new_capital_lose = capital - bet
                    next_state_lose = self.state_space[new_capital_lose]
                    self.P.set_probability(state, action, next_state_lose, 0.0, 1 - self.p_h)
                    self.R.set_reward(state, action, next_state_lose, 0.0)
    
    def get_valid_actions(self, state: State) -> List[Action]:
        """
        获取某状态下的合法动作
        Get valid actions for a state
        
        合法下注额：0到min(资本, 100-资本)
        Valid bets: 0 to min(capital, 100-capital)
        """
        if state.is_terminal:
            return []
        
        capital = state.features['capital']
        max_bet = min(capital, self.goal - capital)
        
        valid_actions = []
        for action in self.action_space:
            bet = action.bet_amount
            if bet <= max_bet:
                valid_actions.append(action)
        
        return valid_actions
    
    def reset(self) -> State:
        """重置环境"""
        self.current_state = self.state_space[1]  # 从资本1开始
        return self.current_state
    
    def step(self, action: Action) -> Tuple[State, float, bool, Dict]:
        """执行动作"""
        # 获取当前资本和下注额
        capital = self.current_state.features['capital']
        bet = action.bet_amount
        
        # 随机决定输赢
        import random
        if random.random() < self.p_h:
            # 赢
            new_capital = min(capital + bet, self.goal)
            reward = 1.0 if new_capital == self.goal else 0.0
        else:
            # 输
            new_capital = capital - bet
            reward = 0.0
        
        # 更新状态
        self.current_state = self.state_space[new_capital]
        done = self.current_state.is_terminal
        
        return self.current_state, reward, done, {'capital': new_capital}
    
    @staticmethod
    def solve_and_visualize(p_h: float = 0.4, gamma: float = 0.99):
        """
        解决并可视化赌徒问题
        Solve and Visualize Gambler's Problem
        
        展示最优策略的有趣特征
        Shows interesting features of optimal policy
        """
        print(f"\n解决赌徒问题 (p_h={p_h})...")
        print(f"Solving Gambler's Problem (p_h={p_h})...")
        
        # 创建问题
        problem = GamblersProblem(p_h=p_h)
        
        # 用价值迭代求解
        vi = ValueIteration(problem, gamma=gamma)
        policy, V = vi.solve(theta=1e-9, verbose=False)
        
        print(f"收敛迭代: {vi.total_iterations}")
        
        # 提取策略和价值
        capitals = list(range(1, 100))
        values = []
        bets = []
        
        for capital in capitals:
            state = problem.state_space[capital]
            values.append(V.get_value(state))
            
            if isinstance(policy, DeterministicPolicy) and state in policy.policy_map:
                action = policy.policy_map[state]
                bet = action.bet_amount
                bets.append(bet)
            else:
                bets.append(0)
        
        # 可视化
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 上图：价值函数
        ax1.plot(capitals, values, 'b-', linewidth=2)
        ax1.set_xlabel('Capital / 资本')
        ax1.set_ylabel('Value / 价值')
        ax1.set_title(f'Value Function (p_h={p_h})')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 100])
        
        # 标注关键点
        ax1.axvline(x=50, color='r', linestyle='--', alpha=0.5, label='Capital=50')
        ax1.legend()
        
        # 下图：最优策略（下注额）
        ax2.bar(capitals, bets, color='green', alpha=0.7)
        ax2.set_xlabel('Capital / 资本')
        ax2.set_ylabel('Optimal Bet / 最优下注')
        ax2.set_title(f'Optimal Policy (p_h={p_h})')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_xlim([0, 100])
        
        # 标注有趣的模式
        if p_h < 0.5:
            ax2.text(50, max(bets)*0.8, 
                    'Conservative when losing odds\n输率高时保守',
                    ha='center', bbox=dict(boxstyle="round,pad=0.3", 
                                         facecolor="yellow", alpha=0.5))
        
        plt.suptitle(f"Gambler's Problem Solution (p_h={p_h}, γ={gamma})",
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig, policy, V
    
    @staticmethod
    def analyze_different_probabilities():
        """
        分析不同获胜概率下的策略
        Analyze Strategies Under Different Win Probabilities
        
        展示p_h如何影响最优策略
        Shows how p_h affects optimal policy
        """
        print("\n" + "="*60)
        print("不同获胜概率的影响")
        print("Effect of Different Win Probabilities")
        print("="*60)
        
        probabilities = [0.25, 0.4, 0.5, 0.55]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, p_h in enumerate(probabilities):
            print(f"\n分析 p_h={p_h}...")
            
            # 创建并解决问题
            problem = GamblersProblem(p_h=p_h)
            vi = ValueIteration(problem, gamma=0.99)
            policy, V = vi.solve(theta=1e-9, verbose=False)
            
            # 提取策略
            capitals = list(range(1, 100))
            bets = []
            for capital in capitals:
                state = problem.state_space[capital]
                if isinstance(policy, DeterministicPolicy) and state in policy.policy_map:
                    action = policy.policy_map[state]
                    bet = action.bet_amount
                    bets.append(bet)
                else:
                    bets.append(0)
            
            # 绘图
            ax = axes[idx]
            ax.bar(capitals, bets, color='steelblue' if p_h < 0.5 else 'coral', alpha=0.7)
            ax.set_xlabel('Capital')
            ax.set_ylabel('Bet')
            ax.set_title(f'p_h = {p_h} {"(Disadvantage)" if p_h < 0.5 else "(Advantage)" if p_h > 0.5 else "(Fair)"}')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_xlim([0, 100])
            ax.set_ylim([0, 55])
            
            # 添加说明
            if p_h < 0.5:
                strategy = "Conservative"
            elif p_h > 0.5:
                strategy = "Aggressive"
            else:
                strategy = "Balanced"
            
            ax.text(50, max(bets) + 2, f'{strategy} Strategy',
                   ha='center', fontsize=10, fontweight='bold')
        
        plt.suptitle("Optimal Betting Strategies for Different Win Probabilities",
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        print("\n关键观察：")
        print("1. p < 0.5：保守策略，小额下注")
        print("2. p = 0.5：平衡策略")
        print("3. p > 0.5：激进策略，大额下注")
        print("4. 策略在某些资本值出现不连续跳跃")
        
        return fig


# ================================================================================
# 第3.5.3节：杰克汽车租赁问题
# Section 3.5.3: Jack's Car Rental
# ================================================================================

class CarRental(MDPEnvironment):
    """
    杰克汽车租赁问题
    Jack's Car Rental Problem
    
    杰克管理两个汽车租赁点，需要决定每晚在两点间转移多少辆车。
    Jack manages two car rental locations and must decide how many cars to move between them each night.
    
    问题特征：
    Problem characteristics:
    - 状态：两个地点的车辆数 (n1, n2)
      States: Number of cars at each location (n1, n2)
    - 动作：从地点1移到地点2的车辆数（可为负）
      Actions: Number of cars to move from location 1 to 2 (can be negative)
    - 随机性：租赁和归还都是泊松分布
      Stochasticity: Rentals and returns follow Poisson distributions
    - 约束：每个地点最多20辆车，每晚最多移5辆
      Constraints: Max 20 cars per location, max 5 cars moved per night
    
    这展示了：
    This demonstrates:
    - 连续状态空间的离散化
      Discretization of continuous state space
    - 资源分配优化
      Resource allocation optimization
    - 处理复杂约束
      Handling complex constraints
    """
    
    def __init__(self, 
                 max_cars: int = 20,
                 max_move: int = 5,
                 rental_reward: float = 10.0,
                 move_cost: float = 2.0,
                 lambdas: Dict[str, float] = None):
        """
        初始化汽车租赁问题
        Initialize Car Rental Problem
        
        Args:
            max_cars: 每个地点最大车辆数
                     Max cars per location
            max_move: 每晚最大转移数
                     Max cars moved per night
            rental_reward: 每辆车租赁收入
                         Rental income per car
            move_cost: 每辆车转移成本
                      Cost per car moved
            lambdas: 泊松参数
                    Poisson parameters
                    {
                        'rental_1': 租赁请求率（地点1）,
                        'rental_2': 租赁请求率（地点2）,
                        'return_1': 归还率（地点1）,
                        'return_2': 归还率（地点2）
                    }
        """
        self.max_cars = max_cars
        self.max_move = max_move
        self.rental_reward = rental_reward
        self.move_cost = move_cost
        
        # 默认泊松参数（来自教材）
        if lambdas is None:
            lambdas = {
                'rental_1': 3,  # 地点1平均每天租出3辆
                'rental_2': 4,  # 地点2平均每天租出4辆
                'return_1': 3,  # 地点1平均每天归还3辆
                'return_2': 2   # 地点2平均每天归还2辆
            }
        self.lambdas = lambdas
        
        # 预计算泊松概率（提高效率）
        self._precompute_poisson()
        
        # 创建状态空间
        self.state_space = []
        self.state_dict = {}  # 快速查找
        
        for n1 in range(max_cars + 1):
            for n2 in range(max_cars + 1):
                state = State(
                    id=f"cars_{n1}_{n2}",
                    features={'cars_1': n1, 'cars_2': n2},
                    is_terminal=False
                )
                self.state_space.append(state)
                self.state_dict[(n1, n2)] = state
        
        # 创建动作空间（正数表示从1到2，负数表示从2到1）
        self.action_space = []
        for move in range(-max_move, max_move + 1):
            action = Action(
                id=f"move_{move}",
                name=f"Move {move} cars"
            )
            action.cars_moved = move  # 将移动数作为属性存储
            self.action_space.append(action)
        
        # 设置动态
        self._setup_dynamics()
        
        super().__init__(
            name="Jack's Car Rental",
            state_space=self.state_space,
            action_space=self.action_space,
            initial_state=self.state_dict[(10, 10)]  # 从各10辆开始
        )
        
        logger.info(f"创建汽车租赁问题: {max_cars}辆/地点, 最多移{max_move}辆")
    
    def _precompute_poisson(self):
        """
        预计算泊松概率
        Precompute Poisson Probabilities
        
        加速转移概率计算
        Speed up transition probability computation
        """
        self.poisson_cache = {}
        
        # 计算每个lambda的概率分布
        for key, lam in self.lambdas.items():
            probs = {}
            # 计算到3倍期望值（覆盖大部分概率质量）
            max_val = min(int(3 * lam), self.max_cars)
            for n in range(max_val + 1):
                probs[n] = poisson.pmf(n, lam)
            # 剩余概率归到最大值
            probs[max_val] += 1 - sum(probs.values())
            self.poisson_cache[key] = probs
    
    def _setup_dynamics(self):
        """
        设置转移概率和奖励
        Setup Transition Probabilities and Rewards
        
        这是最复杂的部分，需要考虑：
        This is the most complex part, considering:
        - 车辆转移
          Car movements
        - 租赁请求（泊松）
          Rental requests (Poisson)
        - 车辆归还（泊松）
          Car returns (Poisson)
        - 容量限制
          Capacity constraints
        """
        self.P = TransitionProbability()
        self.R = RewardFunction()
        
        # 简化：只计算主要转移（完整计算太耗时）
        # Simplification: Only compute major transitions
        print("设置汽车租赁动态（简化版）...")
        
        for state in self.state_space:
            n1 = state.features['cars_1']
            n2 = state.features['cars_2']
            
            for action in self.action_space:
                move = action.cars_moved
                
                # 检查移动是否可行
                if not self._is_valid_move(n1, n2, move):
                    # 非法移动，保持原状态
                    self.P.set_probability(state, action, state, -100, 1.0)
                    self.R.set_reward(state, action, state, -100)  # 大惩罚
                    continue
                
                # 移动后的车辆数
                n1_after_move = n1 - move
                n2_after_move = n2 + move
                
                # 简化：使用期望值而非完整分布
                # 期望租赁数（受可用车辆限制）
                exp_rental_1 = min(n1_after_move, self.lambdas['rental_1'])
                exp_rental_2 = min(n2_after_move, self.lambdas['rental_2'])
                
                # 期望归还数
                exp_return_1 = self.lambdas['return_1']
                exp_return_2 = self.lambdas['return_2']
                
                # 计算期望的下一状态
                n1_next = int(min(self.max_cars, 
                                 max(0, n1_after_move - exp_rental_1 + exp_return_1)))
                n2_next = int(min(self.max_cars,
                                 max(0, n2_after_move - exp_rental_2 + exp_return_2)))
                
                next_state = self.state_dict[(n1_next, n2_next)]
                
                # 计算奖励
                reward = (exp_rental_1 + exp_rental_2) * self.rental_reward
                reward -= abs(move) * self.move_cost
                
                # 设置（简化的）转移
                self.P.set_probability(state, action, next_state, reward, 1.0)
                self.R.set_reward(state, action, next_state, reward)
    
    def _is_valid_move(self, n1: int, n2: int, move: int) -> bool:
        """
        检查移动是否合法
        Check if Move is Valid
        """
        if move > 0:  # 从1到2
            return move <= n1 and (n2 + move) <= self.max_cars
        else:  # 从2到1
            return -move <= n2 and (n1 - move) <= self.max_cars
    
    def reset(self) -> State:
        """重置环境"""
        self.current_state = self.state_dict[(10, 10)]
        return self.current_state
    
    def step(self, action: Action) -> Tuple[State, float, bool, Dict]:
        """执行动作（简化版）"""
        n1 = self.current_state.features['cars_1']
        n2 = self.current_state.features['cars_2']
        move = action.cars_moved
        
        # 简化：直接使用期望转移
        if not self._is_valid_move(n1, n2, move):
            return self.current_state, -100, False, {'cars_1': n1, 'cars_2': n2}
        
        # 移动车辆
        n1_after = n1 - move
        n2_after = n2 + move
        
        # 计算期望奖励（简化）
        exp_rental_1 = min(n1_after, self.lambdas['rental_1'])
        exp_rental_2 = min(n2_after, self.lambdas['rental_2'])
        reward = (exp_rental_1 + exp_rental_2) * self.rental_reward
        reward -= abs(move) * self.move_cost
        
        # 更新到期望的下一状态
        n1_next = int(min(self.max_cars, max(0, n1_after - exp_rental_1 + self.lambdas['return_1'])))
        n2_next = int(min(self.max_cars, max(0, n2_after - exp_rental_2 + self.lambdas['return_2'])))
        
        self.current_state = self.state_dict[(n1_next, n2_next)]
        
        return self.current_state, reward, False, {'cars_1': n1_next, 'cars_2': n2_next}
    
    @staticmethod
    def solve_and_visualize():
        """
        解决并可视化汽车租赁问题
        Solve and Visualize Car Rental Problem
        
        展示资源分配的最优策略
        Shows optimal strategy for resource allocation
        """
        print("\n解决汽车租赁问题（简化版）...")
        print("Solving Car Rental Problem (Simplified)...")
        
        # 创建小规模问题（完整版太大）
        problem = CarRental(max_cars=10, max_move=3)
        
        # 用策略迭代求解
        pi = PolicyIteration(problem, gamma=0.9)
        policy, V = pi.solve(max_iterations=20, verbose=False)
        
        print(f"收敛迭代: {pi.total_iterations}")
        
        # 提取策略矩阵
        max_cars = 10
        policy_matrix = np.zeros((max_cars + 1, max_cars + 1))
        value_matrix = np.zeros((max_cars + 1, max_cars + 1))
        
        for n1 in range(max_cars + 1):
            for n2 in range(max_cars + 1):
                state = problem.state_dict[(n1, n2)]
                value_matrix[n1, n2] = V.get_value(state)
                
                if isinstance(policy, DeterministicPolicy) and state in policy.policy_map:
                    action = policy.policy_map[state]
                    move = action.cars_moved
                    policy_matrix[n1, n2] = move
        
        # 可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 左图：最优策略
        im1 = ax1.imshow(policy_matrix, cmap='RdBu', vmin=-3, vmax=3, aspect='equal')
        ax1.set_xlabel('Cars at Location 2')
        ax1.set_ylabel('Cars at Location 1')
        ax1.set_title('Optimal Policy\n(+: move 1→2, -: move 2→1)')
        
        # 添加数值
        for i in range(max_cars + 1):
            for j in range(max_cars + 1):
                move = int(policy_matrix[i, j])
                color = 'white' if abs(move) > 1 else 'black'
                ax1.text(j, i, str(move), ha='center', va='center',
                        color=color, fontsize=8)
        
        plt.colorbar(im1, ax=ax1, label='Cars Moved')
        
        # 右图：价值函数
        im2 = ax2.imshow(value_matrix, cmap='YlOrRd', aspect='equal')
        ax2.set_xlabel('Cars at Location 2')
        ax2.set_ylabel('Cars at Location 1')
        ax2.set_title('Value Function')
        
        plt.colorbar(im2, ax=ax2, label='Expected Return')
        
        plt.suptitle("Jack's Car Rental - Optimal Solution", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig, policy, V


# ================================================================================
# 第3.5.4节：DP例子运行器
# Section 3.5.4: DP Examples Runner
# ================================================================================

class DPExampleRunner:
    """
    DP例子运行器
    DP Examples Runner
    
    统一运行所有经典例子
    Run all classic examples uniformly
    """
    
    @staticmethod
    def run_all_examples():
        """
        运行所有DP例子
        Run All DP Examples
        """
        print("\n" + "="*80)
        print("动态规划经典例子")
        print("Classic Dynamic Programming Examples")
        print("="*80)
        
        # 1. 网格世界
        print("\n" + "="*60)
        print("例子1：网格世界")
        print("Example 1: Grid World")
        print("="*60)
        
        grid = GridWorldDP(rows=5, cols=5,
                          obstacles={(1,2), (2,2), (3,2)})
        
        # 比较两种算法
        print("\n策略迭代 vs 价值迭代")
        policy_pi, V_pi = grid.solve_with_policy_iteration(gamma=0.9, verbose=False)
        policy_vi, V_vi = grid.solve_with_value_iteration(gamma=0.9, verbose=False)
        
        # 可视化
        grid.visualize_solution(policy_vi, V_vi, 
                               "Grid World - Optimal Solution")
        
        # 演示gamma影响
        GridWorldDP.demonstrate_gamma_effect()
        
        # 2. 赌徒问题
        print("\n" + "="*60)
        print("例子2：赌徒问题")
        print("Example 2: Gambler's Problem")
        print("="*60)
        
        # 解决标准问题
        GamblersProblem.solve_and_visualize(p_h=0.4)
        
        # 分析不同概率
        GamblersProblem.analyze_different_probabilities()
        
        # 3. 汽车租赁（简化版）
        print("\n" + "="*60)
        print("例子3：杰克汽车租赁（简化版）")
        print("Example 3: Jack's Car Rental (Simplified)")
        print("="*60)
        
        CarRental.solve_and_visualize()
        
        print("\n" + "="*80)
        print("所有DP例子完成！")
        print("All DP Examples Complete!")
        print("\n关键学习点：")
        print("Key Learning Points:")
        print("1. DP可以解决各种类型的序列决策问题")
        print("   DP can solve various types of sequential decision problems")
        print("2. 问题的结构决定了最优策略的特征")
        print("   Problem structure determines optimal policy characteristics")
        print("3. 参数（如γ、概率）显著影响最优解")
        print("   Parameters (γ, probabilities) significantly affect optimal solution")
        print("4. 可视化帮助理解算法行为")
        print("   Visualization helps understand algorithm behavior")
        print("="*80)
        
        plt.show()


# ================================================================================
# 主函数
# Main Function
# ================================================================================

def main():
    """
    运行DP经典例子演示
    Run DP Classic Examples Demo
    """
    runner = DPExampleRunner()
    runner.run_all_examples()


if __name__ == "__main__":
    main()