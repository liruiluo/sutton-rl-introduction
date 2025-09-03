"""
================================================================================
第2.4节：网格世界 - 经典MDP示例
Section 2.4: Grid World - Classic MDP Example
================================================================================

网格世界是强化学习中最经典的例子！
Grid World is the most classic example in RL!

它简单直观，但包含了MDP的所有要素：
It's simple and intuitive, but contains all elements of MDP:
- 状态空间：网格中的位置
- 动作空间：上下左右移动
- 奖励：到达目标获得奖励，撞墙有惩罚
- 策略：如何导航到目标

通过网格世界，我们可以直观理解所有RL算法！
Through Grid World, we can intuitively understand all RL algorithms!
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from .mdp_framework import (
    State, Action, MDPEnvironment, MDPAgent,
    TransitionProbability, RewardFunction
)
from .policies_and_values import (
    Policy, StateValueFunction, ActionValueFunction,
    PolicyEvaluation, PolicyImprovement
)

# 设置日志
logger = logging.getLogger(__name__)


# ================================================================================
# 第2.4.1节：网格世界环境
# Section 2.4.1: Grid World Environment
# ================================================================================

class GridWorld(MDPEnvironment):
    """
    网格世界环境
    Grid World Environment
    
    这是Sutton & Barto书中的经典例子！
    This is the classic example from Sutton & Barto!
    
    特点 Features:
    1. 离散状态空间（网格位置）
       Discrete state space (grid positions)
    2. 离散动作空间（四个方向）
       Discrete action space (four directions)
    3. 确定性或随机转移
       Deterministic or stochastic transitions
    4. 稀疏奖励（只在特定位置）
       Sparse rewards (only at specific positions)
    
    用途 Uses:
    - 测试和理解算法
      Test and understand algorithms
    - 可视化价值函数和策略
      Visualize value functions and policies
    - 教学演示
      Teaching demonstrations
    """
    
    def __init__(self, 
                 rows: int = 5,
                 cols: int = 5,
                 start_pos: Tuple[int, int] = (0, 0),
                 goal_pos: Optional[Tuple[int, int]] = None,
                 obstacles: Optional[Set[Tuple[int, int]]] = None,
                 stochastic: bool = False,
                 wind: Optional[np.ndarray] = None):
        """
        初始化网格世界
        Initialize Grid World
        
        Args:
            rows: 行数
            cols: 列数
            start_pos: 起始位置
            goal_pos: 目标位置（None表示右下角）
            obstacles: 障碍物位置集合
            stochastic: 是否随机（风的影响）
            wind: 风场（每个位置的风力向量）
        
        例子 Example:
        5×5网格，起点(0,0)，终点(4,4)
        5×5 grid, start at (0,0), goal at (4,4)
        
        图示 Diagram:
        S . . . .
        . . # . .
        . . # . .
        . . . . .
        . . . . G
        
        S=起点 Start, G=终点 Goal, #=障碍 Obstacle
        """
        super().__init__(name="Grid World")
        
        self.rows = rows
        self.cols = cols
        self.start_pos = start_pos
        self.goal_pos = goal_pos if goal_pos else (rows-1, cols-1)
        self.obstacles = obstacles if obstacles else set()
        self.stochastic = stochastic
        self.wind = wind if wind is not None else np.zeros((rows, cols, 2))
        
        # 创建状态和动作空间
        self._create_spaces()
        
        # 设置动态
        self._setup_dynamics()
        
        # 当前位置
        self.current_pos = start_pos
        
        logger.info(f"初始化{rows}×{cols}网格世界，"
                   f"起点{start_pos}，终点{self.goal_pos}")
    
    def _create_spaces(self):
        """
        创建状态和动作空间
        Create state and action spaces
        """
        # 状态空间：所有网格位置
        # State space: all grid positions
        self.state_space = []
        self.pos_to_state = {}  # 位置到状态的映射
        
        for i in range(self.rows):
            for j in range(self.cols):
                pos = (i, j)
                if pos not in self.obstacles:
                    is_terminal = (pos == self.goal_pos)
                    state = State(
                        id=f"s_{i}_{j}",
                        features=np.array([i, j]),
                        is_terminal=is_terminal,
                        info={'position': pos}
                    )
                    self.state_space.append(state)
                    self.pos_to_state[pos] = state
        
        # 动作空间：四个方向
        # Action space: four directions
        self.action_space = [
            Action(id='up', name='上', parameters=np.array([-1, 0])),
            Action(id='down', name='下', parameters=np.array([1, 0])),
            Action(id='left', name='左', parameters=np.array([0, -1])),
            Action(id='right', name='右', parameters=np.array([0, 1]))
        ]
        
        logger.info(f"创建状态空间: {len(self.state_space)}个状态, "
                   f"动作空间: {len(self.action_space)}个动作")
    
    def _setup_dynamics(self):
        """
        设置环境动态
        Setup environment dynamics
        
        定义转移概率和奖励函数
        Define transition probabilities and reward function
        """
        # 遍历所有状态-动作对
        for state in self.state_space:
            pos = state.info['position']
            
            # 终止状态没有转移
            if state.is_terminal:
                continue
            
            for action in self.action_space:
                # 计算下一位置
                if self.stochastic:
                    # 随机环境：80%按意图移动，20%随机
                    self._setup_stochastic_transition(state, action, pos)
                else:
                    # 确定性环境
                    self._setup_deterministic_transition(state, action, pos)
    
    def _setup_deterministic_transition(self, state: State, action: Action, pos: Tuple):
        """
        设置确定性转移
        Setup deterministic transition
        """
        # 计算目标位置
        move = action.parameters
        next_pos = (pos[0] + int(move[0]), pos[1] + int(move[1]))
        
        # 考虑风的影响
        wind_effect = self.wind[pos[0], pos[1]]
        next_pos = (
            next_pos[0] + int(wind_effect[0]),
            next_pos[1] + int(wind_effect[1])
        )
        
        # 检查边界和障碍
        if self._is_valid_position(next_pos):
            next_state = self.pos_to_state[next_pos]
        else:
            # 撞墙或障碍，留在原地
            next_state = state
            next_pos = pos
        
        # 设置奖励
        reward = self._get_reward(pos, action, next_pos)
        
        # 设置转移概率（确定性=1.0）
        self.P.set_probability(state, action, next_state, reward, 1.0)
    
    def _setup_stochastic_transition(self, state: State, action: Action, pos: Tuple):
        """
        设置随机转移
        Setup stochastic transition
        
        80%概率按意图移动，20%概率垂直方向偏移
        80% probability move as intended, 20% perpendicular drift
        """
        # 主要转移（80%）
        main_prob = 0.8
        side_prob = 0.1  # 每个垂直方向10%
        
        # 获取所有可能的下一状态
        transitions = []
        
        # 意图移动
        move = action.parameters
        intended_next = (pos[0] + int(move[0]), pos[1] + int(move[1]))
        
        # 垂直方向
        if action.id in ['up', 'down']:
            perpendicular_moves = [np.array([0, -1]), np.array([0, 1])]  # 左右
        else:
            perpendicular_moves = [np.array([-1, 0]), np.array([1, 0])]  # 上下
        
        # 添加主要转移
        if self._is_valid_position(intended_next):
            next_state = self.pos_to_state[intended_next]
            reward = self._get_reward(pos, action, intended_next)
            transitions.append((next_state, reward, main_prob))
        else:
            # 撞墙，留在原地
            transitions.append((state, -1, main_prob))
        
        # 添加垂直方向转移
        for perp_move in perpendicular_moves:
            perp_next = (pos[0] + int(perp_move[0]), pos[1] + int(perp_move[1]))
            if self._is_valid_position(perp_next):
                next_state = self.pos_to_state[perp_next]
                reward = self._get_reward(pos, action, perp_next)
                transitions.append((next_state, reward, side_prob))
            else:
                transitions.append((state, -1, side_prob))
        
        # 设置所有转移
        for next_state, reward, prob in transitions:
            self.P.set_probability(state, action, next_state, reward, prob)
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """
        检查位置是否有效
        Check if position is valid
        """
        row, col = pos
        return (0 <= row < self.rows and 
                0 <= col < self.cols and 
                pos not in self.obstacles)
    
    def _get_reward(self, current_pos: Tuple, action: Action, 
                   next_pos: Tuple) -> float:
        """
        获取奖励
        Get reward
        
        奖励设计 Reward Design:
        - 到达目标: +10
        - 撞墙/障碍: -1
        - 普通移动: -0.1（鼓励找最短路径）
        """
        if next_pos == self.goal_pos:
            return 10.0  # 到达目标
        elif next_pos == current_pos:
            return -1.0  # 撞墙或障碍
        else:
            return -0.1  # 普通移动（生存成本）
    
    def reset(self) -> State:
        """重置环境"""
        self.current_pos = self.start_pos
        self.current_state = self.pos_to_state[self.current_pos]
        self.step_count = 0
        self.episode_count += 1
        
        logger.info(f"Episode {self.episode_count}: 重置到位置 {self.start_pos}")
        return self.current_state
    
    def step(self, action: Action) -> Tuple[State, float, bool, Dict]:
        """
        执行动作
        Execute action
        """
        if self.current_state is None:
            raise ValueError("Environment not reset. Call reset() first.")
        
        # 从转移概率采样
        next_state, reward = self.P.sample(self.current_state, action)
        
        # 更新位置
        self.current_state = next_state
        self.current_pos = next_state.info['position']
        self.step_count += 1
        
        # 检查终止
        done = next_state.is_terminal
        
        # 额外信息
        info = {
            'position': self.current_pos,
            'step': self.step_count
        }
        
        logger.debug(f"Step {self.step_count}: {action.name} -> "
                    f"位置{self.current_pos}, 奖励{reward:.1f}")
        
        return next_state, reward, done, info
    
    def render(self, mode: str = 'human', values: Optional[StateValueFunction] = None,
              policy: Optional[Policy] = None):
        """
        渲染环境
        Render environment
        
        可以显示：
        - 当前位置
        - 价值函数（热力图）
        - 策略（箭头）
        
        Args:
            mode: 渲染模式
            values: 要显示的价值函数
            policy: 要显示的策略
        """
        if mode == 'human':
            # 文本渲染
            print("\n" + "="*30)
            print(f"Grid World (Step {self.step_count})")
            print("="*30)
            
            for i in range(self.rows):
                row_str = ""
                for j in range(self.cols):
                    pos = (i, j)
                    
                    if pos == self.current_pos:
                        row_str += " A "  # Agent
                    elif pos == self.goal_pos:
                        row_str += " G "  # Goal
                    elif pos == self.start_pos:
                        row_str += " S "  # Start
                    elif pos in self.obstacles:
                        row_str += " # "  # Obstacle
                    else:
                        row_str += " . "  # Empty
                
                print(row_str)
            print()
        
        elif mode == 'rgb_array':
            # 图形渲染
            return self._render_graphical(values, policy)
    
    def _render_graphical(self, values: Optional[StateValueFunction] = None,
                         policy: Optional[Policy] = None):
        """
        图形化渲染
        Graphical rendering
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 绘制网格
        for i in range(self.rows + 1):
            ax.axhline(y=i, color='black', linewidth=1)
        for j in range(self.cols + 1):
            ax.axvline(x=j, color='black', linewidth=1)
        
        # 如果有价值函数，绘制热力图
        if values:
            value_matrix = np.zeros((self.rows, self.cols))
            for state in self.state_space:
                pos = state.info['position']
                value_matrix[pos[0], pos[1]] = values.get_value(state)
            
            # 归一化到[0, 1]
            vmin, vmax = value_matrix.min(), value_matrix.max()
            if vmax > vmin:
                value_matrix = (value_matrix - vmin) / (vmax - vmin)
            
            # 绘制热力图
            for i in range(self.rows):
                for j in range(self.cols):
                    if (i, j) not in self.obstacles:
                        color_val = value_matrix[i, j]
                        rect = patches.Rectangle((j, self.rows-i-1), 1, 1,
                                                linewidth=0,
                                                facecolor=plt.cm.coolwarm(color_val),
                                                alpha=0.5)
                        ax.add_patch(rect)
                        
                        # 添加价值文本
                        if (i, j) in self.pos_to_state:
                            state = self.pos_to_state[(i, j)]
                            val = values.get_value(state)
                            ax.text(j+0.5, self.rows-i-0.5, f'{val:.1f}',
                                   ha='center', va='center', fontsize=8)
        
        # 如果有策略，绘制箭头
        if policy:
            arrow_map = {
                'up': (0, 0.3),
                'down': (0, -0.3),
                'left': (-0.3, 0),
                'right': (0.3, 0)
            }
            
            for state in self.state_space:
                if not state.is_terminal:
                    pos = state.info['position']
                    i, j = pos
                    
                    # 获取动作概率
                    action_probs = policy.get_action_probabilities(state)
                    
                    # 绘制箭头（按概率）
                    for action, prob in action_probs.items():
                        if prob > 0.1:  # 只显示显著的动作
                            dx, dy = arrow_map.get(action.id, (0, 0))
                            ax.arrow(j+0.5, self.rows-i-0.5, 
                                   dx*prob, dy*prob,
                                   head_width=0.1*prob, 
                                   head_length=0.1*prob,
                                   fc='blue', ec='blue',
                                   alpha=0.7)
        
        # 标记特殊位置
        # 起点
        rect = patches.Rectangle((self.start_pos[1], self.rows-self.start_pos[0]-1),
                                1, 1, linewidth=2, edgecolor='green',
                                facecolor='lightgreen', alpha=0.3)
        ax.add_patch(rect)
        ax.text(self.start_pos[1]+0.5, self.rows-self.start_pos[0]-0.5, 'S',
               ha='center', va='center', fontsize=12, fontweight='bold')
        
        # 终点
        rect = patches.Rectangle((self.goal_pos[1], self.rows-self.goal_pos[0]-1),
                                1, 1, linewidth=2, edgecolor='red',
                                facecolor='lightcoral', alpha=0.3)
        ax.add_patch(rect)
        ax.text(self.goal_pos[1]+0.5, self.rows-self.goal_pos[0]-0.5, 'G',
               ha='center', va='center', fontsize=12, fontweight='bold')
        
        # 障碍物
        for obs_pos in self.obstacles:
            rect = patches.Rectangle((obs_pos[1], self.rows-obs_pos[0]-1),
                                    1, 1, linewidth=1, edgecolor='black',
                                    facecolor='gray', alpha=0.8)
            ax.add_patch(rect)
        
        # 当前位置（如果不是起点）
        if self.current_pos != self.start_pos:
            circle = patches.Circle((self.current_pos[1]+0.5, 
                                    self.rows-self.current_pos[0]-0.5),
                                   0.2, color='blue', alpha=0.8)
            ax.add_patch(circle)
        
        # 设置坐标轴
        ax.set_xlim(0, self.cols)
        ax.set_ylim(0, self.rows)
        ax.set_aspect('equal')
        ax.set_xticks(range(self.cols + 1))
        ax.set_yticks(range(self.rows + 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(f'Grid World - Step {self.step_count}')
        
        plt.tight_layout()
        return fig


# ================================================================================
# 第2.4.2节：网格世界智能体
# Section 2.4.2: Grid World Agent
# ================================================================================

class GridWorldAgent(MDPAgent):
    """
    网格世界智能体
    Grid World Agent
    
    专门为网格世界设计的智能体
    Agent specifically designed for Grid World
    """
    
    def __init__(self, environment: GridWorld, 
                 learning_algorithm: str = "q_learning"):
        """
        初始化网格世界智能体
        
        Args:
            environment: 网格世界环境
            learning_algorithm: 学习算法
                - "q_learning": Q学习
                - "sarsa": SARSA
                - "policy_iteration": 策略迭代
                - "value_iteration": 价值迭代
        """
        super().__init__(name=f"GridWorld {learning_algorithm} Agent")
        
        self.env = environment
        self.learning_algorithm = learning_algorithm
        
        # 初始化Q函数
        self.Q = ActionValueFunction(
            environment.state_space,
            environment.action_space,
            initial_value=0.0
        )
        
        # 初始化V函数（用于基于模型的方法）
        self.V = StateValueFunction(
            environment.state_space,
            initial_value=0.0
        )
        
        logger.info(f"初始化网格世界智能体，算法: {learning_algorithm}")
    
    def select_action(self, state: State) -> Action:
        """
        选择动作（ε-贪婪）
        Select action (ε-greedy)
        """
        return self.Q.get_epsilon_greedy_action(state, self.epsilon)
    
    def update(self, state: State, action: Action,
              reward: float, next_state: State, done: bool):
        """
        更新智能体
        Update agent
        
        根据算法选择更新方式
        Update method depends on algorithm
        """
        # 保存经验
        self.save_experience(state, action, reward, next_state, done)
        
        if self.learning_algorithm == "q_learning":
            self._q_learning_update(state, action, reward, next_state, done)
        elif self.learning_algorithm == "sarsa":
            self._sarsa_update(state, action, reward, next_state, done)
        else:
            # 基于模型的方法在回合结束后批量更新
            pass
    
    def _q_learning_update(self, state: State, action: Action,
                          reward: float, next_state: State, done: bool):
        """
        Q学习更新
        Q-Learning update
        
        Q(s,a) ← Q(s,a) + α[r + γmax_a' Q(s',a') - Q(s,a)]
        
        离策略学习！
        Off-policy learning!
        """
        current_q = self.Q.get_value(state, action)
        
        if done:
            target = reward
        else:
            # 找最大Q值
            next_q_values = self.Q.get_state_values(next_state)
            max_next_q = max(next_q_values.values()) if next_q_values else 0
            target = reward + self.gamma * max_next_q
        
        # TD误差
        td_error = target - current_q
        
        # 更新Q值
        self.Q.update_value(state, action, td_error, self.alpha)
    
    def _sarsa_update(self, state: State, action: Action,
                     reward: float, next_state: State, done: bool):
        """
        SARSA更新
        SARSA update
        
        Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        
        在策略学习！
        On-policy learning!
        """
        current_q = self.Q.get_value(state, action)
        
        if done:
            target = reward
        else:
            # 选择下一动作（ε-贪婪）
            next_action = self.select_action(next_state)
            next_q = self.Q.get_value(next_state, next_action)
            target = reward + self.gamma * next_q
        
        # TD误差
        td_error = target - current_q
        
        # 更新Q值
        self.Q.update_value(state, action, td_error, self.alpha)
    
    def run_policy_iteration(self, theta: float = 1e-6):
        """
        运行策略迭代
        Run policy iteration
        
        基于模型的方法！
        Model-based method!
        """
        if self.learning_algorithm != "policy_iteration":
            logger.warning("Agent not configured for policy iteration")
            return
        
        from .policies_and_values import UniformRandomPolicy
        
        # 初始化随机策略
        policy = UniformRandomPolicy(self.env.action_space)
        
        iteration = 0
        policy_stable = False
        
        while not policy_stable:
            iteration += 1
            logger.info(f"策略迭代 {iteration}")
            
            # 策略评估
            self.V = PolicyEvaluation.evaluate_policy(
                policy, self.env, self.gamma, theta
            )
            
            # 策略改进
            new_policy, changed = PolicyImprovement.improve_policy(
                self.V, self.env, self.gamma
            )
            
            policy_stable = not changed
            policy = new_policy
        
        logger.info(f"策略迭代收敛于第{iteration}次迭代")
        
        # 从V函数导出Q函数
        self._derive_q_from_v()
        
        return policy
    
    def run_value_iteration(self, theta: float = 1e-6,
                          max_iterations: int = 1000):
        """
        运行价值迭代
        Run value iteration
        
        更高效的基于模型方法！
        More efficient model-based method!
        """
        if self.learning_algorithm != "value_iteration":
            logger.warning("Agent not configured for value iteration")
            return
        
        logger.info("开始价值迭代...")
        
        P, R = self.env.get_dynamics()
        
        for iteration in range(max_iterations):
            delta = 0
            
            for state in self.env.state_space:
                if state.is_terminal:
                    continue
                
                old_value = self.V.get_value(state)
                
                # 贝尔曼最优方程
                from .policies_and_values import BellmanEquations
                new_value = BellmanEquations.bellman_optimality_v(
                    state, P, R, self.V, self.env.action_space, self.gamma
                )
                
                self.V.set_value(state, new_value)
                delta = max(delta, abs(old_value - new_value))
            
            if delta < theta:
                logger.info(f"价值迭代收敛于第{iteration+1}次迭代")
                break
        
        # 从V函数导出Q函数和策略
        self._derive_q_from_v()
        
        return self.V.get_greedy_policy(self.Q)
    
    def _derive_q_from_v(self):
        """
        从V函数导出Q函数
        Derive Q-function from V-function
        """
        P, R = self.env.get_dynamics()
        
        for state in self.env.state_space:
            for action in self.env.action_space:
                q_value = 0
                transitions = P.get_transitions(state, action)
                
                for next_state, reward, trans_prob in transitions:
                    q_value += trans_prob * (reward + self.gamma * self.V.get_value(next_state))
                
                self.Q.set_value(state, action, q_value)


# ================================================================================
# 第2.4.3节：网格世界可视化器
# Section 2.4.3: Grid World Visualizer
# ================================================================================

class GridWorldVisualizer:
    """
    网格世界可视化器
    Grid World Visualizer
    
    提供丰富的可视化功能！
    Provides rich visualization capabilities!
    """
    
    @staticmethod
    def visualize_episode(env: GridWorld, agent: GridWorldAgent, 
                         max_steps: int = 100):
        """
        可视化一个回合
        Visualize one episode
        
        显示智能体的轨迹
        Show agent's trajectory
        """
        state = env.reset()
        trajectory = [env.current_pos]
        total_reward = 0
        
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            trajectory.append(env.current_pos)
            total_reward += reward
            
            agent.update(state, action, reward, next_state, done)
            
            if done:
                break
            
            state = next_state
        
        # 绘制轨迹
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 绘制网格
        for i in range(env.rows + 1):
            ax.axhline(y=i, color='gray', linewidth=0.5)
        for j in range(env.cols + 1):
            ax.axvline(x=j, color='gray', linewidth=0.5)
        
        # 绘制轨迹
        traj_x = [pos[1] + 0.5 for pos in trajectory]
        traj_y = [env.rows - pos[0] - 0.5 for pos in trajectory]
        
        ax.plot(traj_x, traj_y, 'b-', linewidth=2, alpha=0.5)
        ax.plot(traj_x, traj_y, 'bo', markersize=8, alpha=0.7)
        
        # 标记起点和终点
        ax.plot(traj_x[0], traj_y[0], 'go', markersize=12, label='Start')
        ax.plot(traj_x[-1], traj_y[-1], 'ro', markersize=12, label='End')
        
        # 标记目标
        goal_x = env.goal_pos[1] + 0.5
        goal_y = env.rows - env.goal_pos[0] - 0.5
        ax.plot(goal_x, goal_y, 'r*', markersize=20, label='Goal')
        
        # 标记障碍物
        for obs_pos in env.obstacles:
            rect = patches.Rectangle((obs_pos[1], env.rows-obs_pos[0]-1),
                                    1, 1, facecolor='gray', alpha=0.8)
            ax.add_patch(rect)
        
        # 设置
        ax.set_xlim(0, env.cols)
        ax.set_ylim(0, env.rows)
        ax.set_aspect('equal')
        ax.set_title(f'Episode Trajectory (Steps: {len(trajectory)-1}, Reward: {total_reward:.1f})')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def compare_algorithms(env: GridWorld, algorithms: List[str],
                          n_episodes: int = 100):
        """
        比较不同算法
        Compare different algorithms
        
        在同一环境中比较性能
        Compare performance in the same environment
        """
        results = {}
        
        for algo in algorithms:
            logger.info(f"训练 {algo} 算法...")
            
            # 创建智能体
            agent = GridWorldAgent(env, learning_algorithm=algo)
            
            # 训练
            episode_rewards = []
            episode_lengths = []
            
            for episode in range(n_episodes):
                state = env.reset()
                total_reward = 0
                steps = 0
                
                for step in range(1000):  # 最大步数
                    action = agent.select_action(state)
                    next_state, reward, done, info = env.step(action)
                    
                    agent.update(state, action, reward, next_state, done)
                    
                    total_reward += reward
                    steps += 1
                    
                    if done:
                        break
                    
                    state = next_state
                
                episode_rewards.append(total_reward)
                episode_lengths.append(steps)
                
                # 衰减探索率
                agent.epsilon = max(0.01, agent.epsilon * 0.995)
            
            results[algo] = {
                'rewards': episode_rewards,
                'lengths': episode_lengths,
                'agent': agent
            }
        
        # 绘制比较图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 学习曲线
        ax1 = axes[0, 0]
        for algo, data in results.items():
            # 移动平均
            window = 10
            rewards = data['rewards']
            ma_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax1.plot(ma_rewards, label=algo, alpha=0.8)
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Average Reward')
        ax1.set_title('Learning Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 回合长度
        ax2 = axes[0, 1]
        for algo, data in results.items():
            lengths = data['lengths']
            ma_lengths = np.convolve(lengths, np.ones(window)/window, mode='valid')
            ax2.plot(ma_lengths, label=algo, alpha=0.8)
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Episode Length')
        ax2.set_title('Episode Lengths')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 最终性能箱线图
        ax3 = axes[1, 0]
        final_rewards = [data['rewards'][-20:] for data in results.values()]
        ax3.boxplot(final_rewards, labels=list(results.keys()))
        ax3.set_ylabel('Final Rewards')
        ax3.set_title('Final Performance (last 20 episodes)')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. 收敛速度
        ax4 = axes[1, 1]
        for algo, data in results.items():
            rewards = data['rewards']
            # 找到第一次达到90%最大奖励的位置
            max_reward = max(rewards)
            threshold = 0.9 * max_reward
            
            converged_episode = None
            for i, r in enumerate(rewards):
                if r >= threshold:
                    converged_episode = i
                    break
            
            if converged_episode:
                ax4.bar(algo, converged_episode, alpha=0.7)
        
        ax4.set_ylabel('Episodes to 90% Performance')
        ax4.set_title('Convergence Speed')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Algorithm Comparison in Grid World')
        plt.tight_layout()
        
        return fig, results


# ================================================================================
# 示例和测试
# Examples and Tests
# ================================================================================

def demonstrate_grid_world():
    """
    演示网格世界
    Demonstrate Grid World
    """
    print("\n" + "="*80)
    print("网格世界演示")
    print("Grid World Demonstration")
    print("="*80)
    
    # 创建网格世界
    print("\n1. 创建5×5网格世界")
    print("1. Create 5×5 Grid World")
    print("-" * 40)
    
    # 设置障碍物
    obstacles = {(1, 2), (2, 2)}  # 中间的墙
    
    env = GridWorld(
        rows=5,
        cols=5,
        start_pos=(0, 0),
        goal_pos=(4, 4),
        obstacles=obstacles,
        stochastic=False  # 先用确定性环境
    )
    
    # 显示环境
    env.render(mode='human')
    
    # 创建智能体
    print("\n2. 创建Q学习智能体")
    print("2. Create Q-Learning Agent")
    print("-" * 40)
    
    agent = GridWorldAgent(env, learning_algorithm="q_learning")
    
    # 运行一个回合
    print("\n3. 运行单个回合")
    print("3. Run Single Episode")
    print("-" * 40)
    
    state = env.reset()
    total_reward = 0
    
    for step in range(50):
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        
        print(f"Step {step+1}: {action.name} -> "
              f"位置{info['position']}, 奖励{reward:.1f}")
        
        agent.update(state, action, reward, next_state, done)
        total_reward += reward
        
        if done:
            print(f"\n到达目标！总奖励: {total_reward:.1f}")
            break
        
        state = next_state
    
    # 可视化轨迹
    print("\n4. 可视化轨迹")
    print("4. Visualize Trajectory")
    print("-" * 40)
    
    visualizer = GridWorldVisualizer()
    fig1 = visualizer.visualize_episode(env, agent, max_steps=50)
    
    # 训练智能体
    print("\n5. 训练智能体(100回合)")
    print("5. Train Agent (100 episodes)")
    print("-" * 40)
    
    from .agent_environment_interface import AgentEnvironmentInterface
    
    interface = AgentEnvironmentInterface(agent, env)
    episodes = interface.run_episodes(n_episodes=100, max_steps_per_episode=100)
    
    # 显示学习进度
    fig2 = interface.plot_learning_curve(window_size=10)
    
    # 可视化学到的价值函数
    print("\n6. 可视化价值函数")
    print("6. Visualize Value Function")
    print("-" * 40)
    
    # 从Q函数导出V函数
    for state in env.state_space:
        q_values = agent.Q.get_state_values(state)
        if q_values:
            max_q = max(q_values.values())
            agent.V.set_value(state, max_q)
    
    fig3 = env._render_graphical(values=agent.V)
    
    # 比较算法
    print("\n7. 比较不同算法")
    print("7. Compare Different Algorithms")
    print("-" * 40)
    
    algorithms = ["q_learning", "sarsa"]
    fig4, results = visualizer.compare_algorithms(env, algorithms, n_episodes=50)
    
    print("\n算法比较结果:")
    print("Algorithm Comparison Results:")
    for algo, data in results.items():
        final_rewards = data['rewards'][-10:]
        print(f"  {algo}: 最终平均奖励 = {np.mean(final_rewards):.2f}")
    
    return [fig1, fig2, fig3, fig4]


def main():
    """主函数"""
    print("\n" + "="*80)
    print("第2.4节：网格世界")
    print("Section 2.4: Grid World")
    print("="*80)
    
    # 运行演示
    figs = demonstrate_grid_world()
    
    print("\n" + "="*80)
    print("网格世界演示完成！")
    print("Grid World Demo Complete!")
    print("\n关键要点：")
    print("Key Takeaways:")
    print("1. 网格世界是理解RL算法的完美平台")
    print("2. 可以直观看到价值函数和策略")
    print("3. 不同算法有不同的学习特性")
    print("4. Q学习是离策略的，SARSA是在策略的")
    print("="*80)
    
    plt.show()
    
    return figs


if __name__ == "__main__":
    main()