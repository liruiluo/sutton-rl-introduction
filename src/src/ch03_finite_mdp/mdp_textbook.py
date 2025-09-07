"""
================================================================================
第3章：有限马尔可夫决策过程 - 强化学习的数学基石
Chapter 3: Finite Markov Decision Processes - Mathematical Foundation of RL

根据 Sutton & Barto《强化学习：导论》第二版 第3章
Based on Sutton & Barto "Reinforcement Learning: An Introduction" Chapter 3
================================================================================

让我用一个故事帮你理解MDP：

想象你是一个送外卖的骑手。

每一刻，你都在某个位置（状态 State）
你需要决定去哪里（动作 Action）
- 可能去接单（收益不确定）
- 可能去送餐（收益确定但要花时间）
- 可能去充电（没收益但必要）

你的决定会带来：
- 立即收益（奖励 Reward）：送完一单的配送费
- 状态改变（转移 Transition）：从A地到B地
- 未来影响：在B地可能有更多订单

关键是：你在B地的机会，只取决于你现在在B地，
而不取决于你是怎么到B地的（马尔可夫性质）！

这就是MDP：用数学描述"序列决策问题"。

================================================================================
为什么MDP如此重要？
Why is MDP So Important?
================================================================================

Sutton & Barto说：
"MDPs are a mathematically idealized form of the reinforcement learning problem."

MDP提供了：
1. 精确的数学语言描述RL问题
2. 贝尔曼方程等强大工具
3. 最优策略的理论保证
4. 几乎所有RL算法的基础

理解MDP，就理解了强化学习的本质！
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import matplotlib.patches as mpatches
from collections import defaultdict


# ================================================================================
# 第3.1节：MDP的基本要素
# Section 3.1: Basic Elements of MDP  
# ================================================================================

@dataclass
class State:
    """
    状态 - MDP的第一要素
    
    状态是什么？
    状态是智能体对环境的感知，包含做决策所需的所有信息。
    
    例子：
    - 棋盘游戏：棋子的位置
    - 自动驾驶：车辆位置、速度、周围环境
    - 股票交易：当前持仓、市场价格、账户余额
    
    关键性质：马尔可夫性
    "The future is independent of the past given the present"
    未来只依赖现在，不依赖过去！
    """
    name: str
    features: Dict[str, Any]  # 状态特征
    is_terminal: bool = False  # 是否终止状态
    
    def __hash__(self):
        """使状态可以作为字典的键"""
        return hash(self.name)
    
    def __eq__(self, other):
        return self.name == other.name
    
    def describe(self):
        """生动地描述这个状态"""
        desc = f"状态 '{self.name}':\n"
        for feature, value in self.features.items():
            desc += f"  - {feature}: {value}\n"
        if self.is_terminal:
            desc += "  [终止状态]"
        return desc


@dataclass  
class Action:
    """
    动作 - MDP的第二要素
    
    动作是智能体能采取的行为。
    
    关键点：
    1. 动作空间可以依赖于状态 A(s)
    2. 动作可以是离散的（上下左右）或连续的（转动30度）
    3. 并非所有动作在所有状态下都可用
    
    例子：
    - 走迷宫：上、下、左、右
    - 开车：加速、刹车、转向
    - 投资：买入、卖出、持有
    """
    name: str
    parameters: Dict[str, Any] = None  # 动作参数
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        return self.name == other.name


class TransitionModel:
    """
    状态转移模型 P(s'|s,a) - MDP的第三要素
    
    这是MDP的"物理规律"！
    给定当前状态s和动作a，下一个状态s'的概率分布。
    
    确定性 vs 随机性：
    - 确定性：下棋，每步的结果是确定的
    - 随机性：掷骰子，结果有随机性
    
    这个概率反映了环境的不确定性！
    """
    
    def __init__(self, name: str = "转移模型"):
        """初始化转移模型"""
        self.name = name
        # 转移概率表：(s, a) -> {s': probability}
        self.transitions: Dict[Tuple[State, Action], Dict[State, float]] = {}
        
    def add_transition(self, state: State, action: Action, 
                      next_states: Dict[State, float]):
        """
        添加转移概率
        
        例子：在位置A执行"向北走"
        - 80%概率到达北边格子
        - 10%概率撞墙留在原地  
        - 10%概率滑到东边（地面很滑）
        """
        # 验证概率和为1
        total_prob = sum(next_states.values())
        assert abs(total_prob - 1.0) < 1e-6, f"概率之和必须为1，当前为{total_prob}"
        
        self.transitions[(state, action)] = next_states
        
    def get_next_states(self, state: State, action: Action) -> Dict[State, float]:
        """获取所有可能的下一状态及其概率"""
        return self.transitions.get((state, action), {})
    
    def sample_next_state(self, state: State, action: Action) -> State:
        """
        采样下一个状态
        
        这模拟了真实环境的随机性！
        就像掷一个有偏的骰子。
        """
        next_states = self.get_next_states(state, action)
        if not next_states:
            raise ValueError(f"没有定义从状态{state.name}执行动作{action.name}的转移")
        
        states = list(next_states.keys())
        probs = list(next_states.values())
        return np.random.choice(states, p=probs)


class RewardFunction:
    """
    奖励函数 R(s,a,s') - MDP的第四要素
    
    奖励是智能体的"指南针"！
    它告诉智能体什么是好的，什么是坏的。
    
    设计奖励函数的艺术：
    1. 稀疏 vs 密集
       - 稀疏：只在目标状态给奖励（难学习）
       - 密集：每步都给奖励（易学习但可能有偏）
    
    2. 塑形奖励（Reward Shaping）
       - 给中间步骤奖励，引导学习
       - 但要小心不要误导！
    
    例子：训练狗
    - 稀疏：只在完成整个动作后给零食
    - 密集：每个正确的小步骤都给零食
    """
    
    def __init__(self, name: str = "奖励函数"):
        """初始化奖励函数"""
        self.name = name
        # 奖励表：(s, a, s') -> reward
        self.rewards: Dict[Tuple[State, Action, State], float] = {}
        
    def set_reward(self, state: State, action: Action, 
                   next_state: State, reward: float):
        """设置奖励值"""
        self.rewards[(state, action, next_state)] = reward
        
    def get_reward(self, state: State, action: Action, 
                   next_state: State) -> float:
        """
        获取奖励
        
        注意：奖励可以是负的（惩罚）！
        - 正奖励：鼓励这种转移
        - 负奖励：避免这种转移
        - 零奖励：中性
        """
        return self.rewards.get((state, action, next_state), 0.0)


# ================================================================================
# 第3.2节：马尔可夫决策过程
# Section 3.2: Markov Decision Process
# ================================================================================

class MarkovDecisionProcess:
    """
    完整的MDP定义
    
    MDP = (S, A, P, R, γ)
    - S: 状态空间
    - A: 动作空间  
    - P: 转移概率
    - R: 奖励函数
    - γ: 折扣因子
    
    为什么叫"马尔可夫"？
    因为满足马尔可夫性质：
    P[S_{t+1}|S_t, A_t, S_{t-1}, A_{t-1}, ...] = P[S_{t+1}|S_t, A_t]
    
    通俗理解：
    "要预测明天的天气，只需要知道今天的天气，
     不需要知道昨天、前天...的天气"
    """
    
    def __init__(self, name: str = "MDP", gamma: float = 0.9):
        """
        初始化MDP
        
        gamma（折扣因子）的意义：
        - γ = 0: 只看眼前（极度短视）
        - γ = 1: 未来和现在同等重要（可能不收敛）
        - γ = 0.9: 未来很重要但要打折（常用）
        
        为什么要打折？
        1. 数学上：保证无限序列收敛
        2. 经济上：今天的1块钱比明天的值钱
        3. 不确定性：未来越远越不确定
        """
        self.name = name
        self.gamma = gamma
        
        # MDP组件
        self.states: List[State] = []
        self.actions: List[Action] = []
        self.transition_model = TransitionModel()
        self.reward_function = RewardFunction()
        
        # 额外信息
        self.initial_state: Optional[State] = None
        self.terminal_states: List[State] = []
        
    def add_state(self, state: State):
        """添加状态到MDP"""
        self.states.append(state)
        if state.is_terminal:
            self.terminal_states.append(state)
            
    def add_action(self, action: Action):
        """添加动作到MDP"""
        self.actions.append(action)
        
    def set_dynamics(self, state: State, action: Action, 
                     outcomes: List[Tuple[State, float, float]]):
        """
        设置动态（转移和奖励）
        
        outcomes: [(next_state, probability, reward), ...]
        
        这是定义"游戏规则"的地方！
        """
        next_states = {}
        for next_state, prob, reward in outcomes:
            next_states[next_state] = prob
            self.reward_function.set_reward(state, action, next_state, reward)
        
        self.transition_model.add_transition(state, action, next_states)
    
    def step(self, state: State, action: Action) -> Tuple[State, float]:
        """
        环境的一步
        
        这是智能体与环境交互的接口！
        给定状态和动作，返回下一状态和奖励。
        """
        # 采样下一状态
        next_state = self.transition_model.sample_next_state(state, action)
        
        # 获取奖励
        reward = self.reward_function.get_reward(state, action, next_state)
        
        return next_state, reward
    
    def is_terminal(self, state: State) -> bool:
        """检查是否终止状态"""
        return state in self.terminal_states
    
    def visualize_dynamics(self, state: State, action: Action):
        """
        可视化状态转移动态
        
        用图形展示从一个状态采取动作后的所有可能结果
        """
        outcomes = self.transition_model.get_next_states(state, action)
        
        if not outcomes:
            print(f"没有定义从{state.name}执行{action.name}的转移")
            return
            
        print(f"\n从状态 '{state.name}' 执行动作 '{action.name}':")
        print("-" * 50)
        
        for next_state, prob in outcomes.items():
            reward = self.reward_function.get_reward(state, action, next_state)
            
            # 用柱状图表示概率
            bar = "█" * int(prob * 20)
            print(f"  → {next_state.name:15} "
                  f"概率:{prob:5.1%} {bar:20} "
                  f"奖励:{reward:+.1f}")


# ================================================================================
# 第3.3节：策略 - 如何做决策
# Section 3.3: Policies - How to Make Decisions
# ================================================================================

class Policy:
    """
    策略 π(a|s) - 智能体的行为准则
    
    策略是什么？
    策略是从状态到动作的映射，告诉智能体在每个状态该做什么。
    
    类型：
    1. 确定性策略：π(s) → a
       每个状态对应一个确定的动作
    
    2. 随机策略：π(a|s) → [0,1]
       每个状态对应动作的概率分布
    
    为什么需要随机策略？
    - 应对对手（石头剪刀布）
    - 探索（尝试不同可能）
    - 部分可观测环境
    """
    
    def __init__(self, name: str = "策略"):
        """初始化策略"""
        self.name = name
        # 策略表：state -> {action: probability}
        self.action_probs: Dict[State, Dict[Action, float]] = {}
        
    def set_action_prob(self, state: State, action_probs: Dict[Action, float]):
        """
        设置状态下的动作概率分布
        
        例子：在十字路口
        - 直行：60%
        - 左转：20%
        - 右转：20%
        """
        total = sum(action_probs.values())
        assert abs(total - 1.0) < 1e-6, f"概率和必须为1，当前为{total}"
        
        self.action_probs[state] = action_probs
        
    def get_action_prob(self, state: State, action: Action) -> float:
        """获取在状态s下选择动作a的概率"""
        if state not in self.action_probs:
            return 0.0
        return self.action_probs[state].get(action, 0.0)
    
    def sample_action(self, state: State) -> Action:
        """
        根据策略采样一个动作
        
        这是策略的执行！
        """
        if state not in self.action_probs:
            raise ValueError(f"策略未定义状态{state.name}的动作")
            
        actions = list(self.action_probs[state].keys())
        probs = list(self.action_probs[state].values())
        return np.random.choice(actions, p=probs)
    
    def make_deterministic(self):
        """
        将随机策略转换为确定性策略
        
        选择每个状态下概率最高的动作
        """
        for state in self.action_probs:
            best_action = max(self.action_probs[state].items(), 
                            key=lambda x: x[1])[0]
            self.action_probs[state] = {best_action: 1.0}
            
    def make_uniform(self, states: List[State], actions: List[Action]):
        """
        创建均匀随机策略
        
        每个动作概率相等，完全随机探索
        """
        for state in states:
            if not state.is_terminal:
                uniform_prob = 1.0 / len(actions)
                self.action_probs[state] = {a: uniform_prob for a in actions}


# ================================================================================
# 第3.4节：价值函数 - 评估策略好坏
# Section 3.4: Value Functions - Evaluating Policies
# ================================================================================

class ValueFunctions:
    """
    价值函数 - 强化学习的核心概念！
    
    两种价值函数：
    
    1. 状态价值函数 V^π(s)
       从状态s开始，遵循策略π，期望获得的累积奖励
       "这个位置有多好？"
    
    2. 动作价值函数 Q^π(s,a)  
       从状态s执行动作a，然后遵循策略π，期望获得的累积奖励
       "在这个位置做这个动作有多好？"
    
    关系：V^π(s) = Σ_a π(a|s) * Q^π(s,a)
    
    价值函数回答了强化学习的核心问题：
    "从长远来看，这个状态/动作有多好？"
    """
    
    def __init__(self, mdp: MarkovDecisionProcess, policy: Policy):
        """初始化价值函数"""
        self.mdp = mdp
        self.policy = policy
        
        # 初始化为0
        self.V = {s: 0.0 for s in mdp.states}
        self.Q = {(s, a): 0.0 for s in mdp.states for a in mdp.actions}
        
    def bellman_expectation_v(self, state: State) -> float:
        """
        贝尔曼期望方程 - 状态价值
        
        V^π(s) = Σ_a π(a|s) Σ_s' P(s'|s,a)[R(s,a,s') + γV^π(s')]
        
        直观理解：
        一个状态的价值 = 立即奖励的期望 + 未来价值的期望（打折）
        
        这是动态规划的基础！
        """
        if state.is_terminal:
            return 0.0
            
        value = 0.0
        for action in self.mdp.actions:
            action_prob = self.policy.get_action_prob(state, action)
            if action_prob == 0:
                continue
                
            # 考虑所有可能的下一状态
            next_states = self.mdp.transition_model.get_next_states(state, action)
            for next_state, trans_prob in next_states.items():
                reward = self.mdp.reward_function.get_reward(state, action, next_state)
                value += action_prob * trans_prob * (
                    reward + self.mdp.gamma * self.V[next_state]
                )
                
        return value
    
    def bellman_expectation_q(self, state: State, action: Action) -> float:
        """
        贝尔曼期望方程 - 动作价值
        
        Q^π(s,a) = Σ_s' P(s'|s,a)[R(s,a,s') + γV^π(s')]
        
        直观理解：
        一个动作的价值 = 执行后的立即奖励 + 到达状态的未来价值
        """
        if state.is_terminal:
            return 0.0
            
        value = 0.0
        next_states = self.mdp.transition_model.get_next_states(state, action)
        
        for next_state, prob in next_states.items():
            reward = self.mdp.reward_function.get_reward(state, action, next_state)
            value += prob * (reward + self.mdp.gamma * self.V[next_state])
            
        return value
    
    def policy_evaluation(self, theta: float = 1e-6, max_iterations: int = 1000):
        """
        策略评估 - 计算给定策略的价值函数
        
        迭代应用贝尔曼期望方程直到收敛
        
        这回答了："我的策略有多好？"
        """
        print(f"\n开始策略评估: {self.policy.name}")
        print("-" * 50)
        
        for iteration in range(max_iterations):
            delta = 0.0
            
            # 更新所有状态的价值
            for state in self.mdp.states:
                if state.is_terminal:
                    continue
                    
                old_value = self.V[state]
                new_value = self.bellman_expectation_v(state)
                self.V[state] = new_value
                
                delta = max(delta, abs(old_value - new_value))
            
            # 同时更新Q值
            for state in self.mdp.states:
                if state.is_terminal:
                    continue
                for action in self.mdp.actions:
                    self.Q[(state, action)] = self.bellman_expectation_q(state, action)
            
            # 检查收敛
            if delta < theta:
                print(f"✓ 策略评估收敛于第{iteration + 1}次迭代")
                break
                
            if (iteration + 1) % 10 == 0:
                print(f"  迭代 {iteration + 1}: delta = {delta:.6f}")
        
        return self.V, self.Q


# ================================================================================
# 第3.5节：最优策略与最优价值函数
# Section 3.5: Optimal Policies and Optimal Value Functions  
# ================================================================================

class OptimalSolution:
    """
    最优解 - MDP的终极目标！
    
    关键定理（Sutton & Barto）：
    1. 存在性：每个MDP都存在至少一个最优策略π*
    2. 唯一性：所有最优策略共享相同的最优价值函数V*和Q*
    3. 贝尔曼最优方程：描述最优价值函数的递归关系
    
    最优意味着什么？
    π* >= π 对所有策略π和所有状态s：V^π*(s) >= V^π(s)
    
    通俗理解：
    最优策略是"最聪明"的玩法，在每个位置都做最好的选择！
    """
    
    @staticmethod
    def bellman_optimality_equation():
        """
        贝尔曼最优方程 - 强化学习的基石！
        
        V*(s) = max_a Σ_s' P(s'|s,a)[R(s,a,s') + γV*(s')]
        Q*(s,a) = Σ_s' P(s'|s,a)[R(s,a,s') + γ max_a' Q*(s',a')]
        
        含义：
        最优价值 = 选择最好动作后的期望回报
        
        这是所有动态规划算法的基础！
        """
        explanation = """
        贝尔曼最优方程告诉我们：
        
        1. 最优状态价值 V*(s)
           = 所有动作中最好的那个动作的价值
           = max_a Q*(s,a)
        
        2. 最优动作价值 Q*(s,a)  
           = 立即奖励 + 未来最优价值的期望
           = R + γ * E[V*(s')]
        
        3. 最优策略 π*(s)
           = argmax_a Q*(s,a)
           = 选择Q值最高的动作
        
        这形成了一个递归关系，可以通过迭代求解！
        """
        print(explanation)


# ================================================================================
# 第3.6节：网格世界 - 经典MDP例子
# Section 3.6: Grid World - Classic MDP Example
# ================================================================================

class GridWorld(MarkovDecisionProcess):
    """
    网格世界 - 最经典的MDP例子！
    
    为什么用网格世界？
    1. 直观：像走迷宫，人人都懂
    2. 可视化：容易画出来看
    3. 可扩展：从简单到复杂
    4. 教学价值：完美展示MDP概念
    
    我们的网格世界故事：
    你是一个机器人，在网格中寻找出口。
    - S: 起点
    - G: 终点（目标）
    - #: 墙壁（不能通过）
    - .: 普通格子
    - !: 陷阱（负奖励）
    """
    
    def __init__(self, grid_map: List[str], gamma: float = 0.9):
        """
        初始化网格世界
        
        grid_map: 字符串列表定义地图
        """
        super().__init__(name="GridWorld", gamma=gamma)
        
        self.grid_map = grid_map
        self.height = len(grid_map)
        self.width = len(grid_map[0]) if self.height > 0 else 0
        
        # 创建状态和动作
        self._create_states()
        self._create_actions()
        self._create_dynamics()
        
    def _create_states(self):
        """创建所有状态"""
        self.grid_states = {}  # (x,y) -> State
        
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid_map[y][x]
                
                if cell == '#':
                    continue  # 墙壁不是状态
                    
                # 创建状态
                features = {
                    'x': x,
                    'y': y, 
                    'type': cell
                }
                
                is_terminal = (cell == 'G')
                state = State(
                    name=f"({x},{y})",
                    features=features,
                    is_terminal=is_terminal
                )
                
                self.add_state(state)
                self.grid_states[(x, y)] = state
                
                if cell == 'S':
                    self.initial_state = state
    
    def _create_actions(self):
        """创建动作（四个方向）"""
        self.add_action(Action("上"))
        self.add_action(Action("下"))
        self.add_action(Action("左"))
        self.add_action(Action("右"))
        
        # 方向映射
        self.directions = {
            Action("上"): (0, -1),
            Action("下"): (0, 1),
            Action("左"): (-1, 0),
            Action("右"): (1, 0)
        }
    
    def _create_dynamics(self):
        """
        创建转移和奖励
        
        规则：
        1. 正常移动：80%成功，20%滑到垂直方向
        2. 撞墙：留在原地
        3. 到达目标：+10奖励
        4. 踩到陷阱：-10奖励
        5. 普通移动：-1奖励（鼓励快速找到出口）
        """
        for state in self.states:
            if state.is_terminal:
                continue
                
            x, y = state.features['x'], state.features['y']
            
            for action in self.actions:
                outcomes = []
                
                # 计算目标位置（80%概率）
                dx, dy = self.directions[action]
                target_x, target_y = x + dx, y + dy
                
                # 计算滑动位置（各10%概率）
                if action.name in ['上', '下']:
                    slip_positions = [(x-1, y), (x+1, y)]  # 左右滑
                else:
                    slip_positions = [(x, y-1), (x, y+1)]  # 上下滑
                
                # 处理主要移动（80%）
                if self._is_valid_position(target_x, target_y):
                    next_state = self.grid_states[(target_x, target_y)]
                    reward = self._get_step_reward(next_state)
                    outcomes.append((next_state, 0.8, reward))
                else:
                    # 撞墙，留在原地
                    outcomes.append((state, 0.8, -1))
                
                # 处理滑动（各10%）
                for slip_x, slip_y in slip_positions:
                    if self._is_valid_position(slip_x, slip_y):
                        next_state = self.grid_states[(slip_x, slip_y)]
                        reward = self._get_step_reward(next_state)
                        outcomes.append((next_state, 0.1, reward))
                    else:
                        # 撞墙，留在原地
                        outcomes.append((state, 0.1, -1))
                
                # 合并相同结果
                merged_outcomes = defaultdict(lambda: [0, 0])
                for next_state, prob, reward in outcomes:
                    merged_outcomes[next_state][0] += prob
                    merged_outcomes[next_state][1] = reward
                
                # 设置动态
                final_outcomes = [
                    (ns, prob, reward) 
                    for ns, (prob, reward) in merged_outcomes.items()
                ]
                self.set_dynamics(state, action, final_outcomes)
    
    def _is_valid_position(self, x: int, y: int) -> bool:
        """检查位置是否有效（不是墙壁且在边界内）"""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return self.grid_map[y][x] != '#'
    
    def _get_step_reward(self, state: State) -> float:
        """获取进入某状态的奖励"""
        cell_type = state.features['type']
        if cell_type == 'G':
            return 10.0  # 到达目标
        elif cell_type == '!':
            return -10.0  # 踩到陷阱
        else:
            return -1.0  # 普通步骤（鼓励快速）
    
    def visualize(self, values: Dict[State, float] = None, 
                  policy: Policy = None):
        """
        可视化网格世界
        
        可以显示：
        - 地图布局
        - 状态价值（热力图）
        - 策略（箭头）
        """
        fig, ax = plt.subplots(figsize=(self.width * 2, self.height * 2))
        
        # 设置坐标系
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(-0.5, self.height - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        # 绘制网格
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid_map[y][x]
                
                # 绘制格子
                if cell == '#':
                    # 墙壁
                    rect = Rectangle((x-0.5, y-0.5), 1, 1, 
                                   facecolor='black', edgecolor='gray')
                    ax.add_patch(rect)
                elif cell == 'S':
                    # 起点
                    rect = Rectangle((x-0.5, y-0.5), 1, 1,
                                   facecolor='green', alpha=0.3, edgecolor='gray')
                    ax.add_patch(rect)
                    ax.text(x, y, 'S', ha='center', va='center', fontsize=20)
                elif cell == 'G':
                    # 终点
                    rect = Rectangle((x-0.5, y-0.5), 1, 1,
                                   facecolor='gold', alpha=0.3, edgecolor='gray')
                    ax.add_patch(rect)
                    ax.text(x, y, 'G', ha='center', va='center', fontsize=20)
                elif cell == '!':
                    # 陷阱
                    rect = Rectangle((x-0.5, y-0.5), 1, 1,
                                   facecolor='red', alpha=0.3, edgecolor='gray')
                    ax.add_patch(rect)
                    ax.text(x, y, '!', ha='center', va='center', fontsize=20)
                else:
                    # 普通格子
                    rect = Rectangle((x-0.5, y-0.5), 1, 1,
                                   facecolor='white', edgecolor='gray')
                    ax.add_patch(rect)
                
                # 显示价值
                if values and (x, y) in self.grid_states:
                    state = self.grid_states[(x, y)]
                    if not state.is_terminal and state in values:
                        value = values[state]
                        ax.text(x, y-0.3, f'{value:.1f}', 
                               ha='center', va='center', fontsize=10)
                
                # 显示策略
                if policy and (x, y) in self.grid_states:
                    state = self.grid_states[(x, y)]
                    if not state.is_terminal and state in policy.action_probs:
                        # 找最可能的动作
                        best_action = max(policy.action_probs[state].items(),
                                        key=lambda x: x[1])[0]
                        
                        # 画箭头
                        dx, dy = self.directions[best_action]
                        ax.arrow(x, y, dx*0.3, dy*0.3,
                               head_width=0.1, head_length=0.1,
                               fc='blue', ec='blue')
        
        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))
        ax.grid(True)
        ax.set_title('Grid World MDP')
        
        plt.tight_layout()
        plt.show()


# ================================================================================
# 第3.7节：完整示例 - 理解MDP
# Section 3.7: Complete Example - Understanding MDP
# ================================================================================

def demonstrate_mdp_concepts():
    """
    演示MDP的核心概念
    
    通过一个具体的网格世界例子，
    展示MDP的所有要素如何协同工作。
    """
    print("="*70)
    print("马尔可夫决策过程(MDP)完整演示")
    print("Complete MDP Demonstration")
    print("="*70)
    
    # 1. 创建网格世界
    print("\n【第1步：定义问题】")
    print("[Step 1: Define the Problem]")
    print("-"*40)
    
    # 定义地图
    # S: 起点, G: 目标, #: 墙壁, !: 陷阱, .: 空地
    grid_map = [
        "S.#..",
        "..#.G",
        "...#.",
        "#.!..",
        ".....",
    ]
    
    print("网格世界地图：")
    for row in grid_map:
        print("  " + " ".join(row))
    
    print("\n任务：从S到达G，避开陷阱!")
    
    # 创建MDP
    mdp = GridWorld(grid_map, gamma=0.9)
    
    print(f"\nMDP规格：")
    print(f"  状态数: {len(mdp.states)}")
    print(f"  动作数: {len(mdp.actions)}")
    print(f"  折扣因子: {mdp.gamma}")
    
    # 2. 展示马尔可夫性质
    print("\n【第2步：理解马尔可夫性质】")
    print("[Step 2: Understanding Markov Property]")
    print("-"*40)
    
    state = mdp.grid_states[(1, 1)]
    action = Action("右")
    
    print(f"当前状态: {state.name}")
    print(f"执行动作: {action.name}")
    
    mdp.visualize_dynamics(state, action)
    
    print("\n关键洞察：")
    print("下一状态只依赖于当前状态和动作，")
    print("不依赖于如何到达当前状态！")
    
    # 3. 创建和评估策略
    print("\n【第3步：策略评估】")
    print("[Step 3: Policy Evaluation]")
    print("-"*40)
    
    # 创建随机策略
    random_policy = Policy("随机策略")
    random_policy.make_uniform(mdp.states, mdp.actions)
    
    # 评估策略
    vf = ValueFunctions(mdp, random_policy)
    V_random, Q_random = vf.policy_evaluation()
    
    print("\n随机策略的状态价值：")
    for state in mdp.states[:5]:  # 显示前5个
        if not state.is_terminal:
            print(f"  V({state.name}) = {V_random[state]:.2f}")
    
    # 4. 可视化
    print("\n【第4步：可视化结果】")
    print("[Step 4: Visualize Results]")
    print("-"*40)
    
    mdp.visualize(values=V_random, policy=random_policy)
    
    # 5. 创建更好的策略
    print("\n【第5步：改进策略】")
    print("[Step 5: Policy Improvement]")
    print("-"*40)
    
    # 创建贪婪策略（总是向目标方向）
    greedy_policy = Policy("向目标贪婪策略")
    
    for state in mdp.states:
        if state.is_terminal:
            continue
            
        x, y = state.features['x'], state.features['y']
        
        # 简单启发式：优先向目标移动
        if x < 4:  # 目标在右边
            greedy_policy.set_action_prob(state, {
                Action("右"): 0.7,
                Action("上"): 0.1,
                Action("下"): 0.1,
                Action("左"): 0.1
            })
        else:
            greedy_policy.set_action_prob(state, {
                Action("上"): 0.4,
                Action("下"): 0.4,
                Action("左"): 0.1,
                Action("右"): 0.1
            })
    
    # 评估新策略
    vf_greedy = ValueFunctions(mdp, greedy_policy)
    V_greedy, Q_greedy = vf_greedy.policy_evaluation()
    
    print("\n策略对比：")
    print("状态      | 随机策略 | 贪婪策略 | 改进")
    print("-"*50)
    
    for state in mdp.states[:5]:
        if not state.is_terminal:
            improvement = V_greedy[state] - V_random[state]
            print(f"{state.name:10} | {V_random[state]:8.2f} | "
                  f"{V_greedy[state]:8.2f} | {improvement:+.2f}")
    
    print("\n贪婪策略明显更好！")
    
    # 6. 总结
    print("\n" + "="*70)
    print("MDP核心概念总结")
    print("MDP Core Concepts Summary")
    print("="*70)
    
    print("""
    1. 马尔可夫性质 Markov Property
       - 未来只依赖现在，不依赖过去
       - 使问题可以用动态规划求解
    
    2. 四要素 Four Elements
       - 状态 States (S)
       - 动作 Actions (A)  
       - 转移概率 Transitions P(s'|s,a)
       - 奖励函数 Rewards R(s,a,s')
    
    3. 策略 Policy π(a|s)
       - 告诉智能体在每个状态该做什么
       - 可以是确定的或随机的
    
    4. 价值函数 Value Functions
       - V^π(s): 状态价值
       - Q^π(s,a): 动作价值
       - 通过贝尔曼方程递归定义
    
    5. 最优解 Optimal Solution
       - 存在最优策略π*
       - 对应最优价值函数V*和Q*
       - 可通过动态规划算法求解
    
    理解了MDP，就掌握了强化学习的数学基础！
    Understanding MDP means mastering the mathematical foundation of RL!
    """)


# ================================================================================
# 主程序入口
# Main Entry Point
# ================================================================================

if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║" + " "*15 + "第3章：有限马尔可夫决策过程".center(38) + " "*15 + "║")
    print("║" + " "*10 + "Chapter 3: Finite Markov Decision Processes".center(48) + " "*10 + "║")
    print("╚" + "═"*68 + "╝")
    
    # 运行完整演示
    demonstrate_mdp_concepts()
    
    print("\n下一章：第4章 - 动态规划")
    print("Next: Chapter 4 - Dynamic Programming")