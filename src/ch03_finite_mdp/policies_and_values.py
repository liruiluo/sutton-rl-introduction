"""
================================================================================
第2.3节：策略与价值函数 - RL的核心概念
Section 2.3: Policies and Value Functions - Core Concepts of RL
================================================================================

本节深入探讨强化学习的两个核心概念：
This section deeply explores two core concepts of RL:

1. 策略(Policy): 行为的规则 π(a|s)
2. 价值函数(Value Function): 状态/动作的长期价值 v(s), q(s,a)

这两个概念的关系是理解RL算法的关键！
The relationship between these two concepts is key to understanding RL algorithms!
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from .mdp_framework import State, Action, MDPEnvironment, TransitionProbability, RewardFunction

# 设置日志
logger = logging.getLogger(__name__)


# ================================================================================
# 第2.3.1节：策略(Policy)
# Section 2.3.1: Policy
# ================================================================================

class Policy(ABC):
    """
    策略基类 - 从状态到动作的映射
    Policy Base Class - Mapping from states to actions
    
    策略是智能体的行为准则！
    Policy is the behavioral rule of the agent!
    
    数学定义 Mathematical Definition:
    π(a|s) = P[A_t = a | S_t = s]
    
    策略告诉智能体在每个状态下该做什么
    Policy tells the agent what to do in each state
    
    分类 Categories:
    1. 确定性策略: a = π(s)
       Deterministic: a = π(s)
    2. 随机策略: π(a|s) ∈ [0,1]
       Stochastic: π(a|s) ∈ [0,1]
    
    为什么需要随机策略？
    Why need stochastic policies?
    - 探索：随机性帮助探索
      Exploration: Randomness helps exploration
    - 鲁棒性：对抗环境的不确定性
      Robustness: Against environment uncertainty
    - 最优性：某些问题的最优策略本身就是随机的
      Optimality: Optimal policy for some problems is inherently stochastic
    """
    
    def __init__(self, name: str = "Policy"):
        """初始化策略"""
        self.name = name
        logger.info(f"初始化策略: {name}")
    
    @abstractmethod
    def get_action_probabilities(self, state: State) -> Dict[Action, float]:
        """
        获取状态下所有动作的概率分布
        Get probability distribution over all actions for a state
        
        Args:
            state: 当前状态
            
        Returns:
            动作概率字典 {action: probability}
        """
        pass
    
    @abstractmethod
    def select_action(self, state: State) -> Action:
        """
        根据策略选择动作
        Select action according to policy
        
        Args:
            state: 当前状态
            
        Returns:
            选择的动作
        """
        pass
    
    def get_probability(self, state: State, action: Action) -> float:
        """
        获取特定状态-动作对的概率
        Get probability for specific state-action pair
        
        π(a|s)
        
        Args:
            state: 状态
            action: 动作
            
        Returns:
            概率值
        """
        probs = self.get_action_probabilities(state)
        return probs.get(action, 0.0)


class DeterministicPolicy(Policy):
    """
    确定性策略
    Deterministic Policy
    
    π: S → A
    
    每个状态只有一个确定的动作
    Each state has only one deterministic action
    """
    
    def __init__(self, policy_map: Dict[State, Action]):
        """
        初始化确定性策略
        
        Args:
            policy_map: 状态到动作的映射
        """
        super().__init__("Deterministic Policy")
        self.policy_map = policy_map
    
    def get_action_probabilities(self, state: State) -> Dict[Action, float]:
        """确定性策略的概率分布是退化的"""
        if state in self.policy_map:
            return {self.policy_map[state]: 1.0}
        return {}
    
    def select_action(self, state: State) -> Action:
        """直接返回映射的动作"""
        if state not in self.policy_map:
            raise ValueError(f"State {state.id} not in policy map")
        return self.policy_map[state]


class StochasticPolicy(Policy):
    """
    随机策略
    Stochastic Policy
    
    π(a|s) ∈ [0,1], Σ_a π(a|s) = 1
    
    每个状态下动作有概率分布
    Actions have probability distribution in each state
    """
    
    def __init__(self, policy_probs: Dict[State, Dict[Action, float]]):
        """
        初始化随机策略
        
        Args:
            policy_probs: 状态到动作概率分布的映射
                         {state: {action: probability}}
        """
        super().__init__("Stochastic Policy")
        self.policy_probs = policy_probs
        
        # 验证概率分布
        self._validate_probabilities()
    
    def _validate_probabilities(self):
        """验证概率分布的合法性"""
        for state, action_probs in self.policy_probs.items():
            total_prob = sum(action_probs.values())
            if not np.isclose(total_prob, 1.0):
                logger.warning(f"State {state.id} probabilities sum to {total_prob}, normalizing")
                # 归一化
                for action in action_probs:
                    action_probs[action] /= total_prob
    
    def get_action_probabilities(self, state: State) -> Dict[Action, float]:
        """返回动作概率分布"""
        if state in self.policy_probs:
            return self.policy_probs[state].copy()
        return {}
    
    def select_action(self, state: State) -> Action:
        """按概率分布采样动作"""
        if state not in self.policy_probs:
            raise ValueError(f"State {state.id} not in policy")
        
        action_probs = self.policy_probs[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        
        # 按概率采样
        return np.random.choice(actions, p=probs)


class UniformRandomPolicy(Policy):
    """
    均匀随机策略
    Uniform Random Policy
    
    π(a|s) = 1/|A(s)|
    
    所有动作等概率，用作基准
    All actions equally likely, used as baseline
    """
    
    def __init__(self, action_space: List[Action]):
        """
        初始化均匀随机策略
        
        Args:
            action_space: 动作空间
        """
        super().__init__("Uniform Random Policy")
        self.action_space = action_space
        self.prob = 1.0 / len(action_space)
    
    def get_action_probabilities(self, state: State) -> Dict[Action, float]:
        """所有动作等概率"""
        return {action: self.prob for action in self.action_space}
    
    def select_action(self, state: State) -> Action:
        """随机选择"""
        return np.random.choice(self.action_space)


# ================================================================================
# 第2.3.2节：价值函数(Value Functions)
# Section 2.3.2: Value Functions
# ================================================================================

class StateValueFunction:
    """
    状态价值函数 V(s)
    State-Value Function V(s)
    
    定义 Definition:
    v_π(s) = E_π[G_t | S_t = s]
           = E_π[Σ_{k=0}^∞ γ^k R_{t+k+1} | S_t = s]
    
    状态价值函数衡量从状态s开始，遵循策略π的期望回报
    State-value function measures expected return starting from state s, following policy π
    
    直观理解 Intuitive Understanding:
    "在这个状态下，如果我遵循策略π，平均能获得多少奖励？"
    "In this state, if I follow policy π, how much reward can I get on average?"
    """
    
    def __init__(self, states: List[State], initial_value: float = 0.0):
        """
        初始化状态价值函数
        
        Args:
            states: 状态空间
            initial_value: 初始价值
        """
        self.V: Dict[State, float] = {s: initial_value for s in states}
        self.states = states
        
        logger.info(f"初始化状态价值函数，{len(states)}个状态")
    
    def get_value(self, state: State) -> float:
        """获取状态价值"""
        return self.V.get(state, 0.0)
    
    def set_value(self, state: State, value: float):
        """设置状态价值"""
        self.V[state] = value
    
    def update_value(self, state: State, delta: float, alpha: float = 1.0):
        """
        增量更新状态价值
        Incrementally update state value
        
        V(s) ← V(s) + α·δ
        
        Args:
            state: 状态
            delta: 更新量
            alpha: 学习率
        """
        old_value = self.get_value(state)
        new_value = old_value + alpha * delta
        self.set_value(state, new_value)
        
        logger.debug(f"更新V({state.id}): {old_value:.3f} -> {new_value:.3f}")
    
    def to_array(self) -> np.ndarray:
        """转换为数组形式"""
        return np.array([self.V[s] for s in self.states])
    
    def from_array(self, values: np.ndarray):
        """从数组加载"""
        for i, state in enumerate(self.states):
            self.V[state] = values[i]
    
    def get_greedy_policy(self, q_function: 'ActionValueFunction') -> DeterministicPolicy:
        """
        从Q函数导出贪婪策略
        Derive greedy policy from Q-function
        
        π(s) = argmax_a Q(s,a)
        
        Args:
            q_function: 动作价值函数
            
        Returns:
            贪婪策略
        """
        policy_map = {}
        
        for state in self.states:
            # 找到最大Q值的动作
            best_action = None
            best_value = float('-inf')
            
            for action in q_function.get_actions(state):
                q_value = q_function.get_value(state, action)
                if q_value > best_value:
                    best_value = q_value
                    best_action = action
            
            if best_action:
                policy_map[state] = best_action
        
        return DeterministicPolicy(policy_map)


class ActionValueFunction:
    """
    动作价值函数 Q(s,a)
    Action-Value Function Q(s,a)
    
    定义 Definition:
    q_π(s,a) = E_π[G_t | S_t = s, A_t = a]
             = E_π[Σ_{k=0}^∞ γ^k R_{t+k+1} | S_t = s, A_t = a]
    
    动作价值函数衡量从状态s执行动作a，然后遵循策略π的期望回报
    Action-value function measures expected return starting from state s, 
    taking action a, then following policy π
    
    直观理解 Intuitive Understanding:
    "在这个状态下执行这个动作，然后遵循策略π，平均能获得多少奖励？"
    "If I take this action in this state, then follow policy π, 
    how much reward can I get on average?"
    
    与V(s)的关系 Relationship with V(s):
    v_π(s) = Σ_a π(a|s)·q_π(s,a)
    q_π(s,a) = r(s,a) + γ·Σ_s' P(s'|s,a)·v_π(s')
    """
    
    def __init__(self, states: List[State], actions: List[Action], 
                 initial_value: float = 0.0):
        """
        初始化动作价值函数
        
        Args:
            states: 状态空间
            actions: 动作空间
            initial_value: 初始价值
        """
        self.Q: Dict[Tuple[State, Action], float] = {}
        
        # 初始化所有状态-动作对
        for s in states:
            for a in actions:
                self.Q[(s, a)] = initial_value
        
        self.states = states
        self.actions = actions
        
        logger.info(f"初始化动作价值函数，{len(states)}×{len(actions)}个状态-动作对")
    
    def get_value(self, state: State, action: Action) -> float:
        """获取动作价值"""
        return self.Q.get((state, action), 0.0)
    
    def set_value(self, state: State, action: Action, value: float):
        """设置动作价值"""
        self.Q[(state, action)] = value
    
    def update_value(self, state: State, action: Action, 
                    delta: float, alpha: float = 1.0):
        """
        增量更新动作价值
        Incrementally update action value
        
        Q(s,a) ← Q(s,a) + α·δ
        
        Args:
            state: 状态
            action: 动作
            delta: 更新量
            alpha: 学习率
        """
        old_value = self.get_value(state, action)
        new_value = old_value + alpha * delta
        self.set_value(state, action, new_value)
        
        logger.debug(f"更新Q({state.id},{action.id}): {old_value:.3f} -> {new_value:.3f}")
    
    def get_actions(self, state: State) -> List[Action]:
        """获取状态下的所有动作"""
        return self.actions
    
    def get_state_values(self, state: State) -> Dict[Action, float]:
        """获取状态下所有动作的价值"""
        return {a: self.get_value(state, a) for a in self.actions}
    
    def get_greedy_action(self, state: State) -> Action:
        """
        获取贪婪动作
        Get greedy action
        
        a* = argmax_a Q(s,a)
        
        Args:
            state: 状态
            
        Returns:
            最佳动作
        """
        action_values = self.get_state_values(state)
        return max(action_values, key=action_values.get)
    
    def get_epsilon_greedy_action(self, state: State, epsilon: float = 0.1) -> Action:
        """
        ε-贪婪动作选择
        ε-greedy action selection
        
        Args:
            state: 状态
            epsilon: 探索率
            
        Returns:
            选择的动作
        """
        if np.random.random() < epsilon:
            # 探索：随机选择
            return np.random.choice(self.actions)
        else:
            # 利用：选择最佳
            return self.get_greedy_action(state)


# ================================================================================
# 第2.3.3节：贝尔曼方程(Bellman Equations)
# Section 2.3.3: Bellman Equations
# ================================================================================

class BellmanEquations:
    """
    贝尔曼方程 - RL的基础方程
    Bellman Equations - Fundamental equations of RL
    
    贝尔曼方程揭示了价值函数的递归结构！
    Bellman equations reveal the recursive structure of value functions!
    
    这是动态规划、蒙特卡洛和TD学习的理论基础
    This is the theoretical foundation of DP, MC, and TD learning
    """
    
    @staticmethod
    def bellman_expectation_v(state: State, 
                             policy: Policy,
                             P: TransitionProbability,
                             R: RewardFunction,
                             V: StateValueFunction,
                             gamma: float = 0.99) -> float:
        """
        贝尔曼期望方程 for V
        Bellman Expectation Equation for V
        
        v_π(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γv_π(s')]
        
        这个方程描述了当前价值与未来价值的关系！
        This equation describes the relationship between current and future values!
        
        Args:
            state: 当前状态
            policy: 策略π
            P: 转移概率
            R: 奖励函数
            V: 当前价值函数估计
            gamma: 折扣因子
            
        Returns:
            状态价值
        """
        value = 0.0
        
        # 遍历所有动作
        action_probs = policy.get_action_probabilities(state)
        
        for action, action_prob in action_probs.items():
            # 计算动作价值
            q_value = 0.0
            
            # 遍历所有可能的转移
            transitions = P.get_transitions(state, action)
            
            for next_state, reward, trans_prob in transitions:
                # 贝尔曼期望方程的核心
                q_value += trans_prob * (reward + gamma * V.get_value(next_state))
            
            # 用策略概率加权
            value += action_prob * q_value
        
        return value
    
    @staticmethod
    def bellman_expectation_q(state: State,
                             action: Action,
                             policy: Policy,
                             P: TransitionProbability,
                             R: RewardFunction,
                             Q: ActionValueFunction,
                             gamma: float = 0.99) -> float:
        """
        贝尔曼期望方程 for Q
        Bellman Expectation Equation for Q
        
        q_π(s,a) = Σ_{s',r} p(s',r|s,a)[r + γΣ_{a'} π(a'|s')q_π(s',a')]
        
        Args:
            state: 当前状态
            action: 当前动作
            policy: 策略π
            P: 转移概率
            R: 奖励函数
            Q: 当前动作价值函数估计
            gamma: 折扣因子
            
        Returns:
            动作价值
        """
        value = 0.0
        
        # 遍历所有可能的转移
        transitions = P.get_transitions(state, action)
        
        for next_state, reward, trans_prob in transitions:
            # 计算下一状态的价值
            next_value = 0.0
            next_action_probs = policy.get_action_probabilities(next_state)
            
            for next_action, next_action_prob in next_action_probs.items():
                next_value += next_action_prob * Q.get_value(next_state, next_action)
            
            # 贝尔曼期望方程的核心
            value += trans_prob * (reward + gamma * next_value)
        
        return value
    
    @staticmethod
    def bellman_optimality_v(state: State,
                            P: TransitionProbability,
                            R: RewardFunction,
                            V: StateValueFunction,
                            action_space: List[Action],
                            gamma: float = 0.99) -> float:
        """
        贝尔曼最优方程 for V
        Bellman Optimality Equation for V
        
        v*(s) = max_a Σ_{s',r} p(s',r|s,a)[r + γv*(s')]
        
        最优价值函数满足这个方程！
        Optimal value function satisfies this equation!
        
        Args:
            state: 当前状态
            P: 转移概率
            R: 奖励函数
            V: 当前价值函数估计
            action_space: 动作空间
            gamma: 折扣因子
            
        Returns:
            最优状态价值
        """
        max_value = float('-inf')
        
        # 遍历所有动作，找最大值
        for action in action_space:
            q_value = 0.0
            
            # 计算动作价值
            transitions = P.get_transitions(state, action)
            
            for next_state, reward, trans_prob in transitions:
                q_value += trans_prob * (reward + gamma * V.get_value(next_state))
            
            max_value = max(max_value, q_value)
        
        return max_value
    
    @staticmethod
    def bellman_optimality_q(state: State,
                            action: Action,
                            P: TransitionProbability,
                            R: RewardFunction,
                            Q: ActionValueFunction,
                            gamma: float = 0.99) -> float:
        """
        贝尔曼最优方程 for Q
        Bellman Optimality Equation for Q
        
        q*(s,a) = Σ_{s',r} p(s',r|s,a)[r + γmax_{a'} q*(s',a')]
        
        Args:
            state: 当前状态
            action: 当前动作
            P: 转移概率
            R: 奖励函数
            Q: 当前动作价值函数估计
            gamma: 折扣因子
            
        Returns:
            最优动作价值
        """
        value = 0.0
        
        # 遍历所有可能的转移
        transitions = P.get_transitions(state, action)
        
        for next_state, reward, trans_prob in transitions:
            # 找下一状态的最大动作价值
            next_action_values = Q.get_state_values(next_state)
            max_next_value = max(next_action_values.values()) if next_action_values else 0
            
            # 贝尔曼最优方程的核心
            value += trans_prob * (reward + gamma * max_next_value)
        
        return value


# ================================================================================
# 第2.3.4节：策略评估与改进
# Section 2.3.4: Policy Evaluation and Improvement
# ================================================================================

class PolicyEvaluation:
    """
    策略评估 - 计算给定策略的价值函数
    Policy Evaluation - Compute value function for given policy
    
    这是策略迭代算法的第一步！
    This is the first step of policy iteration!
    
    通过反复应用贝尔曼期望方程来计算v_π
    Compute v_π by repeatedly applying Bellman expectation equation
    """
    
    @staticmethod
    def evaluate_policy(policy: Policy,
                       environment: MDPEnvironment,
                       gamma: float = 0.99,
                       theta: float = 1e-6,
                       max_iterations: int = 1000) -> StateValueFunction:
        """
        迭代策略评估
        Iterative Policy Evaluation
        
        算法 Algorithm:
        1. 初始化V(s)任意
        2. 重复直到收敛：
           对每个状态s：
             v ← Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γV(s')]
             V(s) ← v
        
        Args:
            policy: 要评估的策略
            environment: MDP环境
            gamma: 折扣因子
            theta: 收敛阈值
            max_iterations: 最大迭代次数
            
        Returns:
            策略的状态价值函数
        """
        # 初始化价值函数
        V = StateValueFunction(environment.state_space, initial_value=0.0)
        
        P, R = environment.get_dynamics()
        
        logger.info("开始策略评估...")
        
        for iteration in range(max_iterations):
            delta = 0  # 最大价值变化
            
            # 对每个状态更新
            for state in environment.state_space:
                if state.is_terminal:
                    continue
                
                # 保存旧值
                old_value = V.get_value(state)
                
                # 应用贝尔曼期望方程
                new_value = BellmanEquations.bellman_expectation_v(
                    state, policy, P, R, V, gamma
                )
                
                # 更新价值
                V.set_value(state, new_value)
                
                # 跟踪最大变化
                delta = max(delta, abs(old_value - new_value))
            
            # 检查收敛
            if delta < theta:
                logger.info(f"策略评估收敛于第{iteration+1}次迭代，delta={delta:.6f}")
                break
            
            if (iteration + 1) % 10 == 0:
                logger.debug(f"迭代{iteration+1}: delta={delta:.6f}")
        
        return V


class PolicyImprovement:
    """
    策略改进 - 基于价值函数改进策略
    Policy Improvement - Improve policy based on value function
    
    这是策略迭代算法的第二步！
    This is the second step of policy iteration!
    
    通过贪婪化价值函数来改进策略
    Improve policy by being greedy with respect to value function
    """
    
    @staticmethod
    def improve_policy(V: StateValueFunction,
                       environment: MDPEnvironment,
                       gamma: float = 0.99) -> Tuple[Policy, bool]:
        """
        策略改进
        Policy Improvement
        
        π'(s) = argmax_a Σ_{s',r} p(s',r|s,a)[r + γV(s')]
        
        定理：如果π'≠π，则π'严格优于π
        Theorem: If π'≠π, then π' is strictly better than π
        
        Args:
            V: 当前策略的价值函数
            environment: MDP环境
            gamma: 折扣因子
            
        Returns:
            (改进的策略, 是否改变)
        """
        P, R = environment.get_dynamics()
        policy_map = {}
        policy_changed = False
        
        logger.info("开始策略改进...")
        
        for state in environment.state_space:
            if state.is_terminal:
                continue
            
            # 找最佳动作
            best_action = None
            best_value = float('-inf')
            
            for action in environment.get_action_space(state):
                # 计算动作价值
                q_value = 0.0
                transitions = P.get_transitions(state, action)
                
                for next_state, reward, trans_prob in transitions:
                    q_value += trans_prob * (reward + gamma * V.get_value(next_state))
                
                if q_value > best_value:
                    best_value = q_value
                    best_action = action
            
            if best_action:
                policy_map[state] = best_action
        
        # 创建新策略
        new_policy = DeterministicPolicy(policy_map)
        
        return new_policy, policy_changed


# ================================================================================
# 第2.3.5节：可视化工具
# Section 2.3.5: Visualization Tools
# ================================================================================

class ValueFunctionVisualizer:
    """
    价值函数可视化工具
    Value Function Visualization Tool
    
    可视化帮助理解算法行为！
    Visualization helps understand algorithm behavior!
    """
    
    @staticmethod
    def plot_state_values(V: StateValueFunction, 
                         title: str = "State Values"):
        """
        绘制状态价值函数
        Plot state value function
        
        Args:
            V: 状态价值函数
            title: 图标题
        """
        states = list(V.V.keys())
        values = list(V.V.values())
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制条形图
        x_pos = range(len(states))
        bars = ax.bar(x_pos, values, alpha=0.7)
        
        # 根据值着色
        norm = plt.Normalize(min(values), max(values))
        colors = plt.cm.coolwarm(norm(values))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # 设置标签
        ax.set_xlabel('States')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([s.id for s in states], rotation=45)
        
        # 添加数值标签
        for i, v in enumerate(values):
            ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')
        
        # 添加网格
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_action_values(Q: ActionValueFunction,
                          title: str = "Action Values"):
        """
        绘制动作价值函数
        Plot action value function
        
        Args:
            Q: 动作价值函数
            title: 图标题
        """
        # 整理数据
        data = []
        state_labels = []
        action_labels = set()
        
        for state in Q.states:
            state_values = Q.get_state_values(state)
            data.append(list(state_values.values()))
            state_labels.append(state.id)
            action_labels.update([a.id for a in state_values.keys()])
        
        data = np.array(data)
        action_labels = sorted(list(action_labels))
        
        # 创建热力图
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(data, cmap='coolwarm', aspect='auto')
        
        # 设置标签
        ax.set_xticks(range(len(action_labels)))
        ax.set_yticks(range(len(state_labels)))
        ax.set_xticklabels(action_labels)
        ax.set_yticklabels(state_labels)
        
        ax.set_xlabel('Actions')
        ax.set_ylabel('States')
        ax.set_title(title)
        
        # 添加数值
        for i in range(len(state_labels)):
            for j in range(len(action_labels)):
                text = ax.text(j, i, f'{data[i, j]:.2f}',
                             ha="center", va="center", color="black")
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, label='Q-value')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_policy(policy: Policy,
                   states: List[State],
                   actions: List[Action],
                   title: str = "Policy"):
        """
        绘制策略
        Plot policy
        
        Args:
            policy: 策略
            states: 状态列表
            actions: 动作列表
            title: 图标题
        """
        # 整理数据
        data = []
        state_labels = []
        action_labels = [a.id for a in actions]
        
        for state in states:
            probs = policy.get_action_probabilities(state)
            row = [probs.get(a, 0.0) for a in actions]
            data.append(row)
            state_labels.append(state.id)
        
        data = np.array(data)
        
        # 创建热力图
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        # 设置标签
        ax.set_xticks(range(len(action_labels)))
        ax.set_yticks(range(len(state_labels)))
        ax.set_xticklabels(action_labels)
        ax.set_yticklabels(state_labels)
        
        ax.set_xlabel('Actions')
        ax.set_ylabel('States')
        ax.set_title(title)
        
        # 添加数值
        for i in range(len(state_labels)):
            for j in range(len(action_labels)):
                if data[i, j] > 0:
                    text = ax.text(j, i, f'{data[i, j]:.2f}',
                                 ha="center", va="center", 
                                 color="white" if data[i, j] > 0.5 else "black")
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, label='π(a|s)')
        
        plt.tight_layout()
        return fig


# ================================================================================
# 示例：简单网格世界
# Example: Simple Grid World
# ================================================================================

def demonstrate_policies_and_values():
    """
    演示策略和价值函数
    Demonstrate policies and value functions
    """
    print("\n" + "="*80)
    print("策略与价值函数演示")
    print("Policies and Value Functions Demonstration")
    print("="*80)
    
    # 导入回收机器人
    from .mdp_framework import RecyclingRobot
    
    # 创建环境
    env = RecyclingRobot()
    
    # 创建简单策略
    print("\n1. 创建策略")
    print("1. Create Policies")
    print("-" * 40)
    
    # 均匀随机策略
    random_policy = UniformRandomPolicy(env.action_space)
    print("均匀随机策略 Uniform Random Policy:")
    for state in env.state_space:
        probs = random_policy.get_action_probabilities(state)
        print(f"  State {state.id}: {probs}")
    
    # 确定性策略（总是搜索）
    from .mdp_framework import Action
    search_action = Action(id='search', name='搜索垃圾')
    wait_action = Action(id='wait', name='等待')
    recharge_action = Action(id='recharge', name='充电')
    
    policy_map = {
        env.state_space[0]: search_action,  # high -> search
        env.state_space[1]: recharge_action  # low -> recharge
    }
    deterministic_policy = DeterministicPolicy(policy_map)
    print("\n确定性策略 Deterministic Policy:")
    for state in env.state_space:
        if state in policy_map:
            action = deterministic_policy.select_action(state)
            print(f"  State {state.id}: {action.name}")
    
    # 评估策略
    print("\n2. 策略评估")
    print("2. Policy Evaluation")
    print("-" * 40)
    
    # 评估随机策略
    print("评估随机策略...")
    V_random = PolicyEvaluation.evaluate_policy(
        random_policy, env, gamma=0.9, theta=1e-4
    )
    
    print("随机策略的状态价值 State values for random policy:")
    for state in env.state_space:
        print(f"  V({state.id}) = {V_random.get_value(state):.3f}")
    
    # 评估确定性策略
    print("\n评估确定性策略...")
    V_det = PolicyEvaluation.evaluate_policy(
        deterministic_policy, env, gamma=0.9, theta=1e-4
    )
    
    print("确定性策略的状态价值 State values for deterministic policy:")
    for state in env.state_space:
        print(f"  V({state.id}) = {V_det.get_value(state):.3f}")
    
    # 比较
    print("\n价值比较 Value Comparison:")
    for state in env.state_space:
        diff = V_det.get_value(state) - V_random.get_value(state)
        print(f"  State {state.id}: 确定性 - 随机 = {diff:.3f}")
    
    # 可视化
    print("\n3. 可视化")
    print("3. Visualization")
    print("-" * 40)
    
    visualizer = ValueFunctionVisualizer()
    
    # 绘制状态价值
    fig1 = visualizer.plot_state_values(V_random, "Random Policy Values")
    fig2 = visualizer.plot_state_values(V_det, "Deterministic Policy Values")
    
    # 绘制策略
    fig3 = visualizer.plot_policy(
        random_policy, env.state_space, env.action_space[:2],
        "Random Policy"
    )
    
    return [fig1, fig2, fig3]


def main():
    """主函数"""
    print("\n" + "="*80)
    print("第2.3节：策略与价值函数")
    print("Section 2.3: Policies and Value Functions")
    print("="*80)
    
    # 运行演示
    figs = demonstrate_policies_and_values()
    
    print("\n" + "="*80)
    print("策略与价值函数演示完成！")
    print("Policies and Value Functions Demo Complete!")
    print("\n关键要点：")
    print("Key Takeaways:")
    print("1. 策略定义行为：π(a|s)")
    print("2. 价值函数评估好坏：V(s), Q(s,a)")
    print("3. 贝尔曼方程连接当前与未来")
    print("4. 策略评估计算v_π，策略改进找更好的π")
    print("="*80)
    
    plt.show()
    
    return figs


if __name__ == "__main__":
    main()