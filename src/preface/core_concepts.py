"""
Preface - Core Concepts of Reinforcement Learning
前言 - 强化学习核心概念

This module implements the fundamental concepts introduced in the preface of
Sutton & Barto's "Reinforcement Learning: An Introduction"

本模块实现了Sutton & Barto《强化学习导论》前言中介绍的基本概念
"""

import numpy as np
from typing import Tuple, Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

# Configure logging / 配置日志
logger = logging.getLogger(__name__)


class RLElement(Enum):
    """
    Basic elements of Reinforcement Learning
    强化学习的基本元素
    """
    AGENT = "agent"  # The learner and decision maker / 学习者和决策者
    ENVIRONMENT = "environment"  # What the agent interacts with / 智能体交互的对象
    STATE = "state"  # Situation the agent finds itself in / 智能体所处的情况
    ACTION = "action"  # What the agent can do / 智能体可以执行的操作
    REWARD = "reward"  # Scalar feedback signal / 标量反馈信号
    POLICY = "policy"  # Mapping from states to actions / 从状态到动作的映射
    VALUE = "value"  # Expected future reward / 期望的未来奖励
    MODEL = "model"  # Agent's representation of environment / 智能体对环境的表示


@dataclass
class Experience:
    """
    Single step experience in RL
    强化学习中的单步经验
    
    Attributes:
        state: Current state S_t / 当前状态
        action: Action taken A_t / 执行的动作
        reward: Reward received R_{t+1} / 收到的奖励
        next_state: Next state S_{t+1} / 下一个状态
        done: Whether episode ended / 回合是否结束
    """
    state: np.ndarray  # State at time t / t时刻的状态
    action: int  # Action taken at time t / t时刻采取的动作
    reward: float  # Reward received at t+1 / t+1时刻收到的奖励
    next_state: np.ndarray  # State at time t+1 / t+1时刻的状态
    done: bool  # Terminal flag / 终止标志
    
    # Additional metadata / 额外元数据
    info: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return (f"Experience(s={self.state.tolist()}, a={self.action}, "
                f"r={self.reward:.2f}, s'={self.next_state.tolist()}, "
                f"done={self.done})")


class Agent:
    """
    Base class for RL agents
    强化学习智能体基类
    
    This class defines the interface all agents must implement
    这个类定义了所有智能体必须实现的接口
    """
    
    def __init__(self, n_states: int, n_actions: int, 
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.9):
        """
        Initialize agent
        初始化智能体
        
        Args:
            n_states: Number of states / 状态数量
            n_actions: Number of actions / 动作数量
            learning_rate: Learning rate α / 学习率
            discount_factor: Discount factor γ / 折扣因子
        """
        self.n_states = n_states  # State space size / 状态空间大小
        self.n_actions = n_actions  # Action space size / 动作空间大小
        self.alpha = learning_rate  # Step size / 步长
        self.gamma = discount_factor  # Discount for future rewards / 未来奖励折扣
        
        # Initialize value function V(s) to zeros
        # 初始化价值函数V(s)为零
        self.V = np.zeros(n_states)  # State value function / 状态价值函数
        
        # Initialize action-value function Q(s,a) to zeros  
        # 初始化动作价值函数Q(s,a)为零
        self.Q = np.zeros((n_states, n_actions))  # Action value function / 动作价值函数
        
        logger.info(f"Agent initialized: {n_states} states, {n_actions} actions")
        logger.info(f"Learning rate α={self.alpha}, Discount γ={self.gamma}")
    
    def select_action(self, state: int, epsilon: float = 0.1) -> int:
        """
        Select action using ε-greedy policy
        使用ε-贪婪策略选择动作
        
        With probability ε: explore (random action)
        With probability 1-ε: exploit (greedy action)
        
        概率ε：探索（随机动作）
        概率1-ε：利用（贪婪动作）
        
        Args:
            state: Current state / 当前状态
            epsilon: Exploration rate / 探索率
            
        Returns:
            Selected action / 选择的动作
        """
        # Exploration-exploitation tradeoff / 探索-利用权衡
        if np.random.random() < epsilon:
            # Explore: random action / 探索：随机动作
            action = np.random.randint(self.n_actions)
            logger.debug(f"Exploring: random action {action}")
        else:
            # Exploit: greedy action based on Q-values / 利用：基于Q值的贪婪动作
            action = np.argmax(self.Q[state])
            logger.debug(f"Exploiting: greedy action {action} "
                        f"(Q={self.Q[state, action]:.3f})")
        
        return action
    
    def update_value(self, state: int, reward: float, next_state: int):
        """
        Update state value function using TD(0) learning
        使用TD(0)学习更新状态价值函数
        
        V(S_t) ← V(S_t) + α[R_{t+1} + γV(S_{t+1}) - V(S_t)]
        
        This is the fundamental TD learning rule
        这是基本的TD学习规则
        
        Args:
            state: Current state S_t / 当前状态
            reward: Received reward R_{t+1} / 收到的奖励
            next_state: Next state S_{t+1} / 下一个状态
        """
        # TD error: δ = R_{t+1} + γV(S_{t+1}) - V(S_t)
        # TD误差：时序差分误差
        td_error = reward + self.gamma * self.V[next_state] - self.V[state]
        
        # Update value: V(S_t) ← V(S_t) + αδ
        # 更新价值：使用TD误差更新
        self.V[state] += self.alpha * td_error
        
        logger.debug(f"Updated V[{state}]: {self.V[state]:.3f} "
                    f"(TD error: {td_error:.3f})")
    
    def update_q_value(self, state: int, action: int, 
                       reward: float, next_state: int):
        """
        Update action-value function using Q-learning (off-policy TD)
        使用Q学习（离策略TD）更新动作价值函数
        
        Q(S_t,A_t) ← Q(S_t,A_t) + α[R_{t+1} + γ max_a Q(S_{t+1},a) - Q(S_t,A_t)]
        
        This is the Q-learning update rule
        这是Q学习更新规则
        
        Args:
            state: Current state S_t / 当前状态
            action: Action taken A_t / 采取的动作
            reward: Received reward R_{t+1} / 收到的奖励
            next_state: Next state S_{t+1} / 下一个状态
        """
        # Current Q-value / 当前Q值
        current_q = self.Q[state, action]
        
        # Maximum Q-value for next state / 下一状态的最大Q值
        max_next_q = np.max(self.Q[next_state])
        
        # TD target: R_{t+1} + γ max_a Q(S_{t+1},a)
        # TD目标：即时奖励加上折扣的未来最大Q值
        td_target = reward + self.gamma * max_next_q
        
        # TD error: δ = TD_target - current_Q
        # TD误差：目标值与当前值的差
        td_error = td_target - current_q
        
        # Update Q-value: Q(S_t,A_t) ← Q(S_t,A_t) + αδ
        # 更新Q值：使用学习率调整
        self.Q[state, action] += self.alpha * td_error
        
        logger.debug(f"Updated Q[{state},{action}]: {self.Q[state, action]:.3f} "
                    f"(TD error: {td_error:.3f})")


class Environment:
    """
    Base class for RL environments
    强化学习环境基类
    
    Defines the interface for environments following the agent-environment loop
    定义遵循智能体-环境循环的环境接口
    """
    
    def __init__(self, n_states: int, n_actions: int):
        """
        Initialize environment
        初始化环境
        
        Args:
            n_states: Number of states / 状态数量
            n_actions: Number of actions / 动作数量
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.current_state = None
        self.step_count = 0
        
        logger.info(f"Environment initialized: {n_states} states, {n_actions} actions")
    
    def reset(self) -> int:
        """
        Reset environment to initial state
        重置环境到初始状态
        
        Returns:
            Initial state / 初始状态
        """
        self.step_count = 0
        self.current_state = self._get_initial_state()
        logger.debug(f"Environment reset to state {self.current_state}")
        return self.current_state
    
    def step(self, action: int) -> Tuple[int, float, bool, Dict]:  # noqa: F821
        """
        Execute one time step in the environment
        在环境中执行一个时间步
        
        This implements the core agent-environment interaction
        这实现了核心的智能体-环境交互
        
        Args:
            action: Action to execute / 要执行的动作
            
        Returns:
            Tuple of (next_state, reward, done, info)
            元组：(下一状态，奖励，是否结束，信息)
        """
        # Validate action / 验证动作
        assert 0 <= action < self.n_actions, f"Invalid action: {action}"
        
        # Execute action and get outcome / 执行动作并获得结果
        next_state, reward, done = self._transition(self.current_state, action)
        
        # Update state and step count / 更新状态和步数
        self.current_state = next_state
        self.step_count += 1
        
        # Additional information / 额外信息
        info = {
            'step_count': self.step_count,
            'state': self.current_state
        }
        
        logger.debug(f"Step {self.step_count}: "
                    f"action={action}, reward={reward:.2f}, "
                    f"next_state={next_state}, done={done}")
        
        return next_state, reward, done, info
    
    def _get_initial_state(self) -> int:
        """
        Get initial state (to be implemented by subclasses)
        获取初始状态（由子类实现）
        """
        raise NotImplementedError
    
    def _transition(self, state: int, action: int) -> Tuple[int, float, bool]:
        """
        State transition function (to be implemented by subclasses)
        状态转移函数（由子类实现）
        
        Args:
            state: Current state / 当前状态
            action: Action taken / 采取的动作
            
        Returns:
            Tuple of (next_state, reward, done)
            元组：(下一状态，奖励，是否结束)
        """
        raise NotImplementedError


def demonstrate_rl_loop(agent: Agent, env: Environment, 
                        n_episodes: int = 100,
                        epsilon: float = 0.1) -> List[float]:  # noqa: F821
    """
    Demonstrate the fundamental RL loop
    演示基本的强化学习循环
    
    This shows the core agent-environment interaction:
    1. Agent observes state
    2. Agent selects action
    3. Environment returns reward and next state
    4. Agent learns from experience
    5. Repeat until episode ends
    
    这展示了核心的智能体-环境交互：
    1. 智能体观察状态
    2. 智能体选择动作
    3. 环境返回奖励和下一状态
    4. 智能体从经验中学习
    5. 重复直到回合结束
    
    Args:
        agent: RL agent / 强化学习智能体
        env: Environment / 环境
        n_episodes: Number of episodes / 回合数
        epsilon: Exploration rate / 探索率
        
    Returns:
        List of episode rewards / 回合奖励列表
    """
    episode_rewards = []  # Track rewards for each episode / 跟踪每回合奖励
    
    for episode in range(n_episodes):
        # Initialize episode / 初始化回合
        state = env.reset()
        total_reward = 0
        done = False
        
        # Run episode until termination / 运行回合直到终止
        while not done:
            # Agent selects action / 智能体选择动作
            action = agent.select_action(state, epsilon)
            
            # Environment responds / 环境响应
            next_state, reward, done, info = env.step(action)
            
            # Agent learns from experience / 智能体从经验中学习
            agent.update_q_value(state, action, reward, next_state)
            
            # Track reward and update state / 跟踪奖励并更新状态
            total_reward += reward
            state = next_state
        
        # Record episode reward / 记录回合奖励
        episode_rewards.append(total_reward)
        
        # Log progress periodically / 定期记录进度
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            logger.info(f"Episode {episode + 1}/{n_episodes}: "
                       f"Avg reward = {avg_reward:.2f}")
    
    return episode_rewards


class RewardHypothesis:
    """
    Demonstrate the Reward Hypothesis
    演示奖励假设
    
    The reward hypothesis states that:
    "All of what we mean by goals and purposes can be well thought of as
    the maximization of the expected value of the cumulative sum of a 
    received scalar signal (called reward)"
    
    奖励假设表明：
    "我们所说的目标和目的都可以被很好地理解为
    最大化接收到的标量信号（称为奖励）累积和的期望值"
    """
    
    @staticmethod
    def demonstrate_goal_as_reward_maximization():
        """
        Show how different goals can be expressed as reward functions
        展示如何将不同目标表达为奖励函数
        """
        examples = {
            "Game Playing / 游戏": {
                "goal": "Win the game / 赢得游戏",
                "reward": "+1 for win, -1 for loss, 0 for draw / 赢+1，输-1，平0"
            },
            "Robot Control / 机器人控制": {
                "goal": "Walk forward / 向前行走",
                "reward": "+distance moved forward / +前进的距离"
            },
            "Resource Management / 资源管理": {
                "goal": "Minimize cost / 最小化成本",
                "reward": "-cost incurred / -产生的成本"
            },
            "Learning / 学习": {
                "goal": "Answer correctly / 正确回答",
                "reward": "+1 for correct, -1 for incorrect / 正确+1，错误-1"
            }
        }
        
        logger.info("=" * 60)
        logger.info("Reward Hypothesis Examples / 奖励假设示例")
        logger.info("=" * 60)
        
        for domain, details in examples.items():
            logger.info(f"\nDomain / 领域: {domain}")
            logger.info(f"  Goal / 目标: {details['goal']}")
            logger.info(f"  Reward / 奖励: {details['reward']}")
        
        return examples


def compare_learning_paradigms():
    """
    Compare RL with supervised and unsupervised learning
    比较强化学习与监督学习和无监督学习
    
    This highlights the unique aspects of RL
    这突出了强化学习的独特方面
    """
    comparisons = {
        "Supervised Learning / 监督学习": {
            "feedback": "Correct labels / 正确标签",
            "goal": "Minimize prediction error / 最小化预测误差",
            "examples": "Classification, Regression / 分类，回归"
        },
        "Unsupervised Learning / 无监督学习": {
            "feedback": "None / 无",
            "goal": "Find hidden structure / 发现隐藏结构",
            "examples": "Clustering, Dimensionality reduction / 聚类，降维"
        },
        "Reinforcement Learning / 强化学习": {
            "feedback": "Reward signal / 奖励信号",
            "goal": "Maximize cumulative reward / 最大化累积奖励",
            "examples": "Game playing, Robot control / 游戏，机器人控制"
        }
    }
    
    logger.info("=" * 60)
    logger.info("Learning Paradigm Comparison / 学习范式比较")
    logger.info("=" * 60)
    
    for paradigm, details in comparisons.items():
        logger.info(f"\n{paradigm}:")
        for key, value in details.items():
            logger.info(f"  {key}: {value}")
    
    # Key differences of RL / 强化学习的关键差异
    logger.info("\n" + "=" * 60)
    logger.info("Unique Aspects of RL / 强化学习的独特方面:")
    logger.info("=" * 60)
    
    unique_aspects = [
        "1. Trial-and-error search / 试错搜索",
        "2. Delayed reward / 延迟奖励", 
        "3. Need to balance exploration and exploitation / 需要平衡探索和利用",
        "4. Actions affect subsequent data / 动作影响后续数据",
        "5. Sequential decision making / 序列决策"
    ]
    
    for aspect in unique_aspects:
        logger.info(f"  {aspect}")
    
    return comparisons