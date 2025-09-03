"""
================================================================================
第2.1节：MDP框架 - 强化学习的数学基础
Section 2.1: MDP Framework - Mathematical Foundation of RL
================================================================================

马尔可夫决策过程(MDP)将第1章的单状态问题扩展到多状态序列决策问题
MDP extends Chapter 1's single-state problem to multi-state sequential decision problems

核心升级 Core Upgrades:
1. 状态(State): 从无状态 → 环境状态空间 S
2. 转移(Transition): 从立即奖励 → 状态转移概率 P
3. 策略(Policy): 从动作选择 → 状态到动作的映射 π
4. 价值(Value): 从Q(a) → V(s)和Q(s,a)

这一章是整个强化学习的理论基石！
This chapter is the theoretical cornerstone of all RL!
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod

# 设置日志
logger = logging.getLogger(__name__)


# ================================================================================
# 第2.1.1节：MDP的形式化定义
# Section 2.1.1: Formal Definition of MDP
# ================================================================================

@dataclass
class State:
    """
    状态 - 环境的完整描述
    State - Complete description of the environment
    
    在MDP中，状态必须满足马尔可夫性质：
    In MDP, state must satisfy Markov property:
    
    P[S_{t+1} | S_t] = P[S_{t+1} | S_1, S_2, ..., S_t]
    
    即：未来只依赖于现在，不依赖于过去
    i.e., The future depends only on the present, not on the past
    
    生活类比 Life Analogy:
    就像下棋，当前棋盘状态包含了所有决策所需信息，
    不需要知道之前每一步是怎么走的
    Like playing chess, the current board state contains all information needed for decisions,
    no need to know how each previous move was made
    """
    
    # 状态标识符
    id: Union[int, str, Tuple]
    
    # 状态特征（可选，用于函数近似）
    features: Optional[np.ndarray] = None
    
    # 是否为终止状态
    is_terminal: bool = False
    
    # 额外信息
    info: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        """使状态可哈希，用作字典键"""
        if isinstance(self.id, (list, np.ndarray)):
            return hash(tuple(self.id))
        return hash(self.id)
    
    def __eq__(self, other):
        """状态相等性比较"""
        if not isinstance(other, State):
            return False
        return self.id == other.id


@dataclass
class Action:
    """
    动作 - 智能体可以执行的操作
    Action - Operations that agent can execute
    
    动作空间可以是：
    Action space can be:
    - 离散的：如{上,下,左,右}
      Discrete: e.g., {up, down, left, right}
    - 连续的：如机器人关节角度
      Continuous: e.g., robot joint angles
    
    本章主要关注离散动作空间
    This chapter focuses on discrete action spaces
    """
    
    # 动作标识符
    id: Union[int, str, Tuple]
    
    # 动作参数（用于连续动作）
    parameters: Optional[np.ndarray] = None
    
    # 动作名称（便于理解）
    name: str = ""
    
    # 额外信息
    info: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        """使动作可哈希"""
        if isinstance(self.id, (list, np.ndarray)):
            return hash(tuple(self.id))
        return hash(self.id)
    
    def __eq__(self, other):
        """动作相等性比较"""
        if not isinstance(other, Action):
            return False
        return self.id == other.id


class PolicyType(Enum):
    """
    策略类型
    Policy Types
    
    强化学习中的策略分类
    Policy categories in RL
    """
    DETERMINISTIC = "deterministic"  # 确定性策略 π: S → A
    STOCHASTIC = "stochastic"        # 随机策略 π: S × A → [0,1]
    EPSILON_GREEDY = "epsilon_greedy"  # ε-贪婪策略
    SOFTMAX = "softmax"              # 基于softmax的策略


# ================================================================================
# 第2.1.2节：MDP的核心组件
# Section 2.1.2: Core Components of MDP
# ================================================================================

class TransitionProbability:
    """
    状态转移概率函数 P
    State Transition Probability Function P
    
    数学定义 Mathematical Definition:
    P(s', r | s, a) = Pr{S_{t+1}=s', R_{t+1}=r | S_t=s, A_t=a}
    
    这定义了MDP的动态特性！
    This defines the dynamics of MDP!
    
    性质 Properties:
    1. 归一化：Σ_{s',r} P(s',r|s,a) = 1, ∀s,a
       Normalization: Sum over all s',r equals 1
    
    2. 马尔可夫性：只依赖当前状态和动作
       Markov property: Depends only on current state and action
    
    深入理解 Deep Understanding:
    转移概率完全描述了环境的行为。如果知道P，
    就可以完美预测环境的反应（虽然可能是随机的）
    Transition probability fully describes environment behavior.
    If we know P, we can perfectly predict environment response (though possibly stochastic)
    """
    
    def __init__(self):
        """初始化转移概率"""
        # 存储格式：P[s][a] = [(s', r, prob), ...]
        # Storage format: P[s][a] = [(s', r, prob), ...]
        self.P: Dict[State, Dict[Action, List[Tuple[State, float, float]]]] = {}
        
        logger.info("初始化转移概率函数")
    
    def set_probability(self, s: State, a: Action, 
                       s_prime: State, r: float, prob: float):
        """
        设置转移概率
        Set transition probability
        
        Args:
            s: 当前状态 Current state
            a: 执行动作 Action taken
            s_prime: 下一状态 Next state
            r: 获得奖励 Reward received
            prob: 转移概率 Transition probability
        """
        if s not in self.P:
            self.P[s] = {}
        if a not in self.P[s]:
            self.P[s][a] = []
        
        self.P[s][a].append((s_prime, r, prob))
        
        logger.debug(f"设置P(s'={s_prime.id}, r={r} | s={s.id}, a={a.id}) = {prob}")
    
    def get_transitions(self, s: State, a: Action) -> List[Tuple[State, float, float]]:
        """
        获取给定(s,a)的所有可能转移
        Get all possible transitions for given (s,a)
        
        Returns:
            [(下一状态, 奖励, 概率), ...]
            [(next_state, reward, probability), ...]
        """
        if s in self.P and a in self.P[s]:
            return self.P[s][a]
        return []
    
    def sample(self, s: State, a: Action) -> Tuple[State, float]:
        """
        根据转移概率采样下一状态和奖励
        Sample next state and reward according to transition probability
        
        这模拟了环境的随机性！
        This simulates environment stochasticity!
        
        Returns:
            (下一状态, 奖励)
            (next_state, reward)
        """
        transitions = self.get_transitions(s, a)
        if not transitions:
            raise ValueError(f"No transitions defined for state {s.id}, action {a.id}")
        
        # 提取概率分布
        probs = [p for _, _, p in transitions]
        probs = np.array(probs) / np.sum(probs)  # 归一化
        
        # 按概率采样
        idx = np.random.choice(len(transitions), p=probs)
        s_prime, r, _ = transitions[idx]
        
        return s_prime, r


class RewardFunction:
    """
    奖励函数 R
    Reward Function R
    
    数学定义 Mathematical Definition:
    1. 四参数形式：r(s, a, s')
       Four-parameter form: r(s, a, s')
    
    2. 期望奖励：r(s, a) = E[R_{t+1} | S_t=s, A_t=a]
       Expected reward: r(s, a) = E[R_{t+1} | S_t=s, A_t=a]
    
    奖励假设 Reward Hypothesis:
    "我们可以认为，所有的目标和目的都可以被描述为
    期望累积奖励总和的最大化"
    "That all of what we mean by goals and purposes can be well thought of as
    the maximization of the expected value of the cumulative sum of rewards"
    
    这是强化学习的核心假设！
    This is the core hypothesis of RL!
    """
    
    def __init__(self, reward_type: str = "deterministic"):
        """
        初始化奖励函数
        
        Args:
            reward_type: "deterministic" 或 "stochastic"
        """
        self.reward_type = reward_type
        # 存储格式取决于类型
        self.R: Dict = {}
        
        logger.info(f"初始化{reward_type}奖励函数")
    
    def set_reward(self, s: State, a: Action, 
                  s_prime: Optional[State] = None,
                  reward: Union[float, Callable] = 0.0):
        """
        设置奖励
        Set reward
        
        支持多种奖励定义方式：
        Supports multiple reward definition methods:
        1. r(s,a,s'): 最一般形式
        2. r(s,a): 期望奖励
        3. r(s): 状态奖励
        """
        key = (s, a, s_prime) if s_prime else (s, a)
        self.R[key] = reward
    
    def get_reward(self, s: State, a: Action, 
                  s_prime: Optional[State] = None) -> float:
        """
        获取奖励
        Get reward
        
        Returns:
            奖励值
            Reward value
        """
        # 尝试不同的键组合
        keys = [
            (s, a, s_prime) if s_prime else None,
            (s, a),
            (s,)
        ]
        
        for key in keys:
            if key and key in self.R:
                reward = self.R[key]
                # 如果是函数，调用它
                if callable(reward):
                    return reward(s, a, s_prime)
                return reward
        
        # 默认奖励
        return 0.0


# ================================================================================
# 第2.1.3节：MDP环境基类
# Section 2.1.3: MDP Environment Base Class
# ================================================================================

class MDPEnvironment(ABC):
    """
    MDP环境基类
    MDP Environment Base Class
    
    这是智能体交互的世界！
    This is the world the agent interacts with!
    
    环境的职责 Environment Responsibilities:
    1. 维护当前状态
       Maintain current state
    2. 接收动作，返回奖励和下一状态
       Receive action, return reward and next state
    3. 判断回合是否结束
       Determine if episode is done
    4. 提供状态和动作空间信息
       Provide state and action space information
    
    设计原则 Design Principles:
    - 环境是被动的，只响应动作
      Environment is passive, only responds to actions
    - 环境不知道智能体的存在
      Environment doesn't know about agent's existence
    - 环境的动态由P和R完全定义
      Environment dynamics fully defined by P and R
    """
    
    def __init__(self, name: str = "MDP Environment"):
        """
        初始化MDP环境
        Initialize MDP environment
        
        Args:
            name: 环境名称
        """
        self.name = name
        
        # 状态和动作空间
        self.state_space: List[State] = []
        self.action_space: List[Action] = []
        
        # 转移概率和奖励函数
        self.P = TransitionProbability()
        self.R = RewardFunction()
        
        # 当前状态
        self.current_state: Optional[State] = None
        
        # 折扣因子（关键参数！）
        self.gamma = 0.99
        
        # 统计信息
        self.step_count = 0
        self.episode_count = 0
        
        logger.info(f"初始化MDP环境: {name}")
    
    @abstractmethod
    def reset(self) -> State:
        """
        重置环境到初始状态
        Reset environment to initial state
        
        每个回合开始时调用
        Called at the beginning of each episode
        
        Returns:
            初始状态
            Initial state
        """
        pass
    
    @abstractmethod
    def step(self, action: Action) -> Tuple[State, float, bool, Dict]:
        """
        执行动作，环境前进一步
        Execute action, environment steps forward
        
        这是环境的核心方法！
        This is the core method of environment!
        
        Args:
            action: 要执行的动作
                   Action to execute
        
        Returns:
            (下一状态, 奖励, 是否结束, 额外信息)
            (next_state, reward, done, info)
        """
        pass
    
    def render(self, mode: str = 'human'):
        """
        渲染环境当前状态
        Render current environment state
        
        Args:
            mode: 渲染模式
                 Rendering mode
        """
        print(f"Current State: {self.current_state.id if self.current_state else 'None'}")
        print(f"Step: {self.step_count}, Episode: {self.episode_count}")
    
    def get_state_space(self) -> List[State]:
        """获取状态空间"""
        return self.state_space
    
    def get_action_space(self, state: Optional[State] = None) -> List[Action]:
        """
        获取动作空间（可能依赖于状态）
        Get action space (may depend on state)
        """
        return self.action_space
    
    def is_terminal(self, state: State) -> bool:
        """
        判断是否为终止状态
        Check if state is terminal
        """
        return state.is_terminal
    
    def get_dynamics(self) -> Tuple[TransitionProbability, RewardFunction]:
        """
        获取环境动态（用于基于模型的方法）
        Get environment dynamics (for model-based methods)
        
        注意：实际环境通常不提供这个！
        Note: Real environments usually don't provide this!
        """
        return self.P, self.R


# ================================================================================
# 第2.1.4节：MDP智能体基类
# Section 2.1.4: MDP Agent Base Class
# ================================================================================

class MDPAgent(ABC):
    """
    MDP智能体基类
    MDP Agent Base Class
    
    智能体是学习和决策的主体！
    Agent is the subject of learning and decision-making!
    
    与第1章的区别 Differences from Chapter 1:
    1. 需要处理状态序列，不只是单个动作
       Need to handle state sequences, not just single actions
    2. 需要学习策略π(a|s)，不只是动作价值Q(a)
       Need to learn policy π(a|s), not just action values Q(a)
    3. 需要考虑未来奖励（折扣），不只是即时奖励
       Need to consider future rewards (discounted), not just immediate rewards
    
    核心组件 Core Components:
    - 策略(Policy): 如何选择动作
    - 价值函数(Value Function): 评估状态或动作的好坏
    - 模型(Model): 对环境的理解（可选）
    """
    
    def __init__(self, name: str = "MDP Agent"):
        """
        初始化MDP智能体
        
        Args:
            name: 智能体名称
        """
        self.name = name
        
        # 策略类型
        self.policy_type = PolicyType.STOCHASTIC
        
        # 学习率
        self.alpha = 0.1
        
        # 折扣因子
        self.gamma = 0.99
        
        # 探索参数
        self.epsilon = 0.1
        
        # 经验缓冲
        self.experience_buffer = []
        
        # 统计信息
        self.total_reward = 0.0
        self.episode_rewards = []
        
        logger.info(f"初始化MDP智能体: {name}")
    
    @abstractmethod
    def select_action(self, state: State) -> Action:
        """
        根据当前状态选择动作
        Select action based on current state
        
        这是策略的体现！
        This embodies the policy!
        
        Args:
            state: 当前状态
                  Current state
        
        Returns:
            选择的动作
            Selected action
        """
        pass
    
    @abstractmethod
    def update(self, state: State, action: Action, 
              reward: float, next_state: State, done: bool):
        """
        根据经验更新智能体
        Update agent based on experience
        
        这是学习的核心！
        This is the core of learning!
        
        Args:
            state: 当前状态 Current state
            action: 执行的动作 Action taken
            reward: 获得的奖励 Reward received
            next_state: 下一状态 Next state
            done: 是否结束 Whether episode is done
        """
        pass
    
    def reset(self):
        """
        重置智能体（新回合开始）
        Reset agent (new episode starts)
        """
        if len(self.experience_buffer) > 0:
            episode_reward = sum(exp[2] for exp in self.experience_buffer)
            self.episode_rewards.append(episode_reward)
            logger.info(f"Episode finished. Total reward: {episode_reward}")
        
        self.experience_buffer = []
    
    def save_experience(self, state: State, action: Action,
                       reward: float, next_state: State, done: bool):
        """
        保存经验
        Save experience
        
        经验回放的基础！
        Foundation for experience replay!
        """
        experience = (state, action, reward, next_state, done)
        self.experience_buffer.append(experience)
        self.total_reward += reward
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        Get statistics
        """
        return {
            'total_reward': self.total_reward,
            'episode_rewards': self.episode_rewards,
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'episodes_completed': len(self.episode_rewards)
        }


# ================================================================================
# 第2.1.5节：MDP问题的数学表述
# Section 2.1.5: Mathematical Formulation of MDP Problem
# ================================================================================

class MDPMathematics:
    """
    MDP的数学理论
    Mathematical Theory of MDP
    
    这部分极其重要，是所有RL算法的理论基础！
    This part is extremely important, the theoretical foundation of all RL algorithms!
    """
    
    @staticmethod
    def explain_mdp_formulation():
        """
        详解MDP的数学表述
        Detailed explanation of MDP mathematical formulation
        """
        print("\n" + "="*80)
        print("MDP的完整数学表述")
        print("Complete Mathematical Formulation of MDP")
        print("="*80)
        
        print("""
        1. MDP的五元组定义 Five-tuple Definition
        ==========================================
        
        MDP = (S, A, P, R, γ)
        
        其中 Where:
        - S: 状态空间（有限集合）
             State space (finite set)
        - A: 动作空间（有限集合）
             Action space (finite set)  
        - P: S × A × S → [0,1]，状态转移概率
             State transition probability
        - R: S × A × S → ℝ，奖励函数
             Reward function
        - γ ∈ [0,1]: 折扣因子
             Discount factor
        
        2. 马尔可夫性质 Markov Property
        ==================================
        
        "未来独立于过去，给定现在"
        "The future is independent of the past given the present"
        
        数学表述：
        P[S_{t+1} | S_t, A_t, S_{t-1}, A_{t-1}, ..., S_0, A_0] = P[S_{t+1} | S_t, A_t]
        
        这个性质极大简化了问题！
        This property greatly simplifies the problem!
        
        3. 轨迹和回报 Trajectory and Return
        =====================================
        
        轨迹 Trajectory:
        τ = S_0, A_0, R_1, S_1, A_1, R_2, ..., S_T
        
        回报 Return (关键概念！):
        G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... 
            = Σ_{k=0}^∞ γ^k R_{t+k+1}
        
        折扣因子γ的作用：
        Role of discount factor γ:
        - γ = 0: 只关心即时奖励（短视）
                Only care about immediate reward (myopic)
        - γ = 1: 所有奖励同等重要（可能不收敛）
                All rewards equally important (may not converge)
        - 0 < γ < 1: 平衡即时和未来奖励
                    Balance immediate and future rewards
        
        4. 策略 Policy
        ===============
        
        确定性策略 Deterministic policy:
        π: S → A
        
        随机策略 Stochastic policy:
        π(a|s) = P[A_t = a | S_t = s]
        
        满足：Σ_a π(a|s) = 1, ∀s ∈ S
        
        5. 价值函数 Value Functions (核心！)
        =====================================
        
        状态价值函数 State-value function:
        v_π(s) = E_π[G_t | S_t = s]
               = E_π[Σ_{k=0}^∞ γ^k R_{t+k+1} | S_t = s]
        
        动作价值函数 Action-value function:
        q_π(s,a) = E_π[G_t | S_t = s, A_t = a]
                 = E_π[Σ_{k=0}^∞ γ^k R_{t+k+1} | S_t = s, A_t = a]
        
        关系 Relationship:
        v_π(s) = Σ_a π(a|s) q_π(s,a)
        
        6. 贝尔曼方程 Bellman Equations (最重要！)
        ===========================================
        
        贝尔曼期望方程 Bellman Expectation Equation:
        
        对于v_π：
        v_π(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γv_π(s')]
        
        对于q_π：
        q_π(s,a) = Σ_{s',r} p(s',r|s,a)[r + γΣ_{a'} π(a'|s')q_π(s',a')]
        
        这些方程揭示了价值函数的递归结构！
        These equations reveal the recursive structure of value functions!
        
        7. 最优性 Optimality
        ====================
        
        最优状态价值函数：
        v*(s) = max_π v_π(s), ∀s ∈ S
        
        最优动作价值函数：
        q*(s,a) = max_π q_π(s,a), ∀s ∈ S, a ∈ A
        
        贝尔曼最优方程 Bellman Optimality Equation:
        v*(s) = max_a Σ_{s',r} p(s',r|s,a)[r + γv*(s')]
        q*(s,a) = Σ_{s',r} p(s',r|s,a)[r + γmax_{a'} q*(s',a')]
        
        最优策略：
        π*(a|s) = 1 if a = argmax_a q*(s,a)
                  0 otherwise
        
        8. 解决MDP的方法 Methods to Solve MDP
        =======================================
        
        已知模型（P和R已知）：
        Known model (P and R known):
        - 动态规划 Dynamic Programming (Chapter 3)
        - 线性规划 Linear Programming
        
        未知模型（需要学习）：
        Unknown model (need to learn):
        - 蒙特卡洛方法 Monte Carlo Methods (Chapter 4)
        - 时序差分学习 Temporal-Difference Learning (Chapter 5)
        - 规划与学习结合 Planning and Learning (Chapter 8)
        """)
    
    @staticmethod
    def demonstrate_value_iteration():
        """
        演示价值函数的迭代计算
        Demonstrate iterative calculation of value function
        
        这是理解贝尔曼方程的关键！
        This is key to understanding Bellman equations!
        """
        print("\n" + "="*80)
        print("演示：价值函数迭代")
        print("Demo: Value Function Iteration")
        print("="*80)
        
        # 创建简单的2状态MDP
        print("\n简单2状态MDP示例：")
        print("Simple 2-state MDP example:")
        print("""
        状态 States: S = {s1, s2}
        动作 Actions: A = {a1, a2}
        
        转移概率 Transition probabilities:
        P(s1|s1,a1) = 0.7, P(s2|s1,a1) = 0.3
        P(s1|s1,a2) = 0.4, P(s2|s1,a2) = 0.6
        P(s1|s2,a1) = 0.5, P(s2|s2,a1) = 0.5
        P(s1|s2,a2) = 0.2, P(s2|s2,a2) = 0.8
        
        奖励 Rewards:
        R(s1,a1) = 1, R(s1,a2) = 0
        R(s2,a1) = 0, R(s2,a2) = 2
        
        策略 Policy: π(a1|s1) = 0.5, π(a2|s1) = 0.5
                    π(a1|s2) = 0.3, π(a2|s2) = 0.7
        
        折扣因子 Discount: γ = 0.9
        """)
        
        # 初始化
        gamma = 0.9
        
        # 转移概率
        P = {
            ('s1', 'a1'): [('s1', 0.7), ('s2', 0.3)],
            ('s1', 'a2'): [('s1', 0.4), ('s2', 0.6)],
            ('s2', 'a1'): [('s1', 0.5), ('s2', 0.5)],
            ('s2', 'a2'): [('s1', 0.2), ('s2', 0.8)]
        }
        
        # 奖励
        R = {
            ('s1', 'a1'): 1,
            ('s1', 'a2'): 0,
            ('s2', 'a1'): 0,
            ('s2', 'a2'): 2
        }
        
        # 策略
        pi = {
            ('s1', 'a1'): 0.5,
            ('s1', 'a2'): 0.5,
            ('s2', 'a1'): 0.3,
            ('s2', 'a2'): 0.7
        }
        
        # 价值函数初始化
        V = {'s1': 0.0, 's2': 0.0}
        
        print("\n价值迭代过程：")
        print("Value iteration process:")
        print("-" * 40)
        
        # 迭代计算
        for iteration in range(10):
            V_new = {}
            
            for s in ['s1', 's2']:
                v = 0
                for a in ['a1', 'a2']:
                    # 计算q(s,a)
                    q = R[(s, a)]
                    for s_next, p_trans in P[(s, a)]:
                        q += gamma * p_trans * V[s_next]
                    
                    # 加权by策略
                    v += pi[(s, a)] * q
                
                V_new[s] = v
            
            # 打印进度
            print(f"Iteration {iteration + 1}:")
            print(f"  V(s1) = {V_new['s1']:.4f}, V(s2) = {V_new['s2']:.4f}")
            print(f"  Change: {abs(V_new['s1'] - V['s1']) + abs(V_new['s2'] - V['s2']):.6f}")
            
            V = V_new
        
        print("\n收敛的价值函数：")
        print("Converged value function:")
        print(f"v_π(s1) = {V['s1']:.4f}")
        print(f"v_π(s2) = {V['s2']:.4f}")
        
        print("""
        观察 Observations:
        1. 价值函数通过迭代逐渐收敛
           Value function gradually converges through iteration
        2. 每次迭代应用贝尔曼期望方程
           Each iteration applies Bellman expectation equation
        3. 收敛速度取决于γ和MDP结构
           Convergence speed depends on γ and MDP structure
        """)


# ================================================================================
# 第2.1.6节：MDP示例 - 回收机器人
# Section 2.1.6: MDP Example - Recycling Robot
# ================================================================================

class RecyclingRobot(MDPEnvironment):
    """
    回收机器人示例（Sutton & Barto 书中例子）
    Recycling Robot Example (from Sutton & Barto book)
    
    场景描述 Scenario Description:
    一个移动机器人的工作是在办公室收集空罐子。
    机器人有充电站，电池电量决定其状态。
    A mobile robot's job is to collect empty cans in an office.
    The robot has a charging station, battery level determines its state.
    
    这个例子展示了：
    This example demonstrates:
    1. 连续决策问题
       Sequential decision problem
    2. 风险与收益权衡
       Risk-reward trade-off
    3. 长期vs短期考虑
       Long-term vs short-term considerations
    """
    
    def __init__(self):
        """初始化回收机器人环境"""
        super().__init__(name="Recycling Robot")
        
        # 定义状态空间
        # Define state space
        self.state_space = [
            State(id='high', info={'description': '高电量'}),  # High battery
            State(id='low', info={'description': '低电量'})    # Low battery
        ]
        
        # 定义动作空间
        # Define action space
        self.action_space = [
            Action(id='search', name='搜索垃圾'),  # Search for cans
            Action(id='wait', name='等待'),        # Wait
            Action(id='recharge', name='充电')     # Recharge
        ]
        
        # 设置转移概率
        # Set transition probabilities
        self._setup_dynamics()
        
        # 初始状态
        self.initial_state = self.state_space[0]  # Start with high battery
        
        logger.info("初始化回收机器人环境完成")
    
    def _setup_dynamics(self):
        """
        设置环境动态
        Setup environment dynamics
        
        这定义了机器人世界的规则！
        This defines the rules of the robot's world!
        """
        high, low = self.state_space
        search, wait, recharge = self.action_space
        
        # 高电量状态的转移
        # Transitions from high battery state
        
        # 搜索：可能保持高电量或降到低电量
        # Search: may stay high or drop to low
        self.P.set_probability(high, search, high, 2.0, 0.7)  # 成功找到，保持高电量
        self.P.set_probability(high, search, low, 2.0, 0.3)   # 成功找到，但电量降低
        
        # 等待：保持高电量
        # Wait: stay high
        self.P.set_probability(high, wait, high, 0.5, 1.0)
        
        # 低电量状态的转移
        # Transitions from low battery state
        
        # 搜索：有风险耗尽电量
        # Search: risk of depleting battery
        self.P.set_probability(low, search, high, -3.0, 0.3)  # 耗尽电量，被救援
        self.P.set_probability(low, search, low, 1.0, 0.7)    # 找到垃圾，保持低电量
        
        # 等待：保持低电量
        # Wait: stay low
        self.P.set_probability(low, wait, low, 0.5, 1.0)
        
        # 充电：回到高电量
        # Recharge: back to high
        self.P.set_probability(low, recharge, high, 0.0, 1.0)
        
        # 注意：高电量状态不能执行充电动作（约束）
        # Note: Cannot recharge in high battery state (constraint)
    
    def reset(self) -> State:
        """重置到初始状态"""
        self.current_state = self.initial_state
        self.step_count = 0
        self.episode_count += 1
        
        logger.info(f"Episode {self.episode_count}: Robot reset to high battery")
        return self.current_state
    
    def step(self, action: Action) -> Tuple[State, float, bool, Dict]:
        """
        执行动作
        Execute action
        
        展示了MDP的核心循环！
        Demonstrates the core MDP loop!
        """
        if self.current_state is None:
            raise ValueError("Environment not reset. Call reset() first.")
        
        # 检查动作合法性
        # Check action validity
        if self.current_state.id == 'high' and action.id == 'recharge':
            raise ValueError("Cannot recharge when battery is high!")
        
        # 从转移概率中采样
        # Sample from transition probability
        next_state, reward = self.P.sample(self.current_state, action)
        
        # 更新状态
        self.current_state = next_state
        self.step_count += 1
        
        # 这个任务是持续的，没有终止状态
        # This is a continuing task, no terminal state
        done = False
        
        # 额外信息
        info = {
            'battery_level': self.current_state.id,
            'action_taken': action.name,
            'reward_received': reward
        }
        
        logger.debug(f"Step {self.step_count}: {action.name} -> "
                    f"Battery: {next_state.id}, Reward: {reward}")
        
        return next_state, reward, done, info
    
    def render(self, mode: str = 'human'):
        """可视化当前状态"""
        if mode == 'human':
            battery_icon = "🔋" if self.current_state.id == 'high' else "🪫"
            print(f"\n回收机器人状态 Robot Status:")
            print(f"  电量 Battery: {battery_icon} {self.current_state.id}")
            print(f"  步数 Steps: {self.step_count}")
            print(f"  可选动作 Available actions:")
            
            for action in self.action_space:
                if not (self.current_state.id == 'high' and action.id == 'recharge'):
                    # 显示每个动作的期望结果
                    transitions = self.P.get_transitions(self.current_state, action)
                    exp_reward = sum(r * p for _, r, p in transitions)
                    print(f"    - {action.name}: 期望奖励 Expected reward = {exp_reward:.2f}")


# ================================================================================
# 示例运行
# Example Run
# ================================================================================

def main():
    """
    运行MDP框架演示
    Run MDP framework demonstration
    """
    print("\n" + "="*80)
    print("第2.1节：MDP框架")
    print("Section 2.1: MDP Framework")
    print("="*80)
    
    # 1. 解释MDP数学
    MDPMathematics.explain_mdp_formulation()
    
    # 2. 演示价值迭代
    MDPMathematics.demonstrate_value_iteration()
    
    # 3. 运行回收机器人示例
    print("\n" + "="*80)
    print("回收机器人示例")
    print("Recycling Robot Example")
    print("="*80)
    
    # 创建环境
    env = RecyclingRobot()
    
    # 简单演示
    print("\n演示随机策略：")
    print("Demo random policy:")
    
    state = env.reset()
    env.render()
    
    for step in range(5):
        # 获取合法动作
        if state.id == 'high':
            valid_actions = [a for a in env.action_space if a.id != 'recharge']
        else:
            valid_actions = env.action_space
        
        # 随机选择
        action = np.random.choice(valid_actions)
        
        print(f"\nStep {step + 1}: 选择动作 Choose action: {action.name}")
        state, reward, done, info = env.step(action)
        print(f"  结果 Result: 奖励={reward:.1f}, 新状态={state.id}")
        
        env.render()
    
    print("\n" + "="*80)
    print("MDP框架演示完成！")
    print("MDP Framework Demo Complete!")
    print("\n下一步：实现智能体-环境接口")
    print("Next: Implement Agent-Environment Interface")
    print("="*80)


if __name__ == "__main__":
    main()