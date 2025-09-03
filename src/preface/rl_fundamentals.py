"""
================================================================================
强化学习导论 - 前言：什么是强化学习？
Reinforcement Learning: An Introduction - Preface: What is RL?
================================================================================

本文件对应教材：Sutton & Barto《强化学习导论》第二版 - 前言部分
This file corresponds to: Sutton & Barto "RL: An Introduction" 2nd Ed - Preface

学习目标 Learning Objectives:
1. 理解强化学习与其他机器学习方法的本质区别
2. 掌握强化学习的核心要素和基本框架
3. 理解价值函数的概念和作用
4. 通过井字棋例子理解时序差分学习

================================================================================
第0.1节：强化学习的独特之处
Section 0.1: What Makes Reinforcement Learning Different
================================================================================

想象你在学习骑自行车：
- 没有老师告诉你每一秒应该怎么做（不是监督学习）
- 你需要自己尝试、摔倒、再尝试（试错学习）
- 你的目标是保持平衡并前进（目标导向）
- 当前的动作影响下一刻的状态（序列决策）

这就是强化学习！

Imagine learning to ride a bicycle:
- No teacher tells you what to do each second (not supervised learning)
- You try, fall, and try again (trial-and-error learning)
- Your goal is to balance and move forward (goal-oriented)
- Current actions affect future states (sequential decisions)

This is reinforcement learning!
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第0.2节：强化学习的三大支柱
# Section 0.2: The Three Pillars of Reinforcement Learning
# ================================================================================

class LearningParadigm(Enum):
    """
    机器学习的三大范式
    The Three Paradigms of Machine Learning
    
    为什么需要区分这三种范式？
    因为它们解决不同类型的问题，需要不同的方法。
    
    Why distinguish these three paradigms?
    Because they solve different types of problems and require different methods.
    """
    
    SUPERVISED = "supervised"      # 有老师的学习 / Learning with a teacher
    UNSUPERVISED = "unsupervised"  # 无老师的学习 / Learning without a teacher  
    REINFORCEMENT = "reinforcement" # 从互动中学习 / Learning from interaction


def compare_learning_paradigms_detailed():
    """
    详细比较三种学习范式
    Detailed Comparison of Three Learning Paradigms
    
    这个比较帮助我们理解：
    1. 为什么需要强化学习？
    2. 它解决什么独特的问题？
    3. 它面临什么独特的挑战？
    
    This comparison helps us understand:
    1. Why do we need reinforcement learning?
    2. What unique problems does it solve?
    3. What unique challenges does it face?
    """
    
    print("\n" + "="*80)
    print("三种机器学习范式的深度比较")
    print("Deep Comparison of Three Machine Learning Paradigms")
    print("="*80)
    
    # 监督学习：像在学校学习
    # Supervised Learning: Like learning in school
    print("\n1. 监督学习 SUPERVISED LEARNING")
    print("-" * 40)
    print("""
    比喻 Analogy: 老师教学生
    - 老师给出问题和标准答案
    - 学生学习模仿正确答案
    - 考试时遇到类似问题就能回答
    
    Teacher teaching students:
    - Teacher provides problems and correct answers
    - Students learn to mimic correct answers
    - Can answer similar problems in exams
    
    数学形式 Mathematical Form:
    给定训练集 Given training set: D = {(x₁,y₁), (x₂,y₂), ..., (xₙ,yₙ)}
    学习函数 Learn function: f: X → Y
    使得 Such that: f(xᵢ) ≈ yᵢ
    
    例子 Examples:
    - 图像分类：输入图片→输出类别
    - 语音识别：输入音频→输出文字
    - 机器翻译：输入英文→输出中文
    
    优势 Advantages:
    ✓ 学习信号明确（有正确答案）
    ✓ 评估简单（预测vs真实）
    ✓ 理论成熟
    
    劣势 Disadvantages:
    ✗ 需要大量标注数据（昂贵）
    ✗ 只能学习已见过的模式
    ✗ 无法处理交互式问题
    """)
    
    # 无监督学习：像探索世界
    # Unsupervised Learning: Like exploring the world
    print("\n2. 无监督学习 UNSUPERVISED LEARNING")
    print("-" * 40)
    print("""
    比喻 Analogy: 婴儿观察世界
    - 没有人告诉什么是什么
    - 自己发现规律和模式
    - 将相似的东西归类
    
    Baby observing the world:
    - No one tells what is what
    - Discovers patterns by itself
    - Groups similar things together
    
    数学形式 Mathematical Form:
    给定数据 Given data: D = {x₁, x₂, ..., xₙ}
    发现结构 Discover structure: P(x) 或 hidden patterns
    
    例子 Examples:
    - 聚类：将客户分组
    - 降维：找到数据的主要特征
    - 异常检测：发现不正常的数据
    
    优势 Advantages:
    ✓ 不需要标签
    ✓ 可以发现未知模式
    ✓ 适合探索性分析
    
    劣势 Disadvantages:
    ✗ 没有明确的优化目标
    ✗ 难以评估好坏
    ✗ 结果可能难以解释
    """)
    
    # 强化学习：像学习技能
    # Reinforcement Learning: Like learning a skill
    print("\n3. 强化学习 REINFORCEMENT LEARNING")
    print("-" * 40)
    print("""
    比喻 Analogy: 学习下棋
    - 没有人告诉每步该怎么走
    - 只有赢了或输了才知道好坏
    - 需要探索不同的策略
    - 当前决策影响未来局面
    
    Learning to play chess:
    - No one tells you each move
    - Only know good/bad when win/lose
    - Need to explore different strategies
    - Current decisions affect future positions
    
    数学形式 Mathematical Form:
    目标 Goal: 最大化累积奖励 Maximize cumulative reward
    G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... = Σ_{k=0}^∞ γᵏR_{t+k+1}
    
    其中 Where:
    - G_t: 回报（未来奖励总和）Return (sum of future rewards)
    - R_t: 时刻t的奖励 Reward at time t
    - γ: 折扣因子(0<γ<1) Discount factor
    
    独特挑战 Unique Challenges:
    
    1. 延迟奖励 Delayed Rewards:
       下棋时，好的开局可能30步后才见效果
       In chess, a good opening may show effect after 30 moves
       
    2. 探索vs利用 Exploration vs Exploitation:
       是用已知的好策略，还是尝试新策略？
       Use known good strategy or try new ones?
       
    3. 信用分配 Credit Assignment:
       赢了棋，是哪些步骤的功劳？
       Won the game, which moves deserve credit?
       
    4. 非静态环境 Non-stationary:
       对手也在学习和改变策略
       Opponent is also learning and changing
    
    例子 Examples:
    - 游戏AI：围棋、星际争霸
    - 机器人控制：行走、抓取
    - 资源管理：电力调度、交通控制
    - 推荐系统：个性化推荐
    
    优势 Advantages:
    ✓ 可以学习复杂策略
    ✓ 适合序列决策
    ✓ 可以超越人类水平
    
    劣势 Disadvantages:
    ✗ 需要大量试错
    ✗ 训练不稳定
    ✗ 难以保证安全性
    """)


# ================================================================================
# 第0.3节：强化学习的核心要素
# Section 0.3: Core Elements of Reinforcement Learning
# ================================================================================

@dataclass
class RLCoreElements:
    """
    强化学习的核心要素详解
    Detailed Explanation of RL Core Elements
    
    强化学习系统必须包含这些要素，缺一不可。
    每个要素都有其独特作用，共同构成完整的学习框架。
    
    An RL system must contain these elements, none can be missing.
    Each element has its unique role, together forming a complete learning framework.
    """
    
    # 1. 智能体 AGENT - 学习者和决策者
    agent: str = """
    智能体是学习的主体，它：
    - 观察环境状态
    - 选择要执行的动作
    - 接收奖励信号
    - 更新自己的策略
    
    The agent is the learner, it:
    - Observes environment states
    - Selects actions to execute
    - Receives reward signals
    - Updates its policy
    
    类比：游戏玩家、自动驾驶系统、交易机器人
    Analogy: Game player, autonomous driving system, trading bot
    """
    
    # 2. 环境 ENVIRONMENT - 智能体交互的世界
    environment: str = """
    环境是智能体之外的一切，它：
    - 接收智能体的动作
    - 返回新的状态
    - 提供奖励信号
    - 定义问题的规则
    
    The environment is everything outside the agent, it:
    - Receives agent's actions
    - Returns new states
    - Provides reward signals
    - Defines problem rules
    
    类比：棋盘、道路系统、金融市场
    Analogy: Chess board, road system, financial market
    """
    
    # 3. 状态 STATE - 环境的描述
    state: str = """
    状态是对环境情况的描述，分为：
    
    环境状态 Environment State (S_t^e):
    - 环境的完整内部表示
    - 包含所有相关信息
    - 通常不完全可观察
    
    观察 Observation (O_t):
    - 智能体实际看到的信息
    - 可能是部分的、有噪声的
    
    智能体状态 Agent State (S_t^a):
    - 智能体的内部表示
    - 用于决策的信息总结
    
    马尔可夫性质 Markov Property:
    P[S_{t+1} | S_t] = P[S_{t+1} | S_1, S_2, ..., S_t]
    未来只依赖当前，不依赖历史
    Future depends only on present, not on history
    
    例子：
    - 围棋：棋盘上所有棋子的位置
    - 自动驾驶：车辆位置、速度、周围物体
    - 股票交易：价格、成交量、技术指标
    """
    
    # 4. 动作 ACTION - 智能体的决策
    action: str = """
    动作是智能体可以执行的操作：
    
    离散动作空间 Discrete Action Space:
    A = {a_1, a_2, ..., a_n}
    例：上下左右、买入卖出持有
    
    连续动作空间 Continuous Action Space:
    A ⊆ ℝⁿ
    例：方向盘角度、油门力度
    
    动作选择策略：
    - 确定性：a = μ(s)
    - 随机性：a ~ π(·|s)
    
    Action selection strategies:
    - Deterministic: a = μ(s)
    - Stochastic: a ~ π(·|s)
    """
    
    # 5. 奖励 REWARD - 学习的信号
    reward: str = """
    奖励是环境给智能体的反馈信号：
    
    R_t ∈ ℝ
    
    奖励假设 Reward Hypothesis:
    '所有目标都可以描述为累积奖励的最大化'
    'All goals can be described as maximization of cumulative reward'
    
    奖励设计原则：
    1. 稀疏vs密集 Sparse vs Dense:
       - 稀疏：只在达到目标时给奖励
       - 密集：持续提供引导信号
    
    2. 塑形 Shaping:
       添加额外奖励引导学习
       Add extra rewards to guide learning
    
    3. 内在vs外在 Intrinsic vs Extrinsic:
       - 外在：环境定义的奖励
       - 内在：智能体自己的好奇心
    
    例子：
    - 游戏：得分+1、死亡-100
    - 机器人：到达目标+10、碰撞-5
    - 投资：盈利为正、亏损为负
    """
    
    # 6. 策略 POLICY - 行为的映射
    policy: str = """
    策略定义智能体的行为方式：
    
    π: S → A (确定性策略 Deterministic)
    π: S × A → [0,1] (随机策略 Stochastic)
    
    π(a|s) = P[A_t = a | S_t = s]
    
    策略的表示：
    1. 查找表 Lookup Table:
       每个状态存储一个动作
    
    2. 参数化 Parameterized:
       π_θ(a|s) with parameters θ
    
    3. 神经网络 Neural Network:
       深度学习表示复杂策略
    
    最优策略 Optimal Policy:
    π* = argmax_π V^π(s) for all s
    """
    
    # 7. 价值函数 VALUE FUNCTION - 长期收益的估计
    value_function: str = """
    价值函数估计长期累积奖励：
    
    状态价值函数 State-Value Function:
    V^π(s) = E_π[G_t | S_t = s]
         = E_π[Σ_{k=0}^∞ γᵏR_{t+k+1} | S_t = s]
    
    动作价值函数 Action-Value Function:
    Q^π(s,a) = E_π[G_t | S_t = s, A_t = a]
    
    贝尔曼方程 Bellman Equation:
    V^π(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γV^π(s')]
    
    价值函数的作用：
    1. 评估策略好坏
    2. 指导策略改进
    3. 实现优化控制
    
    递归性质：
    当前价值 = 即时奖励 + 折扣的未来价值
    Current value = Immediate reward + Discounted future value
    """
    
    # 8. 模型 MODEL - 对环境的理解
    model: str = """
    模型是智能体对环境的理解：
    
    转移模型 Transition Model:
    P_{ss'}^a = P[S_{t+1} = s' | S_t = s, A_t = a]
    
    奖励模型 Reward Model:
    R_s^a = E[R_{t+1} | S_t = s, A_t = a]
    
    分类：
    - 基于模型 Model-based:
      学习环境模型，用于规划
      
    - 无模型 Model-free:
      直接学习策略或价值函数
      
    模型的用途：
    1. 规划：在头脑中模拟
    2. 学习：从想象中学习
    3. 迁移：利用已有知识
    """
    
    def demonstrate_all_elements(self):
        """
        演示所有核心要素及其关系
        Demonstrate all core elements and their relationships
        """
        print("\n" + "="*80)
        print("强化学习八大核心要素")
        print("Eight Core Elements of Reinforcement Learning")
        print("="*80)
        
        elements = [
            ("AGENT 智能体", self.agent),
            ("ENVIRONMENT 环境", self.environment),
            ("STATE 状态", self.state),
            ("ACTION 动作", self.action),
            ("REWARD 奖励", self.reward),
            ("POLICY 策略", self.policy),
            ("VALUE 价值", self.value_function),
            ("MODEL 模型", self.model)
        ]
        
        for i, (name, description) in enumerate(elements, 1):
            print(f"\n{i}. {name}")
            print("-" * 40)
            print(description)
            
        # 展示要素之间的关系
        print("\n" + "="*80)
        print("要素关系图 Element Relationships")
        print("="*80)
        print("""
        智能体 AGENT
            ↓ 执行动作 executes action
        环境 ENVIRONMENT
            ↓ 返回 returns
        (新状态, 奖励) (new state, reward)
            ↓ 观察 observes
        智能体 AGENT
            ↓ 更新 updates
        (策略, 价值函数) (policy, value function)
        
        循环往复，直到学会最优策略
        Cycle repeats until optimal policy is learned
        """)


# ================================================================================
# 第0.4节：强化学习的数学基础
# Section 0.4: Mathematical Foundations of RL
# ================================================================================

class RLMathFoundations:
    """
    强化学习的数学基础详解
    Detailed Mathematical Foundations of RL
    
    这些数学概念是理解强化学习算法的基础。
    我们将通过直观解释和严格定义来掌握它们。
    
    These mathematical concepts are the foundation for understanding RL algorithms.
    We'll master them through intuitive explanations and rigorous definitions.
    """
    
    @staticmethod
    def explain_return_and_discounting():
        """
        回报与折扣详解
        Detailed Explanation of Return and Discounting
        
        核心问题：如何衡量未来奖励的价值？
        Core question: How to measure the value of future rewards?
        """
        print("\n" + "="*80)
        print("回报与折扣 Return and Discounting")
        print("="*80)
        
        print("""
        1. 为什么需要折扣？Why Discount?
        ---------------------------------
        
        考虑两种奖励序列：
        Consider two reward sequences:
        
        A: [0, 0, 0, 100]  (第4步得到100)
        B: [100, 0, 0, 0]  (第1步得到100)
        
        哪个更好？显然B更好，因为：
        Which is better? Obviously B, because:
        
        • 即时满足 Immediate satisfaction
        • 不确定性 Uncertainty (未来可能失败)
        • 经济学原理 Economics (时间价值)
        
        2. 折扣回报 Discounted Return
        ------------------------------
        
        G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...
            = Σ_{k=0}^∞ γᵏR_{t+k+1}
        
        其中 γ ∈ [0, 1] 是折扣因子
        Where γ ∈ [0, 1] is the discount factor
        
        γ的含义：
        • γ = 0: 只关心即时奖励（近视）
        • γ = 1: 平等对待所有奖励（远视）
        • γ = 0.9: 平衡（常用值）
        
        3. 数学性质 Mathematical Properties
        -----------------------------------
        
        有限性条件 Finiteness Condition:
        如果 |R_max| < ∞ 且 γ < 1，则：
        If |R_max| < ∞ and γ < 1, then:
        
        |G_t| ≤ R_max/(1-γ)
        
        递归性质 Recursive Property:
        G_t = R_{t+1} + γG_{t+1}
        
        这个递归性质是TD学习的基础！
        This recursive property is the foundation of TD learning!
        """)
        
        # 可视化折扣效果
        print("\n折扣效果示例 Discount Effect Example:")
        print("-" * 40)
        
        rewards = [10, 10, 10, 10, 10]  # 相同的奖励序列
        gammas = [0, 0.5, 0.9, 0.99, 1.0]
        
        for gamma in gammas:
            discounted = sum(r * (gamma ** i) for i, r in enumerate(rewards))
            print(f"γ = {gamma:4.2f}: G = {discounted:6.2f}")
            
        print("""
        观察：
        • γ越小，越重视近期奖励
        • γ越大，越重视长期奖励
        • γ=1时可能发散（需要episode终止）
        """)
    
    @staticmethod
    def explain_bellman_equation():
        """
        贝尔曼方程详解
        Detailed Explanation of Bellman Equation
        
        贝尔曼方程是强化学习的核心方程，
        几乎所有算法都基于它。
        
        The Bellman equation is the core equation of RL,
        almost all algorithms are based on it.
        """
        print("\n" + "="*80)
        print("贝尔曼方程 Bellman Equation")
        print("="*80)
        
        print("""
        1. 直观理解 Intuitive Understanding
        ------------------------------------
        
        一个状态的价值 = 即时奖励 + 下一个状态的价值
        Value of a state = Immediate reward + Value of next state
        
        就像爬山：
        当前位置的价值 = 这一步的收获 + 下一个位置的价值
        
        Like climbing a mountain:
        Value of current position = Gain from this step + Value of next position
        
        2. 贝尔曼期望方程 Bellman Expectation Equation
        -----------------------------------------------
        
        对于策略π，状态价值函数满足：
        For policy π, state-value function satisfies:
        
        V^π(s) = E_π[R_{t+1} + γV^π(S_{t+1}) | S_t = s]
        
        展开期望：
        Expanding the expectation:
        
        V^π(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γV^π(s')]
        
        其中：
        • π(a|s): 策略（在状态s选择动作a的概率）
        • p(s',r|s,a): 转移概率（环境动力学）
        • γ: 折扣因子
        
        3. 贝尔曼最优方程 Bellman Optimality Equation
        ----------------------------------------------
        
        最优价值函数满足：
        Optimal value function satisfies:
        
        V*(s) = max_a Σ_{s',r} p(s',r|s,a)[r + γV*(s')]
        
        Q*(s,a) = Σ_{s',r} p(s',r|s,a)[r + γ max_{a'} Q*(s',a')]
        
        这告诉我们：
        • 最优价值是所有动作中最好的
        • 贪婪策略是最优的
        
        This tells us:
        • Optimal value is the best among all actions
        • Greedy policy is optimal
        
        4. 为什么重要？Why Important?
        ------------------------------
        
        贝尔曼方程提供了：
        • 价值函数的递归结构
        • 策略评估的方法（求解线性方程组）
        • 策略改进的方向（选择更高价值的动作）
        • TD学习的更新规则（利用递归性质）
        
        Bellman equation provides:
        • Recursive structure of value functions
        • Method for policy evaluation (solving linear equations)
        • Direction for policy improvement (choosing higher value actions)
        • Update rule for TD learning (using recursive property)
        """)
    
    @staticmethod
    def explain_markov_property():
        """
        马尔可夫性质详解
        Detailed Explanation of Markov Property
        
        马尔可夫性质是强化学习的基本假设。
        The Markov property is the basic assumption of RL.
        """
        print("\n" + "="*80)
        print("马尔可夫性质 Markov Property")
        print("="*80)
        
        print("""
        1. 定义 Definition
        ------------------
        
        状态S_t是马尔可夫的，当且仅当：
        State S_t is Markov if and only if:
        
        P[S_{t+1} | S_t] = P[S_{t+1} | S_1, S_2, ..., S_t]
        
        用人话说：
        '未来只依赖现在，不依赖过去'
        '现在包含了预测未来所需的所有信息'
        
        In plain language:
        'The future depends only on the present, not on the past'
        'The present contains all information needed to predict the future'
        
        2. 例子 Examples
        ----------------
        
        马尔可夫的 Markov:
        • 围棋棋盘（当前局面决定一切）
        • 物理系统的完整状态（位置+速度）
        
        非马尔可夫的 Non-Markov:
        • 只看位置不看速度（缺少信息）
        • 股价（依赖历史趋势）
        
        3. 马尔可夫决策过程 MDP
        ------------------------
        
        MDP是强化学习的标准框架：
        MDP is the standard framework for RL:
        
        MDP = (S, A, P, R, γ)
        
        • S: 状态空间 State space
        • A: 动作空间 Action space
        • P: 转移概率 P(s'|s,a)
        • R: 奖励函数 R(s,a,s')
        • γ: 折扣因子 Discount factor
        
        4. 处理非马尔可夫情况
        ----------------------
        
        如果环境不是马尔可夫的：
        If environment is not Markov:
        
        方法1：状态增强
        包含历史信息：s_t = (o_t, o_{t-1}, ..., o_{t-k})
        
        方法2：函数近似
        用RNN/LSTM学习历史表示
        
        方法3：部分可观察MDP (POMDP)
        维护信念状态（概率分布）
        """)


# ================================================================================
# 第0.5节：智能体-环境交互循环
# Section 0.5: Agent-Environment Interaction Loop
# ================================================================================

class AgentEnvironmentInteraction:
    """
    智能体-环境交互详解
    Detailed Agent-Environment Interaction
    
    这是强化学习的核心循环，理解它就理解了RL的本质。
    This is the core loop of RL, understanding it means understanding the essence of RL.
    """
    
    @staticmethod
    def explain_interaction_loop():
        """
        交互循环详解
        Detailed Explanation of Interaction Loop
        """
        print("\n" + "="*80)
        print("智能体-环境交互循环")
        print("Agent-Environment Interaction Loop")
        print("="*80)
        
        print("""
        标准交互协议 Standard Interaction Protocol:
        ------------------------------------------
        
        时刻t At time t:
        
        1. 智能体观察状态 Agent observes state: S_t
        
        2. 智能体选择动作 Agent selects action: A_t ~ π(·|S_t)
        
        3. 环境接收动作并转移 Environment receives action and transitions:
           S_{t+1} ~ P(·|S_t, A_t)
           
        4. 环境返回奖励 Environment returns reward: R_{t+1}
        
        5. 智能体接收 (S_{t+1}, R_{t+1})
        
        6. 智能体更新策略/价值 Agent updates policy/value
        
        7. t ← t+1, 回到步骤1 Go back to step 1
        
        
        数据流 Data Flow:
        ----------------
        
        Agent                Environment
          |                       |
          |-------- A_t -------->|
          |                       |
          |<-- S_{t+1}, R_{t+1} --|
          |                       |
        
        
        关键概念 Key Concepts:
        ---------------------
        
        1. Episode（回合）:
           从初始状态到终止状态的完整交互序列
           Complete interaction sequence from initial to terminal state
           
           例：一局游戏、一次对话、一个任务
           
        2. Trajectory（轨迹）:
           τ = (S_0, A_0, R_1, S_1, A_1, R_2, ..., S_T)
           
        3. Experience（经验）:
           单步转移 (S_t, A_t, R_{t+1}, S_{t+1})
           用于学习的基本单位
           
        4. Horizon（时域）:
           - Finite: T < ∞ (有限期)
           - Infinite: T = ∞ (无限期)
        """)
    
    @staticmethod
    def implement_basic_loop():
        """
        实现基本交互循环
        Implement Basic Interaction Loop
        """
        print("\n" + "="*80)
        print("基本交互循环实现")
        print("Basic Interaction Loop Implementation")
        print("="*80)
        
        print("""
        def interaction_loop(agent, environment, n_episodes):
            '''
            强化学习核心循环
            Core RL Loop
            
            这个循环是所有RL算法的基础框架
            This loop is the basic framework for all RL algorithms
            '''
            
            for episode in range(n_episodes):
                # 1. 初始化回合 Initialize episode
                state = environment.reset()
                total_reward = 0
                done = False
                
                while not done:
                    # 2. 智能体选择动作 Agent selects action
                    action = agent.select_action(state)
                    
                    # 3. 环境执行动作 Environment executes action
                    next_state, reward, done, info = environment.step(action)
                    
                    # 4. 智能体学习 Agent learns
                    agent.update(state, action, reward, next_state, done)
                    
                    # 5. 记录和更新 Record and update
                    total_reward += reward
                    state = next_state
                
                # 6. 回合结束处理 End of episode processing
                print(f"Episode {episode}: Total Reward = {total_reward}")
        
        这个简单的循环包含了RL的全部精髓！
        This simple loop contains all the essence of RL!
        """)


# ================================================================================
# 第0.6节：经验(Experience)与学习
# Section 0.6: Experience and Learning
# ================================================================================

@dataclass
class Experience:
    """
    经验：强化学习的基本学习单位
    Experience: The Basic Learning Unit of RL
    
    每个经验包含一次完整的交互信息。
    智能体通过积累和学习经验来改进策略。
    
    Each experience contains complete interaction information.
    Agents improve policies by accumulating and learning from experiences.
    """
    
    # 基本要素 Basic Elements
    state: np.ndarray       # S_t: 当前状态 Current state
    action: int             # A_t: 采取的动作 Action taken
    reward: float           # R_{t+1}: 获得的奖励 Reward received
    next_state: np.ndarray  # S_{t+1}: 下一状态 Next state
    done: bool              # 是否终止 Whether terminal
    
    # 额外信息 Additional Information
    info: Dict[str, Any] = field(default_factory=dict)
    
    # 学习相关 Learning Related
    value_target: Optional[float] = None    # 价值目标 Value target
    advantage: Optional[float] = None       # 优势函数 Advantage function
    probability: Optional[float] = None     # 动作概率 Action probability
    
    def __post_init__(self):
        """
        后处理：验证和计算派生值
        Post-processing: Validate and compute derived values
        """
        # 确保维度一致性
        if isinstance(self.state, list):
            self.state = np.array(self.state)
        if isinstance(self.next_state, list):
            self.next_state = np.array(self.next_state)
            
    def compute_return(self, gamma: float = 0.99, 
                      next_value: float = 0.0) -> float:
        """
        计算这一步的回报
        Compute the return from this step
        
        G_t = R_{t+1} + γV(S_{t+1})
        
        Args:
            gamma: 折扣因子 Discount factor
            next_value: 下一状态的价值 Value of next state
            
        Returns:
            回报 Return
        """
        if self.done:
            return self.reward  # 终止状态无未来价值
        else:
            return self.reward + gamma * next_value
    
    def compute_td_error(self, current_value: float, 
                        next_value: float, 
                        gamma: float = 0.99) -> float:
        """
        计算TD误差
        Compute TD error
        
        δ = R_{t+1} + γV(S_{t+1}) - V(S_t)
        
        这是TD学习的核心！
        This is the core of TD learning!
        
        Args:
            current_value: V(S_t)
            next_value: V(S_{t+1})
            gamma: 折扣因子
            
        Returns:
            TD误差 TD error
        """
        target = self.compute_return(gamma, next_value)
        return target - current_value
    
    def to_tensor(self) -> Dict[str, np.ndarray]:
        """
        转换为张量格式（用于批量处理）
        Convert to tensor format (for batch processing)
        """
        return {
            'states': self.state,
            'actions': np.array(self.action),
            'rewards': np.array(self.reward),
            'next_states': self.next_state,
            'dones': np.array(self.done, dtype=np.float32)
        }


class ExperienceBuffer:
    """
    经验缓冲区：存储和管理经验
    Experience Buffer: Store and Manage Experiences
    
    不同的RL算法需要不同的经验管理策略。
    Different RL algorithms require different experience management strategies.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        初始化经验缓冲区
        Initialize experience buffer
        
        Args:
            capacity: 最大容量 Maximum capacity
        """
        self.capacity = capacity
        self.buffer: List[Experience] = []
        self.position = 0
        
        logger.info(f"经验缓冲区初始化，容量: {capacity}")
        
    def push(self, experience: Experience):
        """
        添加经验（循环缓冲）
        Add experience (circular buffer)
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            # 覆盖最旧的经验
            self.buffer[self.position] = experience
        
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> List[Experience]:
        """
        随机采样一批经验
        Randomly sample a batch of experiences
        
        用于经验回放 (Experience Replay)
        Used for experience replay
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def get_recent(self, n: int) -> List[Experience]:
        """
        获取最近的n个经验
        Get recent n experiences
        
        用于on-policy学习
        Used for on-policy learning
        """
        if len(self.buffer) < n:
            return self.buffer[:]
        else:
            return self.buffer[-n:]
    
    def compute_returns(self, gamma: float = 0.99):
        """
        计算所有经验的回报（从后向前）
        Compute returns for all experiences (backward)
        
        这是蒙特卡洛方法的核心
        This is the core of Monte Carlo methods
        """
        returns = []
        G = 0
        
        # 从后向前计算回报
        for exp in reversed(self.buffer):
            if exp.done:
                G = 0  # 回合结束，重置回报
            G = exp.reward + gamma * G
            returns.append(G)
            
        returns.reverse()
        
        # 更新经验中的价值目标
        for exp, G in zip(self.buffer, returns):
            exp.value_target = G
            
        return returns
    
    def compute_advantages(self, value_function: Callable, gamma: float = 0.99):
        """
        计算优势函数
        Compute advantage function
        
        A(s,a) = Q(s,a) - V(s)
        
        优势函数告诉我们：这个动作比平均好多少？
        Advantage tells us: How much better is this action than average?
        """
        for i, exp in enumerate(self.buffer):
            V_current = value_function(exp.state)
            
            if exp.done:
                V_next = 0
            else:
                V_next = value_function(exp.next_state)
            
            # TD目标
            td_target = exp.reward + gamma * V_next
            
            # 优势 = TD目标 - 当前价值
            exp.advantage = td_target - V_current
            
        return [exp.advantage for exp in self.buffer]
    
    def clear(self):
        """清空缓冲区 Clear buffer"""
        self.buffer.clear()
        self.position = 0
        
    def __len__(self):
        return len(self.buffer)
    
    def __repr__(self):
        return f"ExperienceBuffer(size={len(self)}/{self.capacity})"


# ================================================================================
# 第0.7节：完整示例 - 把所有概念串起来
# Section 0.7: Complete Example - Connecting All Concepts
# ================================================================================

def complete_rl_demo():
    """
    完整的强化学习演示
    Complete RL Demonstration
    
    这个例子展示了所有核心概念如何协同工作。
    This example shows how all core concepts work together.
    """
    print("\n" + "="*80)
    print("完整强化学习系统演示")
    print("Complete RL System Demonstration")
    print("="*80)
    
    # 1. 详细比较学习范式
    compare_learning_paradigms_detailed()
    
    # 2. 展示核心要素
    elements = RLCoreElements()
    elements.demonstrate_all_elements()
    
    # 3. 数学基础
    math_foundations = RLMathFoundations()
    math_foundations.explain_return_and_discounting()
    math_foundations.explain_bellman_equation()
    math_foundations.explain_markov_property()
    
    # 4. 交互循环
    interaction = AgentEnvironmentInteraction()
    interaction.explain_interaction_loop()
    interaction.implement_basic_loop()
    
    # 5. 经验管理示例
    print("\n" + "="*80)
    print("经验管理示例")
    print("Experience Management Example")
    print("="*80)
    
    # 创建一些示例经验
    buffer = ExperienceBuffer(capacity=5)
    
    for i in range(3):
        exp = Experience(
            state=np.array([i, i]),
            action=i % 2,
            reward=float(i),
            next_state=np.array([i+1, i+1]),
            done=(i == 2)
        )
        buffer.push(exp)
        print(f"添加经验 {i+1}: state={exp.state}, reward={exp.reward}")
    
    # 计算回报
    returns = buffer.compute_returns(gamma=0.9)
    print(f"\n计算的回报: {returns}")
    
    print("\n" + "="*80)
    print("总结 Summary")
    print("="*80)
    print("""
    我们已经学习了强化学习的所有基础概念：
    
    1. 强化学习vs其他学习范式
       - 从交互中学习
       - 延迟奖励
       - 探索与利用
    
    2. 八大核心要素
       - Agent, Environment, State, Action
       - Reward, Policy, Value, Model
    
    3. 数学基础
       - 回报与折扣
       - 贝尔曼方程
       - 马尔可夫性质
    
    4. 交互循环
       - 标准RL循环
       - 经验收集
       - 策略改进
    
    这些概念将贯穿整本书的学习！
    These concepts will run through the entire book!
    
    下一步：通过井字棋游戏实践这些概念
    Next: Practice these concepts through Tic-Tac-Toe
    """)


# ================================================================================
# 主函数：运行所有演示
# Main Function: Run All Demonstrations  
# ================================================================================

if __name__ == "__main__":
    """
    运行前言部分的所有概念演示
    Run all concept demonstrations from the preface
    """
    print("\n" + "="*80)
    print("Sutton & Barto《强化学习导论》")
    print("前言：强化学习基础概念")
    print("\nReinforcement Learning: An Introduction")
    print("Preface: Fundamental Concepts of RL")
    print("="*80)
    
    # 运行完整演示
    complete_rl_demo()
    
    print("\n" + "="*80)
    print("恭喜！你已经掌握了强化学习的基础概念！")
    print("Congratulations! You've mastered the basic concepts of RL!")
    print("\n接下来，我们将通过井字棋游戏来实践这些概念。")
    print("Next, we'll practice these concepts through Tic-Tac-Toe game.")
    print("="*80)