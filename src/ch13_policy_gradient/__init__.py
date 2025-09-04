"""
第13章：策略梯度方法
Chapter 13: Policy Gradient Methods

直接优化策略参数！
Directly optimizing policy parameters!

核心概念 Core Concepts:
1. 策略梯度定理
   Policy Gradient Theorem
   - 无需值函数的梯度计算
     Gradient computation without value function
   
2. REINFORCE
   - 蒙特卡洛策略梯度
     Monte Carlo policy gradient
   - 基线减少方差
     Baseline for variance reduction

3. Actor-Critic
   - 结合策略梯度和TD学习
     Combining policy gradient with TD learning
   - 在线更新
     Online updates

4. 自然策略梯度
   Natural Policy Gradient
   - Fisher信息矩阵
     Fisher Information Matrix
   - TRPO和PPO
     TRPO and PPO

关键优势 Key Advantages:
- 连续动作空间
  Continuous action spaces
- 随机策略
  Stochastic policies
- 更好的收敛性质
  Better convergence properties
- 端到端优化
  End-to-end optimization

现代深度RL的基础！
Foundation of modern deep RL!
"""

from .policy_gradient_theorem import (
    SoftmaxPolicy,
    GaussianPolicy,
    PolicyGradientTheorem,
    AdvantageFunction,
    demonstrate_policy_gradient_theorem
)

from .reinforce import (
    REINFORCE,
    REINFORCEWithBaseline,
    AllActionsREINFORCE,
    SimpleValueFunction,
    SimpleQFunction,
    demonstrate_reinforce
)

from .actor_critic import (
    OneStepActorCritic,
    ActorCriticWithTraces,
    A2C,
    SimpleActor,
    SimpleCritic,
    demonstrate_actor_critic
)

from .natural_policy_gradient import (
    NaturalPolicyGradient,
    TRPO,
    PPO,
    demonstrate_natural_policy_gradient
)

__all__ = [
    # Policy Gradient Theorem
    'SoftmaxPolicy',
    'GaussianPolicy',
    'PolicyGradientTheorem',
    'AdvantageFunction',
    'demonstrate_policy_gradient_theorem',
    
    # REINFORCE
    'REINFORCE',
    'REINFORCEWithBaseline',
    'AllActionsREINFORCE',
    'SimpleValueFunction',
    'SimpleQFunction',
    'demonstrate_reinforce',
    
    # Actor-Critic
    'OneStepActorCritic',
    'ActorCriticWithTraces',
    'A2C',
    'SimpleActor',
    'SimpleCritic',
    'demonstrate_actor_critic',
    
    # Natural Policy Gradient
    'NaturalPolicyGradient',
    'TRPO',
    'PPO',
    'demonstrate_natural_policy_gradient'
]