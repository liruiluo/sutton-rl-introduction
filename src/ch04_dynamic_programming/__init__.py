"""
第3章：动态规划
Chapter 3: Dynamic Programming

动态规划(DP)是强化学习的理论基础！
Dynamic Programming is the theoretical foundation of RL!

本章假设我们完全了解环境（已知MDP模型），学习如何计算最优策略。
This chapter assumes we have complete knowledge of the environment (known MDP model),
and learns how to compute optimal policies.

虽然现实中很少有完全已知的环境，但DP提供了重要的理论基础：
Although real environments are rarely fully known, DP provides important theoretical foundations:
1. 其他方法都可以看作是DP的近似
   Other methods can be viewed as approximations of DP
2. DP提供了算法的理论上限
   DP provides theoretical upper bounds for algorithms
3. DP的思想贯穿整个RL
   DP ideas permeate all of RL
"""

from .dp_foundations import (
    DynamicProgrammingFoundations,
    PolicyEvaluationDP,
    PolicyImprovementDP,
    BellmanOperator
)

from .policy_iteration import (
    PolicyIteration,
    PolicyIterationVisualizer,
    PolicyIterationAnalysis
)

from .value_iteration import (
    ValueIteration,
    ValueIterationVisualizer,
    ValueIterationAnalysis,
    AsynchronousValueIteration
)

from .generalized_policy_iteration import (
    GeneralizedPolicyIteration,
    GPIPattern,
    GPIVisualizer
)

from .dp_examples import (
    GridWorldDP,
    GamblersProblem,
    CarRental,
    DPExampleRunner
)

__all__ = [
    # 基础
    'DynamicProgrammingFoundations',
    'PolicyEvaluationDP',
    'PolicyImprovementDP',
    'BellmanOperator',
    
    # 策略迭代
    'PolicyIteration',
    'PolicyIterationVisualizer',
    'PolicyIterationAnalysis',
    
    # 价值迭代
    'ValueIteration',
    'ValueIterationVisualizer',
    'ValueIterationAnalysis',
    'AsynchronousValueIteration',
    
    # 广义策略迭代
    'GeneralizedPolicyIteration',
    'GPIPattern',
    'GPIVisualizer',
    
    # 示例
    'GridWorldDP',
    'GamblersProblem',
    'CarRental',
    'DPExampleRunner'
]