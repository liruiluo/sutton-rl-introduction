"""
第11章：离策略方法与近似
Chapter 11: Off-policy Methods with Approximation

解决强化学习最难的问题！
Solving the hardest problems in RL!

核心挑战 Core Challenges:
1. 致命三要素 (Deadly Triad)
   - 函数近似 (Function Approximation)
   - 自举 (Bootstrapping)
   - 离策略 (Off-policy)

2. 分布偏移 (Distribution Shift)
   - 行为策略与目标策略的分布不同
   - Behavior and target policy distributions differ

主要方法 Main Methods:
- 重要性采样 (Importance Sampling)
- 梯度TD方法 (Gradient TD Methods)
- 强调TD方法 (Emphatic TD Methods)

关键算法 Key Algorithms:
- 加权重要性采样 (Weighted Importance Sampling)
- GTD2 & TDC
- 强调TD(λ) (Emphatic TD(λ))
- ELSTD
"""

from .importance_sampling import (
    Trajectory,
    ImportanceSampling,
    SemiGradientOffPolicyTD,
    PerDecisionImportanceSampling,
    NStepOffPolicyTD,
    demonstrate_importance_sampling
)

from .gradient_td import (
    ProjectedBellmanError,
    GTD2,
    TDC,
    HTD,
    GradientLSTD,
    demonstrate_gradient_td
)

from .emphatic_td import (
    EmphasisWeights,
    EmphaticTDLambda,
    EmphaticTDC,
    ELSTD,
    TrueOnlineEmphaticTD,
    demonstrate_emphatic_td
)

__all__ = [
    # Importance Sampling
    'Trajectory',
    'ImportanceSampling',
    'SemiGradientOffPolicyTD',
    'PerDecisionImportanceSampling',
    'NStepOffPolicyTD',
    'demonstrate_importance_sampling',
    
    # Gradient TD
    'ProjectedBellmanError',
    'GTD2',
    'TDC',
    'HTD',
    'GradientLSTD',
    'demonstrate_gradient_td',
    
    # Emphatic TD
    'EmphasisWeights',
    'EmphaticTDLambda',
    'EmphaticTDC',
    'ELSTD',
    'TrueOnlineEmphaticTD',
    'demonstrate_emphatic_td'
]