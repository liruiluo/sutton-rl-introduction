"""
第1章：多臂赌博机
Chapter 1: Multi-Armed Bandits

探索与利用的基本权衡
The fundamental trade-off between exploration and exploitation
"""

# 环境和基础概念
from .bandit_introduction import (
    BanditProblemDefinition,
    MultiArmedBandit,
    ActionValueEstimation,
    BaseBanditAgent,
    demonstrate_chapter1_basics
)

# ε-贪婪算法
from .epsilon_greedy import (
    EpsilonGreedyAgent,
    AdaptiveEpsilonGreedy,
    DecayingEpsilonGreedy,
    EpsilonGreedyAnalysis,
    compare_epsilon_greedy_variants
)

# UCB算法
from .ucb_algorithm import (
    UCBAgent,
    UCB2Agent,
    BayesianUCBAgent,
    UCBTunedAgent,
    UCBPrinciple,
    UCBTheoreticalAnalysis,
    compare_ucb_variants
)

# 梯度赌博机算法
from .gradient_bandit import (
    GradientBanditAgent,
    NaturalGradientBandit,
    AdaptiveGradientBandit,
    EntropyRegularizedGradientBandit,
    GradientBanditPrinciple,
    GradientBanditAnalysis,
    compare_all_algorithms
)

# 注：章节已完整实现，无需额外运行脚本

__all__ = [
    # 环境
    'MultiArmedBandit',
    'BaseBanditAgent',
    
    # 算法
    'EpsilonGreedyAgent',
    'UCBAgent',
    'GradientBanditAgent',
    
    # 分析工具
    'EpsilonGreedyAnalysis',
    'UCBTheoreticalAnalysis',
    'GradientBanditAnalysis',
    
    # 演示函数
    'demonstrate_chapter1_basics',
    'compare_epsilon_greedy_variants',
    'compare_ucb_variants',
    'compare_all_algorithms'
]