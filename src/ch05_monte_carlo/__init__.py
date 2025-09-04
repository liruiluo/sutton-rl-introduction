"""
第4章：蒙特卡洛方法
Chapter 4: Monte Carlo Methods

蒙特卡洛(MC)方法是第一类无模型(model-free)的强化学习方法。
Monte Carlo (MC) methods are the first model-free RL methods.

核心思想：通过采样完整的轨迹来估计价值函数
Core idea: Estimate value functions by sampling complete episodes

为什么叫"蒙特卡洛"？
Why called "Monte Carlo"?
源于摩纳哥的赌场，暗示使用随机性来解决问题。
Named after the casino in Monaco, suggesting using randomness to solve problems.

MC vs DP的关键区别：
Key differences between MC and DP:
1. MC不需要环境模型，只需要经验
   MC doesn't need environment model, only needs experience
2. MC使用采样而非全宽度更新
   MC uses sampling instead of full-width updates
3. MC只适用于回合式任务
   MC only works for episodic tasks
4. MC不自举（不使用其他估计）
   MC doesn't bootstrap (doesn't use other estimates)

本章内容：
Chapter contents:
1. MC预测：估计给定策略的价值函数
   MC Prediction: Estimate value function for given policy
2. MC控制：寻找最优策略
   MC Control: Find optimal policy
3. 重要性采样：从一个策略的经验学习另一个策略
   Importance Sampling: Learn one policy from another's experience
4. 实际应用：21点、赛道问题
   Applications: Blackjack, Racetrack problem
"""

from .mc_foundations import (
    MCFoundations,
    Episode,
    Return,
    MCStatistics,
    LawOfLargeNumbers
)

from .mc_prediction import (
    MCPrediction,
    FirstVisitMC,
    EveryVisitMC,
    IncrementalMC,
    MCPredictionVisualizer
)

from .mc_control import (
    MCControl,
    OnPolicyMCControl,
    OffPolicyMCControl,
    EpsilonGreedyPolicy,
    ExploringStarts,
    MCControlVisualizer
)

from .importance_sampling import (
    ImportanceSampling,
    OrdinaryImportanceSampling,
    WeightedImportanceSampling,
    IncrementalISMC,
    ISVisualizer
)

from .mc_examples import (
    Blackjack,
    BlackjackPolicy,
    RaceTrack,
    MCExampleRunner
)

__all__ = [
    # 基础
    'MCFoundations',
    'Episode',
    'Return',
    'MCStatistics',
    'LawOfLargeNumbers',
    
    # MC预测
    'MCPrediction',
    'FirstVisitMC',
    'EveryVisitMC',
    'IncrementalMC',
    'MCPredictionVisualizer',
    
    # MC控制
    'MCControl',
    'OnPolicyMCControl',
    'OffPolicyMCControl',
    'EpsilonGreedyPolicy',
    'ExploringStarts',
    'MCControlVisualizer',
    
    # 重要性采样
    'ImportanceSampling',
    'OrdinaryImportanceSampling',
    'WeightedImportanceSampling',
    'IncrementalISMC',
    'ISVisualizer',
    
    # 示例
    'Blackjack',
    'BlackjackPolicy',
    'RaceTrack',
    'MCExampleRunner'
]