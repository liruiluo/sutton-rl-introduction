"""
================================================================================
第8章：使用表格方法的规划与学习
Chapter 8: Planning and Learning with Tabular Methods
================================================================================

统一基于模型和无模型方法！
Unifying model-based and model-free methods!

本章内容 Chapter Contents:
1. 模型与规划 Models and Planning
2. Dyna：集成规划、动作和学习 Dyna: Integrating Planning, Acting, and Learning  
3. 当模型错误时 When the Model Is Wrong
4. 优先级扫描 Prioritized Sweeping
5. 期望更新vs样本更新 Expected vs Sample Updates
6. 轨迹采样 Trajectory Sampling
7. 实时动态规划 Real-time Dynamic Programming
8. 规划的决策时间 Planning at Decision Time
9. 启发式搜索 Heuristic Search
10. 蒙特卡洛树搜索 Monte Carlo Tree Search

核心概念 Core Concepts:
- 模型学习 Model Learning
- 规划 Planning
- Dyna架构 Dyna Architecture
- 优先级扫描 Prioritized Sweeping
- MCTS算法 MCTS Algorithm

统一的视角：
所有方法都可以看作在估计价值函数！
All methods can be viewed as estimating value functions!
"""

# 导出主要类和函数
# Export main classes and functions

from .models_and_planning import (
    TabularModel,
    DeterministicModel,
    StochasticModel,
    PlanningAgent,
    demonstrate_models_and_planning
)

from .dyna_q import (
    DynaQ,
    DynaQPlus,
    DynaQComparator,
    SimulatedExperience,
    demonstrate_dyna_q
)

from .prioritized_sweeping import (
    PriorityQueue,
    PrioritizedSweeping,
    PrioritizedDynaQ,
    demonstrate_prioritized_sweeping
)

from .expected_vs_sample import (
    ExpectedUpdate,
    SampleUpdate,
    UpdateComparator,
    demonstrate_expected_vs_sample
)

from .trajectory_sampling import (
    TrajectorySampling,
    UniformSampling,
    OnPolicySampling,
    demonstrate_trajectory_sampling
)

from .mcts import (
    MCTSNode,
    MonteCarloTreeSearch,
    UCTSelection,
    demonstrate_mcts
)

from .test_chapter8 import main as test_chapter8

__all__ = [
    # 模型与规划
    'TabularModel',
    'DeterministicModel', 
    'StochasticModel',
    'PlanningAgent',
    
    # Dyna算法
    'DynaQ',
    'DynaQPlus',
    'DynaQComparator',
    'SimulatedExperience',
    
    # 优先级扫描
    'PriorityQueue',
    'PrioritizedSweeping',
    'PrioritizedDynaQ',
    
    # 期望vs样本更新
    'ExpectedUpdate',
    'SampleUpdate',
    'UpdateComparator',
    
    # 轨迹采样
    'TrajectorySampling',
    'UniformSampling',
    'OnPolicySampling',
    
    # MCTS
    'MCTSNode',
    'MonteCarloTreeSearch',
    'UCTSelection',
    
    # 演示函数
    'demonstrate_models_and_planning',
    'demonstrate_dyna_q',
    'demonstrate_prioritized_sweeping',
    'demonstrate_expected_vs_sample',
    'demonstrate_trajectory_sampling',
    'demonstrate_mcts',
    
    # 测试
    'test_chapter8'
]