"""
第10章：使用近似的同策略控制
Chapter 10: On-policy Control with Approximation

从预测到控制的扩展！
Extension from prediction to control!

核心概念 Core Concepts:
1. 回合式半梯度控制
   Episodic semi-gradient control
2. 连续任务的控制
   Control for continuing tasks
3. 平均奖励设置
   Average reward setting

主要算法 Main Algorithms:
- 半梯度Sarsa (Semi-gradient Sarsa)
- 半梯度Expected Sarsa
- 半梯度n-step Sarsa
- 差分半梯度Sarsa (Differential semi-gradient Sarsa)

关键挑战 Key Challenges:
- 控制问题的收敛性更难保证
  Convergence harder to guarantee in control
- 探索与利用的平衡
  Balance exploration and exploitation
- 连续任务的特殊处理
  Special handling for continuing tasks
"""

from .episodic_semi_gradient import (
    SemiGradientSarsa,
    SemiGradientExpectedSarsa,
    SemiGradientNStepSarsa,
    MountainCarTileCoding,
    demonstrate_episodic_control
)

from .continuous_tasks import (
    AccessControlQueuing,
    DifferentialSemiGradientSarsa,
    AverageRewardSetting,
    demonstrate_continuous_control
)

from .control_with_fa import (
    ActionValueApproximator,
    LinearActionValueFunction,
    ControlWithFA,
    ActorCriticWithFA,
    demonstrate_control_with_fa
)

__all__ = [
    # Episodic Control
    'SemiGradientSarsa',
    'SemiGradientExpectedSarsa',
    'SemiGradientNStepSarsa',
    'MountainCarTileCoding',
    'demonstrate_episodic_control',
    
    # Continuous Tasks
    'AccessControlQueuing',
    'DifferentialSemiGradientSarsa',
    'AverageRewardSetting',
    'demonstrate_continuous_control',
    
    # Control with FA
    'ActionValueApproximator',
    'LinearActionValueFunction',
    'ControlWithFA',
    'ActorCriticWithFA',
    'demonstrate_control_with_fa'
]