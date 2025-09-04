"""
第12章：资格迹
Chapter 12: Eligibility Traces

统一TD和MC的优雅框架！
Elegant framework unifying TD and MC!

核心概念 Core Concepts:
1. λ-return
   - 前向视角
     Forward view
   - n-step returns的加权平均
     Weighted average of n-step returns

2. 资格迹
   Eligibility Traces
   - 后向视角
     Backward view
   - 记录状态的"资格"
     Record state's "eligibility"

3. 等价性
   Equivalence
   - 前向视角 ≡ 后向视角
     Forward view ≡ Backward view

主要算法 Main Algorithms:
- TD(λ)
- Sarsa(λ)
- Q(λ) (Watkins's, Peng's)
- 真正的在线TD(λ)
  True Online TD(λ)

资格迹类型 Trace Types:
- 累积迹 (Accumulating traces)
- 替换迹 (Replacing traces)
- Dutch迹 (Dutch traces)

关键优势 Key Advantages:
- 更快的信用分配
  Faster credit assignment
- 更好的样本效率
  Better sample efficiency
- 灵活的偏差-方差权衡
  Flexible bias-variance tradeoff
"""

from .lambda_return import (
    Episode,
    LambdaReturn,
    OfflineLambdaReturn,
    SemiGradientLambdaReturn,
    TTD,
    demonstrate_lambda_return
)

from .td_lambda import (
    TDLambda,
    TrueOnlineTDLambda,
    TruncatedTDLambda,
    VariableLambdaTD,
    demonstrate_td_lambda
)

from .control_traces import (
    SarsaLambda,
    WatkinsQLambda,
    PengQLambda,
    TrueOnlineSarsaLambda,
    demonstrate_control_traces
)

__all__ = [
    # Lambda-return
    'Episode',
    'LambdaReturn',
    'OfflineLambdaReturn',
    'SemiGradientLambdaReturn',
    'TTD',
    'demonstrate_lambda_return',
    
    # TD(lambda)
    'TDLambda',
    'TrueOnlineTDLambda',
    'TruncatedTDLambda',
    'VariableLambdaTD',
    'demonstrate_td_lambda',
    
    # Control with traces
    'SarsaLambda',
    'WatkinsQLambda',
    'PengQLambda',
    'TrueOnlineSarsaLambda',
    'demonstrate_control_traces'
]