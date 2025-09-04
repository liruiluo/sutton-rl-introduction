"""
第9章：使用近似的同策略预测
Chapter 9: On-policy Prediction with Approximation

从表格方法到函数近似！
From tabular methods to function approximation!

核心概念 Core Concepts:
1. 梯度下降基础
   Gradient descent foundations
2. 线性函数近似
   Linear function approximation
3. 特征构造方法
   Feature construction methods
4. 神经网络近似
   Neural network approximation

主要算法 Main Algorithms:
- 梯度MC (Gradient Monte Carlo)
- 半梯度TD (Semi-gradient TD)
- 最小二乘TD (Least-Squares TD)
- 梯度TD with Eligibility Traces

关键优势 Key Advantages:
- 处理大/连续状态空间
  Handle large/continuous state spaces
- 泛化到未见状态
  Generalize to unseen states
- 内存效率高
  Memory efficient
"""

from .gradient_descent import (
    ValueFunctionApproximator,
    GradientDescent,
    StochasticGradientDescent,
    MiniBatchGradientDescent,
    demonstrate_gradient_descent
)

from .linear_approximation import (
    LinearFeatures,
    LinearValueFunction,
    GradientMonteCarlo,
    SemiGradientTD,
    SemiGradientTDLambda,
    demonstrate_linear_approximation
)

from .feature_construction import (
    PolynomialFeatures,
    FourierBasis,
    RadialBasisFunction,
    TileCoding,
    Iht,
    demonstrate_feature_construction
)

from .least_squares_td import (
    LeastSquaresTD,
    LeastSquaresTDLambda,
    RecursiveLeastSquaresTD,
    demonstrate_lstd
)

from .neural_approximation import (
    NeuralNetwork,
    DeepQNetwork,
    GradientTDNN,
    demonstrate_neural_approximation
)

__all__ = [
    # Gradient Descent
    'ValueFunctionApproximator',
    'GradientDescent',
    'StochasticGradientDescent',
    'MiniBatchGradientDescent',
    'demonstrate_gradient_descent',
    
    # Linear Approximation
    'LinearFeatures',
    'LinearValueFunction',
    'GradientMonteCarlo',
    'SemiGradientTD',
    'SemiGradientTDLambda',
    'demonstrate_linear_approximation',
    
    # Feature Construction
    'PolynomialFeatures',
    'FourierBasis',
    'RadialBasisFunction',
    'TileCoding',
    'Iht',
    'demonstrate_feature_construction',
    
    # Least Squares TD
    'LeastSquaresTD',
    'LeastSquaresTDLambda',
    'RecursiveLeastSquaresTD',
    'demonstrate_lstd',
    
    # Neural Approximation
    'NeuralNetwork',
    'DeepQNetwork',
    'GradientTDNN',
    'demonstrate_neural_approximation',
]