"""
第7章：n步自举方法
Chapter 7: n-step Bootstrapping

n步方法统一了TD和MC！
n-step methods unify TD and MC!

核心思想 Core Ideas:
1. n=1: TD(0)，一步预测
        One-step lookahead
2. n=∞: MC，完整回报
        Complete return
3. 1<n<∞: 介于TD和MC之间
          Between TD and MC

优势 Advantages:
- 灵活的偏差-方差权衡
  Flexible bias-variance tradeoff
- 更快的学习
  Faster learning
- 无需等待回合结束
  No need to wait for episode end

本章实现 This Chapter Implements:
- n步TD预测
  n-step TD prediction
- n步SARSA控制
  n-step SARSA control
- n步期望SARSA
  n-step Expected SARSA
- 重要性采样修正
  Importance sampling corrections
- n步Tree Backup
  n-step Tree Backup
- n步Q(σ)
  n-step Q(σ)
"""

from .n_step_td import (
    NStepTD,
    NStepReturn,
    NStepBuffer,
    demonstrate_n_step_td
)

from .n_step_sarsa import (
    NStepSARSA,
    NStepExpectedSARSA,
    NStepQSigma,
    demonstrate_n_step_sarsa
)

from .off_policy_n_step import (
    OffPolicyNStepTD,
    OffPolicyNStepSARSA,
    ImportanceSamplingCorrection,
    demonstrate_off_policy_n_step
)

from .tree_backup import (
    NStepTreeBackup,
    TreeBackupNode,
    demonstrate_tree_backup
)

__all__ = [
    # n-step TD
    'NStepTD',
    'NStepReturn',
    'NStepBuffer',
    
    # n-step SARSA
    'NStepSARSA',
    'NStepExpectedSARSA',
    'NStepQSigma',
    
    # Off-policy
    'OffPolicyNStepTD',
    'OffPolicyNStepSARSA',
    'ImportanceSamplingCorrection',
    
    # Tree Backup
    'NStepTreeBackup',
    'TreeBackupNode',
    
    # Demonstrations
    'demonstrate_n_step_td',
    'demonstrate_n_step_sarsa',
    'demonstrate_off_policy_n_step',
    'demonstrate_tree_backup'
]