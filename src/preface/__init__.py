"""
Preface Module - Introduction to Reinforcement Learning
前言模块 - 强化学习导论

This module contains implementations of concepts from the preface of
Sutton & Barto's "Reinforcement Learning: An Introduction"

本模块包含Sutton & Barto《强化学习导论》前言中概念的实现
"""

from .core_concepts import (
    RLElement,
    Experience,
    Agent,
    Environment,
    demonstrate_rl_loop,
    RewardHypothesis,
    compare_learning_paradigms
)

from .tictactoe import (
    TicTacToeState,
    TicTacToePlayer,
    TicTacToeEnvironment,
    train_players,
    visualize_training_progress,
    demonstrate_value_function,
    human_vs_ai_game
)

__all__ = [
    # Core concepts / 核心概念
    'RLElement',
    'Experience',
    'Agent',
    'Environment',
    'demonstrate_rl_loop',
    'RewardHypothesis',
    'compare_learning_paradigms',
    
    # Tic-tac-toe / 井字棋
    'TicTacToeState',
    'TicTacToePlayer',
    'TicTacToeEnvironment',
    'train_players',
    'visualize_training_progress',
    'demonstrate_value_function',
    'human_vs_ai_game',
]