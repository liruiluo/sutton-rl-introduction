#!/usr/bin/env python
"""
测试第1章所有引言模块
Test all Chapter 1 Introduction modules

确保所有基础概念实现正确
Ensure all fundamental concepts are correctly implemented
"""

import sys
import traceback
import numpy as np
from pathlib import Path

# Add project root to path
# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_rl_fundamentals():
    """
    测试RL基础概念
    Test RL Fundamentals
    """
    print("\n" + "="*60)
    print("测试RL基础概念...")
    print("Testing RL Fundamentals...")
    print("="*60)
    
    try:
        from src.ch01_introduction.rl_fundamentals import (
            RLProblem, Agent, Environment, RewardSignal,
            ValueFunction, Policy, Model
        )
        
        # Test RL Problem
        # 测试RL问题
        print("\n测试RLProblem类...")
        problem = RLProblem("Test Problem")
        assert problem.name == "Test Problem", "Problem name error"
        print("  ✓ RLProblem创建成功")
        
        # Test Value Function
        # 测试价值函数
        print("\n测试ValueFunction类...")
        vf = ValueFunction(initial_value=0.0)
        vf.update_value("state1", 1.0, alpha=0.5)
        value = vf.get_value("state1")
        assert 0.4 < value < 0.6, f"Value update error: {value}"
        print(f"  ✓ ValueFunction更新成功: {value:.3f}")
        
        # Test Policy
        # 测试策略
        print("\n测试Policy类...")
        policy = Policy(exploration_rate=0.1)
        actions = ["up", "down", "left", "right"]
        action = policy.select_action("state1", actions, vf)
        assert action in actions, "Invalid action selected"
        print(f"  ✓ Policy选择动作: {action}")
        
        # Test Model
        # 测试模型
        print("\n测试Model类...")
        model = Model()
        model.update("s1", "a1", 1.0, "s2")
        next_state, reward = model.predict("s1", "a1")
        assert next_state == "s2", "Model prediction error"
        assert abs(reward - 1.0) < 0.01, "Reward prediction error"
        print("  ✓ Model预测成功")
        
        # Test Reward Signal
        # 测试奖励信号
        print("\n测试RewardSignal类...")
        rs = RewardSignal()
        rs.record_reward(1.0)
        rs.record_reward(2.0)
        assert rs.total_reward == 3.0, "Reward recording error"
        print(f"  ✓ RewardSignal记录成功: total={rs.total_reward}")
        
        print("\n✅ RL基础概念测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ RL基础概念测试失败: {e}")
        traceback.print_exc()
        return False


def test_tic_tac_toe():
    """
    测试井字游戏
    Test Tic-Tac-Toe
    """
    print("\n" + "="*60)
    print("测试井字游戏...")
    print("Testing Tic-Tac-Toe...")
    print("="*60)
    
    try:
        from src.ch01_introduction.tic_tac_toe import (
            TicTacToeGame, ValueFunctionPlayer, RandomPlayer
        )
        
        # Test game mechanics
        # 测试游戏机制
        print("\n测试游戏机制...")
        game = TicTacToeGame()
        game.reset()
        
        # Test initial state
        # 测试初始状态
        assert game.current_player == 1, "Initial player should be 1"
        assert len(game.get_available_actions()) == 9, "Should have 9 available positions"
        print("  ✓ 游戏初始化成功")
        
        # Test making moves
        # 测试下棋
        board, reward, done = game.make_move((1, 1))  # Center
        assert game.board[1, 1] == 1, "Move not recorded"
        assert game.current_player == -1, "Player not switched"
        assert not done, "Game shouldn't be over"
        print("  ✓ 下棋机制正确")
        
        # Test win detection
        # 测试获胜检测
        game.reset()
        # Create winning position for X
        # 为X创建获胜位置
        game.make_move((0, 0))  # X
        game.make_move((1, 0))  # O
        game.make_move((0, 1))  # X
        game.make_move((1, 1))  # O
        board, reward, done = game.make_move((0, 2))  # X wins
        
        assert done, "Game should be over"
        assert game.winner == 1, "X should win"
        print("  ✓ 获胜检测正确")
        
        # Test players
        # 测试玩家
        print("\n测试玩家类...")
        vf_player = ValueFunctionPlayer("VF Player", 1, epsilon=0.1)
        random_player = RandomPlayer("Random", -1)
        
        game.reset()
        action = random_player.choose_action(game)
        assert action in game.get_available_actions(), "Invalid action from random player"
        print("  ✓ 随机玩家正常")
        
        # Test value function player
        # 测试价值函数玩家
        action = vf_player.choose_action(game)
        assert action in game.get_available_actions(), "Invalid action from VF player"
        
        # Test value update
        # 测试价值更新
        state_hash = game.get_state_hash()
        vf_player.add_state(state_hash)
        vf_player.update_values(1.0)
        
        value = vf_player.get_value(state_hash)
        assert value != 0.5, "Value should be updated"
        print(f"  ✓ 价值函数玩家正常, value={value:.3f}")
        
        # Quick training test
        # 快速训练测试
        print("\n测试训练过程...")
        from src.ch01_introduction.tic_tac_toe import train_value_function_player
        trained = train_value_function_player(n_games=100)
        
        assert len(trained.values) > 0, "No values learned"
        print(f"  ✓ 训练成功，学习了{len(trained.values)}个状态")
        
        print("\n✅ 井字游戏测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 井字游戏测试失败: {e}")
        traceback.print_exc()
        return False


def test_history_and_concepts():
    """
    测试历史和概念
    Test History and Concepts
    """
    print("\n" + "="*60)
    print("测试历史和概念...")
    print("Testing History and Concepts...")
    print("="*60)
    
    try:
        from src.ch01_introduction.history_and_concepts import (
            RLHistory, KeyConcepts, EarlyHistory, ModernDevelopments
        )
        
        # Test RLHistory
        # 测试RL历史
        print("\n测试RLHistory...")
        history = RLHistory()
        
        assert len(history.milestones) > 0, "No milestones"
        assert len(history.key_papers) > 0, "No key papers"
        
        # Test era classification
        # 测试时代分类
        era_1950 = history.get_era(1950)
        assert era_1950 == "Classical Era", f"Wrong era for 1950: {era_1950}"
        
        era_2020 = history.get_era(2020)
        assert era_2020 == "Deep RL Era", f"Wrong era for 2020: {era_2020}"
        
        print(f"  ✓ 历史记录: {len(history.milestones)}个里程碑")
        
        # Test KeyConcepts
        # 测试关键概念
        print("\n测试KeyConcepts...")
        concepts = KeyConcepts()
        
        assert len(concepts.concepts) > 0, "No concepts"
        
        # Test specific concept
        # 测试特定概念
        summary = concepts.get_concept_summary("exploration_exploitation")
        assert "Exploration vs Exploitation" in summary, "Concept not found"
        print(f"  ✓ 概念库: {len(concepts.concepts)}个核心概念")
        
        # Test EarlyHistory
        # 测试早期历史
        print("\n测试EarlyHistory...")
        early = EarlyHistory()
        
        assert len(early.animal_learning) > 0, "No animal learning history"
        assert len(early.optimal_control) > 0, "No optimal control history"
        assert "trial_and_error" in early.animal_learning, "Missing key concept"
        print(f"  ✓ 早期历史: {len(early.animal_learning)}个动物学习概念")
        
        # Test ModernDevelopments
        # 测试现代发展
        print("\n测试ModernDevelopments...")
        modern = ModernDevelopments()
        
        assert len(modern.deep_rl) > 0, "No deep RL developments"
        assert len(modern.applications) > 0, "No applications"
        
        # Check for key algorithms
        # 检查关键算法
        assert "dqn" in modern.deep_rl, "Missing DQN"
        assert "alphago" in modern.deep_rl, "Missing AlphaGo"
        
        dqn_info = modern.deep_rl["dqn"]
        assert dqn_info["year"] == 2013, "Wrong DQN year"
        
        print(f"  ✓ 现代发展: {len(modern.deep_rl)}个深度RL突破")
        print(f"  ✓ 应用领域: {len(modern.applications)}个主要领域")
        
        print("\n✅ 历史和概念测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 历史和概念测试失败: {e}")
        traceback.print_exc()
        return False


def test_demonstrations():
    """
    测试演示函数
    Test demonstration functions
    """
    print("\n" + "="*60)
    print("测试演示函数...")
    print("Testing Demonstration Functions...")
    print("="*60)
    
    try:
        # Test RL fundamentals demonstration
        # 测试RL基础演示
        print("\n测试RL基础演示...")
        from src.ch01_introduction.rl_fundamentals import demonstrate_rl_fundamentals
        
        # Just check it runs without error
        # 只检查运行无错误
        import io
        import contextlib
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            demonstrate_rl_fundamentals()
        
        output = f.getvalue()
        assert "Reinforcement Learning Fundamentals" in output, "Demo didn't run"
        print("  ✓ RL基础演示成功")
        
        # Test history demonstration
        # 测试历史演示
        print("\n测试历史概念演示...")
        from src.ch01_introduction.history_and_concepts import demonstrate_history_and_concepts
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            demonstrate_history_and_concepts()
        
        output = f.getvalue()
        assert "History and Concepts" in output, "History demo didn't run"
        print("  ✓ 历史概念演示成功")
        
        print("\n✅ 所有演示测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 演示测试失败: {e}")
        traceback.print_exc()
        return False


def main():
    """
    运行所有测试
    Run all tests
    """
    print("\n" + "="*80)
    print("第1章：引言 - 模块测试")
    print("Chapter 1: Introduction - Module Tests")
    print("="*80)
    
    tests = [
        ("RL基础概念", test_rl_fundamentals),
        ("井字游戏", test_tic_tac_toe),
        ("历史和概念", test_history_and_concepts),
        ("演示函数", test_demonstrations),
    ]
    
    results = []
    
    for name, test_func in tests:
        print(f"\n运行测试: {name}")
        success = test_func()
        results.append((name, success))
        
        if not success:
            print(f"\n⚠️ {name}测试失败")
    
    # Summary
    # 总结
    print("\n" + "="*80)
    print("测试总结 Test Summary")
    print("="*80)
    
    all_passed = True
    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 第1章所有引言模块测试通过！")
        print("🎉 All Chapter 1 Introduction modules passed!")
        print("\n第1章完成内容：")
        print("✓ RL基本要素（Agent, Environment, Reward, Value, Policy, Model）")
        print("✓ 井字游戏示例（价值函数学习）")
        print("✓ RL历史（从动物学习到深度RL）")
        print("✓ 关键概念（探索vs利用、价值函数、策略梯度等）")
    else:
        print("\n⚠️ 有些测试失败，请检查代码")
        print("⚠️ Some tests failed, please check the code")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)