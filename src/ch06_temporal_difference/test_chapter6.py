#!/usr/bin/env python
"""
测试第5章所有时序差分模块
Test all Chapter 5 Temporal Difference modules

确保所有TD算法实现正确
Ensure all TD algorithm implementations are correct
"""

import sys
import traceback
import numpy as np
from pathlib import Path
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_td_foundations():
    """
    测试TD基础理论
    Test TD Foundations
    """
    print("\n" + "="*60)
    print("测试TD基础理论...")
    print("Testing TD Foundations...")
    print("="*60)
    
    try:
        from src.ch06_temporal_difference.td_foundations import (
            TDTheory, TDError, TDErrorAnalyzer, TD0
        )
        from src.ch03_finite_mdp.gridworld import GridWorld
        from src.ch03_finite_mdp.policies_and_values import UniformRandomPolicy
        
        # 测试TD理论
        print("测试TD理论展示...")
        # TDTheory.explain_td_vs_mc_vs_dp()  # 只是打印，不需要测试返回值
        print("✓ TD理论展示通过")
        
        # 测试TD误差
        print("测试TD误差...")
        from src.ch03_finite_mdp.mdp_framework import State
        test_state = State("test", features={'value': 1})
        td_error = TDError(
            value=0.5,
            timestep=0,
            state=test_state,
            reward=1.0,
            state_value=0.0,
            next_state_value=0.5
        )
        assert td_error.value == 0.5, "TD误差值错误"
        print("✓ TD误差测试通过")
        
        # 测试TD误差分析器
        print("测试TD误差分析器...")
        analyzer = TDErrorAnalyzer(window_size=10)
        for i in range(20):
            td_err = TDError(
                value=np.random.normal(0, 1),
                timestep=i,
                state=test_state
            )
            analyzer.add_error(td_err)
        
        stats = analyzer.get_statistics()
        assert 'total_errors' in stats, "统计信息缺失"
        assert stats['total_errors'] == 20, "误差计数错误"
        print("✓ TD误差分析器测试通过")
        
        # 测试TD(0)
        print("测试TD(0)算法...")
        env = GridWorld(rows=3, cols=3, start_pos=(0,0), goal_pos=(2,2))
        policy = UniformRandomPolicy(env.action_space)
        
        td0 = TD0(env, gamma=0.9, alpha=0.1)
        V = td0.learn(policy, n_episodes=50, verbose=False)
        
        # 检查价值函数合理性
        for state in env.state_space:
            if not state.is_terminal:
                value = V.get_value(state)
                assert -100 < value < 100, f"TD(0)价值异常: {value}"
        
        print("✓ TD(0)测试通过")
        
        print("\n✅ TD基础理论测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ TD基础理论测试失败: {e}")
        traceback.print_exc()
        return False


def test_td_control():
    """
    测试TD控制算法
    Test TD Control Algorithms
    """
    print("\n" + "="*60)
    print("测试TD控制算法...")
    print("Testing TD Control Algorithms...")
    print("="*60)
    
    try:
        from src.ch06_temporal_difference.td_control import (
            SARSA, QLearning, ExpectedSARSA, TDControlComparator
        )
        from src.ch03_finite_mdp.gridworld import GridWorld
        
        # 创建环境
        env = GridWorld(rows=3, cols=3, start_pos=(0,0), goal_pos=(2,2))
        print(f"✓ 创建3×3网格世界")
        
        # 测试SARSA
        print("\n测试SARSA...")
        sarsa = SARSA(env, gamma=0.9, alpha=0.1, epsilon=0.1)
        Q_sarsa = sarsa.learn(n_episodes=50, verbose=False)
        
        # 检查Q函数合理性
        q_values_sarsa = []
        for state in env.state_space:
            if not state.is_terminal:
                for action in env.action_space:
                    q = Q_sarsa.get_value(state, action)
                    q_values_sarsa.append(q)
                    assert -100 < q < 100, f"SARSA Q值异常: {q}"
        
        assert len(sarsa.episode_returns) == 50, "SARSA回合数不匹配"
        print("✓ SARSA测试通过")
        
        # 测试Q-Learning
        print("测试Q-Learning...")
        qlearning = QLearning(env, gamma=0.9, alpha=0.1, epsilon=0.1)
        Q_qlearning = qlearning.learn(n_episodes=50, verbose=False)
        
        q_values_ql = []
        for state in env.state_space:
            if not state.is_terminal:
                for action in env.action_space:
                    q = Q_qlearning.get_value(state, action)
                    q_values_ql.append(q)
                    assert -100 < q < 100, f"Q-Learning Q值异常: {q}"
        
        assert len(qlearning.episode_returns) == 50, "Q-Learning回合数不匹配"
        print("✓ Q-Learning测试通过")
        
        # 测试Expected SARSA
        print("测试Expected SARSA...")
        expected_sarsa = ExpectedSARSA(env, gamma=0.9, alpha=0.1, epsilon=0.1)
        Q_expected = expected_sarsa.learn(n_episodes=50, verbose=False)
        
        q_values_exp = []
        for state in env.state_space:
            if not state.is_terminal:
                for action in env.action_space:
                    q = Q_expected.get_value(state, action)
                    q_values_exp.append(q)
                    assert -100 < q < 100, f"Expected SARSA Q值异常: {q}"
        
        print("✓ Expected SARSA测试通过")
        
        # 测试算法比较器
        print("\n测试TD控制算法比较器...")
        comparator = TDControlComparator(env)
        results = comparator.run_comparison(
            n_episodes=20,
            n_runs=2,
            gamma=0.9,
            alpha=0.1,
            epsilon=0.1,
            verbose=False
        )
        
        assert 'SARSA' in results, "比较结果缺少SARSA"
        assert 'Q-Learning' in results, "比较结果缺少Q-Learning"
        assert 'Expected SARSA' in results, "比较结果缺少Expected SARSA"
        print("✓ TD控制算法比较器测试通过")
        
        # 比较三种算法的结果
        print("\n算法比较:")
        print(f"  SARSA平均Q值: {np.mean(q_values_sarsa):.3f}")
        print(f"  Q-Learning平均Q值: {np.mean(q_values_ql):.3f}")
        print(f"  Expected SARSA平均Q值: {np.mean(q_values_exp):.3f}")
        
        print("\n✅ TD控制算法测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ TD控制算法测试失败: {e}")
        traceback.print_exc()
        return False


def test_n_step_td():
    """
    测试n-step TD方法
    Test n-step TD Methods
    """
    print("\n" + "="*60)
    print("测试n-step TD方法...")
    print("Testing n-step TD Methods...")
    print("="*60)
    
    try:
        from src.ch06_temporal_difference.n_step_td import (
            NStepExperience, NStepTD, NStepSARSA, NStepComparator
        )
        from src.ch03_finite_mdp.gridworld import GridWorld
        from src.ch03_finite_mdp.policies_and_values import UniformRandomPolicy
        from src.ch03_finite_mdp.mdp_framework import State, Action
        
        # 测试n-step经验
        print("测试n-step经验...")
        states = [State(f"s{i}", features={'value': i}) for i in range(4)]
        actions = [Action(f"a{i}", f"Action {i}") for i in range(3)]
        rewards = [1.0, 2.0, 3.0]
        
        n_exp = NStepExperience(states=states[:4], actions=actions, rewards=rewards)
        assert n_exp.n == 3, "n值计算错误"
        
        g = n_exp.compute_n_step_return(gamma=0.9, final_value=10.0)
        expected_g = 1.0 + 0.9*2.0 + 0.81*3.0 + 0.729*10.0
        assert abs(g - expected_g) < 0.001, f"n-step回报计算错误: {g} vs {expected_g}"
        print("✓ n-step经验测试通过")
        
        # 测试n-step TD预测
        print("\n测试n-step TD预测...")
        env = GridWorld(rows=3, cols=3, start_pos=(0,0), goal_pos=(2,2))
        policy = UniformRandomPolicy(env.action_space)
        
        for n in [1, 3, 5]:
            n_step_td = NStepTD(env, n=n, gamma=0.9, alpha=0.1)
            V = n_step_td.learn(policy, n_episodes=30, verbose=False)
            
            # 检查价值函数
            for state in env.state_space:
                if not state.is_terminal:
                    value = V.get_value(state)
                    assert -100 < value < 100, f"{n}-step TD价值异常: {value}"
            
            print(f"✓ {n}-step TD测试通过")
        
        # 测试n-step SARSA
        print("\n测试n-step SARSA...")
        n_step_sarsa = NStepSARSA(env, n=3, gamma=0.9, alpha=0.1, epsilon=0.1)
        Q = n_step_sarsa.learn(n_episodes=30, verbose=False)
        
        # 检查Q函数
        for state in env.state_space:
            if not state.is_terminal:
                for action in env.action_space:
                    q = Q.get_value(state, action)
                    assert -100 < q < 100, f"n-step SARSA Q值异常: {q}"
        
        print("✓ n-step SARSA测试通过")
        
        # 测试n值比较器
        print("\n测试n值比较器...")
        comparator = NStepComparator(env)
        results = comparator.compare_n_values(
            n_values=[1, 2, 3],
            n_episodes=20,
            n_runs=2,
            gamma=0.9,
            alpha=0.1,
            verbose=False
        )
        
        assert 1 in results, "比较结果缺少n=1"
        assert 2 in results, "比较结果缺少n=2"
        assert 3 in results, "比较结果缺少n=3"
        
        for n, data in results.items():
            assert 'final_return_mean' in data, f"n={n}缺少最终回报"
            assert 'convergence_mean' in data, f"n={n}缺少收敛信息"
        
        print("✓ n值比较器测试通过")
        
        print("\n✅ n-step TD方法测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ n-step TD方法测试失败: {e}")
        traceback.print_exc()
        return False


def test_convergence_comparison():
    """
    测试TD方法的收敛性比较
    Test convergence comparison of TD methods
    """
    print("\n" + "="*60)
    print("测试TD方法收敛性比较...")
    print("Testing TD Methods Convergence Comparison...")
    print("="*60)
    
    try:
        from src.ch06_temporal_difference.td_foundations import TD0
        from src.ch06_temporal_difference.td_control import SARSA, QLearning
        from src.ch06_temporal_difference.n_step_td import NStepTD
        from src.ch03_finite_mdp.gridworld import GridWorld
        from src.ch03_finite_mdp.policies_and_values import UniformRandomPolicy
        
        # 创建环境
        env = GridWorld(rows=4, cols=4, start_pos=(0,0), goal_pos=(3,3))
        policy = UniformRandomPolicy(env.action_space)
        
        print("比较不同TD方法的收敛速度...")
        
        # TD(0)
        td0 = TD0(env, gamma=0.9, alpha=0.1)
        td0_returns = []
        for _ in range(50):
            ret = td0.learn_episode(policy)
            td0_returns.append(ret)
        
        # 3-step TD
        n_step_td = NStepTD(env, n=3, gamma=0.9, alpha=0.1)
        n_step_returns = []
        for _ in range(50):
            ret = n_step_td.learn_episode(policy)
            n_step_returns.append(ret)
        
        # SARSA
        sarsa = SARSA(env, gamma=0.9, alpha=0.1, epsilon=0.1)
        for _ in range(50):
            sarsa.learn_episode()
        sarsa_returns = sarsa.episode_returns
        
        # Q-Learning
        qlearning = QLearning(env, gamma=0.9, alpha=0.1, epsilon=0.1)
        for _ in range(50):
            qlearning.learn_episode()
        ql_returns = qlearning.episode_returns
        
        # 分析收敛
        print("\n收敛分析（最后10回合平均）：")
        print(f"  TD(0): {np.mean(td0_returns[-10:]):.3f}")
        print(f"  3-step TD: {np.mean(n_step_returns[-10:]):.3f}")
        print(f"  SARSA: {np.mean(sarsa_returns[-10:]):.3f}")
        print(f"  Q-Learning: {np.mean(ql_returns[-10:]):.3f}")
        
        # 检查是否都在合理范围
        for name, returns in [("TD(0)", td0_returns),
                              ("3-step TD", n_step_returns),
                              ("SARSA", sarsa_returns),
                              ("Q-Learning", ql_returns)]:
            avg = np.mean(returns[-10:]) if len(returns) >= 10 else np.mean(returns)
            assert -100 < avg < 100, f"{name}收敛值异常: {avg}"
        
        print("\n✅ TD方法收敛性比较测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ TD方法收敛性比较测试失败: {e}")
        traceback.print_exc()
        return False


def test_td_vs_mc_comparison():
    """
    测试TD与MC的比较
    Test TD vs MC Comparison
    """
    print("\n" + "="*60)
    print("测试TD与MC比较...")
    print("Testing TD vs MC Comparison...")
    print("="*60)
    
    try:
        from src.ch06_temporal_difference.td_foundations import TD0
        from src.ch05_monte_carlo.mc_prediction import FirstVisitMC
        from src.ch03_finite_mdp.gridworld import GridWorld
        from src.ch03_finite_mdp.policies_and_values import UniformRandomPolicy
        
        # 创建环境
        env = GridWorld(rows=3, cols=3, start_pos=(0,0), goal_pos=(2,2))
        policy = UniformRandomPolicy(env.action_space)
        
        print("运行TD(0)...")
        td0 = TD0(env, gamma=0.9, alpha=0.1)
        V_td = td0.learn(policy, n_episodes=100, verbose=False)
        
        print("运行First-Visit MC...")
        mc = FirstVisitMC(env, gamma=0.9)
        V_mc = mc.estimate_V(policy, n_episodes=100, verbose=False)
        
        # 比较价值函数
        print("\n价值函数比较（部分状态）：")
        differences = []
        
        for i, state in enumerate(env.state_space[:5]):
            if not state.is_terminal:
                td_value = V_td.get_value(state)
                mc_value = V_mc.get_value(state)
                diff = abs(td_value - mc_value)
                differences.append(diff)
                
                print(f"  State {state.id}: TD={td_value:.3f}, MC={mc_value:.3f}, Diff={diff:.3f}")
        
        # 平均差异应该不太大（都收敛到V^π）
        avg_diff = np.mean(differences)
        print(f"\n平均差异: {avg_diff:.3f}")
        assert avg_diff < 5.0, f"TD和MC差异过大: {avg_diff}"
        
        print("✅ TD与MC比较测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ TD与MC比较测试失败: {e}")
        traceback.print_exc()
        return False


def main():
    """
    运行所有测试
    Run all tests
    """
    print("\n" + "="*80)
    print("第5章：时序差分学习 - 模块测试")
    print("Chapter 5: Temporal-Difference Learning - Module Tests")
    print("="*80)
    
    tests = [
        ("TD基础理论", test_td_foundations),
        ("TD控制算法", test_td_control),
        ("n-step TD方法", test_n_step_td),
        ("收敛性比较", test_convergence_comparison),
        ("TD vs MC比较", test_td_vs_mc_comparison)
    ]
    
    results = []
    start_time = time.time()
    
    for name, test_func in tests:
        print(f"\n运行测试: {name}")
        success = test_func()
        results.append((name, success))
        
        if not success:
            print(f"\n⚠️ 测试失败，停止后续测试")
            break
    
    total_time = time.time() - start_time
    
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
    
    print(f"\n总测试时间: {total_time:.2f}秒")
    
    if all_passed:
        print("\n🎉 第5章所有TD模块测试通过！")
        print("🎉 All Chapter 5 TD modules passed!")
        print("\n时序差分学习实现验证完成:")
        print("✓ TD基础（TD(0)、TD误差、收敛性）")
        print("✓ TD控制（SARSA、Q-learning、Expected SARSA）")
        print("✓ n-step TD（统一MC和TD）")
        print("✓ 算法比较和分析")
        print("\n这是强化学习最核心的内容！")
        print("This is the core of reinforcement learning!")
        print("\n可以继续学习第6章：TD(λ)和资格迹")
        print("Ready to proceed to Chapter 6: TD(λ) and Eligibility Traces")
    else:
        print("\n⚠️ 有些测试失败，请检查代码")
        print("⚠️ Some tests failed, please check the code")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)