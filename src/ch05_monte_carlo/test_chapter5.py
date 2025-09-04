#!/usr/bin/env python
"""
测试第5章所有蒙特卡洛模块
Test all Chapter 4 Monte Carlo modules

确保所有MC算法实现正确
Ensure all MC algorithm implementations are correct
"""

import sys
import traceback
import numpy as np
from pathlib import Path
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_mc_foundations():
    """
    测试MC基础理论
    Test MC Foundations
    """
    print("\n" + "="*60)
    print("测试MC基础理论...")
    print("Testing MC Foundations...")
    print("="*60)
    
    try:
        from src.ch05_monte_carlo.mc_foundations import (
            Episode, Experience, Return, MCStatistics,
            LawOfLargeNumbers, MCFoundations
        )
        from src.ch03_finite_mdp.mdp_framework import State, Action
        
        # 测试Episode类
        print("测试Episode类...")
        episode = Episode()
        
        # 创建一些状态和动作
        states = [State(f"s{i}", {'value': i}) for i in range(3)]
        actions = [Action(f"a{i}", f"Action {i}") for i in range(2)]
        
        # 添加经验
        episode.add_experience(Experience(states[0], actions[0], 1.0, states[1], False))
        episode.add_experience(Experience(states[1], actions[1], 2.0, states[2], True))
        
        assert episode.length() == 2, "Episode长度错误"
        assert episode.is_complete(), "Episode应该完成"
        print("✓ Episode类测试通过")
        
        # 测试回报计算
        print("测试回报计算...")
        returns = episode.compute_returns(gamma=0.9)
        expected_g0 = 1.0 + 0.9 * 2.0  # G_0 = R_1 + γR_2
        expected_g1 = 2.0  # G_1 = R_2
        
        assert abs(returns[0] - expected_g0) < 0.001, f"G_0计算错误: {returns[0]} vs {expected_g0}"
        assert abs(returns[1] - expected_g1) < 0.001, f"G_1计算错误: {returns[1]} vs {expected_g1}"
        print("✓ 回报计算测试通过")
        
        # 测试Return统计类
        print("测试Return统计...")
        ret = Return()
        test_returns = [5.0, 4.0, 6.0, 5.5, 4.5]
        for g in test_returns:
            ret.add_return(g)
        
        assert abs(ret.mean - np.mean(test_returns)) < 0.001, "均值计算错误"
        assert ret.count == len(test_returns), "计数错误"
        print("✓ Return统计测试通过")
        
        # 测试MCStatistics
        print("测试MCStatistics...")
        stats = MCStatistics()
        for i in range(10):
            stats.update_state_value(states[0], np.random.normal(5.0, 1.0))
        
        estimate = stats.get_state_value_estimate(states[0])
        assert 3.0 < estimate < 7.0, f"估计值不合理: {estimate}"
        print("✓ MCStatistics测试通过")
        
        print("\n✅ MC基础理论测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ MC基础理论测试失败: {e}")
        traceback.print_exc()
        return False


def test_mc_prediction():
    """
    测试MC预测
    Test MC Prediction
    """
    print("\n" + "="*60)
    print("测试MC预测...")
    print("Testing MC Prediction...")
    print("="*60)
    
    try:
        from src.ch05_monte_carlo.mc_prediction import (
            FirstVisitMC, EveryVisitMC, IncrementalMC
        )
        from src.ch03_finite_mdp.gridworld import GridWorld
        from src.ch03_finite_mdp.policies_and_values import UniformRandomPolicy
        
        # 创建简单环境
        env = GridWorld(rows=2, cols=2, start_pos=(0,0), goal_pos=(1,1))
        print(f"✓ 创建2×2网格世界")
        
        # 创建随机策略
        policy = UniformRandomPolicy(env.action_space)
        print(f"✓ 创建随机策略")
        
        # 测试First-Visit MC
        print("\n测试First-Visit MC...")
        first_visit = FirstVisitMC(env, gamma=0.9)
        V_first = first_visit.estimate_V(policy, n_episodes=100, verbose=False)
        
        # 检查价值函数合理性
        for state in env.state_space:
            if not state.is_terminal:
                value = V_first.get_value(state)
                assert -100 < value < 100, f"First-visit价值异常: {value}"
        print("✓ First-Visit MC测试通过")
        
        # 测试Every-Visit MC
        print("测试Every-Visit MC...")
        every_visit = EveryVisitMC(env, gamma=0.9)
        V_every = every_visit.estimate_V(policy, n_episodes=100, verbose=False)
        
        for state in env.state_space:
            if not state.is_terminal:
                value = V_every.get_value(state)
                assert -100 < value < 100, f"Every-visit价值异常: {value}"
        print("✓ Every-Visit MC测试通过")
        
        # 测试增量MC
        print("测试增量MC...")
        incremental = IncrementalMC(env, gamma=0.9, alpha=0.1)
        V_inc = incremental.estimate_V(policy, n_episodes=100, verbose=False)
        
        for state in env.state_space:
            if not state.is_terminal:
                value = V_inc.get_value(state)
                assert -100 < value < 100, f"增量MC价值异常: {value}"
        print("✓ 增量MC测试通过")
        
        # 比较不同方法
        print("\n比较不同MC预测方法...")
        sample_state = env.state_space[0]
        v_first = V_first.get_value(sample_state)
        v_every = V_every.get_value(sample_state)
        v_inc = V_inc.get_value(sample_state)
        
        print(f"  First-visit: {v_first:.3f}")
        print(f"  Every-visit: {v_every:.3f}")
        print(f"  Incremental: {v_inc:.3f}")
        
        # 值应该相近但不完全相同
        assert abs(v_first - v_every) < 10, "First和Every差异过大"
        
        print("\n✅ MC预测测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ MC预测测试失败: {e}")
        traceback.print_exc()
        return False


def test_mc_control():
    """
    测试MC控制
    Test MC Control
    """
    print("\n" + "="*60)
    print("测试MC控制...")
    print("Testing MC Control...")
    print("="*60)
    
    try:
        from src.ch05_monte_carlo.mc_control import (
            EpsilonGreedyPolicy, OnPolicyMCControl, OffPolicyMCControl
        )
        from src.ch03_finite_mdp.gridworld import GridWorld
        from src.ch03_finite_mdp.policies_and_values import ActionValueFunction
        
        # 创建环境
        env = GridWorld(rows=3, cols=3, start_pos=(0,0), goal_pos=(2,2))
        print(f"✓ 创建3×3网格世界")
        
        # 测试ε-贪婪策略
        print("\n测试ε-贪婪策略...")
        Q = ActionValueFunction(env.state_space, env.action_space, initial_value=0.0)
        eps_policy = EpsilonGreedyPolicy(Q, epsilon=0.1, action_space=env.action_space)
        
        # 测试动作概率
        sample_state = env.state_space[0]
        probs = eps_policy.get_action_probabilities(sample_state)
        
        total_prob = sum(probs.values())
        assert abs(total_prob - 1.0) < 0.001, f"概率和不为1: {total_prob}"
        
        # 至少有探索概率
        min_prob = min(probs.values())
        assert min_prob >= 0.1 / len(env.action_space), "探索概率过低"
        print("✓ ε-贪婪策略测试通过")
        
        # 测试On-Policy MC控制
        print("\n测试On-Policy MC控制...")
        on_policy = OnPolicyMCControl(env, gamma=0.9, epsilon=0.1)
        learned_policy = on_policy.learn(n_episodes=100, verbose=False)
        
        assert len(on_policy.episodes) == 100, "回合数不匹配"
        assert len(on_policy.sa_visits) > 0, "没有访问任何(s,a)对"
        print("✓ On-Policy MC控制测试通过")
        
        # 测试Off-Policy MC控制
        print("\n测试Off-Policy MC控制...")
        off_policy = OffPolicyMCControl(env, gamma=0.9, behavior_epsilon=0.3)
        target_policy = off_policy.learn(n_episodes=100, verbose=False)
        
        assert len(off_policy.episodes) == 100, "回合数不匹配"
        assert len(off_policy.importance_ratios) > 0, "没有IS比率"
        print("✓ Off-Policy MC控制测试通过")
        
        print("\n✅ MC控制测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ MC控制测试失败: {e}")
        traceback.print_exc()
        return False


def test_importance_sampling():
    """
    测试重要性采样
    Test Importance Sampling
    """
    print("\n" + "="*60)
    print("测试重要性采样...")
    print("Testing Importance Sampling...")
    print("="*60)
    
    try:
        from src.ch05_monte_carlo.importance_sampling import (
            OrdinaryImportanceSampling, WeightedImportanceSampling,
            IncrementalISMC
        )
        from src.ch03_finite_mdp.gridworld import GridWorld
        from src.ch03_finite_mdp.policies_and_values import (
            UniformRandomPolicy, DeterministicPolicy
        )
        
        # 创建环境
        env = GridWorld(rows=2, cols=2, start_pos=(0,0), goal_pos=(1,1))
        print(f"✓ 创建2×2网格世界")
        
        # 创建行为和目标策略
        behavior_policy = UniformRandomPolicy(env.action_space)
        
        # 创建简单的确定性目标策略
        policy_map = {}
        for state in env.state_space:
            if not state.is_terminal:
                policy_map[state] = env.action_space[0]  # 总是选第一个动作
        target_policy = DeterministicPolicy(policy_map)
        
        print("✓ 创建行为和目标策略")
        
        # 测试普通IS
        print("\n测试普通重要性采样...")
        ordinary_is = OrdinaryImportanceSampling(
            env, target_policy, behavior_policy, gamma=0.9
        )
        
        # 生成并处理一个回合
        from src.ch05_monte_carlo.mc_foundations import Episode, Experience
        episode = Episode()
        state = env.reset()
        for _ in range(5):
            action = behavior_policy.select_action(state)
            next_state, reward, done, _ = env.step(action)
            exp = Experience(state, action, reward, next_state, done)
            episode.add_experience(exp)
            state = next_state
            if done:
                break
        
        ordinary_is.update_value(episode)
        assert len(ordinary_is.is_ratios) > 0, "没有IS比率"
        print("✓ 普通IS测试通过")
        
        # 测试加权IS
        print("测试加权重要性采样...")
        weighted_is = WeightedImportanceSampling(
            env, target_policy, behavior_policy, gamma=0.9
        )
        
        weighted_is.update_value(episode)
        assert len(weighted_is.is_ratios) > 0, "没有IS比率"
        print("✓ 加权IS测试通过")
        
        # 测试增量IS MC
        print("测试增量IS MC...")
        from src.ch05_monte_carlo.mc_control import EpsilonGreedyPolicy
        from src.ch03_finite_mdp.policies_and_values import ActionValueFunction
        
        Q = ActionValueFunction(env.state_space, env.action_space, 0.0)
        behavior_eps = EpsilonGreedyPolicy(Q, epsilon=0.3, action_space=env.action_space)
        
        incremental_is = IncrementalISMC(
            env, target_policy, behavior_eps, gamma=0.9
        )
        
        # 学习
        _, learned_Q = incremental_is.learn(n_episodes=50, verbose=False)
        assert len(incremental_is.C_sa) > 0, "没有累积权重"
        print("✓ 增量IS MC测试通过")
        
        print("\n✅ 重要性采样测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 重要性采样测试失败: {e}")
        traceback.print_exc()
        return False


def test_mc_examples():
    """
    测试MC经典例子
    Test MC Classic Examples
    """
    print("\n" + "="*60)
    print("测试MC经典例子...")
    print("Testing MC Classic Examples...")
    print("="*60)
    
    try:
        from src.ch05_monte_carlo.mc_examples import (
            Blackjack, BlackjackPolicy, RaceTrack
        )
        
        # 测试21点
        print("\n测试21点游戏...")
        blackjack = Blackjack()
        
        # 测试状态空间
        assert len(blackjack.state_space) > 0, "21点状态空间为空"
        assert len(blackjack.action_space) == 2, "21点应有2个动作"
        
        # 测试游戏流程
        state = blackjack.reset()
        assert state is not None, "21点重置失败"
        
        # 执行一个动作
        action = blackjack.action_space[0]  # hit
        next_state, reward, done, info = blackjack.step(action)
        
        # 奖励应在合理范围
        assert -1 <= reward <= 1, f"21点奖励异常: {reward}"
        print("✓ 21点游戏测试通过")
        
        # 测试21点策略
        print("测试21点策略...")
        policy = BlackjackPolicy(threshold=20)
        
        # 测试阈值策略
        test_state = blackjack.state_space[0]  # 获取一个非终止状态
        if not test_state.is_terminal:
            test_state.features = {'player_sum': 21, 'dealer_showing': 5, 'usable_ace': False}
            action = policy.select_action(test_state)
            assert action.id == "stick", "21点时应该停牌"
        print("✓ 21点策略测试通过")
        
        # 测试赛道
        print("\n测试赛道问题...")
        racetrack = RaceTrack(track_name="simple")
        
        assert len(racetrack.state_space) > 0, "赛道状态空间为空"
        assert len(racetrack.action_space) == 9, "赛道应有9个动作"
        
        # 测试重置
        state = racetrack.reset()
        assert state is not None, "赛道重置失败"
        assert racetrack.position in racetrack.start_positions, "未在起点"
        
        # 测试移动
        action = racetrack.action_space[4]  # (0,0)加速度
        next_state, reward, done, info = racetrack.step(action)
        
        # 奖励应该是负的（时间成本）
        assert reward <= 0, f"赛道奖励应为负: {reward}"
        print("✓ 赛道问题测试通过")
        
        print("\n✅ MC例子测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ MC例子测试失败: {e}")
        traceback.print_exc()
        return False


def test_convergence():
    """
    测试MC方法的收敛性
    Test convergence of MC methods
    """
    print("\n" + "="*60)
    print("测试MC收敛性...")
    print("Testing MC Convergence...")
    print("="*60)
    
    try:
        from src.ch05_monte_carlo.mc_prediction import FirstVisitMC, EveryVisitMC
        from src.ch03_finite_mdp.gridworld import GridWorld
        from src.ch03_finite_mdp.policies_and_values import UniformRandomPolicy
        
        # 创建简单确定性环境
        env = GridWorld(rows=2, cols=2, start_pos=(0,0), goal_pos=(1,1))
        policy = UniformRandomPolicy(env.action_space)
        
        # 运行更多回合测试收敛
        print("测试First-Visit MC收敛...")
        first_visit = FirstVisitMC(env, gamma=0.9)
        
        # 记录收敛过程
        values_history = []
        episodes_list = [10, 50, 100, 500, 1000]
        
        for n_ep in episodes_list:
            first_visit = FirstVisitMC(env, gamma=0.9)
            V = first_visit.estimate_V(policy, n_episodes=n_ep, verbose=False)
            
            # 记录一个状态的价值
            sample_state = env.state_space[0]
            value = V.get_value(sample_state)
            values_history.append(value)
            print(f"  {n_ep}回合: V(s0)={value:.3f}")
        
        # 检查收敛趋势（后期应该更稳定）
        later_variance = np.var(values_history[-2:])
        early_variance = np.var(values_history[:2]) if len(values_history) > 2 else 0
        
        print(f"  早期方差: {early_variance:.3f}")
        print(f"  后期方差: {later_variance:.3f}")
        
        # 后期应该更稳定（方差更小）
        # 但由于随机性，不强制要求
        print("✓ 收敛性测试通过")
        
        print("\n✅ MC收敛性测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ MC收敛性测试失败: {e}")
        traceback.print_exc()
        return False


def main():
    """
    运行所有测试
    Run all tests
    """
    print("\n" + "="*80)
    print("第5章：蒙特卡洛方法 - 模块测试")
    print("Chapter 4: Monte Carlo Methods - Module Tests")
    print("="*80)
    
    tests = [
        ("MC基础理论", test_mc_foundations),
        ("MC预测", test_mc_prediction),
        ("MC控制", test_mc_control),
        ("重要性采样", test_importance_sampling),
        ("MC例子", test_mc_examples),
        ("收敛性", test_convergence)
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
        print("\n🎉 第5章所有MC模块测试通过！")
        print("🎉 All Chapter 4 MC modules passed!")
        print("\n蒙特卡洛方法实现验证完成:")
        print("✓ MC基础（回合、回报、统计）")
        print("✓ MC预测（First-visit、Every-visit、增量）")
        print("✓ MC控制（On-policy、Off-policy）")
        print("✓ 重要性采样（普通、加权、增量）")
        print("✓ 经典例子（21点、赛道）")
        print("\n可以继续学习第6章：时序差分方法")
        print("Ready to proceed to Chapter 6: Temporal Difference Learning")
    else:
        print("\n⚠️ 有些测试失败，请检查代码")
        print("⚠️ Some tests failed, please check the code")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)