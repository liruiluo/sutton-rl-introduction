#!/usr/bin/env python
"""
测试第7章所有n步自举方法模块
Test all Chapter 7 n-step Bootstrapping modules

确保所有n步算法实现正确
Ensure all n-step algorithm implementations are correct
"""

import sys
import traceback
import numpy as np
from pathlib import Path
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_n_step_td():
    """
    测试n步TD预测
    Test n-step TD Prediction
    """
    print("\n" + "="*60)
    print("测试n步TD预测...")
    print("Testing n-step TD Prediction...")
    print("="*60)
    
    try:
        from src.ch07_n_step_bootstrapping.n_step_td import (
            NStepReturn, NStepBuffer, NStepTD, NStepTDComparator,
            NStepExperience
        )
        from src.ch02_mdp.mdp_framework import State
        from src.ch02_mdp.gridworld import GridWorld
        from src.ch02_mdp.policies_and_values import UniformRandomPolicy
        
        # 测试n步回报计算
        print("测试n步回报计算...")
        rewards = [0, 0, 1, 0, 0]
        values = [0.1, 0.2, 0.5, 0.3, 0.1, 0]
        gamma = 0.9
        
        # 测试不同n值
        for n in [1, 2, 3]:
            g_n = NStepReturn.compute_n_step_return(rewards, values, n, gamma, t=0)
            print(f"  {n}步回报: {g_n:.3f}")
            assert isinstance(g_n, float), f"{n}步回报类型错误"
        print("  ✓ n步回报计算测试通过")
        
        # 测试n步缓冲区
        print("\n测试n步缓冲区...")
        buffer = NStepBuffer(n=3)
        
        # 添加经验
        for i in range(5):
            state = State(f"s{i}", features={'value': i})
            exp = NStepExperience(
                state=state,
                action=None,
                reward=i * 0.1,
                next_state=State(f"s{i+1}", features={'value': i+1}),
                done=(i == 4),
                value=0.1 * (i + 1)
            )
            buffer.add(exp)
        
        assert buffer.is_ready(), "缓冲区应该准备好"
        g_buffer = buffer.compute_n_step_return(gamma)
        assert isinstance(g_buffer, float), "缓冲区回报类型错误"
        print(f"  缓冲区n步回报: {g_buffer:.3f}")
        print("  ✓ n步缓冲区测试通过")
        
        # 测试n步TD算法
        print("\n测试n步TD算法...")
        env = GridWorld(rows=3, cols=3, start_pos=(0,0), goal_pos=(2,2))
        policy = UniformRandomPolicy(env.action_space)
        
        for n in [1, 2, 4]:
            n_step_td = NStepTD(env, n=n, gamma=0.9, alpha=0.1)
            
            # 学习几个回合
            for _ in range(10):
                ret = n_step_td.learn_episode(policy)
                assert -100 < ret < 100, f"{n}步TD回报异常: {ret}"
            
            # 检查价值函数
            for state in env.state_space[:3]:
                if not state.is_terminal:
                    value = n_step_td.V.get_value(state)
                    assert -100 < value < 100, f"{n}步TD价值异常: {value}"
            
            assert len(n_step_td.episode_returns) == 10, f"{n}步TD回合数不匹配"
            print(f"  ✓ {n}步TD测试通过")
        
        # 测试比较器
        print("\n测试n步TD比较器...")
        comparator = NStepTDComparator(env)
        results = comparator.compare_n_values(
            n_values=[1, 2, 4],
            n_episodes=20,
            n_runs=2,
            verbose=False
        )
        
        assert len(results) == 3, "比较结果数量错误"
        for n in [1, 2, 4]:
            assert n in results, f"缺少n={n}的结果"
            assert 'final_return_mean' in results[n], f"n={n}缺少最终回报"
            assert 'convergence_mean' in results[n], f"n={n}缺少收敛信息"
        print("  ✓ n步TD比较器测试通过")
        
        print("\n✅ n步TD预测测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ n步TD预测测试失败: {e}")
        traceback.print_exc()
        return False


def test_n_step_sarsa():
    """
    测试n步SARSA控制
    Test n-step SARSA Control
    """
    print("\n" + "="*60)
    print("测试n步SARSA控制...")
    print("Testing n-step SARSA Control...")
    print("="*60)
    
    try:
        from src.ch07_n_step_bootstrapping.n_step_sarsa import (
            NStepSARSA, NStepExpectedSARSA, NStepQSigma
        )
        from src.ch02_mdp.gridworld import GridWorld
        
        # 创建环境
        env = GridWorld(rows=3, cols=3, start_pos=(0,0), goal_pos=(2,2))
        print("✓ 创建3×3网格世界")
        
        # 测试n步SARSA
        print("\n测试n步SARSA...")
        for n in [1, 2, 4]:
            sarsa = NStepSARSA(env, n=n, gamma=0.9, alpha=0.1, epsilon=0.1)
            
            # 学习几个回合
            for _ in range(20):
                ret, length = sarsa.learn_episode()
                assert -100 < ret < 100, f"{n}步SARSA回报异常: {ret}"
                assert 0 < length < 1000, f"{n}步SARSA长度异常: {length}"
            
            # 检查Q函数
            for state in env.state_space[:3]:
                if not state.is_terminal:
                    for action in env.action_space:
                        q = sarsa.Q.get_value(state, action)
                        assert -100 < q < 100, f"{n}步SARSA Q值异常: {q}"
            
            assert len(sarsa.episode_returns) == 20, f"{n}步SARSA回合数不匹配"
            print(f"  ✓ {n}步SARSA测试通过")
        
        # 测试n步期望SARSA
        print("\n测试n步期望SARSA...")
        expected_sarsa = NStepExpectedSARSA(env, n=4, gamma=0.9, alpha=0.1, epsilon=0.1)
        
        # 测试期望值计算
        test_state = env.state_space[0]
        if not test_state.is_terminal:
            expected_value = expected_sarsa.compute_expected_value(test_state)
            assert isinstance(expected_value, float), "期望值类型错误"
            assert -100 < expected_value < 100, f"期望值异常: {expected_value}"
        
        # 学习
        for _ in range(20):
            ret, length = expected_sarsa.learn_episode()
            assert -100 < ret < 100, f"期望SARSA回报异常: {ret}"
            assert 0 < length < 1000, f"期望SARSA长度异常: {length}"
        
        assert len(expected_sarsa.episode_returns) == 20, "期望SARSA回合数不匹配"
        print("  ✓ n步期望SARSA测试通过")
        
        # 测试n步Q(σ)
        print("\n测试n步Q(σ)...")
        q_sigma = NStepQSigma(env, n=4, gamma=0.9, alpha=0.1, sigma=0.5, epsilon=0.1)
        
        # 测试σ参数
        sigma_value = q_sigma.sigma_func(0)
        assert 0 <= sigma_value <= 1, f"σ值异常: {sigma_value}"
        
        # 测试n步回报计算
        states = [env.state_space[i] for i in range(3)]
        actions = [env.action_space[0], env.action_space[1]]
        rewards = [0.0, 1.0]
        
        g_sigma = q_sigma.compute_n_step_return(states, actions, rewards, tau=0, T=2)
        assert isinstance(g_sigma, float), "Q(σ)回报类型错误"
        print(f"  Q(σ)回报: {g_sigma:.3f}")
        print("  ✓ n步Q(σ)测试通过")
        
        print("\n✅ n步SARSA控制测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ n步SARSA控制测试失败: {e}")
        traceback.print_exc()
        return False


def test_off_policy_n_step():
    """
    测试Off-Policy n步方法
    Test Off-Policy n-step Methods
    """
    print("\n" + "="*60)
    print("测试Off-Policy n步方法...")
    print("Testing Off-Policy n-step Methods...")
    print("="*60)
    
    try:
        from src.ch07_n_step_bootstrapping.off_policy_n_step import (
            ImportanceSamplingCorrection, OffPolicyNStepTD, OffPolicyNStepSARSA
        )
        from src.ch02_mdp.gridworld import GridWorld
        from src.ch02_mdp.policies_and_values import UniformRandomPolicy
        from src.ch04_monte_carlo.mc_control import EpsilonGreedyPolicy
        
        # 测试重要性采样修正
        print("测试重要性采样修正...")
        
        # 单步比率
        ratio = ImportanceSamplingCorrection.compute_importance_ratio(
            target_prob=0.8, behavior_prob=0.4, truncate=5.0
        )
        assert ratio == 2.0, f"单步比率错误: {ratio}"
        print(f"  单步比率: {ratio:.2f}")
        
        # 累积比率
        target_probs = [0.8, 0.7, 0.9]
        behavior_probs = [0.4, 0.5, 0.3]
        
        cumulative_ratio = ImportanceSamplingCorrection.compute_cumulative_ratio(
            target_probs, behavior_probs, truncate=10.0
        )
        expected_ratio = (0.8/0.4) * (0.7/0.5) * (0.9/0.3)
        assert abs(cumulative_ratio - expected_ratio) < 0.01, "累积比率错误"
        print(f"  累积比率: {cumulative_ratio:.2f}")
        
        # Per-decision比率
        per_decision = ImportanceSamplingCorrection.compute_per_decision_ratios(
            target_probs, behavior_probs, gamma=0.9, truncate=10.0
        )
        assert len(per_decision) == 3, "Per-decision比率数量错误"
        print(f"  Per-decision比率: {[f'{r:.2f}' for r in per_decision]}")
        print("  ✓ 重要性采样修正测试通过")
        
        # 测试Off-Policy n步TD
        print("\n测试Off-Policy n步TD...")
        env = GridWorld(rows=3, cols=3, start_pos=(0,0), goal_pos=(2,2))
        
        # 创建行为和目标策略
        behavior_policy = UniformRandomPolicy(env.action_space)
        
        from src.ch02_mdp.policies_and_values import ActionValueFunction
        Q_target = ActionValueFunction(env.state_space, env.action_space, 0.0)
        target_policy = EpsilonGreedyPolicy(
            Q_target, epsilon=0.1, epsilon_decay=1.0,
            epsilon_min=0.1, action_space=env.action_space
        )
        
        off_policy_td = OffPolicyNStepTD(env, n=4, gamma=0.9, alpha=0.1)
        
        # 学习
        for _ in range(10):
            ret = off_policy_td.learn_episode(behavior_policy, target_policy)
            assert -100 < ret < 100, f"Off-Policy TD回报异常: {ret}"
        
        # 检查价值函数
        for state in env.state_space[:3]:
            if not state.is_terminal:
                value = off_policy_td.V.get_value(state)
                assert -100 < value < 100, f"Off-Policy TD价值异常: {value}"
        
        assert len(off_policy_td.episode_returns) == 10, "Off-Policy TD回合数不匹配"
        assert len(off_policy_td.importance_ratios_history) > 0, "缺少IS比率记录"
        print("  ✓ Off-Policy n步TD测试通过")
        
        # 测试Off-Policy n步SARSA
        print("\n测试Off-Policy n步SARSA...")
        off_policy_sarsa = OffPolicyNStepSARSA(
            env, n=4, gamma=0.9, alpha=0.1,
            epsilon_behavior=0.3, epsilon_target=0.1
        )
        
        # 测试动作概率计算
        test_state = env.state_space[0]
        test_action = env.action_space[0]
        prob = off_policy_sarsa.get_action_probability(
            off_policy_sarsa.behavior_policy, test_state, test_action
        )
        assert 0 <= prob <= 1, f"动作概率异常: {prob}"
        
        # 学习
        for _ in range(20):
            ret, length = off_policy_sarsa.learn_episode()
            assert -100 < ret < 100, f"Off-Policy SARSA回报异常: {ret}"
            assert 0 < length < 1000, f"Off-Policy SARSA长度异常: {length}"
        
        assert len(off_policy_sarsa.episode_returns) == 20, "Off-Policy SARSA回合数不匹配"
        print("  ✓ Off-Policy n步SARSA测试通过")
        
        print("\n✅ Off-Policy n步方法测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ Off-Policy n步方法测试失败: {e}")
        traceback.print_exc()
        return False


def test_tree_backup():
    """
    测试Tree Backup算法
    Test Tree Backup Algorithm
    """
    print("\n" + "="*60)
    print("测试Tree Backup算法...")
    print("Testing Tree Backup Algorithm...")
    print("="*60)
    
    try:
        from src.ch07_n_step_bootstrapping.tree_backup import (
            TreeBackupNode, NStepTreeBackup, TreeBackupVisualizer
        )
        from src.ch02_mdp.mdp_framework import State, Action
        from src.ch02_mdp.gridworld import GridWorld
        from src.ch02_mdp.policies_and_values import UniformRandomPolicy
        
        # 测试Tree Backup节点
        print("测试Tree Backup节点...")
        state = State("test", features={'value': 0})
        node = TreeBackupNode(
            state=state,
            depth=0,
            is_leaf=False,
            expected_value=1.0
        )
        
        # 添加动作值
        action1 = Action("a1")
        action2 = Action("a2")
        node.action_values = {action1: 0.5, action2: 0.8}
        node.action_probabilities = {action1: 0.3, action2: 0.7}
        node.taken_action = action1
        
        backup_value = node.compute_backup_value(gamma=0.9)
        assert isinstance(backup_value, float), "Backup值类型错误"
        print(f"  节点backup值: {backup_value:.3f}")
        print("  ✓ Tree Backup节点测试通过")
        
        # 测试n步Tree Backup算法
        print("\n测试n步Tree Backup算法...")
        env = GridWorld(rows=3, cols=3, start_pos=(0,0), goal_pos=(2,2))
        
        for n in [1, 2, 4]:
            tree_backup = NStepTreeBackup(env, n=n, gamma=0.9, alpha=0.1, epsilon=0.1)
            
            # 测试动作概率计算
            test_state = env.state_space[0]
            test_action = env.action_space[0]
            prob = tree_backup.get_action_probability(test_state, test_action)
            assert 0 <= prob <= 1, f"动作概率异常: {prob}"
            
            # 测试tree backup回报计算
            states = [env.state_space[i] for i in range(min(3, len(env.state_space)))]
            actions = [env.action_space[0], env.action_space[1]]
            rewards = [0.0, 1.0]
            
            g_tree = tree_backup.compute_tree_backup_return(
                states, actions, rewards, tau=0, T=2
            )
            assert isinstance(g_tree, float), "Tree backup回报类型错误"
            
            # 学习
            for _ in range(20):
                ret, length = tree_backup.learn_episode()
                assert -100 < ret < 100, f"{n}步Tree Backup回报异常: {ret}"
                assert 0 < length < 1000, f"{n}步Tree Backup长度异常: {length}"
            
            # 检查Q函数
            for state in env.state_space[:3]:
                if not state.is_terminal:
                    for action in env.action_space:
                        q = tree_backup.Q.get_value(state, action)
                        assert -100 < q < 100, f"{n}步Tree Backup Q值异常: {q}"
            
            assert len(tree_backup.episode_returns) == 20, f"{n}步Tree Backup回合数不匹配"
            print(f"  ✓ {n}步Tree Backup测试通过")
        
        # 测试off-policy学习
        print("\n测试Tree Backup的off-policy学习...")
        behavior_policy = UniformRandomPolicy(env.action_space)
        tree_backup_offpolicy = NStepTreeBackup(
            env, n=4, gamma=0.9, alpha=0.1, epsilon=0.05
        )
        
        # 使用随机策略学习
        for _ in range(20):
            ret, length = tree_backup_offpolicy.learn_episode(behavior_policy)
            assert -100 < ret < 100, "Off-policy Tree Backup回报异常"
            assert 0 < length < 1000, "Off-policy Tree Backup长度异常"
        
        assert len(tree_backup_offpolicy.episode_returns) == 20, "Off-policy回合数不匹配"
        print("  ✓ Off-policy Tree Backup测试通过")
        
        print("\n✅ Tree Backup算法测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ Tree Backup算法测试失败: {e}")
        traceback.print_exc()
        return False


def test_n_step_comparison():
    """
    测试n步方法的比较
    Test comparison of n-step methods
    """
    print("\n" + "="*60)
    print("测试n步方法比较...")
    print("Testing n-step Methods Comparison...")
    print("="*60)
    
    try:
        from src.ch07_n_step_bootstrapping.n_step_td import NStepTD
        from src.ch07_n_step_bootstrapping.n_step_sarsa import NStepSARSA, NStepExpectedSARSA
        from src.ch07_n_step_bootstrapping.tree_backup import NStepTreeBackup
        from src.ch02_mdp.gridworld import GridWorld
        from src.ch02_mdp.policies_and_values import UniformRandomPolicy
        
        # 创建环境
        env = GridWorld(rows=3, cols=3, start_pos=(0,0), goal_pos=(2,2))
        policy = UniformRandomPolicy(env.action_space)
        
        n = 4
        n_episodes = 50
        
        print(f"比较不同的{n}步方法...")
        
        methods = {
            'n-step TD': NStepTD(env, n=n, gamma=0.9, alpha=0.1),
            'n-step SARSA': NStepSARSA(env, n=n, gamma=0.9, alpha=0.1, epsilon=0.1),
            'n-step Expected SARSA': NStepExpectedSARSA(env, n=n, gamma=0.9, alpha=0.1, epsilon=0.1),
            'n-step Tree Backup': NStepTreeBackup(env, n=n, gamma=0.9, alpha=0.1, epsilon=0.1)
        }
        
        results = {}
        
        for name, algo in methods.items():
            returns = []
            
            if name == 'n-step TD':
                # TD预测
                for _ in range(n_episodes):
                    ret = algo.learn_episode(policy)
                    returns.append(ret)
            else:
                # 控制算法
                for _ in range(n_episodes):
                    ret, _ = algo.learn_episode()
                    returns.append(ret)
            
            results[name] = {
                'final_return': returns[-1],
                'avg_return': np.mean(returns[-10:])
            }
            
            print(f"  {name}: 最终回报={returns[-1]:.2f}, "
                  f"平均回报={results[name]['avg_return']:.2f}")
        
        # 验证所有方法都收敛到合理值
        for name, data in results.items():
            assert -100 < data['final_return'] < 100, f"{name}收敛值异常"
        
        print("\n✅ n步方法比较测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ n步方法比较测试失败: {e}")
        traceback.print_exc()
        return False


def main():
    """
    运行所有测试
    Run all tests
    """
    print("\n" + "="*80)
    print("第7章：n步自举方法 - 模块测试")
    print("Chapter 7: n-step Bootstrapping - Module Tests")
    print("="*80)
    
    tests = [
        ("n步TD预测", test_n_step_td),
        ("n步SARSA控制", test_n_step_sarsa),
        ("Off-Policy n步方法", test_off_policy_n_step),
        ("Tree Backup算法", test_tree_backup),
        ("n步方法比较", test_n_step_comparison)
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
        print("\n🎉 第7章所有n步自举方法模块测试通过！")
        print("🎉 All Chapter 7 n-step Bootstrapping modules passed!")
        print("\nn步方法实现验证完成:")
        print("✓ n步TD预测")
        print("✓ n步SARSA和期望SARSA")
        print("✓ n步Q(σ)统一算法")
        print("✓ Off-policy n步方法with重要性采样")
        print("✓ Tree Backup算法（无需IS）")
        print("\nn步方法优雅地统一了TD和MC！")
        print("n-step methods elegantly unify TD and MC!")
        print("\n可以继续学习第8章或开始实际项目")
        print("Ready to proceed to Chapter 8 or start practical projects")
    else:
        print("\n⚠️ 有些测试失败，请检查代码")
        print("⚠️ Some tests failed, please check the code")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)