#!/usr/bin/env python
"""
测试第12章所有资格迹模块
Test all Chapter 12 Eligibility Traces modules

确保所有资格迹算法实现正确
Ensure all eligibility trace algorithm implementations are correct
"""

import sys
import traceback
import numpy as np
from pathlib import Path
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_lambda_return():
    """
    测试λ-return方法
    Test λ-return methods
    """
    print("\n" + "="*60)
    print("测试λ-return...")
    print("Testing λ-return...")
    print("="*60)
    
    try:
        from src.ch12_eligibility_traces.lambda_return import (
            Episode, LambdaReturn, OfflineLambdaReturn,
            SemiGradientLambdaReturn, TTD
        )
        
        # 测试Episode
        print("\n测试Episode数据结构...")
        episode = Episode(
            states=[0, 1, 2, 3],
            actions=[0, 1, 0, 1],
            rewards=[1.0, -1.0, 2.0, -0.5]
        )
        
        assert episode.length == 4
        g = episode.compute_return(0, gamma=0.9)
        assert isinstance(g, float)
        print(f"  ✓ Episode测试通过，回报={g:.3f}")
        
        # 测试LambdaReturn计算器
        print("\n测试λ-return计算器...")
        lambda_calc = LambdaReturn(lambda_=0.8, gamma=0.9)
        g_lambda = lambda_calc.compute_lambda_return(episode, t=0)
        assert isinstance(g_lambda, float)
        print(f"  ✓ λ-return计算测试通过，G^λ={g_lambda:.3f}")
        
        # 测试离线λ-return
        print("\n测试离线λ-return...")
        n_features = 8
        
        def simple_features(state):
            features = np.zeros(n_features)
            if isinstance(state, int):
                features[state % n_features] = 1.0
            return features
        
        offline_lambda = OfflineLambdaReturn(
            n_features=n_features,
            feature_extractor=simple_features,
            lambda_=0.9,
            alpha=0.1,
            gamma=0.9
        )
        
        offline_lambda.learn_episode(episode)
        assert offline_lambda.episode_count == 1
        assert offline_lambda.total_updates > 0
        print(f"  ✓ 离线λ-return测试通过，更新{offline_lambda.total_updates}次")
        
        # 测试半梯度λ-return
        print("\n测试半梯度λ-return...")
        sg_lambda = SemiGradientLambdaReturn(
            n_features=n_features,
            feature_extractor=simple_features,
            lambda_=0.8,
            alpha=0.05,
            gamma=0.9
        )
        
        sg_lambda.start_episode()
        for i in range(3):
            sg_lambda.step(i, -1.0, i+1, i == 2)
        
        assert sg_lambda.update_count > 0
        print(f"  ✓ 半梯度λ-return测试通过")
        
        # 测试TTD
        print("\n测试TTD...")
        ttd = TTD(
            n_features=n_features,
            feature_extractor=simple_features,
            lambda_=0.9,
            alpha=0.05,
            gamma=0.9,
            horizon=3
        )
        
        for i in range(5):
            ttd.step(i % 3, -1.0, (i+1) % 3, False)
        
        assert ttd.step_count == 5
        print(f"  ✓ TTD测试通过，步数={ttd.step_count}")
        
        print("\n✅ λ-return测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ λ-return测试失败: {e}")
        traceback.print_exc()
        return False


def test_td_lambda():
    """
    测试TD(λ)算法
    Test TD(λ) algorithms
    """
    print("\n" + "="*60)
    print("测试TD(λ)算法...")
    print("Testing TD(λ) Algorithms...")
    print("="*60)
    
    try:
        from src.ch12_eligibility_traces.td_lambda import (
            TDLambda, TrueOnlineTDLambda, TruncatedTDLambda, VariableLambdaTD
        )
        
        n_features = 10
        
        def tile_features(state):
            features = np.zeros(n_features)
            if isinstance(state, int):
                features[state % n_features] = 1.0
                features[(state * 2) % n_features] = 0.5
            return features
        
        # 测试基础TD(λ)
        print("\n测试基础TD(λ)...")
        td_lambda = TDLambda(
            n_features=n_features,
            feature_extractor=tile_features,
            lambda_=0.9,
            alpha=0.05,
            gamma=0.95,
            trace_type='accumulating'
        )
        
        for i in range(10):
            td_error = td_lambda.update(i % 5, -1.0, (i+1) % 5, i == 9)
            assert isinstance(td_error, float)
        
        stats = td_lambda.get_statistics()
        assert 'mean_td_error' in stats
        assert td_lambda.update_count == 10
        print(f"  ✓ 基础TD(λ)测试通过，迹大小={stats['mean_trace_magnitude']:.3f}")
        
        # 测试替换迹
        print("\n测试替换迹TD(λ)...")
        td_lambda_rep = TDLambda(
            n_features=n_features,
            feature_extractor=tile_features,
            lambda_=0.9,
            alpha=0.05,
            gamma=0.95,
            trace_type='replacing'
        )
        
        for i in range(10):
            td_lambda_rep.update(i % 5, -1.0, (i+1) % 5, i == 9)
        
        assert td_lambda_rep.update_count == 10
        print(f"  ✓ 替换迹TD(λ)测试通过")
        
        # 测试真正的在线TD(λ)
        print("\n测试真正的在线TD(λ)...")
        true_online_td = TrueOnlineTDLambda(
            n_features=n_features,
            feature_extractor=tile_features,
            lambda_=0.9,
            alpha=0.05,
            gamma=0.95
        )
        
        for i in range(10):
            td_error = true_online_td.update(i % 5, -1.0, (i+1) % 5, i == 9)
            assert isinstance(td_error, float)
        
        assert true_online_td.update_count == 10
        print(f"  ✓ 真正的在线TD(λ)测试通过")
        
        # 测试截断TD(λ)
        print("\n测试截断TD(λ)...")
        truncated_td = TruncatedTDLambda(
            n_features=n_features,
            feature_extractor=tile_features,
            lambda_=0.9,
            alpha=0.05,
            gamma=0.95,
            trace_threshold=0.01
        )
        
        for i in range(20):
            truncated_td.update(i % 5, -1.0, (i+1) % 5, i == 19)
        
        stats = truncated_td.get_statistics()
        assert 'mean_active_traces' in stats
        print(f"  ✓ 截断TD(λ)测试通过，活跃迹={stats['mean_active_traces']:.1f}")
        
        # 测试变λTD
        print("\n测试变λTD...")
        def lambda_func(state):
            if isinstance(state, int):
                return 0.9 if state % 5 == 0 else 0.5
            return 0.7
        
        variable_td = VariableLambdaTD(
            n_features=n_features,
            feature_extractor=tile_features,
            lambda_function=lambda_func,
            alpha=0.05,
            gamma=0.95
        )
        
        for i in range(10):
            variable_td.update(i % 5, -1.0, (i+1) % 5, i == 9)
        
        assert variable_td.update_count == 10
        assert len(variable_td.lambda_history) == 10
        print(f"  ✓ 变λTD测试通过")
        
        print("\n✅ TD(λ)算法测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ TD(λ)算法测试失败: {e}")
        traceback.print_exc()
        return False


def test_control_traces():
    """
    测试控制算法的资格迹
    Test control algorithms with eligibility traces
    """
    print("\n" + "="*60)
    print("测试控制算法资格迹...")
    print("Testing Control with Eligibility Traces...")
    print("="*60)
    
    try:
        from src.ch12_eligibility_traces.control_traces import (
            SarsaLambda, WatkinsQLambda, PengQLambda, TrueOnlineSarsaLambda
        )
        
        n_features = 10
        n_actions = 3
        
        def sa_features(state, action):
            features = np.zeros(n_features)
            if isinstance(state, int):
                base_idx = (state * n_actions + action) % n_features
                features[base_idx] = 1.0
            return features
        
        # 测试Sarsa(λ)
        print("\n测试Sarsa(λ)...")
        sarsa_lambda = SarsaLambda(
            n_features=n_features,
            n_actions=n_actions,
            feature_extractor=sa_features,
            lambda_=0.9,
            alpha=0.1,
            gamma=0.95,
            epsilon=0.1,
            trace_type='accumulating'
        )
        
        # 模拟更新
        for i in range(10):
            state = i % 5
            action = sarsa_lambda.select_action(state)
            next_state = (i + 1) % 5
            next_action = sarsa_lambda.select_action(next_state)
            
            td_error = sarsa_lambda.update(
                state, action, -1.0, next_state, next_action, i == 9
            )
            assert isinstance(td_error, float)
        
        assert sarsa_lambda.update_count == 10
        print(f"  ✓ Sarsa(λ)测试通过")
        
        # 测试Watkins's Q(λ)
        print("\n测试Watkins's Q(λ)...")
        watkins_q = WatkinsQLambda(
            n_features=n_features,
            n_actions=n_actions,
            feature_extractor=sa_features,
            lambda_=0.9,
            alpha=0.1,
            gamma=0.95,
            epsilon=0.1
        )
        
        for i in range(10):
            state = i % 5
            action, was_greedy = watkins_q.select_action(state)
            next_state = (i + 1) % 5
            
            td_error = watkins_q.update(
                state, action, -1.0, next_state, i == 9, was_greedy
            )
            assert isinstance(td_error, float)
        
        assert watkins_q.update_count == 10
        print(f"  ✓ Watkins's Q(λ)测试通过，迹截断{watkins_q.trace_cuts}次")
        
        # 测试Peng's Q(λ)
        print("\n测试Peng's Q(λ)...")
        peng_q = PengQLambda(
            n_features=n_features,
            n_actions=n_actions,
            feature_extractor=sa_features,
            lambda_=0.9,
            alpha=0.1,
            gamma=0.95,
            epsilon=0.1
        )
        
        for i in range(10):
            state = i % 5
            action = peng_q.select_action(state)
            next_state = (i + 1) % 5
            
            td_error = peng_q.update(state, action, -1.0, next_state, i == 9)
            assert isinstance(td_error, float)
        
        assert peng_q.update_count == 10
        print(f"  ✓ Peng's Q(λ)测试通过")
        
        # 测试真正的在线Sarsa(λ)
        print("\n测试真正的在线Sarsa(λ)...")
        true_online_sarsa = TrueOnlineSarsaLambda(
            n_features=n_features,
            n_actions=n_actions,
            feature_extractor=sa_features,
            lambda_=0.9,
            alpha=0.1,
            gamma=0.95,
            epsilon=0.1
        )
        
        for i in range(10):
            state = i % 5
            action = true_online_sarsa.select_action(state)
            next_state = (i + 1) % 5
            next_action = true_online_sarsa.select_action(next_state)
            
            td_error = true_online_sarsa.update(
                state, action, -1.0, next_state, next_action, i == 9
            )
            assert isinstance(td_error, float)
        
        assert true_online_sarsa.update_count == 10
        print(f"  ✓ 真正的在线Sarsa(λ)测试通过")
        
        print("\n✅ 控制算法资格迹测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 控制算法资格迹测试失败: {e}")
        traceback.print_exc()
        return False


def test_integration():
    """
    测试集成场景
    Test integration scenarios
    """
    print("\n" + "="*60)
    print("测试集成场景...")
    print("Testing Integration Scenarios...")
    print("="*60)
    
    try:
        from src.ch12_eligibility_traces.lambda_return import Episode, LambdaReturn
        from src.ch12_eligibility_traces.td_lambda import TDLambda, TrueOnlineTDLambda
        from src.ch12_eligibility_traces.control_traces import SarsaLambda
        
        n_features = 8
        n_actions = 2
        
        # 特征提取器
        def features(state):
            f = np.zeros(n_features)
            if isinstance(state, int):
                f[state % n_features] = 1.0
            return f
        
        def sa_features(state, action):
            f = np.zeros(n_features)
            if isinstance(state, int):
                idx = (state * n_actions + action) % n_features
                f[idx] = 1.0
            return f
        
        # 创建测试回合
        print("\n创建测试回合...")
        episode = Episode(
            states=[0, 1, 2, 1, 0],
            actions=[0, 1, 0, 1, 0],
            rewards=[-1.0, -1.0, 5.0, -1.0, -1.0]
        )
        
        # 计算不同λ的回报
        print("\n不同λ值的λ-return:")
        for lambda_val in [0.0, 0.5, 1.0]:
            calc = LambdaReturn(lambda_=lambda_val, gamma=0.9)
            g = calc.compute_lambda_return(episode, t=0)
            print(f"  λ={lambda_val}: G^λ={g:.3f}")
        
        # 比较TD(λ)算法
        print("\n训练不同TD(λ)算法...")
        
        # 基础TD(λ)
        td_lambda = TDLambda(
            n_features=n_features,
            feature_extractor=features,
            lambda_=0.9,
            alpha=0.1,
            gamma=0.9
        )
        
        # 真正的在线TD(λ)
        true_online = TrueOnlineTDLambda(
            n_features=n_features,
            feature_extractor=features,
            lambda_=0.9,
            alpha=0.1,
            gamma=0.9
        )
        
        # 训练
        for _ in range(2):
            for i in range(len(episode.states) - 1):
                state = episode.states[i]
                reward = episode.rewards[i]
                next_state = episode.states[i + 1]
                done = (i == len(episode.states) - 2)
                
                td_lambda.update(state, reward, next_state, done)
                true_online.update(state, reward, next_state, done)
        
        # 比较价值估计
        print("\n价值估计比较:")
        print("状态  TD(λ)   真正在线")
        print("-" * 25)
        for state in range(3):
            v_td = td_lambda.get_value(state)
            v_online = true_online.get_value(state)
            print(f"{state:3d}  {v_td:7.3f}  {v_online:7.3f}")
        
        # 测试Sarsa(λ)
        print("\n测试Sarsa(λ)控制...")
        sarsa = SarsaLambda(
            n_features=n_features,
            n_actions=n_actions,
            feature_extractor=sa_features,
            lambda_=0.9,
            alpha=0.1,
            gamma=0.9,
            epsilon=0.1
        )
        
        # 简单环境
        class SimpleEnv:
            def __init__(self):
                self.state = 0
            
            def reset(self):
                self.state = 0
                return self.state
            
            def step(self, action):
                if action == 0:
                    self.state = max(0, self.state - 1)
                else:
                    self.state = min(2, self.state + 1)
                
                reward = 5.0 if self.state == 2 else -1.0
                done = self.state == 2
                
                return self.state, reward, done, {}
        
        env = SimpleEnv()
        total_return, steps = sarsa.learn_episode(env, max_steps=20)
        print(f"\nSarsa(λ)回合: 回报={total_return:.1f}, 步数={steps}")
        
        print("\n✅ 集成场景测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 集成场景测试失败: {e}")
        traceback.print_exc()
        return False


def main():
    """
    运行所有测试
    Run all tests
    """
    print("\n" + "="*80)
    print("第12章：资格迹 - 模块测试")
    print("Chapter 12: Eligibility Traces - Module Tests")
    print("="*80)
    
    tests = [
        ("λ-return", test_lambda_return),
        ("TD(λ)算法", test_td_lambda),
        ("控制算法资格迹", test_control_traces),
        ("集成场景", test_integration)
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
        print("\n🎉 第12章所有资格迹模块测试通过！")
        print("🎉 All Chapter 12 Eligibility Traces modules passed!")
        print("\n资格迹实现验证完成:")
        print("✓ λ-return方法")
        print("✓ TD(λ)算法")
        print("✓ Sarsa(λ)和Q(λ)")
        print("✓ 真正的在线算法")
        print("\n统一了TD和MC的优雅框架！")
        print("Elegant framework unifying TD and MC!")
        print("\n准备进入第13章：策略梯度方法")
        print("Ready to proceed to Chapter 13: Policy Gradient Methods")
    else:
        print("\n⚠️ 有些测试失败，请检查代码")
        print("⚠️ Some tests failed, please check the code")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)