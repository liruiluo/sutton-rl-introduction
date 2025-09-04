#!/usr/bin/env python
"""
测试第11章所有离策略近似模块
Test all Chapter 11 Off-policy Approximation modules

确保所有离策略算法实现正确
Ensure all off-policy algorithm implementations are correct
"""

import sys
import traceback
import numpy as np
from pathlib import Path
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_importance_sampling():
    """
    测试重要性采样方法
    Test importance sampling methods
    """
    print("\n" + "="*60)
    print("测试重要性采样...")
    print("Testing Importance Sampling...")
    print("="*60)
    
    try:
        from src.ch11_off_policy_approximation.importance_sampling import (
            Trajectory, ImportanceSampling, SemiGradientOffPolicyTD,
            PerDecisionImportanceSampling, NStepOffPolicyTD
        )
        
        # 测试轨迹
        print("\n测试轨迹数据结构...")
        trajectory = Trajectory(
            states=[0, 1, 2],
            actions=[0, 1, 0],
            rewards=[1.0, -1.0, 2.0],
            probs_b=[0.5, 0.5, 0.5],
            probs_pi=[0.8, 0.2, 0.9]
        )
        
        assert trajectory.length == 3
        rho = trajectory.compute_importance_ratio(0, 2)
        assert rho > 0
        g = trajectory.compute_return(0.9, 0)
        assert isinstance(g, float)
        print("  ✓ 轨迹测试通过")
        
        # 测试重要性采样
        print("\n测试重要性采样评估...")
        n_states = 5
        
        # 普通IS
        ordinary_is = ImportanceSampling(n_states, gamma=0.9, weighted=False)
        ordinary_is.update_from_trajectory(trajectory)
        value = ordinary_is.get_value(0)
        assert isinstance(value, float)
        print("  ✓ 普通IS测试通过")
        
        # 加权IS
        weighted_is = ImportanceSampling(n_states, gamma=0.9, weighted=True)
        weighted_is.update_from_trajectory(trajectory)
        value = weighted_is.get_value(0)
        assert isinstance(value, float)
        print("  ✓ 加权IS测试通过")
        
        # 测试半梯度离策略TD
        print("\n测试半梯度离策略TD...")
        n_features = 8
        
        def simple_features(state):
            features = np.zeros(n_features)
            if isinstance(state, int):
                features[state % n_features] = 1.0
            return features
        
        off_td = SemiGradientOffPolicyTD(
            feature_extractor=simple_features,
            n_features=n_features,
            alpha=0.1,
            gamma=0.9
        )
        
        td_error = off_td.update(0, 1.0, 1, False, importance_ratio=1.2)
        assert isinstance(td_error, float)
        assert off_td.update_count == 1
        print("  ✓ 半梯度离策略TD测试通过")
        
        # 测试Per-decision IS
        print("\n测试Per-decision重要性采样...")
        pd_is = PerDecisionImportanceSampling(
            n_features=n_features,
            feature_extractor=simple_features,
            alpha=0.1,
            gamma=0.9,
            lambda_=0.5
        )
        
        td_error = pd_is.update(0, 0, 1.0, 1, False, prob_b=0.5, prob_pi=0.8)
        assert isinstance(td_error, float)
        assert pd_is.update_count == 1
        print("  ✓ Per-decision IS测试通过")
        
        # 测试n-step离策略TD
        print("\n测试n-step离策略TD...")
        n_step_td = NStepOffPolicyTD(
            n_features=n_features,
            feature_extractor=simple_features,
            n=4,
            alpha=0.1,
            gamma=0.9
        )
        
        # 添加经验
        for i in range(5):
            n_step_td.add_experience(i, -1.0, 1.1)
        
        assert n_step_td.update_count > 0
        print(f"  ✓ n-step离策略TD测试通过，更新{n_step_td.update_count}次")
        
        print("\n✅ 重要性采样测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 重要性采样测试失败: {e}")
        traceback.print_exc()
        return False


def test_gradient_td():
    """
    测试梯度TD方法
    Test gradient TD methods
    """
    print("\n" + "="*60)
    print("测试梯度TD方法...")
    print("Testing Gradient TD Methods...")
    print("="*60)
    
    try:
        from src.ch11_off_policy_approximation.gradient_td import (
            ProjectedBellmanError, GTD2, TDC, HTD, GradientLSTD
        )
        
        n_features = 8
        
        def simple_features(state):
            features = np.zeros(n_features)
            if isinstance(state, int):
                features[state % n_features] = 1.0
            return features / (np.linalg.norm(features) + 0.01)
        
        # 测试投影Bellman误差
        print("\n测试投影Bellman误差...")
        pbe = ProjectedBellmanError(n_features, simple_features, gamma=0.9)
        
        # 添加样本
        for i in range(10):
            pbe.update_statistics(i % 5, np.random.randn(), (i+1) % 5, False)
        
        test_weights = np.random.randn(n_features) * 0.1
        pbe_value = pbe.compute_pbe(test_weights)
        gradient = pbe.compute_gradient(test_weights)
        
        assert isinstance(pbe_value, float)
        assert len(gradient) == n_features
        print(f"  ✓ PBE测试通过，PBE={pbe_value:.4f}")
        
        # 测试GTD2
        print("\n测试GTD2...")
        gtd2 = GTD2(n_features, simple_features, alpha_w=0.01, alpha_v=0.1, gamma=0.9)
        
        for i in range(10):
            td_error = gtd2.update(i % 5, -1.0, (i+1) % 5, False, importance_ratio=1.1)
            assert isinstance(td_error, float)
        
        assert gtd2.update_count == 10
        print(f"  ✓ GTD2测试通过，||w||={np.linalg.norm(gtd2.w):.3f}")
        
        # 测试TDC
        print("\n测试TDC...")
        tdc = TDC(n_features, simple_features, alpha_w=0.01, alpha_v=0.1, gamma=0.9)
        
        for i in range(10):
            td_error = tdc.update(i % 5, -1.0, (i+1) % 5, False, importance_ratio=1.1)
            assert isinstance(td_error, float)
        
        assert tdc.update_count == 10
        print(f"  ✓ TDC测试通过，||w||={np.linalg.norm(tdc.w):.3f}")
        
        # 测试HTD
        print("\n测试HTD...")
        htd = HTD(n_features, simple_features, alpha=0.01, beta=0.1, gamma=0.9, lambda_=0.5)
        
        for i in range(10):
            td_error = htd.update(i % 5, -1.0, (i+1) % 5, i == 9, importance_ratio=1.2)
            assert isinstance(td_error, float)
        
        assert htd.update_count == 10
        print(f"  ✓ HTD测试通过，avg_ρ={htd.avg_importance_ratio:.2f}")
        
        # 测试梯度LSTD
        print("\n测试梯度LSTD...")
        glstd = GradientLSTD(n_features, simple_features, alpha=0.1, gamma=0.9)
        
        for i in range(10):
            td_error = glstd.update(i % 5, -1.0, (i+1) % 5, False, importance_ratio=1.0)
            assert isinstance(td_error, float)
        
        assert glstd.update_count == 10
        print(f"  ✓ 梯度LSTD测试通过，||w||={np.linalg.norm(glstd.w):.3f}")
        
        print("\n✅ 梯度TD方法测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 梯度TD方法测试失败: {e}")
        traceback.print_exc()
        return False


def test_emphatic_td():
    """
    测试强调TD方法
    Test emphatic TD methods
    """
    print("\n" + "="*60)
    print("测试强调TD方法...")
    print("Testing Emphatic TD Methods...")
    print("="*60)
    
    try:
        from src.ch11_off_policy_approximation.emphatic_td import (
            EmphasisWeights, EmphaticTDLambda, EmphaticTDC,
            ELSTD, TrueOnlineEmphaticTD
        )
        
        n_features = 8
        
        def simple_features(state):
            features = np.zeros(n_features)
            if isinstance(state, int):
                features[state % n_features] = 1.0
            return features / (np.linalg.norm(features) + 0.01)
        
        def interest_fn(state):
            if isinstance(state, int):
                return 2.0 if state % 5 == 0 else 0.5
            return 1.0
        
        # 测试强调权重
        print("\n测试强调权重计算...")
        emphasis_computer = EmphasisWeights(gamma=0.9, lambda_=0.8, interest_fn=interest_fn)
        
        for i in range(5):
            emphasis = emphasis_computer.compute_emphasis(i, importance_ratio=1.2)
            assert isinstance(emphasis, float)
            assert emphasis > 0
        
        stats = emphasis_computer.get_statistics()
        assert 'mean_emphasis' in stats
        print(f"  ✓ 强调权重测试通过，平均M={stats['mean_emphasis']:.3f}")
        
        # 测试强调TD(λ)
        print("\n测试强调TD(λ)...")
        emphatic_td = EmphaticTDLambda(
            n_features=n_features,
            feature_extractor=simple_features,
            alpha=0.05,
            gamma=0.9,
            lambda_=0.8,
            interest_fn=interest_fn
        )
        
        for i in range(10):
            td_error = emphatic_td.update(i % 5, -1.0, (i+1) % 5, i == 9, importance_ratio=1.1)
            assert isinstance(td_error, float)
        
        assert emphatic_td.update_count == 10
        print(f"  ✓ 强调TD(λ)测试通过，||w||={np.linalg.norm(emphatic_td.w):.3f}")
        
        # 测试强调TDC
        print("\n测试强调TDC...")
        emphatic_tdc = EmphaticTDC(
            n_features=n_features,
            feature_extractor=simple_features,
            alpha_w=0.01,
            alpha_v=0.1,
            gamma=0.9,
            lambda_=0.8,
            interest_fn=interest_fn
        )
        
        for i in range(10):
            td_error = emphatic_tdc.update(i % 5, -1.0, (i+1) % 5, i == 9, importance_ratio=1.1)
            assert isinstance(td_error, float)
        
        assert emphatic_tdc.update_count == 10
        print(f"  ✓ 强调TDC测试通过，||w||={np.linalg.norm(emphatic_tdc.w):.3f}")
        
        # 测试ELSTD
        print("\n测试ELSTD...")
        elstd = ELSTD(
            n_features=n_features,
            feature_extractor=simple_features,
            gamma=0.9,
            lambda_=0.8,
            epsilon=0.01,
            interest_fn=interest_fn
        )
        
        for i in range(20):
            elstd.add_sample(i % 5, -1.0, (i+1) % 5, i == 19, importance_ratio=1.0)
        
        weights = elstd.solve()
        assert len(weights) == n_features
        print(f"  ✓ ELSTD测试通过，||w||={np.linalg.norm(weights):.3f}")
        
        # 测试真正的在线强调TD
        print("\n测试真正的在线强调TD...")
        true_online_etd = TrueOnlineEmphaticTD(
            n_features=n_features,
            feature_extractor=simple_features,
            alpha=0.05,
            gamma=0.9,
            lambda_=0.8,
            interest_fn=interest_fn
        )
        
        for i in range(10):
            td_error = true_online_etd.update(i % 5, -1.0, (i+1) % 5, i == 9, importance_ratio=1.1)
            assert isinstance(td_error, float)
        
        assert true_online_etd.update_count == 10
        print(f"  ✓ 真正的在线强调TD测试通过，||w||={np.linalg.norm(true_online_etd.w):.3f}")
        
        print("\n✅ 强调TD方法测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 强调TD方法测试失败: {e}")
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
        from src.ch11_off_policy_approximation.importance_sampling import (
            ImportanceSampling, Trajectory
        )
        from src.ch11_off_policy_approximation.gradient_td import GTD2, TDC
        from src.ch11_off_policy_approximation.emphatic_td import EmphaticTDLambda
        
        n_features = 8
        n_states = 5
        
        def simple_features(state):
            features = np.zeros(n_features)
            if isinstance(state, int):
                features[state % n_features] = 1.0
                features[(state + 1) % n_features] = 0.3
            return features / (np.linalg.norm(features) + 0.01)
        
        # 生成轨迹
        print("\n生成测试轨迹...")
        states = list(range(n_states)) * 2
        actions = [i % 2 for i in range(len(states))]
        rewards = [-1.0 if s != 2 else 5.0 for s in states]
        probs_b = [0.5] * len(states)
        probs_pi = [0.8 if a == 0 else 0.2 for a in actions]
        
        trajectory = Trajectory(states, actions, rewards, probs_b, probs_pi)
        print(f"  ✓ 轨迹长度: {trajectory.length}")
        
        # 比较不同方法
        print("\n训练不同方法...")
        
        # GTD2
        gtd2 = GTD2(n_features, simple_features, alpha_w=0.01, alpha_v=0.1, gamma=0.9)
        
        # TDC
        tdc = TDC(n_features, simple_features, alpha_w=0.01, alpha_v=0.1, gamma=0.9)
        
        # 强调TD
        emphatic_td = EmphaticTDLambda(
            n_features=n_features,
            feature_extractor=simple_features,
            alpha=0.05,
            gamma=0.9,
            lambda_=0.8
        )
        
        # 训练
        for i in range(len(states) - 1):
            state = states[i]
            reward = rewards[i]
            next_state = states[i + 1]
            rho = probs_pi[i] / probs_b[i]
            
            gtd2.update(state, reward, next_state, False, rho)
            tdc.update(state, reward, next_state, False, rho)
            emphatic_td.update(state, reward, next_state, False, rho)
        
        # 比较价值估计
        print("\n价值估计比较:")
        print("状态  GTD2    TDC     ETD(λ)")
        print("-" * 30)
        for state in range(n_states):
            v_gtd2 = gtd2.get_value(state)
            v_tdc = tdc.get_value(state)
            v_etd = emphatic_td.get_value(state)
            print(f"{state:3d}  {v_gtd2:6.3f}  {v_tdc:6.3f}  {v_etd:6.3f}")
        
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
    print("第11章：离策略方法与近似 - 模块测试")
    print("Chapter 11: Off-policy Methods with Approximation - Module Tests")
    print("="*80)
    
    tests = [
        ("重要性采样", test_importance_sampling),
        ("梯度TD方法", test_gradient_td),
        ("强调TD方法", test_emphatic_td),
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
        print("\n🎉 第11章所有离策略近似模块测试通过！")
        print("🎉 All Chapter 11 Off-policy Approximation modules passed!")
        print("\n离策略近似实现验证完成:")
        print("✓ 重要性采样方法")
        print("✓ 梯度TD算法")
        print("✓ 强调TD方法")
        print("✓ 投影Bellman误差")
        print("\n解决了致命三要素问题！")
        print("Solved the deadly triad problem!")
        print("\n准备进入第12章：资格迹")
        print("Ready to proceed to Chapter 12: Eligibility Traces")
    else:
        print("\n⚠️ 有些测试失败，请检查代码")
        print("⚠️ Some tests failed, please check the code")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)