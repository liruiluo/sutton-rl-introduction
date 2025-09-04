#!/usr/bin/env python
"""
测试第10章所有控制近似模块
Test all Chapter 10 Control Approximation modules

确保所有控制算法实现正确
Ensure all control algorithm implementations are correct
"""

import sys
import traceback
import numpy as np
from pathlib import Path
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_episodic_semi_gradient():
    """
    测试回合式半梯度控制
    Test episodic semi-gradient control
    """
    print("\n" + "="*60)
    print("测试回合式半梯度控制...")
    print("Testing Episodic Semi-gradient Control...")
    print("="*60)
    
    try:
        from src.ch10_on_policy_control_approximation.episodic_semi_gradient import (
            SemiGradientSarsa, SemiGradientExpectedSarsa, 
            SemiGradientNStepSarsa, MountainCarTileCoding, MountainCarState
        )
        
        n_features = 16
        n_actions = 3
        
        # 测试半梯度Sarsa
        print("\n测试半梯度Sarsa...")
        sarsa = SemiGradientSarsa(
            n_features=n_features,
            n_actions=n_actions,
            alpha=0.1,
            gamma=0.9,
            epsilon=0.1
        )
        
        # 模拟更新
        state = np.random.randn(n_features)
        action = sarsa.select_action(state)
        reward = -1.0
        next_state = np.random.randn(n_features)
        next_action = sarsa.select_action(next_state)
        
        sarsa.update(state, action, reward, next_state, next_action, False)
        assert sarsa.step_count == 1
        print(f"  ✓ 半梯度Sarsa更新测试通过，步数={sarsa.step_count}")
        
        # 测试Expected Sarsa
        print("\n测试Expected Sarsa...")
        expected_sarsa = SemiGradientExpectedSarsa(
            n_features=n_features,
            n_actions=n_actions,
            alpha=0.1,
            gamma=0.9,
            epsilon=0.1
        )
        
        # 计算期望价值
        expected_value = expected_sarsa.get_expected_value(state)
        assert isinstance(expected_value, (float, np.floating))
        
        expected_sarsa.update(state, action, reward, next_state, False)
        assert expected_sarsa.step_count == 1
        print(f"  ✓ Expected Sarsa测试通过，期望价值={expected_value:.3f}")
        
        # 测试n-step Sarsa
        print("\n测试n-step Sarsa...")
        n_step = SemiGradientNStepSarsa(
            n_features=n_features,
            n_actions=n_actions,
            n=4,
            alpha=0.1,
            gamma=0.9,
            epsilon=0.1
        )
        
        # 填充缓冲
        for i in range(6):
            s = np.random.randn(n_features)
            a = n_step.select_action(s)
            r = -1.0
            
            n_step.state_buffer.append(s)
            n_step.action_buffer.append(a)
            n_step.reward_buffer.append(r)
        
        # 计算n步回报
        if len(n_step.reward_buffer) >= n_step.n:
            G = n_step.compute_n_step_return(0)
            assert isinstance(G, (float, np.floating))
            print(f"  ✓ {n_step.n}-step Sarsa测试通过，n步回报={G:.3f}")
        
        # 测试Mountain Car瓦片编码
        print("\n测试Mountain Car瓦片编码...")
        mc_coder = MountainCarTileCoding(
            n_tilings=8,
            tiles_per_dim=8,
            iht_size=512
        )
        
        # 测试状态
        test_state = MountainCarState(position=-0.5, velocity=0.0)
        active_tiles = mc_coder.get_active_tiles(test_state)
        features = mc_coder.get_features(test_state)
        
        assert len(active_tiles) == mc_coder.n_tilings
        assert len(features) == 512
        assert np.sum(features) == mc_coder.n_tilings
        
        print(f"  ✓ Mountain Car瓦片编码测试通过")
        print(f"    活跃瓦片数: {len(active_tiles)}")
        print(f"    特征维度: {len(features)}")
        print(f"    稀疏度: {1 - np.sum(features > 0) / len(features):.1%}")
        
        print("\n✅ 回合式半梯度控制测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 回合式半梯度控制测试失败: {e}")
        traceback.print_exc()
        return False


def test_continuous_tasks():
    """
    测试连续任务算法
    Test continuing task algorithms
    """
    print("\n" + "="*60)
    print("测试连续任务算法...")
    print("Testing Continuing Task Algorithms...")
    print("="*60)
    
    try:
        from src.ch10_on_policy_control_approximation.continuous_tasks import (
            AverageRewardSetting, DifferentialSemiGradientSarsa,
            AccessControlQueuing
        )
        
        # 测试平均奖励设置
        print("\n测试平均奖励设置...")
        avg_reward = AverageRewardSetting(alpha=0.01)
        
        # 模拟奖励流
        rewards = []
        for t in range(100):
            reward = 5.0 + 2.0 * np.sin(t * 0.1)
            rewards.append(reward)
            avg_reward.update_average(reward)
        
        true_avg = avg_reward.get_true_average()
        est_avg = avg_reward.average_reward
        recent_avg = avg_reward.get_recent_average(50)
        
        assert abs(true_avg - np.mean(rewards)) < 0.01
        print(f"  ✓ 平均奖励设置测试通过")
        print(f"    真实平均: {true_avg:.3f}")
        print(f"    估计平均: {est_avg:.3f}")
        print(f"    最近平均: {recent_avg:.3f}")
        
        # 测试差分Sarsa
        print("\n测试差分半梯度Sarsa...")
        n_features = 8
        n_actions = 2
        
        diff_sarsa = DifferentialSemiGradientSarsa(
            n_features=n_features,
            n_actions=n_actions,
            alpha=0.1,
            beta=0.01,
            epsilon=0.1
        )
        
        # 模拟学习
        for step in range(10):
            state = np.random.randn(n_features)
            action = diff_sarsa.select_action(state)
            reward = np.random.randn() + 1.0
            next_state = np.random.randn(n_features)
            next_action = diff_sarsa.select_action(next_state)
            
            diff_sarsa.update(state, action, reward, next_state, next_action)
        
        assert diff_sarsa.step_count == 10
        assert len(diff_sarsa.td_errors) == 10
        print(f"  ✓ 差分Sarsa测试通过")
        print(f"    平均奖励: {diff_sarsa.average_reward:.3f}")
        print(f"    TD误差: {np.mean(np.abs(diff_sarsa.td_errors)):.3f}")
        
        # 测试队列系统
        print("\n测试Access-Control队列...")
        queue_env = AccessControlQueuing(
            n_servers=5,
            n_priorities=4,
            queue_capacity=10,
            arrival_prob=0.6
        )
        
        state = queue_env.reset()
        assert len(state) == 1 + queue_env.n_priorities
        
        # 运行几步
        total_reward = 0.0
        for _ in range(20):
            action = np.random.choice([0, 1])  # 随机接受/拒绝
            next_state, reward, done, info = queue_env.step(action)
            total_reward += reward
            assert not done  # 连续任务永不结束
            state = next_state
        
        print(f"  ✓ 队列系统测试通过")
        print(f"    接受顾客: {queue_env.accepted_customers}")
        print(f"    拒绝顾客: {queue_env.rejected_customers}")
        print(f"    总奖励: {total_reward:.1f}")
        
        print("\n✅ 连续任务算法测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 连续任务算法测试失败: {e}")
        traceback.print_exc()
        return False


def test_control_with_fa():
    """
    测试通用控制框架
    Test general control framework
    """
    print("\n" + "="*60)
    print("测试通用控制框架...")
    print("Testing General Control Framework...")
    print("="*60)
    
    try:
        from src.ch10_on_policy_control_approximation.control_with_fa import (
            LinearActionValueFunction, ControlWithFA, ActorCriticWithFA
        )
        
        n_features = 8
        n_actions = 3
        
        # 简单特征提取器
        def simple_features(state, action):
            if isinstance(state, np.ndarray):
                return state[:n_features]
            return np.random.randn(n_features)
        
        # 测试线性动作价值函数
        print("\n测试线性动作价值函数...")
        linear_q = LinearActionValueFunction(
            feature_extractor=simple_features,
            n_features=n_features,
            n_actions=n_actions
        )
        
        test_state = np.random.randn(n_features)
        
        # 测试预测
        for a in range(n_actions):
            q_val = linear_q.predict(test_state, a)
            assert isinstance(q_val, (float, np.floating))
        
        # 测试更新
        action = 0
        target = 1.0
        td_error = linear_q.update(test_state, action, target, alpha=0.1)
        assert isinstance(td_error, (float, np.floating))
        assert linear_q.update_count == 1
        
        print(f"  ✓ 线性动作价值函数测试通过")
        
        # 测试控制框架 - Sarsa
        print("\n测试控制框架(Sarsa)...")
        sarsa_controller = ControlWithFA(
            approximator=linear_q,
            n_actions=n_actions,
            alpha=0.1,
            gamma=0.9,
            epsilon=0.1,
            method='sarsa'
        )
        
        state = np.random.randn(n_features)
        action = sarsa_controller.select_action(state)
        reward = -1.0
        next_state = np.random.randn(n_features)
        
        td_error = sarsa_controller.learn_step(state, action, reward, next_state, False)
        assert isinstance(td_error, (float, np.floating))
        assert sarsa_controller.step_count == 1
        
        print(f"  ✓ Sarsa控制器测试通过")
        
        # 测试控制框架 - Q-learning
        print("\n测试控制框架(Q-learning)...")
        q_approximator = LinearActionValueFunction(
            feature_extractor=simple_features,
            n_features=n_features,
            n_actions=n_actions
        )
        
        q_controller = ControlWithFA(
            approximator=q_approximator,
            n_actions=n_actions,
            alpha=0.1,
            gamma=0.9,
            epsilon=0.1,
            method='q_learning'
        )
        
        td_error = q_controller.learn_step(state, action, reward, next_state, False)
        assert isinstance(td_error, (float, np.floating))
        
        print(f"  ✓ Q-learning控制器测试通过")
        
        # 测试控制框架 - Expected Sarsa
        print("\n测试控制框架(Expected Sarsa)...")
        exp_approximator = LinearActionValueFunction(
            feature_extractor=simple_features,
            n_features=n_features,
            n_actions=n_actions
        )
        
        exp_controller = ControlWithFA(
            approximator=exp_approximator,
            n_actions=n_actions,
            alpha=0.1,
            gamma=0.9,
            epsilon=0.1,
            method='expected_sarsa'
        )
        
        td_error = exp_controller.learn_step(state, action, reward, next_state, False)
        assert isinstance(td_error, (float, np.floating))
        
        print(f"  ✓ Expected Sarsa控制器测试通过")
        
        # 测试Actor-Critic
        print("\n测试Actor-Critic架构...")
        state_dim = 4
        ac = ActorCriticWithFA(
            state_dim=state_dim,
            n_actions=n_actions,
            actor_lr=0.01,
            critic_lr=0.1,
            gamma=0.9
        )
        
        test_state = np.random.randn(state_dim)
        
        # 测试动作概率
        probs = ac.get_action_probabilities(test_state)
        assert len(probs) == n_actions
        assert abs(np.sum(probs) - 1.0) < 1e-6
        
        # 测试动作选择
        action = ac.select_action(test_state)
        assert 0 <= action < n_actions
        
        # 测试状态价值
        value = ac.get_state_value(test_state)
        assert isinstance(value, (float, np.floating))
        
        # 测试更新
        reward = 1.0
        next_state = np.random.randn(state_dim)
        ac.update(test_state, action, reward, next_state, False)
        
        print(f"  ✓ Actor-Critic测试通过")
        print(f"    动作概率: {[f'{p:.3f}' for p in probs]}")
        print(f"    状态价值: {value:.3f}")
        
        print("\n✅ 通用控制框架测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 通用控制框架测试失败: {e}")
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
        from src.ch10_on_policy_control_approximation.episodic_semi_gradient import (
            SemiGradientSarsa
        )
        from src.ch10_on_policy_control_approximation.continuous_tasks import (
            DifferentialSemiGradientSarsa, AccessControlQueuing
        )
        
        # 测试回合式到连续任务的转换
        print("\n测试任务类型转换...")
        
        # 回合式任务
        n_features = 8
        n_actions = 2
        
        episodic_agent = SemiGradientSarsa(
            n_features=n_features,
            n_actions=n_actions,
            alpha=0.1,
            gamma=0.9,
            epsilon=0.1
        )
        
        # 连续任务
        continuous_agent = DifferentialSemiGradientSarsa(
            n_features=n_features,
            n_actions=n_actions,
            alpha=0.1,
            beta=0.01,
            epsilon=0.1
        )
        
        # 模拟转换
        state = np.random.randn(n_features)
        
        # 两种agent选择动作
        episodic_action = episodic_agent.select_action(state)
        continuous_action = continuous_agent.select_action(state)
        
        assert 0 <= episodic_action < n_actions
        assert 0 <= continuous_action < n_actions
        
        print(f"  ✓ 任务类型转换测试通过")
        print(f"    回合式选择: 动作{episodic_action}")
        print(f"    连续任务选择: 动作{continuous_action}")
        
        # 测试队列系统与差分Sarsa集成
        print("\n测试队列系统集成...")
        queue_env = AccessControlQueuing(
            n_servers=3,
            n_priorities=2,
            queue_capacity=5
        )
        
        state_size = 1 + queue_env.n_priorities
        learner = DifferentialSemiGradientSarsa(
            n_features=state_size,
            n_actions=2,
            alpha=0.1,
            beta=0.01,
            epsilon=0.2
        )
        
        # 学习几步
        state = queue_env.reset()
        action = learner.select_action(state)
        
        for step in range(20):
            next_state, reward, _, _ = queue_env.step(action)
            next_action = learner.select_action(next_state)
            
            learner.update(state, action, reward, next_state, next_action)
            
            state = next_state
            action = next_action
        
        assert learner.step_count == 20
        print(f"  ✓ 队列系统集成测试通过")
        print(f"    学习步数: {learner.step_count}")
        print(f"    平均奖励: {learner.average_reward:.3f}")
        
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
    print("第10章：使用近似的同策略控制 - 模块测试")
    print("Chapter 10: On-policy Control with Approximation - Module Tests")
    print("="*80)
    
    tests = [
        ("回合式半梯度控制", test_episodic_semi_gradient),
        ("连续任务算法", test_continuous_tasks),
        ("通用控制框架", test_control_with_fa),
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
        print("\n🎉 第10章所有控制近似模块测试通过！")
        print("🎉 All Chapter 10 Control Approximation modules passed!")
        print("\n控制近似实现验证完成:")
        print("✓ 回合式半梯度控制")
        print("✓ 连续任务与平均奖励")
        print("✓ 通用控制框架")
        print("✓ Actor-Critic架构")
        print("\n从预测到控制的成功扩展！")
        print("Successful extension from prediction to control!")
        print("\n准备进入第11章：离策略方法")
        print("Ready to proceed to Chapter 11: Off-policy Methods")
    else:
        print("\n⚠️ 有些测试失败，请检查代码")
        print("⚠️ Some tests failed, please check the code")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)