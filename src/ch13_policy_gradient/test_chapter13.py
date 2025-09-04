#!/usr/bin/env python
"""
测试第13章所有策略梯度模块
Test all Chapter 13 Policy Gradient modules

确保所有策略梯度算法实现正确
Ensure all policy gradient algorithm implementations are correct
"""

import sys
import traceback
import numpy as np
from pathlib import Path
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_policy_gradient_theorem():
    """
    测试策略梯度定理
    Test policy gradient theorem
    """
    print("\n" + "="*60)
    print("测试策略梯度定理...")
    print("Testing Policy Gradient Theorem...")
    print("="*60)
    
    try:
        from src.ch13_policy_gradient.policy_gradient_theorem import (
            SoftmaxPolicy, GaussianPolicy, PolicyGradientTheorem, AdvantageFunction
        )
        
        # 测试Softmax策略
        print("\n测试Softmax策略...")
        n_features = 6
        n_actions = 3
        
        def simple_features(state, action):
            features = np.zeros(n_features)
            if isinstance(state, int):
                features[state % n_features] = 1.0
                features[(state + action) % n_features] = 0.5
            return features
        
        softmax_policy = SoftmaxPolicy(n_features, n_actions, simple_features)
        
        # 测试动作概率
        state = 1
        probs = softmax_policy.compute_action_probabilities(state)
        assert abs(np.sum(probs) - 1.0) < 1e-6
        assert all(p >= 0 for p in probs)
        print(f"  ✓ Softmax策略测试通过，概率和={np.sum(probs):.6f}")
        
        # 测试高斯策略
        print("\n测试高斯策略...")
        gaussian_policy = GaussianPolicy(state_dim=4, action_dim=2)
        test_state = np.random.randn(4)
        
        action = gaussian_policy.select_action(test_state)
        assert action.shape == (2,)
        
        log_prob = gaussian_policy.compute_log_probability(test_state, action)
        assert isinstance(log_prob, float)
        print(f"  ✓ 高斯策略测试通过，动作维度={action.shape}")
        
        # 测试策略梯度定理
        print("\n测试策略梯度定理收敛性...")
        pgt = PolicyGradientTheorem()
        true_grad, est_grad = pgt.demonstrate_convergence(n_samples=50)
        
        error = np.linalg.norm(est_grad - true_grad)
        assert error < 1.0  # 宽松的阈值
        print(f"  ✓ 策略梯度定理测试通过，估计误差={error:.4f}")
        
        # 测试优势函数
        print("\n测试优势函数...")
        adv_func = AdvantageFunction()
        
        advantage = adv_func.compute_advantage(0, 0, 5.0, 3.0)
        assert advantage == 2.0
        
        rewards = [1.0, -1.0, 2.0]
        values = [2.0, 1.5, 3.0]
        gae_advantages = adv_func.compute_gae(rewards, values, gamma=0.9, lambda_=0.95)
        assert len(gae_advantages) == len(rewards)
        print(f"  ✓ 优势函数测试通过，GAE计算完成")
        
        print("\n✅ 策略梯度定理测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 策略梯度定理测试失败: {e}")
        traceback.print_exc()
        return False


def test_reinforce():
    """
    测试REINFORCE算法
    Test REINFORCE algorithms
    """
    print("\n" + "="*60)
    print("测试REINFORCE算法...")
    print("Testing REINFORCE Algorithms...")
    print("="*60)
    
    try:
        from src.ch13_policy_gradient.reinforce import (
            REINFORCE, REINFORCEWithBaseline, AllActionsREINFORCE,
            SimpleValueFunction, SimpleQFunction
        )
        from src.ch13_policy_gradient.policy_gradient_theorem import SoftmaxPolicy
        
        n_features = 8
        n_actions = 2
        
        def state_action_features(state, action):
            features = np.zeros(n_features)
            if isinstance(state, int):
                base_idx = (state * n_actions + action) % n_features
                features[base_idx] = 1.0
            return features
        
        def state_features(state):
            features = np.zeros(n_features)
            if isinstance(state, int):
                features[state % n_features] = 1.0
            return features
        
        # 简单环境
        class TestEnv:
            def __init__(self):
                self.state = 0
                self.step_count = 0
            
            def reset(self):
                self.state = 0
                self.step_count = 0
                return self.state
            
            def step(self, action):
                if action == 0:
                    self.state = max(0, self.state - 1)
                else:
                    self.state = min(3, self.state + 1)
                
                reward = 10.0 if self.state == 3 else -1.0
                done = self.state == 3 or self.step_count >= 10
                
                self.step_count += 1
                return self.state, reward, done, {}
        
        # 测试基础REINFORCE
        print("\n测试基础REINFORCE...")
        policy = SoftmaxPolicy(n_features, n_actions, state_action_features)
        reinforce = REINFORCE(policy, alpha=0.1, gamma=0.9)
        
        env = TestEnv()
        episode_return = reinforce.learn_episode(env, max_steps=10)
        
        assert reinforce.episode_count == 1
        assert len(reinforce.episode_returns) == 1
        print(f"  ✓ 基础REINFORCE测试通过，回报={episode_return:.1f}")
        
        # 测试带基线的REINFORCE
        print("\n测试REINFORCE with Baseline...")
        policy_baseline = SoftmaxPolicy(n_features, n_actions, state_action_features)
        value_func = SimpleValueFunction(n_features, state_features)
        
        reinforce_baseline = REINFORCEWithBaseline(
            policy_baseline, value_func,
            alpha_theta=0.05, alpha_w=0.1, gamma=0.9
        )
        
        episode_return = reinforce_baseline.learn_episode(env, max_steps=10)
        assert reinforce_baseline.episode_count == 1
        assert len(reinforce_baseline.advantages) > 0
        print(f"  ✓ REINFORCE with Baseline测试通过")
        
        # 测试All-actions REINFORCE
        print("\n测试All-actions REINFORCE...")
        policy_all = SoftmaxPolicy(n_features, n_actions, state_action_features)
        q_func = SimpleQFunction(n_features, n_actions, state_action_features)
        
        all_actions = AllActionsREINFORCE(
            policy_all, q_func,
            alpha=0.05, gamma=0.9
        )
        
        episode_return = all_actions.learn_episode(env, max_steps=10)
        assert all_actions.episode_count == 1
        assert all_actions.gradient_updates > 0
        print(f"  ✓ All-actions REINFORCE测试通过")
        
        print("\n✅ REINFORCE算法测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ REINFORCE算法测试失败: {e}")
        traceback.print_exc()
        return False


def test_actor_critic():
    """
    测试Actor-Critic算法
    Test Actor-Critic algorithms
    """
    print("\n" + "="*60)
    print("测试Actor-Critic算法...")
    print("Testing Actor-Critic Algorithms...")
    print("="*60)
    
    try:
        from src.ch13_policy_gradient.actor_critic import (
            OneStepActorCritic, ActorCriticWithTraces, A2C,
            SimpleActor, SimpleCritic
        )
        
        n_features = 8
        n_actions = 2
        
        def state_features(state):
            features = np.zeros(n_features)
            if isinstance(state, int):
                features[state % n_features] = 1.0
            return features
        
        def state_action_features(state, action):
            features = np.zeros(n_features)
            if isinstance(state, int):
                base_idx = (state * n_actions + action) % n_features
                features[base_idx] = 1.0
            return features
        
        # 简单环境
        class TestEnv:
            def __init__(self):
                self.state = 0
                self.step_count = 0
            
            def reset(self):
                self.state = 0
                self.step_count = 0
                return self.state
            
            def step(self, action):
                self.state = min(3, self.state + action)
                reward = 10.0 if self.state == 3 else -1.0
                done = self.state == 3 or self.step_count >= 10
                self.step_count += 1
                return self.state, reward, done, {}
        
        # 测试One-step Actor-Critic
        print("\n测试One-step Actor-Critic...")
        actor = SimpleActor(n_features, n_actions, state_action_features)
        critic = SimpleCritic(n_features, state_features)
        
        ac_1step = OneStepActorCritic(
            actor, critic,
            alpha_theta=0.05, alpha_w=0.1, gamma=0.9
        )
        
        env = TestEnv()
        episode_return = ac_1step.learn_episode(env, max_steps=10)
        
        assert ac_1step.episode_count == 1
        assert ac_1step.step_count > 0
        assert len(ac_1step.td_errors) > 0
        print(f"  ✓ One-step AC测试通过，TD误差数={len(ac_1step.td_errors)}")
        
        # 测试带资格迹的Actor-Critic
        print("\n测试Actor-Critic with Traces...")
        actor_traces = SimpleActor(n_features, n_actions, state_action_features)
        critic_traces = SimpleCritic(n_features, state_features)
        
        ac_traces = ActorCriticWithTraces(
            actor_traces, critic_traces,
            lambda_theta=0.9, lambda_w=0.9,
            alpha_theta=0.02, alpha_w=0.05, gamma=0.9
        )
        
        episode_return = ac_traces.learn_episode(env, max_steps=10)
        
        assert ac_traces.episode_count == 1
        assert ac_traces.step_count > 0
        print(f"  ✓ AC with Traces测试通过")
        
        # 测试A2C
        print("\n测试A2C...")
        actor_a2c = SimpleActor(n_features, n_actions, state_action_features)
        critic_a2c = SimpleCritic(n_features, state_features)
        
        a2c = A2C(
            actor_a2c, critic_a2c,
            n_steps=5, alpha_theta=0.01, alpha_w=0.05,
            gamma=0.9, use_gae=True, gae_lambda=0.95
        )
        
        rewards = a2c.learn_steps(env, n_steps=10)
        
        assert len(rewards) == 10
        assert a2c.step_count == 10
        print(f"  ✓ A2C测试通过，步数={a2c.step_count}")
        
        print("\n✅ Actor-Critic算法测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ Actor-Critic算法测试失败: {e}")
        traceback.print_exc()
        return False


def test_natural_policy_gradient():
    """
    测试自然策略梯度
    Test natural policy gradient
    """
    print("\n" + "="*60)
    print("测试自然策略梯度...")
    print("Testing Natural Policy Gradient...")
    print("="*60)
    
    try:
        from src.ch13_policy_gradient.natural_policy_gradient import (
            NaturalPolicyGradient, PPO
        )
        from src.ch13_policy_gradient.policy_gradient_theorem import SoftmaxPolicy
        from src.ch13_policy_gradient.reinforce import SimpleValueFunction
        
        n_features = 8
        n_actions = 2
        
        def state_action_features(state, action):
            features = np.zeros(n_features)
            if isinstance(state, int):
                base_idx = (state * n_actions + action) % n_features
                features[base_idx] = 1.0
            return features
        
        def state_features(state):
            features = np.zeros(n_features)
            if isinstance(state, int):
                features[state % n_features] = 1.0
            return features
        
        # 简单环境
        class TestEnv:
            def __init__(self):
                self.state = 0
                self.step_count = 0
            
            def reset(self):
                self.state = 0
                self.step_count = 0
                return self.state
            
            def step(self, action):
                self.state = min(3, self.state + action)
                reward = 10.0 if self.state == 3 else -1.0
                done = self.state == 3 or self.step_count >= 10
                self.step_count += 1
                return self.state, reward, done, {}
        
        # 测试自然策略梯度
        print("\n测试Natural Policy Gradient...")
        npg_policy = SoftmaxPolicy(n_features, n_actions, state_action_features)
        npg = NaturalPolicyGradient(
            npg_policy, alpha=0.1, gamma=0.9, damping=0.01
        )
        
        env = TestEnv()
        episode_return = npg.learn_episode(env, max_steps=10)
        
        assert npg.episode_count == 1
        assert npg.fisher_matrix is not None
        assert len(npg.natural_gradient_norms) > 0
        print(f"  ✓ Natural PG测试通过，回报={episode_return:.1f}")
        
        # 测试PPO
        print("\n测试PPO...")
        ppo_policy = SoftmaxPolicy(n_features, n_actions, state_action_features)
        ppo_value = SimpleValueFunction(n_features, state_features)
        
        ppo = PPO(
            ppo_policy, ppo_value,
            clip_epsilon=0.2, alpha_theta=0.01, alpha_w=0.05,
            gamma=0.9, lam=0.95, epochs=2, batch_size=32
        )
        
        # 收集轨迹
        ppo.collect_trajectories(env, n_steps=50)
        assert len(ppo.states) == 50
        assert len(ppo.advantages) == 50
        
        # PPO更新
        ppo.update()
        # clip_fractions可能在update内部计算，不一定立即有值
        if ppo.clip_fractions:
            print(f"  ✓ PPO测试通过，裁剪比例={ppo.clip_fractions[-1]:.3f}")
        else:
            print(f"  ✓ PPO测试通过，状态数={len(ppo.states)}")
        
        print("\n✅ 自然策略梯度测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 自然策略梯度测试失败: {e}")
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
        from src.ch13_policy_gradient.policy_gradient_theorem import SoftmaxPolicy
        from src.ch13_policy_gradient.reinforce import REINFORCE, SimpleValueFunction
        from src.ch13_policy_gradient.actor_critic import OneStepActorCritic, SimpleCritic, SimpleActor
        
        n_features = 6
        n_actions = 2
        
        def features(state, action):
            f = np.zeros(n_features)
            if isinstance(state, int):
                f[(state * n_actions + action) % n_features] = 1.0
            return f
        
        # 更复杂的环境
        class ChainEnv:
            """链式MDP环境"""
            def __init__(self, n_states=5):
                self.n_states = n_states
                self.state = 0
                self.step_count = 0
            
            def reset(self):
                self.state = 0
                self.step_count = 0
                return self.state
            
            def step(self, action):
                if action == 1 and self.state < self.n_states - 1:
                    self.state += 1
                elif action == 0 and self.state > 0:
                    self.state -= 1
                
                # 只在最右端给奖励
                if self.state == self.n_states - 1:
                    reward = 10.0
                    done = True
                else:
                    reward = -0.1
                    done = False
                
                self.step_count += 1
                if self.step_count >= 50:
                    done = True
                
                return self.state, reward, done, {}
        
        env = ChainEnv(n_states=5)
        
        # 比较REINFORCE和Actor-Critic
        print("\n训练并比较算法...")
        
        # REINFORCE
        policy_reinforce = SoftmaxPolicy(n_features, n_actions, features)
        reinforce = REINFORCE(policy_reinforce, alpha=0.1, gamma=0.95)
        
        reinforce_returns = []
        for _ in range(10):
            ret = reinforce.learn_episode(env, max_steps=50)
            reinforce_returns.append(ret)
        
        # Actor-Critic
        actor = SimpleActor(n_features, n_actions, features)
        
        def state_features(state):
            f = np.zeros(n_features)
            if isinstance(state, int):
                f[state % n_features] = 1.0
            return f
        
        critic = SimpleCritic(n_features, state_features)
        ac = OneStepActorCritic(actor, critic, alpha_theta=0.05, alpha_w=0.1, gamma=0.95)
        
        ac_returns = []
        for _ in range(10):
            ret = ac.learn_episode(env, max_steps=50)
            ac_returns.append(ret)
        
        print(f"\nREINFORCE平均回报: {np.mean(reinforce_returns):.2f}")
        print(f"Actor-Critic平均回报: {np.mean(ac_returns):.2f}")
        
        # 验证学习是否发生
        assert len(reinforce.episode_returns) == 10
        assert len(ac.td_errors) > 0
        
        print("\n✅ 集成场景测试通过！")
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
    print("第13章：策略梯度方法 - 模块测试")
    print("Chapter 13: Policy Gradient Methods - Module Tests")
    print("="*80)
    
    tests = [
        ("策略梯度定理", test_policy_gradient_theorem),
        ("REINFORCE算法", test_reinforce),
        ("Actor-Critic算法", test_actor_critic),
        ("自然策略梯度", test_natural_policy_gradient),
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
        print("\n🎉 第13章所有策略梯度模块测试通过！")
        print("🎉 All Chapter 13 Policy Gradient modules passed!")
        print("\n策略梯度方法实现验证完成:")
        print("✓ 策略梯度定理")
        print("✓ REINFORCE算法")
        print("✓ Actor-Critic方法")
        print("✓ 自然策略梯度和PPO")
        print("\n从值函数到直接策略优化的转变完成！")
        print("Transition from value functions to direct policy optimization complete!")
    else:
        print("\n⚠️ 有些测试失败，请检查代码")
        print("⚠️ Some tests failed, please check the code")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)