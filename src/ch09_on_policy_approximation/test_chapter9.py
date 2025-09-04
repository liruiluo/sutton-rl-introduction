#!/usr/bin/env python
"""
测试第9章所有函数近似模块
Test all Chapter 9 Function Approximation modules

确保所有算法实现正确
Ensure all algorithm implementations are correct
"""

import sys
import traceback
import numpy as np
from pathlib import Path
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_gradient_descent():
    """
    测试梯度下降基础
    Test gradient descent foundations
    """
    print("\n" + "="*60)
    print("测试梯度下降...")
    print("Testing Gradient Descent...")
    print("="*60)
    
    try:
        from src.ch09_on_policy_approximation.gradient_descent import (
            ValueFunctionApproximator, SimpleLinearApproximator,
            GradientDescent, StochasticGradientDescent, MiniBatchGradientDescent
        )
        
        # 创建简单的线性近似器
        print("\n测试值函数近似器...")
        n_features = 10
        
        def poly_features(s):
            return np.array([s**i for i in range(n_features)])
        
        approx = SimpleLinearApproximator(n_features, poly_features)
        print("✓ 创建线性近似器")
        
        # 生成测试数据
        n_samples = 50
        x = np.linspace(-1, 1, n_samples)
        y_true = [np.sin(2 * xi) for xi in x]
        
        # 测试批量梯度下降
        print("\n测试批量梯度下降...")
        bgd = GradientDescent(learning_rate=0.1)
        results = bgd.optimize(approx, x, y_true, n_epochs=20, tolerance=1e-6)
        assert 'losses' in results
        assert len(results['losses']) > 0
        print(f"  ✓ 批量GD收敛于{results['n_iterations']}次迭代")
        
        # 测试SGD
        print("\n测试随机梯度下降...")
        approx2 = SimpleLinearApproximator(n_features, poly_features)
        sgd = StochasticGradientDescent(learning_rate=0.1, decay_rate=0.99)
        sgd_results = sgd.train(approx2, list(x), y_true, n_epochs=5)
        assert 'losses' in sgd_results
        print(f"  ✓ SGD完成，最终学习率={sgd_results['final_learning_rate']:.4f}")
        
        # 测试小批量梯度下降
        print("\n测试小批量梯度下降...")
        approx3 = SimpleLinearApproximator(n_features, poly_features)
        mbgd = MiniBatchGradientDescent(learning_rate=0.1, batch_size=10, momentum=0.9)
        mbgd_results = mbgd.train(approx3, list(x), y_true, n_epochs=5)
        assert 'losses' in mbgd_results
        print(f"  ✓ 小批量GD完成，迭代次数={mbgd_results['n_iterations']}")
        
        print("\n✅ 梯度下降测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 梯度下降测试失败: {e}")
        traceback.print_exc()
        return False


def test_linear_approximation():
    """
    测试线性函数近似
    Test linear function approximation
    """
    print("\n" + "="*60)
    print("测试线性函数近似...")
    print("Testing Linear Function Approximation...")
    print("="*60)
    
    try:
        from src.ch09_on_policy_approximation.linear_approximation import (
            LinearFeatures, LinearValueFunction,
            GradientMonteCarlo, SemiGradientTD, SemiGradientTDLambda
        )
        from src.ch03_finite_mdp.gridworld import GridWorld
        from src.ch03_finite_mdp.policies_and_values import UniformRandomPolicy
        
        # 创建环境
        env = GridWorld(rows=3, cols=3, start_pos=(0,0), goal_pos=(2,2))
        print("✓ 创建3×3网格世界")
        
        # 创建特征提取器
        n_features = 9
        feature_extractor = LinearFeatures(n_features)
        print(f"✓ 创建{n_features}维特征提取器")
        
        # 测试特征提取
        state = env.state_space[0]
        features = feature_extractor.extract(state)
        assert len(features) == n_features
        print("  ✓ 特征提取测试通过")
        
        # 创建策略
        policy = UniformRandomPolicy(env.action_space)
        
        # 测试线性值函数
        print("\n测试线性值函数...")
        value_func = LinearValueFunction(feature_extractor)
        prediction = value_func.predict(state)
        assert isinstance(prediction, (float, np.floating))
        
        td_error = value_func.update(state, 1.0, 0.01)
        assert isinstance(td_error, (float, np.floating))
        print("  ✓ 线性值函数测试通过")
        
        # 测试梯度蒙特卡洛
        print("\n测试梯度蒙特卡洛...")
        gmc = GradientMonteCarlo(env, feature_extractor, gamma=0.9, alpha=0.01)
        v_gmc = gmc.learn(policy, n_episodes=10, verbose=False)
        assert v_gmc.update_count > 0
        print(f"  ✓ 梯度MC完成，更新次数={v_gmc.update_count}")
        
        # 测试半梯度TD
        print("\n测试半梯度TD(0)...")
        feature_extractor2 = LinearFeatures(n_features)
        sgtd = SemiGradientTD(env, feature_extractor2, gamma=0.9, alpha=0.01)
        v_sgtd = sgtd.learn(policy, n_episodes=10, verbose=False)
        assert sgtd.step_count > 0
        print(f"  ✓ 半梯度TD完成，步数={sgtd.step_count}")
        
        # 测试半梯度TD(λ)
        print("\n测试半梯度TD(λ)...")
        feature_extractor3 = LinearFeatures(n_features)
        sgtd_lambda = SemiGradientTDLambda(
            env, feature_extractor3, gamma=0.9, alpha=0.01, lambda_=0.5
        )
        v_lambda = sgtd_lambda.learn(policy, n_episodes=10, verbose=False)
        assert sgtd_lambda.step_count > 0
        print(f"  ✓ 半梯度TD(λ)完成，步数={sgtd_lambda.step_count}")
        
        print("\n✅ 线性函数近似测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 线性函数近似测试失败: {e}")
        traceback.print_exc()
        return False


def test_feature_construction():
    """
    测试特征构造方法
    Test feature construction methods
    """
    print("\n" + "="*60)
    print("测试特征构造...")
    print("Testing Feature Construction...")
    print("="*60)
    
    try:
        from src.ch09_on_policy_approximation.feature_construction import (
            PolynomialFeatures, FourierBasis, RadialBasisFunction,
            TileCoding, Iht
        )
        
        # 测试状态空间
        bounds = [(0, 10), (0, 10)]
        test_state = np.array([5.0, 5.0])
        
        # 测试多项式特征
        print("\n测试多项式特征...")
        poly = PolynomialFeatures(degree=2, include_bias=True)
        poly_features = poly.transform(test_state)
        # 如果返回2D数组，取第一行
        if poly_features.ndim == 2:
            poly_features = poly_features[0]
        # 多项式特征应该包含：1, x1, x2, x1^2, x1*x2, x2^2 = 6个特征
        assert len(poly_features) == 6
        print(f"  ✓ 多项式特征: 输入2维 -> 输出{len(poly_features)}维")
        
        # 测试傅里叶基
        print("\n测试傅里叶基...")
        fourier = FourierBasis(n_features=16, bounds=bounds)
        fourier_features = fourier.transform(test_state)
        assert len(fourier_features) == 16
        assert -1 <= fourier_features.min() <= fourier_features.max() <= 1
        print(f"  ✓ 傅里叶特征: 16维，范围[{fourier_features.min():.2f}, {fourier_features.max():.2f}]")
        
        # 测试径向基函数
        print("\n测试径向基函数...")
        rbf = RadialBasisFunction(n_features=9, bounds=bounds)
        rbf_features = rbf.transform(test_state)
        assert len(rbf_features) == 9
        assert 0 <= rbf_features.min() <= rbf_features.max() <= 1
        active_features = np.sum(rbf_features > 0.1)
        print(f"  ✓ RBF特征: 9维，活跃特征数={active_features}")
        
        # 测试瓦片编码
        print("\n测试瓦片编码...")
        tiles = TileCoding(n_tilings=8, bounds=bounds, n_tiles_per_dim=4, iht_size=256)
        active_tiles = tiles.get_tiles(test_state)
        assert len(active_tiles) == 8  # n_tilings
        
        tile_features = tiles.transform(test_state)
        assert len(tile_features) == 256  # iht_size
        assert np.sum(tile_features) == 8  # 每个瓦片一个活跃特征
        print(f"  ✓ 瓦片编码: {len(tile_features)}维，稀疏度={1 - np.sum(tile_features > 0) / len(tile_features):.1%}")
        
        print("\n✅ 特征构造测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 特征构造测试失败: {e}")
        traceback.print_exc()
        return False


def test_least_squares_td():
    """
    测试最小二乘TD
    Test Least-Squares TD
    """
    print("\n" + "="*60)
    print("测试最小二乘TD...")
    print("Testing Least-Squares TD...")
    print("="*60)
    
    try:
        from src.ch09_on_policy_approximation.least_squares_td import (
            LeastSquaresTD, LeastSquaresTDLambda, RecursiveLeastSquaresTD
        )
        from src.ch09_on_policy_approximation.linear_approximation import LinearFeatures
        from src.ch03_finite_mdp.gridworld import GridWorld
        from src.ch03_finite_mdp.policies_and_values import UniformRandomPolicy
        
        # 创建环境
        env = GridWorld(rows=3, cols=3, start_pos=(0,0), goal_pos=(2,2))
        print("✓ 创建3×3网格世界")
        
        # 创建特征提取器
        n_features = 9
        feature_extractor = LinearFeatures(n_features)
        
        # 创建策略
        policy = UniformRandomPolicy(env.action_space)
        
        # 收集样本
        print("\n收集训练样本...")
        samples = []
        for _ in range(20):
            state = env.reset()
            while not state.is_terminal:
                action = policy.select_action(state)
                next_state, reward, done, _ = env.step(action)
                samples.append((state, reward, next_state))
                state = next_state
                if done:
                    break
        print(f"  ✓ 收集了{len(samples)}个样本")
        
        # 测试LSTD(0)
        print("\n测试LSTD(0)...")
        lstd = LeastSquaresTD(feature_extractor, gamma=0.9, epsilon=0.01)
        for state, reward, next_state in samples:
            lstd.add_sample(state, reward, next_state)
        
        weights = lstd.solve()
        assert len(weights) == n_features
        print(f"  ✓ LSTD(0)求解完成，权重范数={np.linalg.norm(weights):.3f}")
        
        # 测试LSTD(λ)
        print("\n测试LSTD(λ)...")
        lstd_lambda = LeastSquaresTDLambda(
            feature_extractor, gamma=0.9, lambda_=0.5, epsilon=0.01
        )
        
        # 重置资格迹在新回合开始
        for i, (state, reward, next_state) in enumerate(samples):
            if i > 0 and samples[i-1][2].is_terminal:
                lstd_lambda.z = np.zeros(n_features)
            lstd_lambda.add_sample(state, reward, next_state)
        
        weights_lambda = lstd_lambda.solve()
        assert len(weights_lambda) == n_features
        print(f"  ✓ LSTD(λ)求解完成，权重范数={np.linalg.norm(weights_lambda):.3f}")
        
        # 测试递归LSTD
        print("\n测试递归LSTD...")
        rlstd = RecursiveLeastSquaresTD(feature_extractor, gamma=0.9, epsilon=0.01)
        
        for i, (state, reward, next_state) in enumerate(samples):
            if i > 0 and samples[i-1][2].is_terminal:
                rlstd.z = np.zeros(n_features)
            rlstd.update(state, reward, next_state)
        
        assert len(rlstd.weights) == n_features
        print(f"  ✓ 递归LSTD完成，权重范数={np.linalg.norm(rlstd.weights):.3f}")
        
        # 比较权重差异
        diff = np.linalg.norm(rlstd.weights - weights)
        print(f"  递归vs批量权重差异: {diff:.4f}")
        
        print("\n✅ 最小二乘TD测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 最小二乘TD测试失败: {e}")
        traceback.print_exc()
        return False


def test_neural_approximation():
    """
    测试神经网络近似
    Test neural network approximation
    """
    print("\n" + "="*60)
    print("测试神经网络近似...")
    print("Testing Neural Network Approximation...")
    print("="*60)
    
    try:
        from src.ch09_on_policy_approximation.neural_approximation import (
            NeuralNetwork, ReplayBuffer, Experience,
            DeepQNetwork, GradientTDNN
        )
        
        # 测试基础神经网络
        print("\n测试神经网络...")
        nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1,
                          learning_rate=0.1)
        
        # 简单的训练数据
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])  # XOR
        
        # 训练几步
        for _ in range(100):
            for i in range(len(X)):
                loss = nn.train_step(X[i], y[i])
        
        # 测试预测
        prediction = nn.predict(X[0])
        assert isinstance(prediction, (float, np.floating))
        print(f"  ✓ 神经网络测试通过，更新次数={nn.update_count}")
        
        # 测试经验回放缓冲
        print("\n测试经验回放缓冲...")
        buffer = ReplayBuffer(capacity=100)
        
        for i in range(50):
            exp = Experience(
                state=np.random.randn(4),
                action=i % 2,
                reward=np.random.randn(),
                next_state=np.random.randn(4),
                done=i % 10 == 0
            )
            buffer.push(exp)
        
        assert len(buffer) == 50
        batch = buffer.sample(10)
        assert len(batch) == 10
        print(f"  ✓ 回放缓冲测试通过，大小={len(buffer)}")
        
        # 测试DQN
        print("\n测试DQN...")
        dqn = DeepQNetwork(
            state_size=4,
            action_size=2,
            hidden_size=16,
            learning_rate=0.001,
            buffer_size=100,
            batch_size=8,
            target_update_freq=10
        )
        
        # 添加经验
        for _ in range(20):
            state = np.random.randn(4)
            action = dqn.select_action(state, epsilon=0.5)
            reward = np.random.randn()
            next_state = np.random.randn(4)
            done = np.random.random() > 0.8
            
            dqn.store_experience(state, action, reward, next_state, done)
        
        # 训练
        loss = dqn.train_batch()
        assert dqn.update_count > 0
        print(f"  ✓ DQN测试通过，更新次数={dqn.update_count}")
        
        # 测试梯度TD神经网络
        print("\n测试梯度TD神经网络...")
        gtd_nn = GradientTDNN(state_size=4, hidden_size=16,
                             learning_rate=0.001, gamma=0.9)
        
        for _ in range(10):
            state = np.random.randn(4)
            reward = np.random.randn()
            next_state = np.random.randn(4)
            done = np.random.random() > 0.8
            
            td_error = gtd_nn.update(state, reward, next_state, done)
            assert isinstance(td_error, (float, np.floating))
        
        assert gtd_nn.update_count > 0
        print(f"  ✓ 梯度TD神经网络测试通过，更新次数={gtd_nn.update_count}")
        
        print("\n✅ 神经网络近似测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 神经网络近似测试失败: {e}")
        traceback.print_exc()
        return False


def main():
    """
    运行所有测试
    Run all tests
    """
    print("\n" + "="*80)
    print("第9章：同策略预测与近似 - 模块测试")
    print("Chapter 9: On-policy Prediction with Approximation - Module Tests")
    print("="*80)
    
    tests = [
        ("梯度下降", test_gradient_descent),
        ("线性函数近似", test_linear_approximation),
        ("特征构造", test_feature_construction),
        ("最小二乘TD", test_least_squares_td),
        ("神经网络近似", test_neural_approximation)
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
        print("\n🎉 第9章所有函数近似模块测试通过！")
        print("🎉 All Chapter 9 Function Approximation modules passed!")
        print("\n函数近似实现验证完成:")
        print("✓ 梯度下降基础")
        print("✓ 线性函数近似")
        print("✓ 特征构造方法")
        print("✓ 最小二乘TD")
        print("✓ 神经网络近似")
        print("\n从表格到函数近似的飞跃！")
        print("The leap from tabular to function approximation!")
        print("\n可以继续学习第10章或开始实际项目")
        print("Ready to proceed to Chapter 10 or start practical projects")
    else:
        print("\n⚠️ 有些测试失败，请检查代码")
        print("⚠️ Some tests failed, please check the code")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)