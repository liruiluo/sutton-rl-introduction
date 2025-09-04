#!/usr/bin/env python
"""
测试第2章所有多臂赌博机模块
Test all Chapter 2 Multi-Armed Bandit modules

确保所有赌博机算法实现正确
Ensure all bandit algorithm implementations are correct
"""

import sys
import traceback
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_epsilon_greedy():
    """
    测试ε-贪婪算法
    Test ε-greedy algorithm
    """
    print("\n" + "="*60)
    print("测试ε-贪婪算法...")
    print("Testing ε-greedy Algorithm...")
    print("="*60)
    
    try:
        from src.ch02_multi_armed_bandits.epsilon_greedy import EpsilonGreedyAgent
        
        # 创建10臂赌博机
        n_arms = 10
        true_values = np.random.randn(n_arms)
        
        # 创建ε-贪婪智能体
        agent = EpsilonGreedyAgent(n_arms=n_arms, epsilon=0.1)
        
        # 运行几步
        for _ in range(100):
            action = agent.select_action()
            reward = true_values[action] + np.random.randn()
            agent.update(action, reward)
        
        print(f"  ✓ ε-贪婪测试通过，动作选择次数: {agent.action_counts}")
        return True
        
    except Exception as e:
        print(f"  ❌ ε-贪婪测试失败: {e}")
        return False


def test_ucb():
    """
    测试UCB算法
    Test UCB algorithm
    """
    print("\n" + "="*60)
    print("测试UCB算法...")
    print("Testing UCB Algorithm...")
    print("="*60)
    
    try:
        from src.ch02_multi_armed_bandits.ucb_algorithm import UCBAgent
        
        # 创建UCB智能体
        agent = UCBAgent(n_arms=10, c=2.0)
        
        # 运行几步
        for t in range(100):
            action = agent.select_action(t+1)
            reward = np.random.randn()
            agent.update(action, reward)
        
        print(f"  ✓ UCB测试通过，探索奖励: c={agent.c}")
        return True
        
    except Exception as e:
        print(f"  ❌ UCB测试失败: {e}")
        return False


def test_gradient_bandit():
    """
    测试梯度赌博机
    Test gradient bandit
    """
    print("\n" + "="*60)
    print("测试梯度赌博机...")
    print("Testing Gradient Bandit...")
    print("="*60)
    
    try:
        from src.ch02_multi_armed_bandits.gradient_bandit import GradientBanditAgent
        
        # 创建梯度赌博机智能体
        agent = GradientBanditAgent(n_arms=10, alpha=0.1, use_baseline=True)
        
        # 运行几步
        for _ in range(100):
            action = agent.select_action()
            reward = np.random.randn() + 2  # 偏移的奖励
            agent.update(action, reward)
        
        print(f"  ✓ 梯度赌博机测试通过，使用基线: {agent.use_baseline}")
        return True
        
    except Exception as e:
        print(f"  ❌ 梯度赌博机测试失败: {e}")
        return False


def main():
    """
    运行所有测试
    Run all tests
    """
    print("\n" + "="*80)
    print("第2章：多臂赌博机 - 模块测试")
    print("Chapter 2: Multi-Armed Bandits - Module Tests")
    print("="*80)
    
    tests = [
        ("ε-贪婪", test_epsilon_greedy),
        ("UCB", test_ucb),
        ("梯度赌博机", test_gradient_bandit),
    ]
    
    results = []
    
    for name, test_func in tests:
        print(f"\n运行测试: {name}")
        success = test_func()
        results.append((name, success))
    
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
        print("\n🎉 第2章所有多臂赌博机模块测试通过！")
        print("🎉 All Chapter 2 Multi-Armed Bandit modules passed!")
    else:
        print("\n⚠️ 有些测试失败，请检查代码")
        print("⚠️ Some tests failed, please check the code")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)