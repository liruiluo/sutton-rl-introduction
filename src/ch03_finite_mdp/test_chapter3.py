#!/usr/bin/env python
"""
测试第2章所有模块
Test all Chapter 2 modules

确保所有实现正确工作
Ensure all implementations work correctly
"""

import sys
import traceback
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_mdp_framework():
    """测试MDP框架"""
    print("\n" + "="*60)
    print("测试MDP框架...")
    print("Testing MDP Framework...")
    print("="*60)
    
    try:
        from src.ch03_finite_mdp.mdp_framework import (
            State, Action, MDPEnvironment, MDPAgent,
            RecyclingRobot, MDPMathematics
        )
        
        # 创建环境
        env = RecyclingRobot()
        print(f"✓ 创建回收机器人环境: {env.name}")
        
        # 重置环境
        state = env.reset()
        print(f"✓ 重置到初始状态: {state.id}")
        
        # 执行动作
        from src.ch03_finite_mdp.mdp_framework import Action
        search_action = Action(id='search', name='搜索垃圾')
        next_state, reward, done, info = env.step(search_action)
        print(f"✓ 执行动作: {search_action.name} -> 奖励={reward}")
        
        print("\n✅ MDP框架测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ MDP框架测试失败: {e}")
        traceback.print_exc()
        return False


def test_agent_environment_interface():
    """测试智能体-环境接口"""
    print("\n" + "="*60)
    print("测试智能体-环境接口...")
    print("Testing Agent-Environment Interface...")
    print("="*60)
    
    try:
        from src.ch03_finite_mdp.agent_environment_interface import (
            Experience, Trajectory, Episode,
            AgentEnvironmentInterface, RandomAgent,
            ExperienceBuffer
        )
        from src.ch03_finite_mdp.mdp_framework import RecyclingRobot, Action
        
        # 创建环境
        env = RecyclingRobot()
        
        # 创建受限的动作空间（高电量时不能充电）
        # 这里我们使用前两个动作（search和wait），避免recharge
        limited_actions = [
            Action(id='search', name='搜索垃圾'),
            Action(id='wait', name='等待')
        ]
        agent = RandomAgent(limited_actions)
        print(f"✓ 创建随机智能体: {agent.name}")
        
        # 创建接口
        interface = AgentEnvironmentInterface(agent, env)
        print(f"✓ 创建智能体-环境接口")
        
        # 运行一个回合
        episode = interface.run_episode(max_steps=10)
        print(f"✓ 运行回合: 步数={len(episode.trajectory)}, "
              f"奖励={episode.return_value:.2f}")
        
        # 测试经验缓冲区
        buffer = ExperienceBuffer(capacity=100)
        buffer.add_trajectory(episode.trajectory)
        print(f"✓ 经验缓冲区: 大小={len(buffer)}")
        
        print("\n✅ 智能体-环境接口测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 智能体-环境接口测试失败: {e}")
        traceback.print_exc()
        return False


def test_policies_and_values():
    """测试策略和价值函数"""
    print("\n" + "="*60)
    print("测试策略和价值函数...")
    print("Testing Policies and Value Functions...")
    print("="*60)
    
    try:
        from src.ch03_finite_mdp.policies_and_values import (
            Policy, DeterministicPolicy, StochasticPolicy,
            UniformRandomPolicy, StateValueFunction,
            ActionValueFunction, BellmanEquations,
            PolicyEvaluation
        )
        from src.ch03_finite_mdp.mdp_framework import RecyclingRobot
        
        # 创建环境
        env = RecyclingRobot()
        
        # 创建随机策略
        random_policy = UniformRandomPolicy(env.action_space)
        print(f"✓ 创建随机策略")
        
        # 获取动作概率
        state = env.state_space[0]
        probs = random_policy.get_action_probabilities(state)
        print(f"✓ 获取动作概率: {len(probs)}个动作")
        
        # 创建价值函数
        V = StateValueFunction(env.state_space, initial_value=0.0)
        print(f"✓ 创建状态价值函数: {len(env.state_space)}个状态")
        
        Q = ActionValueFunction(env.state_space, env.action_space, initial_value=0.0)
        print(f"✓ 创建动作价值函数")
        
        # 策略评估（简单测试）
        print("✓ 测试策略评估...")
        V_evaluated = PolicyEvaluation.evaluate_policy(
            random_policy, env, gamma=0.9, theta=0.01, max_iterations=10
        )
        print(f"  评估后V(high)={V_evaluated.get_value(env.state_space[0]):.3f}")
        
        print("\n✅ 策略和价值函数测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 策略和价值函数测试失败: {e}")
        traceback.print_exc()
        return False


def test_gridworld():
    """测试网格世界"""
    print("\n" + "="*60)
    print("测试网格世界...")
    print("Testing Grid World...")
    print("="*60)
    
    try:
        from src.ch03_finite_mdp.gridworld import (
            GridWorld, GridWorldAgent, GridWorldVisualizer
        )
        
        # 创建网格世界
        env = GridWorld(rows=3, cols=3, start_pos=(0, 0), goal_pos=(2, 2))
        print(f"✓ 创建3×3网格世界")
        
        # 重置环境
        state = env.reset()
        print(f"✓ 重置到起始位置: {env.current_pos}")
        
        # 创建智能体
        agent = GridWorldAgent(env, learning_algorithm="q_learning")
        print(f"✓ 创建Q学习智能体")
        
        # 执行几步
        for i in range(3):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            print(f"  步骤{i+1}: {action.name} -> 位置{info['position']}")
            if done:
                print(f"✓ 到达目标！")
                break
        
        # 渲染（文本模式）
        print("\n网格世界状态:")
        env.render(mode='human')
        
        print("\n✅ 网格世界测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 网格世界测试失败: {e}")
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("第2章：有限马尔可夫决策过程 - 模块测试")
    print("Chapter 2: Finite MDPs - Module Tests")
    print("="*80)
    
    tests = [
        ("MDP框架", test_mdp_framework),
        ("智能体-环境接口", test_agent_environment_interface),
        ("策略和价值函数", test_policies_and_values),
        ("网格世界", test_gridworld)
    ]
    
    results = []
    for name, test_func in tests:
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
        print("\n🎉 第2章所有模块测试通过！")
        print("🎉 All Chapter 2 modules passed!")
        print("\n可以继续学习第3章：动态规划")
        print("Ready to proceed to Chapter 3: Dynamic Programming")
    else:
        print("\n⚠️ 有些测试失败，请检查代码")
        print("⚠️ Some tests failed, please check the code")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)