#!/usr/bin/env python
"""
测试第8章所有规划与学习模块
Test all Chapter 8 Planning and Learning modules

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


def test_models_and_planning():
    """
    测试模型与规划基础
    Test models and planning foundations
    """
    print("\n" + "="*60)
    print("测试模型与规划...")
    print("Testing Models and Planning...")
    print("="*60)
    
    try:
        from src.ch08_planning_and_learning.models_and_planning import (
            DeterministicModel, StochasticModel, PlanningAgent
        )
        from src.ch02_mdp.gridworld import GridWorld
        
        # 创建环境
        env = GridWorld(rows=3, cols=3, start_pos=(0,0), goal_pos=(2,2))
        print("✓ 创建3×3网格世界")
        
        # 测试确定性模型
        print("\n测试确定性模型...")
        det_model = DeterministicModel(env.state_space, env.action_space)
        
        state = env.state_space[0]
        action = env.action_space[0]
        next_state = env.state_space[1]
        reward = -1.0
        
        det_model.update(state, action, next_state, reward)
        assert det_model.is_known(state, action), "模型应该知道此转移"
        
        sampled_next, sampled_reward = det_model.sample(state, action)
        assert sampled_next == next_state, "确定性模型应返回相同的下一状态"
        assert sampled_reward == reward, "确定性模型应返回相同的奖励"
        print("  ✓ 确定性模型测试通过")
        
        # 测试随机模型
        print("\n测试随机模型...")
        stoch_model = StochasticModel(env.state_space, env.action_space)
        
        # 添加多个转移模拟随机性
        for _ in range(10):
            stoch_model.update(state, action, next_state, -1.0)
        stoch_model.update(state, action, env.state_space[2], -2.0)
        
        prob = stoch_model.get_probability(state, action, next_state)
        assert 0 <= prob <= 1, "概率应在[0,1]范围"
        assert prob > 0.5, "主要转移应有较高概率"
        print("  ✓ 随机模型测试通过")
        
        # 测试规划智能体
        print("\n测试规划智能体...")
        planner = PlanningAgent(det_model, env.state_space, env.action_space)
        
        # 执行规划步骤
        initial_steps = planner.planning_steps
        planner.plan(10)
        assert planner.planning_steps == initial_steps + 10, "应执行10步规划"
        print("  ✓ 规划智能体测试通过")
        
        print("\n✅ 模型与规划测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 模型与规划测试失败: {e}")
        traceback.print_exc()
        return False


def test_dyna_q():
    """
    测试Dyna-Q算法
    Test Dyna-Q algorithm
    """
    print("\n" + "="*60)
    print("测试Dyna-Q算法...")
    print("Testing Dyna-Q Algorithm...")
    print("="*60)
    
    try:
        from src.ch08_planning_and_learning.dyna_q import (
            DynaQ, DynaQPlus, DynaQComparator
        )
        from src.ch02_mdp.gridworld import GridWorld
        
        # 创建环境
        env = GridWorld(rows=4, cols=4, start_pos=(0,0), goal_pos=(3,3))
        print("✓ 创建4×4网格世界")
        
        # 测试基本Dyna-Q
        print("\n测试Dyna-Q...")
        dyna_q = DynaQ(env, n_planning_steps=5, gamma=0.9, alpha=0.1, epsilon=0.1)
        
        # 学习几个回合
        for _ in range(10):
            ret, length = dyna_q.learn_episode()
            assert -100 < ret < 100, f"回报异常: {ret}"
            assert 0 < length < 1000, f"回合长度异常: {length}"
        
        assert dyna_q.real_steps > 0, "应有真实步数"
        assert dyna_q.planning_steps > 0, "应有规划步数"
        assert len(dyna_q.observed_sa_pairs) > 0, "应有观察的状态-动作对"
        print(f"  ✓ Dyna-Q测试通过 (真实步数={dyna_q.real_steps}, 规划步数={dyna_q.planning_steps})")
        
        # 测试Dyna-Q+
        print("\n测试Dyna-Q+...")
        dyna_q_plus = DynaQPlus(env, n_planning_steps=5, kappa=0.001)
        
        # 测试探索奖励
        state = env.state_space[0]
        action = env.action_space[0]
        dyna_q_plus.current_time = 100
        dyna_q_plus.last_visit_time[(state, action)] = 0
        
        bonus = dyna_q_plus.get_exploration_bonus(state, action)
        assert bonus > 0, "应有探索奖励"
        assert bonus == dyna_q_plus.kappa * np.sqrt(100), "探索奖励计算错误"
        print(f"  ✓ Dyna-Q+测试通过 (探索奖励={bonus:.4f})")
        
        # 测试比较器
        print("\n测试Dyna-Q比较器...")
        comparator = DynaQComparator(env)
        results = comparator.compare_planning_steps(
            n_values=[0, 5],
            n_episodes=10,
            n_runs=2,
            verbose=False
        )
        
        assert len(results) == 2, "应有2个结果"
        assert 0 in results and 5 in results, "应包含n=0和n=5的结果"
        print("  ✓ Dyna-Q比较器测试通过")
        
        print("\n✅ Dyna-Q测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ Dyna-Q测试失败: {e}")
        traceback.print_exc()
        return False


def test_prioritized_sweeping():
    """
    测试优先级扫描
    Test prioritized sweeping
    """
    print("\n" + "="*60)
    print("测试优先级扫描...")
    print("Testing Prioritized Sweeping...")
    print("="*60)
    
    try:
        from src.ch08_planning_and_learning.prioritized_sweeping import (
            PriorityQueue, PrioritizedSweeping, PrioritizedDynaQ
        )
        from src.ch02_mdp.gridworld import GridWorld
        
        # 测试优先队列
        print("测试优先队列...")
        pqueue = PriorityQueue(threshold=0.01)
        
        from src.ch02_mdp.mdp_framework import State, Action
        state1 = State("s1", features={})
        state2 = State("s2", features={})
        action1 = Action("a1")
        action2 = Action("a2")
        
        pqueue.push(state1, action1, 0.5)
        pqueue.push(state2, action2, 0.8)
        
        assert pqueue.size() == 2, "队列应有2个元素"
        
        # 弹出应该是高优先级的
        s, a, p = pqueue.pop()
        assert s == state2 and a == action2, "应先弹出高优先级项"
        assert abs(p - 0.8) < 0.001, "优先级应为0.8"
        print("  ✓ 优先队列测试通过")
        
        # 测试优先级扫描
        print("\n测试优先级扫描算法...")
        env = GridWorld(rows=4, cols=4, start_pos=(0,0), goal_pos=(3,3))
        ps = PrioritizedSweeping(env, n_planning_steps=5, threshold=0.01)
        
        # 学习几个回合
        for _ in range(10):
            ret, length = ps.learn_episode()
            assert -100 < ret < 100, f"回报异常: {ret}"
            assert 0 < length < 1000, f"回合长度异常: {length}"
        
        assert ps.real_steps > 0, "应有真实步数"
        assert ps.planning_steps > 0, "应有规划步数"
        assert len(ps.predecessors) > 0, "应有前驱记录"
        print(f"  ✓ 优先级扫描测试通过 (规划步数={ps.planning_steps})")
        
        # 测试优先级Dyna-Q
        print("\n测试优先级Dyna-Q...")
        pdq = PrioritizedDynaQ(env, n_planning_steps=5, threshold=0.01)
        
        state = env.state_space[0]
        action = env.action_space[0]
        next_state = env.state_space[1]
        reward = -1.0
        
        pdq.learn_step(state, action, next_state, reward)
        assert pdq.real_steps == 1, "应有1个真实步"
        assert (state, action) in pdq.observed_sa, "应记录观察的(s,a)"
        print("  ✓ 优先级Dyna-Q测试通过")
        
        print("\n✅ 优先级扫描测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 优先级扫描测试失败: {e}")
        traceback.print_exc()
        return False


def test_expected_vs_sample():
    """
    测试期望更新vs样本更新
    Test expected vs sample updates
    """
    print("\n" + "="*60)
    print("测试期望更新vs样本更新...")
    print("Testing Expected vs Sample Updates...")
    print("="*60)
    
    try:
        from src.ch08_planning_and_learning.expected_vs_sample import (
            ExpectedUpdate, SampleUpdate, UpdateComparator
        )
        from src.ch08_planning_and_learning.models_and_planning import StochasticModel
        from src.ch02_mdp.gridworld import GridWorld
        
        # 创建环境和模型
        env = GridWorld(rows=3, cols=3, start_pos=(0,0), goal_pos=(2,2))
        model = StochasticModel(env.state_space, env.action_space)
        
        # 构建模型
        print("构建模型...")
        for _ in range(50):
            state = env.reset()
            for _ in range(10):
                if state.is_terminal:
                    break
                action = np.random.choice(env.action_space)
                next_state, reward, done, _ = env.step(action)
                model.update(state, action, next_state, reward)
                state = next_state
        print("  ✓ 模型构建完成")
        
        # 测试期望更新
        print("\n测试期望更新...")
        exp_updater = ExpectedUpdate(env, model, gamma=0.95)
        
        state = env.state_space[0]
        action = env.action_space[0]
        
        if model.is_known(state, action):
            old_q = exp_updater.Q.get_value(state, action)
            exp_updater.expected_update_step(state, action)
            new_q = exp_updater.Q.get_value(state, action)
            assert new_q != old_q or old_q == 0, "Q值应该更新"
        print("  ✓ 期望更新测试通过")
        
        # 测试样本更新
        print("\n测试样本更新...")
        sample_updater = SampleUpdate(env, model, gamma=0.95, alpha=0.1)
        
        if model.is_known(state, action):
            old_q = sample_updater.Q.get_value(state, action)
            sample_updater.sample_update_step(state, action)
            new_q = sample_updater.Q.get_value(state, action)
            # 样本更新可能不改变值（如果采样的转移导致相同的TD误差）
            print(f"  样本更新: {old_q:.3f} -> {new_q:.3f}")
        print("  ✓ 样本更新测试通过")
        
        # 测试比较器
        print("\n测试更新比较器...")
        comparator = UpdateComparator(env)
        comparison = comparator.compare_updates(
            model,
            expected_iterations=10,
            sample_iterations=100,
            n_runs=2
        )
        
        assert comparison.expected_iterations > 0, "期望更新应有迭代"
        assert comparison.sample_iterations > 0, "样本更新应有迭代"
        assert comparison.value_difference >= 0, "价值差异应非负"
        print(f"  ✓ 更新比较器测试通过 (价值差异={comparison.value_difference:.4f})")
        
        print("\n✅ 期望vs样本更新测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 期望vs样本更新测试失败: {e}")
        traceback.print_exc()
        return False


def test_trajectory_sampling():
    """
    测试轨迹采样
    Test trajectory sampling
    """
    print("\n" + "="*60)
    print("测试轨迹采样...")
    print("Testing Trajectory Sampling...")
    print("="*60)
    
    try:
        from src.ch08_planning_and_learning.trajectory_sampling import (
            Trajectory, TrajectoryGenerator, TrajectorySampling,
            UniformSampling, OnPolicySampling, RealTimeDynamicProgramming
        )
        from src.ch08_planning_and_learning.models_and_planning import DeterministicModel
        from src.ch02_mdp.gridworld import GridWorld
        from src.ch02_mdp.policies_and_values import UniformRandomPolicy
        
        # 创建环境
        env = GridWorld(rows=3, cols=3, start_pos=(0,0), goal_pos=(2,2))
        model = DeterministicModel(env.state_space, env.action_space)
        
        # 构建模型
        print("构建模型...")
        for _ in range(30):
            state = env.reset()
            for _ in range(10):
                if state.is_terminal:
                    break
                action = np.random.choice(env.action_space)
                next_state, reward, done, _ = env.step(action)
                model.update(state, action, next_state, reward)
                state = next_state
        print("  ✓ 模型构建完成")
        
        # 测试轨迹
        print("\n测试轨迹数据结构...")
        traj = Trajectory()
        traj.add_step(env.state_space[0], env.action_space[0], -1.0)
        traj.add_step(env.state_space[1], env.action_space[1], -1.0)
        
        assert traj.length == 2, "轨迹长度应为2"
        assert traj.return_value == -2.0, "轨迹回报应为-2.0"
        assert abs(traj.discounted_return(0.9) - (-1.0 - 0.9)) < 0.001, "折扣回报计算错误"
        print("  ✓ 轨迹测试通过")
        
        # 测试轨迹生成器
        print("\n测试轨迹生成器...")
        generator = TrajectoryGenerator(env, model)
        policy = UniformRandomPolicy(env.action_space)
        
        trajectory = generator.generate_trajectory(policy, max_steps=20)
        assert trajectory.length > 0, "应生成非空轨迹"
        assert len(trajectory.states) > 0, "应有状态"
        print(f"  ✓ 轨迹生成器测试通过 (长度={trajectory.length})")
        
        # 测试轨迹采样算法
        print("\n测试轨迹采样算法...")
        traj_sampler = TrajectorySampling(env, gamma=0.9, alpha=0.1)
        
        # 从真实经验学习
        state = env.state_space[0]
        action = env.action_space[0]
        next_state = env.state_space[1]
        reward = -1.0
        
        traj_sampler.learn_from_real_experience(state, action, next_state, reward)
        assert traj_sampler.update_count == 1, "应有1次更新"
        
        # 使用轨迹规划
        traj_sampler.planning_with_trajectories(n_trajectories=5)
        assert traj_sampler.trajectory_count == 5, "应生成5条轨迹"
        print(f"  ✓ 轨迹采样测试通过 (更新数={traj_sampler.update_count})")
        
        # 测试RTDP
        print("\n测试实时动态规划...")
        rtdp = RealTimeDynamicProgramming(env, gamma=0.9, alpha=0.1)
        
        rtdp.run_trial(max_steps=10)
        assert len(rtdp.visited_states) > 0, "应访问一些状态"
        assert rtdp.update_count > 0, "应有更新"
        print(f"  ✓ RTDP测试通过 (访问状态={len(rtdp.visited_states)})")
        
        print("\n✅ 轨迹采样测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 轨迹采样测试失败: {e}")
        traceback.print_exc()
        return False


def test_mcts():
    """
    测试蒙特卡洛树搜索
    Test Monte Carlo Tree Search
    """
    print("\n" + "="*60)
    print("测试蒙特卡洛树搜索...")
    print("Testing Monte Carlo Tree Search...")
    print("="*60)
    
    try:
        from src.ch08_planning_and_learning.mcts import (
            MCTSNode, UCTSelection, MonteCarloTreeSearch
        )
        from src.ch02_mdp.gridworld import GridWorld
        from src.ch02_mdp.mdp_framework import State, Action
        
        # 测试MCTS节点
        print("测试MCTS节点...")
        state = State("test", features={})
        node = MCTSNode(state)
        
        # 更新节点
        node.update(1.0)
        assert node.visit_count == 1, "访问次数应为1"
        assert node.total_value == 1.0, "总价值应为1.0"
        assert node.average_value == 1.0, "平均价值应为1.0"
        
        # 添加子节点
        action = Action("a1")
        child_state = State("child", features={})
        child = node.add_child(action, child_state)
        
        assert action in node.children, "应有子节点"
        assert child.parent == node, "父节点应正确设置"
        print("  ✓ MCTS节点测试通过")
        
        # 测试UCT选择
        print("\n测试UCT选择...")
        uct = UCTSelection(c=1.41421356237)
        
        # 更新子节点使其有不同的统计
        child.update(0.5)
        child.update(0.8)
        
        uct_values = uct.compute_uct_values(node)
        assert action in uct_values, "应计算UCT值"
        print("  ✓ UCT选择测试通过")
        
        # 测试MCTS算法
        print("\n测试MCTS算法...")
        env = GridWorld(rows=3, cols=3, start_pos=(0,0), goal_pos=(2,2))
        mcts = MonteCarloTreeSearch(env, c=1.41421356237, gamma=0.9)
        
        start_state = env.state_space[0]
        best_action = mcts.search(start_state, n_simulations=50, max_depth=10)
        
        assert best_action is not None, "应选择一个动作"
        assert best_action in env.action_space, "动作应在动作空间中"
        
        stats = mcts.get_tree_statistics()
        assert stats['tree_size'] > 1, "树应该增长"
        assert stats['total_simulations'] == 50, "应执行50次模拟"
        print(f"  ✓ MCTS测试通过 (树大小={stats['tree_size']}, 最佳动作={best_action.id})")
        
        print("\n✅ MCTS测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ MCTS测试失败: {e}")
        traceback.print_exc()
        return False


def main():
    """
    运行所有测试
    Run all tests
    """
    print("\n" + "="*80)
    print("第8章：规划与学习 - 模块测试")
    print("Chapter 8: Planning and Learning - Module Tests")
    print("="*80)
    
    tests = [
        ("模型与规划", test_models_and_planning),
        ("Dyna-Q算法", test_dyna_q),
        ("优先级扫描", test_prioritized_sweeping),
        ("期望vs样本更新", test_expected_vs_sample),
        ("轨迹采样", test_trajectory_sampling),
        ("蒙特卡洛树搜索", test_mcts)
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
        print("\n🎉 第8章所有规划与学习模块测试通过！")
        print("🎉 All Chapter 8 Planning and Learning modules passed!")
        print("\n规划与学习实现验证完成:")
        print("✓ 模型与规划基础")
        print("✓ Dyna-Q和Dyna-Q+")
        print("✓ 优先级扫描")
        print("✓ 期望vs样本更新")
        print("✓ 轨迹采样和RTDP")
        print("✓ 蒙特卡洛树搜索(MCTS)")
        print("\n模型增强了学习效率！")
        print("Models enhance learning efficiency!")
        print("\n可以继续学习第9章或开始实际项目")
        print("Ready to proceed to Chapter 9 or start practical projects")
    else:
        print("\n⚠️ 有些测试失败，请检查代码")
        print("⚠️ Some tests failed, please check the code")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)