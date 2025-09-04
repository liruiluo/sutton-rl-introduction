#!/usr/bin/env python
"""
================================================================================
Sutton & Barto 强化学习导论 - 综合测试
Sutton & Barto Reinforcement Learning: An Introduction - Comprehensive Test
================================================================================

测试所有章节的实现
Test all chapter implementations

验证从多臂赌博机到策略梯度的完整学习路径
Verify complete learning path from bandits to policy gradients
"""

import sys
import time
import traceback
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_chapter(chapter_num: int, module_path: str, test_file: str) -> bool:
    """
    测试单个章节
    Test single chapter
    
    Args:
        chapter_num: 章节号
                    Chapter number
        module_path: 模块路径
                    Module path
        test_file: 测试文件名
                  Test file name
    
    Returns:
        是否通过测试
        Whether test passed
    """
    print(f"\n{'='*80}")
    print(f"测试第{chapter_num}章 Testing Chapter {chapter_num}")
    print(f"{'='*80}")
    
    try:
        # 动态导入测试模块
        test_module = __import__(f"{module_path}.{test_file}", fromlist=['main'])
        
        # 运行测试
        if hasattr(test_module, 'main'):
            success = test_module.main()
            return success
        else:
            print(f"⚠️ 第{chapter_num}章没有main测试函数")
            return False
            
    except ImportError as e:
        print(f"⚠️ 无法导入第{chapter_num}章测试: {e}")
        return False
    except Exception as e:
        print(f"❌ 第{chapter_num}章测试失败: {e}")
        traceback.print_exc()
        return False


def main():
    """
    运行所有章节测试
    Run all chapter tests
    """
    print("\n" + "="*80)
    print("Sutton & Barto 强化学习导论 - 综合测试")
    print("Sutton & Barto RL Introduction - Comprehensive Test")
    print("="*80)
    
    # 定义所有章节及其测试
    chapters = [
        (1, "src.ch01_introduction", "test_chapter1"),
        (2, "src.ch02_multi_armed_bandits", "test_chapter2"),
        (3, "src.ch03_finite_mdp", "test_chapter3"),
        (4, "src.ch04_dynamic_programming", "test_chapter4"),
        (5, "src.ch05_monte_carlo", "test_chapter5"),
        (6, "src.ch06_temporal_difference", "test_chapter6"),
        (7, "src.ch07_n_step_bootstrapping", "test_chapter7"),
        (8, "src.ch08_planning_and_learning", "test_chapter8"),
        (9, "src.ch09_on_policy_approximation", "test_chapter9"),
        (10, "src.ch10_on_policy_control_approximation", "test_chapter10"),
        (11, "src.ch11_off_policy_approximation", "test_chapter11"),
        (12, "src.ch12_eligibility_traces", "test_chapter12"),
        (13, "src.ch13_policy_gradient", "test_chapter13"),
    ]
    
    results = []
    total_time = time.time()
    
    print("\n开始测试所有章节...")
    print("Starting tests for all chapters...")
    
    for chapter_num, module_path, test_file in chapters:
        start_time = time.time()
        success = test_chapter(chapter_num, module_path, test_file)
        elapsed = time.time() - start_time
        results.append((chapter_num, success, elapsed))
        
        if not success:
            print(f"\n⚠️ 第{chapter_num}章测试失败，继续下一章...")
    
    total_elapsed = time.time() - total_time
    
    # 打印总结
    print("\n" + "="*80)
    print("测试总结 Test Summary")
    print("="*80)
    
    passed_count = 0
    failed_chapters = []
    
    print("\n章节测试结果:")
    print("-" * 40)
    for chapter_num, success, elapsed in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"第{chapter_num:2d}章: {status} (耗时: {elapsed:.2f}秒)")
        
        if success:
            passed_count += 1
        else:
            failed_chapters.append(chapter_num)
    
    print(f"\n总测试时间: {total_elapsed:.2f}秒")
    print(f"通过率: {passed_count}/{len(chapters)} ({100*passed_count/len(chapters):.1f}%)")
    
    if passed_count == len(chapters):
        print("\n" + "🎉"*20)
        print("恭喜！所有章节测试通过！")
        print("Congratulations! All chapters passed!")
        print("🎉"*20)
        
        print("\n已完成的强化学习算法实现:")
        print("="*40)
        print("""
        基础方法 Fundamental Methods:
        ✓ 多臂赌博机 (Multi-armed Bandits)
        ✓ 动态规划 (Dynamic Programming)
        ✓ 蒙特卡洛 (Monte Carlo)
        ✓ 时序差分 (Temporal Difference)
        
        高级方法 Advanced Methods:
        ✓ n步自举 (n-step Bootstrapping)
        ✓ 规划与学习 (Planning and Learning)
        ✓ 函数逼近 (Function Approximation)
        ✓ 资格迹 (Eligibility Traces)
        ✓ 策略梯度 (Policy Gradient)
        
        现代算法 Modern Algorithms:
        ✓ DQN基础 (DQN Basics)
        ✓ Actor-Critic
        ✓ PPO (Proximal Policy Optimization)
        ✓ 自然策略梯度 (Natural Policy Gradient)
        """)
        
        print("\n学习路径完成:")
        print("="*40)
        print("""
        1. 探索与利用 → 2. 马尔可夫决策过程
        3. 规划方法 → 4. 无模型学习
        5. 自举方法 → 6. 函数逼近
        7. 策略梯度 → 8. 深度强化学习基础
        """)
        
    else:
        print(f"\n⚠️ 有{len(failed_chapters)}个章节测试失败:")
        print(f"失败章节: {failed_chapters}")
        print("\n请检查这些章节的实现")
    
    return passed_count == len(chapters)


def quick_test():
    """
    快速测试（只测试关键章节）
    Quick test (test key chapters only)
    """
    print("\n" + "="*80)
    print("快速测试模式 Quick Test Mode")
    print("="*80)
    
    key_chapters = [
        (1, "src.ch01_introduction", "test_chapter1"),          # 引言
        (2, "src.ch02_multi_armed_bandits", "test_chapter2"),  # 基础
        (6, "src.ch06_temporal_difference", "test_chapter6"),   # TD学习
        (9, "src.ch09_on_policy_approximation", "test_chapter9"), # 函数逼近
        (13, "src.ch13_policy_gradient", "test_chapter13"),     # 策略梯度
    ]
    
    print("\n测试关键章节: 1, 2, 6, 9, 13")
    
    all_passed = True
    for chapter_num, module_path, test_file in key_chapters:
        success = test_chapter(chapter_num, module_path, test_file)
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n✅ 关键章节测试全部通过!")
    else:
        print("\n❌ 有关键章节测试失败")
    
    return all_passed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试强化学习实现")
    parser.add_argument('--quick', action='store_true', 
                       help='快速测试（只测试关键章节）')
    parser.add_argument('--chapter', type=int, 
                       help='测试指定章节')
    
    args = parser.parse_args()
    
    if args.quick:
        success = quick_test()
    elif args.chapter:
        # 测试单个章节
        chapters_map = {
            1: ("src.ch01_introduction", "test_chapter1"),
            2: ("src.ch02_multi_armed_bandits", "test_chapter2"),
            3: ("src.ch03_finite_mdp", "test_chapter3"),
            4: ("src.ch04_dynamic_programming", "test_chapter4"),
            5: ("src.ch05_monte_carlo", "test_chapter5"),
            6: ("src.ch06_temporal_difference", "test_chapter6"),
            7: ("src.ch07_n_step_bootstrapping", "test_chapter7"),
            8: ("src.ch08_planning_and_learning", "test_chapter8"),
            9: ("src.ch09_on_policy_approximation", "test_chapter9"),
            10: ("src.ch10_on_policy_control_approximation", "test_chapter10"),
            11: ("src.ch11_off_policy_approximation", "test_chapter11"),
            12: ("src.ch12_eligibility_traces", "test_chapter12"),
            13: ("src.ch13_policy_gradient", "test_chapter13"),
        }
        
        if args.chapter in chapters_map:
            module_path, test_file = chapters_map[args.chapter]
            success = test_chapter(args.chapter, module_path, test_file)
        else:
            print(f"❌ 章节{args.chapter}不存在")
            success = False
    else:
        success = main()
    
    sys.exit(0 if success else 1)