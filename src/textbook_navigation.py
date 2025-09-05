"""
================================================================================
Sutton & Barto《强化学习：导论》完整教科书式代码实现
Complete Textbook-Style Implementation of Sutton & Barto's RL: An Introduction

作者说明 Author's Note:
这不仅仅是代码实现，更是一本可以独立阅读的强化学习教科书。
每个文件都包含详细的解释、数学推导、直观例子和可运行的代码。
读完这些代码，你就掌握了强化学习的精髓。

This is not just code implementation, but a complete RL textbook you can read.
Every file contains detailed explanations, math derivations, intuitive examples, 
and runnable code. After reading this code, you'll master the essence of RL.
================================================================================

如何使用本项目 How to Use This Project:
1. 按顺序阅读每个章节的代码
2. 运行每个模块的演示
3. 修改参数，观察变化
4. 实现课后练习

Read each chapter's code in order
Run demonstrations in each module  
Modify parameters and observe changes
Implement exercises
================================================================================
"""

import os
import sys
from typing import Dict, List, Tuple
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class RLTextbook:
    """
    强化学习教科书导航系统
    RL Textbook Navigation System
    
    这个类提供整本"代码教科书"的导航功能
    """
    
    def __init__(self):
        """初始化教科书结构"""
        self.chapters = self._build_chapter_structure()
        
    def _build_chapter_structure(self) -> Dict:
        """
        构建完整的章节结构
        
        每个章节都对应Sutton & Barto书中的内容
        """
        return {
            "00_preface": {
                "title": "前言：什么是强化学习",
                "title_en": "Preface: What is Reinforcement Learning",
                "modules": {
                    "what_is_rl.py": "强化学习的本质与挑战",
                },
                "key_concepts": [
                    "试错学习 Trial-and-error learning",
                    "延迟奖励 Delayed reward", 
                    "探索与利用 Exploration vs Exploitation",
                    "信用分配 Credit assignment"
                ],
                "learning_goal": "理解强化学习与其他学习范式的区别"
            },
            
            "01_introduction": {
                "title": "第1章：引言",
                "title_en": "Chapter 1: Introduction", 
                "modules": {
                    "rl_fundamentals.py": "强化学习基础概念",
                    "tic_tac_toe.py": "井字棋完整实现",
                    "history_and_concepts.py": "强化学习的历史与核心概念"
                },
                "key_concepts": [
                    "智能体与环境 Agent and Environment",
                    "奖励信号 Reward Signal",
                    "价值函数 Value Function",
                    "策略 Policy",
                    "模型 Model"
                ],
                "learning_goal": "掌握强化学习的基本要素和框架"
            },
            
            "02_multi_armed_bandits": {
                "title": "第2章：多臂赌博机",
                "title_en": "Chapter 2: Multi-Armed Bandits",
                "modules": {
                    "bandit_introduction.py": "赌博机问题定义",
                    "epsilon_greedy.py": "ε-贪婪算法（深度解析）",
                    "ucb_algorithm.py": "上置信界算法",
                    "gradient_bandit.py": "梯度赌博机算法"
                },
                "key_concepts": [
                    "动作价值 Action Values q*(a)",
                    "增量更新 Incremental Updates",
                    "ε-贪婪方法 ε-greedy Methods",
                    "乐观初始值 Optimistic Initial Values",
                    "上置信界 Upper Confidence Bounds",
                    "梯度上升 Gradient Ascent"
                ],
                "learning_goal": "深入理解探索与利用的平衡"
            },
            
            "03_finite_mdp": {
                "title": "第3章：有限马尔可夫决策过程",
                "title_en": "Chapter 3: Finite Markov Decision Processes",
                "modules": {
                    "mdp_framework.py": "MDP框架实现",
                    "agent_environment_interface.py": "智能体-环境接口",
                    "policies_and_values.py": "策略与价值函数",
                    "gridworld.py": "网格世界环境"
                },
                "key_concepts": [
                    "马尔可夫性质 Markov Property",
                    "状态转移概率 State Transition Probability p(s'|s,a)",
                    "贝尔曼期望方程 Bellman Expectation Equation",
                    "贝尔曼最优方程 Bellman Optimality Equation",
                    "最优策略 Optimal Policy π*",
                    "最优价值函数 Optimal Value Functions v*, q*"
                ],
                "learning_goal": "掌握强化学习的数学框架"
            },
            
            "04_dynamic_programming": {
                "title": "第4章：动态规划",
                "title_en": "Chapter 4: Dynamic Programming",
                "modules": {
                    "dp_foundations.py": "动态规划基础",
                    "policy_iteration.py": "策略迭代算法",
                    "value_iteration.py": "价值迭代算法",
                    "generalized_policy_iteration.py": "广义策略迭代",
                    "dp_examples.py": "经典DP问题"
                },
                "key_concepts": [
                    "策略评估 Policy Evaluation",
                    "策略改进 Policy Improvement",
                    "策略迭代 Policy Iteration",
                    "价值迭代 Value Iteration",
                    "异步DP Asynchronous DP",
                    "广义策略迭代 GPI"
                ],
                "learning_goal": "理解完美模型下的最优解法"
            },
            
            "05_monte_carlo": {
                "title": "第5章：蒙特卡洛方法",
                "title_en": "Chapter 5: Monte Carlo Methods",
                "modules": {
                    "mc_foundations.py": "MC方法基础",
                    "mc_prediction.py": "MC预测",
                    "mc_control.py": "MC控制",
                    "importance_sampling.py": "重要性采样",
                    "mc_examples.py": "21点游戏"
                },
                "key_concepts": [
                    "首次访问MC First-Visit MC",
                    "每次访问MC Every-Visit MC",
                    "探索性启动 Exploring Starts",
                    "软策略 Soft Policies",
                    "离策略学习 Off-Policy Learning",
                    "重要性采样 Importance Sampling"
                ],
                "learning_goal": "从完整经验序列中学习"
            },
            
            "06_temporal_difference": {
                "title": "第6章：时序差分学习",
                "title_en": "Chapter 6: Temporal-Difference Learning",
                "modules": {
                    "td_foundations.py": "TD学习基础",
                    "td_control.py": "TD控制算法",
                    "n_step_td.py": "n步TD方法"
                },
                "key_concepts": [
                    "TD(0)算法",
                    "SARSA算法",
                    "Q-learning算法",
                    "Expected SARSA",
                    "Double Q-learning",
                    "TD误差 TD Error δ"
                ],
                "learning_goal": "结合DP和MC的优势"
            },
            
            "07_n_step_bootstrapping": {
                "title": "第7章：n步自举",
                "title_en": "Chapter 7: n-step Bootstrapping",
                "modules": {
                    "n_step_td.py": "n步TD预测",
                    "n_step_sarsa.py": "n步SARSA",
                    "off_policy_n_step.py": "n步离策略学习",
                    "tree_backup.py": "树备份算法"
                },
                "key_concepts": [
                    "n步回报 n-step Return G_t:t+n",
                    "n步TD预测 n-step TD Prediction",
                    "n步SARSA n-step SARSA",
                    "n步期望SARSA n-step Expected SARSA",
                    "n步树备份 n-step Tree Backup",
                    "统一算法 Unifying Algorithm"
                ],
                "learning_goal": "统一视角看待MC和TD"
            },
            
            "08_planning_and_learning": {
                "title": "第8章：规划与学习的整合",
                "title_en": "Chapter 8: Planning and Learning with Tabular Methods",
                "modules": {
                    "models_and_planning.py": "模型与规划",
                    "dyna_q.py": "Dyna-Q算法",
                    "prioritized_sweeping.py": "优先扫描",
                    "expected_vs_sample.py": "期望更新vs样本更新",
                    "trajectory_sampling.py": "轨迹采样",
                    "mcts.py": "蒙特卡洛树搜索"
                },
                "key_concepts": [
                    "模型 Model",
                    "规划 Planning",
                    "Dyna架构 Dyna Architecture",
                    "优先扫描 Prioritized Sweeping",
                    "MCTS蒙特卡洛树搜索"
                ],
                "learning_goal": "整合基于模型和无模型方法"
            },
            
            "09_on_policy_approximation": {
                "title": "第9章：在策略预测的近似方法",
                "title_en": "Chapter 9: On-policy Prediction with Approximation",
                "modules": {
                    "gradient_descent.py": "梯度下降基础",
                    "linear_approximation.py": "线性函数近似",
                    "feature_construction.py": "特征构造",
                    "least_squares_td.py": "最小二乘TD",
                    "neural_approximation.py": "神经网络近似"
                },
                "key_concepts": [
                    "函数近似 Function Approximation",
                    "随机梯度下降 SGD",
                    "半梯度方法 Semi-gradient Methods",
                    "特征向量 Feature Vectors",
                    "LSTD最小二乘TD"
                ],
                "learning_goal": "处理大规模状态空间"
            },
            
            "10_on_policy_control_approximation": {
                "title": "第10章：在策略控制的近似方法",
                "title_en": "Chapter 10: On-policy Control with Approximation",
                "modules": {
                    "control_with_fa.py": "函数近似控制",
                    "episodic_semi_gradient.py": "回合式半梯度控制",
                    "continuous_tasks.py": "连续任务"
                },
                "key_concepts": [
                    "山车问题 Mountain Car",
                    "回合式半梯度Sarsa",
                    "平均奖励设定 Average Reward",
                    "差分半梯度Sarsa"
                ],
                "learning_goal": "近似方法的控制算法"
            },
            
            "11_off_policy_approximation": {
                "title": "第11章：离策略方法的近似",
                "title_en": "Chapter 11: Off-policy Methods with Approximation",
                "modules": {
                    "importance_sampling.py": "重要性采样",
                    "gradient_td.py": "梯度TD方法",
                    "emphatic_td.py": "强调TD"
                },
                "key_concepts": [
                    "半梯度离策略TD",
                    "梯度TD GTD/TDC",
                    "强调TD Emphatic TD",
                    "致命三要素 Deadly Triad"
                ],
                "learning_goal": "离策略学习的稳定性"
            },
            
            "12_eligibility_traces": {
                "title": "第12章：资格迹",
                "title_en": "Chapter 12: Eligibility Traces",
                "modules": {
                    "lambda_return.py": "λ-回报",
                    "td_lambda.py": "TD(λ)算法",
                    "control_traces.py": "控制算法的资格迹"
                },
                "key_concepts": [
                    "资格迹 Eligibility Traces e(s)",
                    "λ-回报 λ-return G^λ_t",
                    "前向视角 Forward View",
                    "后向视角 Backward View",
                    "TD(λ)算法",
                    "True Online TD(λ)"
                ],
                "learning_goal": "统一TD和MC的视角"
            },
            
            "13_policy_gradient": {
                "title": "第13章：策略梯度方法",
                "title_en": "Chapter 13: Policy Gradient Methods",
                "modules": {
                    "policy_gradient_theorem.py": "策略梯度定理",
                    "reinforce.py": "REINFORCE算法",
                    "actor_critic.py": "Actor-Critic方法",
                    "natural_policy_gradient.py": "自然策略梯度"
                },
                "key_concepts": [
                    "策略梯度定理 Policy Gradient Theorem",
                    "REINFORCE算法",
                    "基线 Baseline",
                    "Actor-Critic架构",
                    "自然梯度 Natural Gradient",
                    "TRPO/PPO算法"
                ],
                "learning_goal": "直接优化策略参数"
            }
        }
    
    def show_table_of_contents(self):
        """显示完整的教科书目录"""
        print("="*80)
        print("📚 Sutton & Barto《强化学习》教科书式代码实现 - 完整目录")
        print("📚 Complete Table of Contents - Textbook-Style Implementation")
        print("="*80)
        
        for chapter_dir, info in self.chapters.items():
            print(f"\n{'='*60}")
            print(f"📖 {info['title']}")
            print(f"   {info['title_en']}")
            print(f"{'='*60}")
            
            print("\n📂 模块文件 Modules:")
            for module, description in info['modules'].items():
                print(f"   • {module:30} - {description}")
            
            print("\n🎯 核心概念 Key Concepts:")
            for concept in info['key_concepts']:
                print(f"   ✓ {concept}")
            
            print(f"\n💡 学习目标: {info['learning_goal']}")
            print(f"   Learning Goal: {info['learning_goal']}")
    
    def show_learning_paths(self):
        """显示不同的学习路径"""
        print("\n" + "="*80)
        print("🛤️ 推荐学习路径 Recommended Learning Paths")
        print("="*80)
        
        paths = {
            "🎓 初学者路径 Beginner Path": [
                "00_preface → 理解什么是强化学习",
                "01_introduction → 掌握基本概念",
                "02_multi_armed_bandits → 探索vs利用",
                "03_finite_mdp → 数学框架",
                "04_dynamic_programming → 理想情况解法",
                "06_temporal_difference → 实用算法"
            ],
            
            "⚡ 快速实践路径 Fast Track": [
                "00_preface → 快速了解",
                "02_multi_armed_bandits → 简单问题",
                "06_temporal_difference → Q-learning",
                "09_on_policy_approximation → 函数近似",
                "13_policy_gradient → 现代方法"
            ],
            
            "🔬 研究者路径 Researcher Path": [
                "按顺序学习所有章节",
                "重点理解数学推导",
                "实现所有算法变体",
                "完成书中练习题",
                "对比不同算法性能"
            ],
            
            "💼 工程师路径 Engineer Path": [
                "00_preface → 概览",
                "03_finite_mdp → 问题建模",
                "06_temporal_difference → DQN基础",
                "08_planning_and_learning → MCTS",
                "13_policy_gradient → PPO/A3C"
            ]
        }
        
        for path_name, steps in paths.items():
            print(f"\n{path_name}:")
            for i, step in enumerate(steps, 1):
                print(f"  {i}. {step}")
    
    def run_chapter_demo(self, chapter_number: int):
        """运行指定章节的演示"""
        chapter_map = {
            0: "00_preface",
            1: "01_introduction",
            2: "02_multi_armed_bandits",
            3: "03_finite_mdp",
            4: "04_dynamic_programming",
            5: "05_monte_carlo",
            6: "06_temporal_difference",
            7: "07_n_step_bootstrapping",
            8: "08_planning_and_learning",
            9: "09_on_policy_approximation",
            10: "10_on_policy_control_approximation",
            11: "11_off_policy_approximation",
            12: "12_eligibility_traces",
            13: "13_policy_gradient"
        }
        
        if chapter_number not in chapter_map:
            print(f"章节 {chapter_number} 不存在")
            return
        
        chapter_dir = chapter_map[chapter_number]
        chapter_info = self.chapters[chapter_dir]
        
        print(f"\n运行章节演示: {chapter_info['title']}")
        print(f"Running Chapter Demo: {chapter_info['title_en']}")
        print("="*60)
        
        # 这里可以导入并运行相应章节的演示代码
        if chapter_number == 0:
            from src.preface import run_preface_demonstrations
            run_preface_demonstrations()
        elif chapter_number == 2:
            from src.ch02_multi_armed_bandits.epsilon_greedy import demonstrate_epsilon_greedy
            demonstrate_epsilon_greedy()
        # ... 其他章节类似
        
    def show_implementation_status(self):
        """显示各章节的实现状态"""
        print("\n" + "="*80)
        print("📊 实现状态 Implementation Status")
        print("="*80)
        
        status = {
            "00_preface": "✅ 完成 Complete",
            "01_introduction": "✅ 完成 Complete",
            "02_multi_armed_bandits": "✅ 完成 Complete (教科书式重构)",
            "03_finite_mdp": "✅ 完成 Complete",
            "04_dynamic_programming": "✅ 完成 Complete",
            "05_monte_carlo": "✅ 完成 Complete",
            "06_temporal_difference": "✅ 完成 Complete",
            "07_n_step_bootstrapping": "✅ 完成 Complete",
            "08_planning_and_learning": "✅ 完成 Complete",
            "09_on_policy_approximation": "✅ 完成 Complete",
            "10_on_policy_control_approximation": "✅ 完成 Complete",
            "11_off_policy_approximation": "✅ 完成 Complete",
            "12_eligibility_traces": "✅ 完成 Complete",
            "13_policy_gradient": "✅ 完成 Complete"
        }
        
        for chapter, stat in status.items():
            chapter_info = self.chapters[chapter]
            print(f"{chapter:30} {stat:20} - {chapter_info['title']}")
        
        print(f"\n总体完成度: 13/13 章节 (100%)")
        print(f"Overall Progress: 13/13 chapters (100%)")


def main():
    """主程序入口"""
    print("╔" + "═"*78 + "╗")
    print("║" + " "*10 + "Sutton & Barto《强化学习》教科书式代码实现".center(58) + " "*10 + "║")
    print("║" + " "*15 + "Textbook-Style RL Implementation Navigator".center(48) + " "*15 + "║")
    print("╚" + "═"*78 + "╝")
    
    textbook = RLTextbook()
    
    while True:
        print("\n" + "="*60)
        print("请选择操作 Select Operation:")
        print("="*60)
        print("1. 📚 查看完整目录 (View Table of Contents)")
        print("2. 🛤️ 查看学习路径 (View Learning Paths)")
        print("3. 📊 查看实现状态 (View Implementation Status)")
        print("4. ▶️ 运行章节演示 (Run Chapter Demo)")
        print("5. ❌ 退出 (Exit)")
        
        choice = input("\n请输入选择 (1-5): ").strip()
        
        if choice == '1':
            textbook.show_table_of_contents()
        elif choice == '2':
            textbook.show_learning_paths()
        elif choice == '3':
            textbook.show_implementation_status()
        elif choice == '4':
            chapter = input("请输入章节号 (0-13): ").strip()
            if chapter.isdigit():
                textbook.run_chapter_demo(int(chapter))
            else:
                print("无效的章节号")
        elif choice == '5':
            print("\n感谢使用！祝学习愉快！")
            print("Thank you! Happy learning!")
            break
        else:
            print("无效选择，请重试")


if __name__ == "__main__":
    main()