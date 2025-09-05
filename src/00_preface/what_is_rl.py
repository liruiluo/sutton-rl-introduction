"""
================================================================================
前言：什么是强化学习？
Preface: What is Reinforcement Learning?

根据 Sutton & Barto《强化学习：导论》第二版
Based on Sutton & Barto "Reinforcement Learning: An Introduction" 2nd Edition
================================================================================

让我通过一个故事开始：

想象你是一个刚出生的婴儿。你不知道如何走路，不知道如何说话，甚至不知道如何抓住眼前的玩具。
但是，你有一个强大的能力：**学习**。

当你试图抓住玩具时：
- 成功了 → 你感到快乐（正奖励）→ 下次更可能用同样的方式
- 失败了 → 你感到沮丧（负奖励）→ 下次会尝试不同的方式

这就是强化学习的本质：**通过与环境的交互来学习**。

没有老师告诉你"手应该这样动"（不是监督学习）
也没有大量的抓取动作数据让你分析模式（不是无监督学习）
你只是不断尝试，从结果中学习。

================================================================================
强化学习的核心要素
Core Elements of Reinforcement Learning
================================================================================

Sutton & Barto 在前言中强调了强化学习的独特之处：

"Reinforcement learning is learning what to do—how to map situations to actions—
so as to maximize a numerical reward signal."

让我们理解这句话的每个部分：
1. "learning what to do" - 学习做什么（不是学习分类或预测）
2. "map situations to actions" - 从情境到动作的映射（决策）
3. "maximize a numerical reward" - 最大化数值奖励（明确的目标）

与其他机器学习范式的根本区别：
- 监督学习：有标准答案 → 学习模仿
- 无监督学习：发现隐藏结构 → 学习理解
- 强化学习：只有奖励信号 → 学习决策
"""

import numpy as np
from typing import Any, Tuple, List, Optional, Dict
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod


# ================================================================================
# 第0.1节：强化学习问题的本质
# Section 0.1: The Nature of Reinforcement Learning Problem
# ================================================================================

class ReinforcementLearningProblem:
    """
    强化学习问题的本质特征
    
    书中强调的四个特征：
    1. 试错搜索 (Trial-and-error search)
    2. 延迟奖励 (Delayed reward)
    3. 时间很重要 (Time really matters)
    4. 智能体的行动影响后续数据 (Agent's actions affect subsequent data)
    
    这些特征使得强化学习与其他学习范式根本不同。
    """
    
    @staticmethod
    def demonstrate_trial_and_error():
        """
        演示试错搜索的本质
        
        想象一个孩子学习骑自行车：
        - 第一次尝试：摔倒（失败）
        - 第二次尝试：骑了1米（小成功）
        - 第三次尝试：骑了5米（更大成功）
        - ...经过多次尝试...
        - 第N次尝试：成功骑行（掌握技能）
        
        关键洞察：
        1. 没有人能准确告诉你如何保持平衡（没有监督信号）
        2. 你必须亲自尝试才能学会（需要探索）
        3. 失败是学习过程的必要部分（从错误中学习）
        """
        print("="*70)
        print("试错搜索演示：学习找到最佳路径")
        print("Trial-and-Error Search: Learning to Find the Best Path")
        print("="*70)
        
        # 简单的迷宫问题
        # S: 起点, G: 目标, #: 墙壁, .: 空地
        maze = [
            ['S', '.', '#', '.', '.'],
            ['.', '.', '#', '.', '.'],
            ['.', '.', '.', '.', '#'],
            ['#', '.', '#', '.', '.'],
            ['.', '.', '.', '.', 'G']
        ]
        
        print("\n迷宫布局 Maze Layout:")
        for row in maze:
            print(' '.join(row))
        
        print("\n学习过程 Learning Process:")
        print("1. 初次尝试：随机探索，撞墙多次")
        print("2. 第10次尝试：记住了一些死路")
        print("3. 第50次尝试：找到一条可行路径")
        print("4. 第100次尝试：发现了最短路径")
        print("\n这就是试错学习：没有地图，只有通过尝试才能学习！")
        print("This is trial-and-error: No map, learn only by trying!")
    
    @staticmethod
    def demonstrate_delayed_reward():
        """
        演示延迟奖励的挑战
        
        下棋是延迟奖励的完美例子：
        - 你走的每一步棋，不会立即知道好坏
        - 只有在游戏结束时（赢/输/和），才知道最终结果
        - 挑战：如何知道哪些步骤导致了最终的胜利或失败？
        
        这就是"时间信用分配问题"(Temporal Credit Assignment Problem)
        """
        print("\n" + "="*70)
        print("延迟奖励演示：下棋的困境")
        print("Delayed Reward: The Chess Dilemma")
        print("="*70)
        
        # 模拟一局简化的棋局
        moves = [
            ("开局：e4", 0),      # 没有立即奖励
            ("对手：e5", 0),      
            ("Nf3", 0),          # 依然没有奖励
            ("对手：Nc6", 0),     
            ("Bb5", 0),          # 西班牙开局，但还是没有奖励
            # ... 很多步之后 ...
            ("将军！", 0),        # 还是没有奖励！
            ("对手认输", +1)      # 终于得到奖励！
        ]
        
        print("\n棋局进展 Game Progress:")
        for i, (move, reward) in enumerate(moves, 1):
            print(f"  第{i}步: {move:15} → 立即奖励: {reward}")
        
        print("\n关键问题 Key Question:")
        print("  哪一步是制胜的关键？开局？中局？还是最后的将军？")
        print("  Which move was crucial? Opening? Middle game? Or checkmate?")
        print("\n这就是延迟奖励的挑战：需要学会长远规划！")
        print("This is the challenge of delayed reward: Learn to plan ahead!")
    
    @staticmethod
    def demonstrate_sequential_nature():
        """
        演示时序性的重要性
        
        强化学习中，时间顺序至关重要：
        - 不同时刻的相同动作可能有完全不同的效果
        - 之前的决策会影响未来的可能性
        """
        print("\n" + "="*70)
        print("时序性演示：时机决定一切")
        print("Sequential Nature: Timing is Everything")
        print("="*70)
        
        print("\n投资决策的例子 Investment Example:")
        print("同样是'买入'动作，时机不同，结果天差地别：")
        
        scenarios = [
            ("2019年买入", "股价: $100", "2020年: $150", "+50% 收益"),
            ("2020年买入", "股价: $150", "2021年: $120", "-20% 亏损"),
            ("2021年买入", "股价: $120", "2022年: $180", "+50% 收益")
        ]
        
        for when, price, future, result in scenarios:
            print(f"  {when}: {price} → {future} → {result}")
        
        print("\n关键洞察 Key Insight:")
        print("  相同的动作（买入），不同的时机，完全不同的结果！")
        print("  Same action (buy), different timing, completely different results!")
        print("\n这就是为什么强化学习需要考虑状态(state)！")
        print("This is why RL needs to consider state!")


# ================================================================================
# 第0.2节：强化学习与人工智能的关系
# Section 0.2: RL and Artificial Intelligence
# ================================================================================

class RLandAI:
    """
    强化学习在人工智能中的地位
    
    Sutton & Barto 认为：
    "强化学习是最接近人类和动物学习方式的机器学习方法"
    
    为什么这么说？让我们深入理解。
    """
    
    @staticmethod
    def compare_learning_paradigms():
        """
        比较不同的学习范式
        
        通过具体例子理解不同学习方式的本质区别
        """
        print("="*70)
        print("学习范式对比：以'学习识别猫'为例")
        print("Learning Paradigms: Learning to Recognize Cats")
        print("="*70)
        
        paradigms = {
            "监督学习 Supervised Learning": {
                "方法": "给你10000张标注好的图片：这是猫，那不是猫",
                "优点": "学习效率高，准确率可以很高",
                "缺点": "需要大量标注数据，只能学会识别，不能学会互动",
                "本质": "模仿学习 (Learning by Imitation)"
            },
            "无监督学习 Unsupervised Learning": {
                "方法": "给你10000张图片，自己发现其中的模式",
                "优点": "不需要标注，可以发现隐藏的结构",
                "缺点": "不知道发现的模式是否有用",
                "本质": "模式发现 (Pattern Discovery)"
            },
            "强化学习 Reinforcement Learning": {
                "方法": "你试图摸猫：摸对了它会咕噜咕噜（奖励），摸错了它会抓你（惩罚）",
                "优点": "学会的不仅是识别，更是如何互动",
                "缺点": "学习过程可能很慢，需要真实互动",
                "本质": "互动学习 (Learning by Interaction)"
            }
        }
        
        for name, details in paradigms.items():
            print(f"\n{name}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
        
        print("\n" + "="*70)
        print("为什么说强化学习更接近真实智能？")
        print("Why is RL Closer to Real Intelligence?")
        print("="*70)
        print("""
        1. 真实世界没有标签
           The real world has no labels
           
        2. 智能体必须通过行动来学习
           Agents must learn through actions
           
        3. 决策的后果可能很久才显现
           Consequences of decisions may appear much later
           
        4. 必须平衡探索新知识和利用已有知识
           Must balance exploring new knowledge and exploiting existing knowledge
        
        这些正是人类和动物面临的学习挑战！
        These are exactly the learning challenges faced by humans and animals!
        """)


# ================================================================================
# 第0.3节：经典强化学习例子 - 井字棋
# Section 0.3: Classic RL Example - Tic-Tac-Toe
# ================================================================================

class TicTacToeIntroduction:
    """
    通过井字棋理解强化学习
    
    Sutton & Barto 在前言中用井字棋作为第一个例子，
    因为它完美展示了强化学习的核心思想。
    
    关键点：
    1. 不需要知道完美策略（不是监督学习）
    2. 通过自我对弈学习（试错）
    3. 只有游戏结束才知道输赢（延迟奖励）
    4. 需要给中间状态赋予价值（价值函数）
    """
    
    def __init__(self):
        """初始化井字棋游戏"""
        self.board = np.zeros((3, 3), dtype=int)  # 0: 空, 1: X, -1: O
        self.value_function = {}  # 状态价值函数
        
    def demonstrate_value_function_learning(self):
        """
        演示价值函数的学习过程
        
        这是强化学习的核心：学习评估每个状态的价值
        """
        print("="*70)
        print("井字棋价值函数学习")
        print("Tic-Tac-Toe Value Function Learning")
        print("="*70)
        
        print("\n初始认知：所有状态价值都是0.5（不知道好坏）")
        print("Initial: All states have value 0.5 (unknown)")
        
        print("\n通过游戏学习后：")
        print("After learning through play:")
        
        # 展示一些典型状态的价值
        example_states = [
            {
                "board": [
                    ['X', '.', '.'],
                    ['.', 'X', '.'],
                    ['.', '.', '.']
                ],
                "value": 0.8,
                "explanation": "X占据了对角线的一部分，优势明显"
            },
            {
                "board": [
                    ['X', 'O', 'X'],
                    ['O', 'X', '.'],
                    ['.', '.', '.']
                ],
                "value": 0.9,
                "explanation": "X即将获胜（下一步下右下角）"
            },
            {
                "board": [
                    ['O', 'O', '.'],
                    ['X', 'X', '.'],
                    ['.', '.', '.']
                ],
                "value": 0.3,
                "explanation": "O威胁很大，X处于劣势"
            }
        ]
        
        for i, state in enumerate(example_states, 1):
            print(f"\n状态{i} State {i}:")
            for row in state["board"]:
                print("  " + " ".join(row))
            print(f"  学习到的价值 Learned Value: {state['value']}")
            print(f"  原因 Reason: {state['explanation']}")
        
        print("\n" + "="*70)
        print("关键洞察：价值函数的意义")
        print("Key Insight: Meaning of Value Function")
        print("="*70)
        print("""
        V(s) = 从状态s开始，采用当前策略，最终获胜的概率
        
        学习过程：
        1. 赢了的游戏：提高路径上所有状态的价值
        2. 输了的游戏：降低路径上所有状态的价值
        3. 平局：小幅调整状态价值
        
        经过成千上万局游戏后，价值函数收敛到真实价值！
        After thousands of games, value function converges to true values!
        """)


# ================================================================================
# 第0.4节：强化学习的挑战
# Section 0.4: Challenges in Reinforcement Learning
# ================================================================================

class RLChallenges:
    """
    强化学习面临的核心挑战
    
    Sutton & Barto 在书中反复强调这些挑战，
    理解它们是掌握强化学习的关键。
    """
    
    @staticmethod
    def exploration_vs_exploitation():
        """
        探索与利用的困境
        
        这可能是强化学习中最基本的权衡
        """
        print("="*70)
        print("探索 vs 利用：永恒的困境")
        print("Exploration vs Exploitation: The Eternal Dilemma")
        print("="*70)
        
        print("\n餐厅选择问题 Restaurant Selection Problem:")
        print("-"*40)
        
        # 模拟10家餐厅，你只知道其中3家
        restaurants = {
            "老王面馆": {"known": True, "rating": 7.5, "visits": 20},
            "小李烧烤": {"known": True, "rating": 8.0, "visits": 15},
            "张姐川菜": {"known": True, "rating": 6.5, "visits": 10},
            "神秘餐厅A": {"known": False, "rating": 9.5, "visits": 0},  # 其实是最好的！
            "神秘餐厅B": {"known": False, "rating": 5.0, "visits": 0},
            "神秘餐厅C": {"known": False, "rating": 7.0, "visits": 0},
            "神秘餐厅D": {"known": False, "rating": 8.5, "visits": 0},
            "神秘餐厅E": {"known": False, "rating": 4.0, "visits": 0},
            "神秘餐厅F": {"known": False, "rating": 6.0, "visits": 0},
            "神秘餐厅G": {"known": False, "rating": 7.8, "visits": 0},
        }
        
        print("你的已知选择 Your Known Options:")
        for name, info in restaurants.items():
            if info["known"]:
                print(f"  {name}: 评分 {info['rating']}/10 (去过{info['visits']}次)")
        
        print("\n未探索的选择 Unexplored Options:")
        unknown_count = sum(1 for r in restaurants.values() if not r["known"])
        print(f"  还有 {unknown_count} 家餐厅你从未尝试过")
        print(f"  There are {unknown_count} restaurants you've never tried")
        
        print("\n困境 The Dilemma:")
        print("  利用 Exploit: 去'小李烧烤'(8.0分) - 保证不错的体验")
        print("  探索 Explore: 尝试新餐厅 - 可能发现更好的（或更差的）")
        
        print("\n真相揭示 Truth Revealed:")
        print("  实际上，'神秘餐厅A'有9.5分，是最好的选择！")
        print("  Actually, 'Mystery Restaurant A' has 9.5 rating, the best!")
        print("  但如果你从不探索，永远不会发现它...")
        print("  But if you never explore, you'll never discover it...")
        
        print("\n" + "="*70)
        print("平衡策略 Balancing Strategies:")
        print("="*70)
        print("""
        1. ε-贪婪：90%时间去已知最好的，10%时间随机探索
        2. 乐观初始化：假设未知的都很好，自然促进探索
        3. UCB：选择"可能是最好"的（考虑不确定性）
        4. Thompson采样：根据概率分布采样
        
        没有完美的答案，需要根据具体问题调整！
        No perfect answer, need to adjust based on specific problems!
        """)
    
    @staticmethod
    def credit_assignment_problem():
        """
        信用分配问题
        
        当获得奖励时，如何知道是哪些行动导致的？
        """
        print("\n" + "="*70)
        print("信用分配问题：功劳归谁？")
        print("Credit Assignment: Who Gets the Credit?")
        print("="*70)
        
        print("\n足球进球的例子 Football Goal Example:")
        print("-"*40)
        
        # 一个进球的过程
        goal_sequence = [
            ("守门员", "开大脚", "看似平常"),
            ("后卫", "头球解围", "化解危机"),
            ("中场", "精准长传", "视野开阔"),
            ("边锋", "高速突破", "撕开防线"),
            ("前锋", "冷静推射", "球进了！"),
        ]
        
        print("进球过程 Goal Sequence:")
        for i, (player, action, note) in enumerate(goal_sequence, 1):
            print(f"  {i}. {player}: {action} ({note})")
        
        print("\n问题 The Problem:")
        print("  团队得分了！+1 奖励")
        print("  Team scored! +1 reward")
        print("\n  但是谁的功劳？")
        print("  But who deserves the credit?")
        print("  - 前锋？（他进的球）")
        print("  - 边锋？（他创造的机会）")
        print("  - 中场？（他发起的进攻）")
        print("  - 所有人？（团队合作）")
        
        print("\n" + "="*70)
        print("强化学习的解决方案 RL Solutions:")
        print("="*70)
        print("""
        1. 时序差分学习（TD）：
           - 将功劳向后传播
           - 离奖励越近的动作获得越多信用
        
        2. 蒙特卡洛方法：
           - 等到回合结束
           - 平均分配功劳给所有动作
        
        3. 资格迹（Eligibility Traces）：
           - 根据时间衰减分配功劳
           - 最近的动作获得更多信用
        
        这些方法各有优缺点，后续章节会详细讲解！
        Each method has pros and cons, detailed in later chapters!
        """)


# ================================================================================
# 第0.5节：本书的学习路径
# Section 0.5: Learning Path of This Book
# ================================================================================

class LearningPath:
    """
    Sutton & Barto 教材的学习路径
    
    理解本书的组织结构，有助于更好地学习
    """
    
    @staticmethod
    def show_book_structure():
        """展示本书的结构和学习路径"""
        print("="*70)
        print("Sutton & Barto《强化学习》学习路径")
        print("Learning Path of Sutton & Barto's RL Book")
        print("="*70)
        
        parts = {
            "第I部分：表格解法 (Tabular Solutions)": [
                "第1章：引言 - RL的基本概念",
                "第2章：多臂赌博机 - 探索vs利用",
                "第3章：有限MDP - 形式化框架",
                "第4章：动态规划 - 完美模型下的解法",
                "第5章：蒙特卡洛方法 - 从经验学习",
                "第6章：时序差分学习 - DP和MC的结合",
                "第7章：n步自举 - TD和MC的统一",
                "第8章：规划和学习 - 模型的使用"
            ],
            "第II部分：近似解法 (Approximate Solutions)": [
                "第9章：在策略预测 - 函数近似",
                "第10章：在策略控制 - 近似控制",
                "第11章：离策略方法 - 从其他策略学习",
                "第12章：资格迹 - 统一的视角"
            ],
            "第III部分：深入探讨 (Looking Deeper)": [
                "第13章：策略梯度方法 - 直接优化策略",
                "第14章：心理学 - RL与动物学习",
                "第15章：神经科学 - RL与大脑",
                "第16章：应用和案例 - 真实世界的RL",
                "第17章：前沿 - 未来的方向"
            ]
        }
        
        for part_name, chapters in parts.items():
            print(f"\n{part_name}")
            print("-" * 50)
            for chapter in chapters:
                print(f"  • {chapter}")
        
        print("\n" + "="*70)
        print("学习建议 Learning Suggestions:")
        print("="*70)
        print("""
        初学者路径 Beginner Path:
        1, 2, 3, 4, 5, 6 → 掌握核心概念
        
        实践者路径 Practitioner Path:
        1, 2, 3, 6, 9, 13 → 快速上手现代方法
        
        研究者路径 Researcher Path:
        全书通读 → 深入理解理论基础
        
        关键是：动手实现每个算法！
        Key point: Implement every algorithm!
        """)


# ================================================================================
# 实践：运行前言演示
# Practice: Run Preface Demonstrations
# ================================================================================

def run_preface_demonstrations():
    """运行前言中的所有演示"""
    
    print("╔" + "═"*68 + "╗")
    print("║" + " "*20 + "强化学习：导论 - 前言".center(28) + " "*20 + "║")
    print("║" + " "*15 + "Reinforcement Learning: Preface".center(38) + " "*15 + "║")
    print("╚" + "═"*68 + "╝")
    
    print("\n欢迎来到强化学习的世界！")
    print("Welcome to the World of Reinforcement Learning!")
    print("\n本演示将帮助你理解什么是强化学习。")
    print("This demonstration will help you understand what RL is.")
    print("="*70)
    
    # 1. 强化学习问题的本质
    print("\n【第1部分：强化学习问题的本质】")
    print("[Part 1: Nature of RL Problems]")
    print("="*70)
    
    problem = ReinforcementLearningProblem()
    problem.demonstrate_trial_and_error()
    input("\n按Enter继续... Press Enter to continue...")
    
    problem.demonstrate_delayed_reward()
    input("\n按Enter继续... Press Enter to continue...")
    
    problem.demonstrate_sequential_nature()
    input("\n按Enter继续... Press Enter to continue...")
    
    # 2. 强化学习与AI
    print("\n【第2部分：强化学习与人工智能】")
    print("[Part 2: RL and Artificial Intelligence]")
    print("="*70)
    
    ai = RLandAI()
    ai.compare_learning_paradigms()
    input("\n按Enter继续... Press Enter to continue...")
    
    # 3. 井字棋例子
    print("\n【第3部分：经典例子 - 井字棋】")
    print("[Part 3: Classic Example - Tic-Tac-Toe]")
    print("="*70)
    
    ttt = TicTacToeIntroduction()
    ttt.demonstrate_value_function_learning()
    input("\n按Enter继续... Press Enter to continue...")
    
    # 4. 强化学习的挑战
    print("\n【第4部分：强化学习的挑战】")
    print("[Part 4: Challenges in RL]")
    print("="*70)
    
    challenges = RLChallenges()
    challenges.exploration_vs_exploitation()
    input("\n按Enter继续... Press Enter to continue...")
    
    challenges.credit_assignment_problem()
    input("\n按Enter继续... Press Enter to continue...")
    
    # 5. 学习路径
    print("\n【第5部分：本书的学习路径】")
    print("[Part 5: Learning Path of This Book]")
    print("="*70)
    
    path = LearningPath()
    path.show_book_structure()
    
    # 总结
    print("\n" + "="*70)
    print("前言总结 Preface Summary")
    print("="*70)
    print("""
    强化学习的精髓：
    
    1. 从互动中学习 (Learning from Interaction)
       - 不是从数据集学习，而是从经验学习
       
    2. 面向目标 (Goal-Directed)
       - 明确的目标：最大化累积奖励
       
    3. 考虑未来 (Considering the Future)
       - 不只看眼前，要考虑长远影响
       
    4. 平衡探索与利用 (Balancing Exploration and Exploitation)
       - 既要尝试新事物，也要利用已知最好的
    
    准备好开始这段激动人心的学习旅程了吗？
    Ready to start this exciting learning journey?
    
    下一章：第1章 - 引言
    Next: Chapter 1 - Introduction
    """)


if __name__ == "__main__":
    # 运行前言演示
    run_preface_demonstrations()