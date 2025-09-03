"""
Run Preface Demonstrations
运行前言演示

This script demonstrates all concepts from the preface:
1. Core RL concepts and terminology
2. Tic-tac-toe with temporal difference learning
3. The reward hypothesis
4. Comparison with other learning paradigms

这个脚本演示前言中的所有概念：
1. 核心强化学习概念和术语
2. 使用时序差分学习的井字棋
3. 奖励假设
4. 与其他学习范式的比较
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import sys
from typing import Optional  # noqa: F401

# Add parent directory to path / 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from preface.core_concepts import (
    RLElement, Experience, Agent, Environment,  # noqa: F401
    demonstrate_rl_loop, RewardHypothesis, compare_learning_paradigms  # noqa: F401
)
from preface.tictactoe import (
    TicTacToeState, TicTacToePlayer, TicTacToeEnvironment,  # noqa: F401
    train_players, visualize_training_progress,
    demonstrate_value_function, human_vs_ai_game
)

# Configure logging / 配置日志
logger = logging.getLogger(__name__)


def demonstrate_core_concepts():
    """
    Demonstrate core RL concepts
    演示核心强化学习概念
    
    This covers the fundamental terminology and ideas
    这涵盖了基本术语和思想
    """
    print("\n" + "=" * 80)
    print("PART 1: CORE REINFORCEMENT LEARNING CONCEPTS")
    print("第1部分：核心强化学习概念")
    print("=" * 80)
    
    # 1. Basic Elements / 基本元素
    print("\n" + "-" * 60)
    print("1.1 Basic Elements of RL / 强化学习基本元素")
    print("-" * 60)
    
    for element in RLElement:
        print(f"\n{element.value.upper()}:")
        
        descriptions = {
            RLElement.AGENT: "The learner and decision maker that interacts with the environment\n"
                           "学习者和决策者，与环境交互",
            RLElement.ENVIRONMENT: "Everything outside the agent, which the agent interacts with\n"
                                 "智能体外部的一切，智能体与之交互",
            RLElement.STATE: "A representation of the environment at a given time\n"
                           "给定时间环境的表示",
            RLElement.ACTION: "A choice made by the agent that affects the environment\n"
                            "智能体做出的影响环境的选择",
            RLElement.REWARD: "A scalar feedback signal indicating how well the agent is doing\n"
                            "指示智能体表现如何的标量反馈信号",
            RLElement.POLICY: "A mapping from states to actions: π(a|s)\n"
                            "从状态到动作的映射：π(a|s)",
            RLElement.VALUE: "Expected future reward from a state: V(s) = E[G_t|S_t=s]\n"
                           "从某状态的期望未来奖励：V(s) = E[G_t|S_t=s]",
            RLElement.MODEL: "The agent's representation of how the environment works\n"
                           "智能体对环境如何工作的表示"
        }
        
        print(f"  {descriptions[element]}")
    
    # 2. Experience Tuple / 经验元组
    print("\n" + "-" * 60)
    print("1.2 Experience Tuple (SARS') / 经验元组")
    print("-" * 60)
    
    # Create example experience / 创建示例经验
    example_exp = Experience(
        state=np.array([1, 0, 0]),
        action=2,
        reward=-0.1,
        next_state=np.array([1, 0, 1]),
        done=False
    )
    
    print(f"\nExample experience / 示例经验:")
    print(f"  State S_t:      {example_exp.state}")
    print(f"  Action A_t:     {example_exp.action}")
    print(f"  Reward R_{{t+1}}: {example_exp.reward}")
    print(f"  Next State S_{{t+1}}: {example_exp.next_state}")
    print(f"  Done:          {example_exp.done}")
    
    print("\nThis tuple (S_t, A_t, R_{t+1}, S_{t+1}) is the fundamental unit of RL")
    print("这个元组是强化学习的基本单位")
    
    # 3. Value Function Equations / 价值函数方程
    print("\n" + "-" * 60)
    print("1.3 Fundamental Equations / 基本方程")
    print("-" * 60)
    
    equations = [
        ("State Value Function / 状态价值函数",
         "V(s) = E[G_t | S_t = s] = E[Σ_{k=0}^∞ γ^k R_{t+k+1} | S_t = s]"),
        
        ("Action Value Function / 动作价值函数",
         "Q(s,a) = E[G_t | S_t = s, A_t = a]"),
        
        ("Bellman Equation / 贝尔曼方程",
         "V(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γV(s')]"),
        
        ("TD Learning Update / TD学习更新",
         "V(S_t) ← V(S_t) + α[R_{t+1} + γV(S_{t+1}) - V(S_t)]"),
        
        ("Q-Learning Update / Q学习更新",
         "Q(S_t,A_t) ← Q(S_t,A_t) + α[R_{t+1} + γ max_a Q(S_{t+1},a) - Q(S_t,A_t)]")
    ]
    
    for name, equation in equations:
        print(f"\n{name}:")
        print(f"  {equation}")
    
    # 4. Exploration vs Exploitation / 探索与利用
    print("\n" + "-" * 60)
    print("1.4 Exploration vs Exploitation / 探索与利用")
    print("-" * 60)
    
    print("\nThe fundamental dilemma in RL / 强化学习中的基本困境:")
    print("  • Exploration: Try new actions to discover better policies")
    print("    探索：尝试新动作以发现更好的策略")
    print("  • Exploitation: Use current knowledge to maximize reward")
    print("    利用：使用当前知识最大化奖励")
    
    print("\nCommon solutions / 常见解决方案:")
    print("  • ε-greedy: Random action with probability ε")
    print("    ε-贪婪：以概率ε随机动作")
    print("  • Softmax: Action probability ∝ exp(Q(s,a)/τ)")
    print("    Softmax：动作概率 ∝ exp(Q(s,a)/τ)")
    print("  • UCB: Balance based on uncertainty")
    print("    UCB：基于不确定性平衡")


def demonstrate_reward_hypothesis():
    """
    Demonstrate the reward hypothesis
    演示奖励假设
    """
    print("\n" + "=" * 80)
    print("PART 2: THE REWARD HYPOTHESIS")
    print("第2部分：奖励假设")
    print("=" * 80)
    
    print("\n" + "-" * 60)
    print("The Central Assumption of RL / 强化学习的核心假设")
    print("-" * 60)
    
    print('\n"All of what we mean by goals and purposes can be well thought of as')
    print('the maximization of the expected value of the cumulative sum of a')
    print('received scalar signal (called reward)."')
    print('\n"我们所说的目标和目的都可以被很好地理解为')
    print('最大化接收到的标量信号（称为奖励）累积和的期望值。"')
    
    # Show examples / 显示示例
    reward_hyp = RewardHypothesis()
    examples = reward_hyp.demonstrate_goal_as_reward_maximization()
    
    print("\n" + "-" * 60)
    print("Examples of Goals as Reward Functions / 目标作为奖励函数的示例")
    print("-" * 60)
    
    # Create detailed examples / 创建详细示例
    detailed_examples = [
        {
            "domain": "Chess / 国际象棋",
            "goal": "Checkmate opponent / 将死对手",
            "reward_design": {
                "Win": "+1",
                "Loss": "-1", 
                "Draw": "0",
                "Piece capture": "+0.01 × piece_value",
                "Check": "+0.001"
            },
            "explanation": "Primary reward for game outcome, small rewards for progress"
                         "\n主要奖励用于游戏结果，小奖励用于进展"
        },
        {
            "domain": "Autonomous Driving / 自动驾驶",
            "goal": "Safe and efficient travel / 安全高效的行驶",
            "reward_design": {
                "Reach destination": "+100",
                "Collision": "-1000",
                "Traffic violation": "-10",
                "Distance traveled": "+0.1/meter",
                "Fuel efficiency": "+0.01 × mpg"
            },
            "explanation": "Heavy penalty for safety violations, reward for progress"
                         "\n对安全违规重罚，对进展奖励"
        },
        {
            "domain": "Portfolio Management / 投资组合管理",
            "goal": "Maximize returns with controlled risk / 在控制风险下最大化回报",
            "reward_design": {
                "Daily return": "+return_percentage",
                "Sharpe ratio": "+0.1 × sharpe",
                "Max drawdown": "-10 × drawdown",
                "Transaction cost": "-cost"
            },
            "explanation": "Balance between returns and risk metrics"
                         "\n在回报和风险指标之间平衡"
        }
    ]
    
    for example in detailed_examples:
        print(f"\n{example['domain']}:")
        print(f"  Goal / 目标: {example['goal']}")
        print(f"  Reward Design / 奖励设计:")
        for component, value in example['reward_design'].items():
            print(f"    • {component}: {value}")
        print(f"  {example['explanation']}")
    
    print("\n" + "-" * 60)
    print("Key Insights / 关键见解")
    print("-" * 60)
    
    insights = [
        "1. Reward shaping is crucial for learning efficiency",
        "   奖励塑造对学习效率至关重要",
        
        "2. Sparse rewards (only at goal) can make learning difficult",
        "   稀疏奖励（仅在目标时）会使学习困难",
        
        "3. Dense rewards provide more learning signal but must align with true goal",
        "   密集奖励提供更多学习信号但必须与真实目标一致",
        
        "4. Reward hacking: Agent may find unintended ways to maximize reward",
        "   奖励黑客：智能体可能找到非预期的方式最大化奖励",
        
        "5. The art of RL often lies in designing the right reward function",
        "   强化学习的艺术往往在于设计正确的奖励函数"
    ]
    
    for insight in insights:
        print(f"\n{insight}")


def demonstrate_learning_paradigms():
    """
    Compare RL with other learning paradigms
    比较强化学习与其他学习范式
    """
    print("\n" + "=" * 80)
    print("PART 3: LEARNING PARADIGM COMPARISON")
    print("第3部分：学习范式比较")
    print("=" * 80)
    
    # Get comparison / 获取比较
    comparisons = compare_learning_paradigms()
    
    # Create detailed comparison table / 创建详细比较表
    print("\n" + "-" * 60)
    print("Detailed Comparison / 详细比较")
    print("-" * 60)
    
    aspects = {
        "Training Data / 训练数据": {
            "Supervised": "Labeled examples (X, y) / 标记示例",
            "Unsupervised": "Unlabeled data X / 未标记数据",
            "Reinforcement": "Experience (s, a, r, s') / 经验"
        },
        "Feedback / 反馈": {
            "Supervised": "Immediate and direct / 即时直接",
            "Unsupervised": "None / 无",
            "Reinforcement": "Delayed and indirect / 延迟间接"
        },
        "Goal / 目标": {
            "Supervised": "Match labels / 匹配标签",
            "Unsupervised": "Find patterns / 发现模式",
            "Reinforcement": "Maximize reward / 最大化奖励"
        },
        "Exploration / 探索": {
            "Supervised": "Not needed / 不需要",
            "Unsupervised": "Not applicable / 不适用",
            "Reinforcement": "Essential / 必要"
        },
        "Sequential / 序列性": {
            "Supervised": "Usually i.i.d. / 通常独立同分布",
            "Unsupervised": "Usually i.i.d. / 通常独立同分布",
            "Reinforcement": "Sequential decisions / 序列决策"
        }
    }
    
    # Print comparison table / 打印比较表
    for aspect, values in aspects.items():
        print(f"\n{aspect}:")
        for paradigm, value in values.items():
            print(f"  • {paradigm:15s}: {value}")
    
    print("\n" + "-" * 60)
    print("When to Use Each Paradigm / 何时使用每种范式")
    print("-" * 60)
    
    use_cases = {
        "Supervised Learning / 监督学习": [
            "• You have labeled training data / 你有标记的训练数据",
            "• The task is classification or regression / 任务是分类或回归",
            "• Examples: Image recognition, spam detection / 示例：图像识别，垃圾邮件检测"
        ],
        "Unsupervised Learning / 无监督学习": [
            "• You want to understand data structure / 你想理解数据结构",
            "• No labels are available / 没有可用标签",
            "• Examples: Customer segmentation, anomaly detection / 示例：客户分割，异常检测"
        ],
        "Reinforcement Learning / 强化学习": [
            "• Sequential decision making / 序列决策",
            "• Learning from interaction / 从交互中学习",
            "• Examples: Game playing, robotics, trading / 示例：游戏，机器人，交易"
        ]
    }
    
    for paradigm, cases in use_cases.items():
        print(f"\n{paradigm}:")
        for case in cases:
            print(f"  {case}")


def run_tictactoe_demo(cfg: DictConfig):
    """
    Run the tic-tac-toe demonstration
    运行井字棋演示
    
    Args:
        cfg: Hydra configuration / Hydra配置
    """
    print("\n" + "=" * 80)
    print("PART 4: TIC-TAC-TOE WITH TEMPORAL DIFFERENCE LEARNING")
    print("第4部分：使用时序差分学习的井字棋")
    print("=" * 80)
    
    # Training parameters from config / 从配置获取训练参数
    n_episodes = cfg.preface.tictactoe.episodes
    epsilon = cfg.preface.tictactoe.epsilon
    alpha = cfg.preface.tictactoe.alpha
    
    print(f"\nTraining Parameters / 训练参数:")
    print(f"  Episodes / 回合数: {n_episodes}")
    print(f"  Exploration ε / 探索率: {epsilon}")
    print(f"  Learning rate α / 学习率: {alpha}")
    
    # Train players / 训练玩家
    print("\n" + "-" * 60)
    print("Training Players / 训练玩家")
    print("-" * 60)
    
    player1, player2, win_rates = train_players(
        n_episodes=n_episodes,
        epsilon=epsilon,
        alpha=alpha
    )
    
    # Visualize training progress / 可视化训练进度
    if cfg.general.save_plots:
        print("\n" + "-" * 60)
        print("Visualizing Training Progress / 可视化训练进度")
        print("-" * 60)
        
        fig = visualize_training_progress(win_rates)
        
        # Save plot / 保存图表
        plot_dir = Path(cfg.visualization.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / "tictactoe_training.png"
        fig.savefig(plot_path, dpi=cfg.visualization.dpi)
        print(f"Training plot saved to / 训练图表已保存到: {plot_path}")
        plt.show()
    
    # Demonstrate value function / 演示价值函数
    print("\n" + "-" * 60)
    print("Demonstrating Learned Value Function / 演示学习到的价值函数")
    print("-" * 60)
    
    demonstrate_value_function(player1)
    
    # Save trained players / 保存训练好的玩家
    save_dir = Path("outputs/models")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    player1_path = save_dir / "tictactoe_player1.pkl"
    player2_path = save_dir / "tictactoe_player2.pkl"
    
    player1.save_policy(str(player1_path))
    player2.save_policy(str(player2_path))
    
    print(f"\nTrained players saved to / 训练好的玩家已保存到:")
    print(f"  Player 1: {player1_path}")
    print(f"  Player 2: {player2_path}")
    
    # Demonstrate a few games / 演示几局游戏
    print("\n" + "-" * 60)
    print("Sample Games / 示例游戏")
    print("-" * 60)
    
    env = TicTacToeEnvironment(player1, player2)
    
    # Disable exploration for demonstration / 演示时禁用探索
    player1.epsilon = 0
    player2.epsilon = 0
    
    print("\nPlaying 3 demonstration games with trained players...")
    print("进行3局训练好的玩家演示游戏...")
    
    for i in range(3):
        print(f"\n{'='*40}")
        print(f"Game {i+1} / 游戏 {i+1}")
        print('='*40)
        
        winner = env.play_episode(training=False, verbose=True)
    
    # Option to play against AI / 与AI对战选项
    print("\n" + "-" * 60)
    print("Interactive Play / 互动游戏")
    print("-" * 60)
    
    play_vs_ai = input("\nWould you like to play against the AI? (y/n) / 想与AI对战吗？(y/n): ")
    if play_vs_ai.lower() == 'y':
        human_vs_ai_game(player2)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Main function to run all preface demonstrations
    运行所有前言演示的主函数
    
    Args:
        cfg: Hydra configuration / Hydra配置
    """
    # Set random seed / 设置随机种子
    np.random.seed(cfg.general.seed)
    
    # Print configuration / 打印配置
    print("\n" + "=" * 80)
    print("REINFORCEMENT LEARNING: AN INTRODUCTION")
    print("强化学习导论")
    print("Sutton & Barto - Second Edition")
    print("=" * 80)
    print("\nPREFACE: FOUNDATIONS OF REINFORCEMENT LEARNING")
    print("前言：强化学习基础")
    print("=" * 80)
    
    print("\nConfiguration / 配置:")
    print(OmegaConf.to_yaml(cfg))
    
    # Run demonstrations / 运行演示
    try:
        # Part 1: Core concepts / 核心概念
        demonstrate_core_concepts()
        
        # Part 2: Reward hypothesis / 奖励假设
        demonstrate_reward_hypothesis()
        
        # Part 3: Learning paradigms / 学习范式
        demonstrate_learning_paradigms()
        
        # Part 4: Tic-tac-toe / 井字棋
        if cfg.get('preface'):
            run_tictactoe_demo(cfg)
        
        print("\n" + "=" * 80)
        print("PREFACE DEMONSTRATIONS COMPLETE")
        print("前言演示完成")
        print("=" * 80)
        
        print("\nKey Takeaways / 关键要点:")
        print("1. RL is about learning from interaction to achieve goals")
        print("   强化学习是关于从交互中学习以实现目标")
        print("2. The reward hypothesis: Goals can be formalized as reward maximization")
        print("   奖励假设：目标可以形式化为奖励最大化")
        print("3. Value functions estimate expected future rewards")
        print("   价值函数估计期望未来奖励")
        print("4. TD learning updates values based on experience")
        print("   TD学习基于经验更新价值")
        print("5. Exploration vs exploitation is a fundamental challenge")
        print("   探索与利用是基本挑战")
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
        raise


if __name__ == "__main__":
    main()