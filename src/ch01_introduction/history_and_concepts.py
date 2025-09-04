"""
History and Key Concepts of Reinforcement Learning
强化学习的历史和关键概念

Overview of RL development and fundamental concepts
RL发展概述和基本概念
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Milestone:
    """
    A milestone in RL history
    RL历史中的里程碑
    """
    year: int
    event: str
    contributors: List[str]
    significance: str


class RLHistory:
    """
    History of Reinforcement Learning
    强化学习的历史
    """
    
    def __init__(self):
        """
        Initialize RL history
        初始化RL历史
        """
        self.milestones = self._create_milestones()
        self.key_papers = self._create_key_papers()
        
    def _create_milestones(self) -> List[Milestone]:
        """
        Create list of major RL milestones
        创建主要RL里程碑列表
        """
        return [
            Milestone(
                1911,
                "Law of Effect",
                ["Edward Thorndike"],
                "Foundation of trial-and-error learning"
            ),
            Milestone(
                1948,
                "Turing's Pleasure-Pain System",
                ["Alan Turing"],
                "Early ideas about machine learning from rewards"
            ),
            Milestone(
                1954,
                "Stochastic Learning Models",
                ["Bush", "Mosteller"],
                "Mathematical foundation for learning theory"
            ),
            Milestone(
                1957,
                "Dynamic Programming",
                ["Richard Bellman"],
                "Bellman equation and optimality principle"
            ),
            Milestone(
                1959,
                "Samuel's Checkers Player",
                ["Arthur Samuel"],
                "First successful RL application in games"
            ),
            Milestone(
                1972,
                "Adaptive Critics",
                ["Klopf"],
                "Precursor to Actor-Critic methods"
            ),
            Milestone(
                1983,
                "Adaptive Heuristic Critic",
                ["Barto", "Sutton", "Anderson"],
                "Early neural network RL"
            ),
            Milestone(
                1989,
                "Q-Learning",
                ["Chris Watkins"],
                "Model-free off-policy control"
            ),
            Milestone(
                1992,
                "TD-Gammon",
                ["Gerald Tesauro"],
                "Backgammon at expert level using TD learning"
            ),
            Milestone(
                1994,
                "SARSA",
                ["Rummery", "Niranjan"],
                "On-policy TD control"
            ),
            Milestone(
                1996,
                "Function Approximation Convergence",
                ["Tsitsiklis", "Van Roy"],
                "Theoretical foundations for FA in RL"
            ),
            Milestone(
                2000,
                "Policy Gradient Theorem",
                ["Sutton", "McAllester", "Singh", "Mansour"],
                "Foundation for modern policy gradient methods"
            ),
            Milestone(
                2003,
                "Natural Policy Gradient",
                ["Kakade"],
                "Fisher information for policy optimization"
            ),
            Milestone(
                2013,
                "DQN (Deep Q-Network)",
                ["Mnih et al.", "DeepMind"],
                "Deep RL revolution begins"
            ),
            Milestone(
                2015,
                "A3C",
                ["Mnih et al.", "DeepMind"],
                "Asynchronous actor-critic"
            ),
            Milestone(
                2015,
                "TRPO",
                ["Schulman et al."],
                "Trust region policy optimization"
            ),
            Milestone(
                2016,
                "AlphaGo",
                ["Silver et al.", "DeepMind"],
                "Defeating world champion in Go"
            ),
            Milestone(
                2017,
                "PPO",
                ["Schulman et al.", "OpenAI"],
                "Proximal policy optimization"
            ),
            Milestone(
                2017,
                "AlphaZero",
                ["Silver et al.", "DeepMind"],
                "Mastering Chess, Shogi, and Go from self-play"
            ),
            Milestone(
                2019,
                "MuZero",
                ["Schrittwieser et al.", "DeepMind"],
                "Planning without a model of the environment"
            )
        ]
        
    def _create_key_papers(self) -> Dict[str, Dict[str, Any]]:
        """
        Create dictionary of key RL papers
        创建关键RL论文字典
        """
        return {
            "Sutton & Barto 1998": {
                "title": "Reinforcement Learning: An Introduction",
                "authors": ["Richard Sutton", "Andrew Barto"],
                "significance": "The definitive textbook on RL",
                "concepts": ["TD learning", "Monte Carlo", "Dynamic Programming"]
            },
            "Watkins 1989": {
                "title": "Learning from Delayed Rewards",
                "authors": ["Chris Watkins"],
                "significance": "Introduced Q-learning",
                "concepts": ["Q-learning", "Off-policy learning"]
            },
            "Williams 1992": {
                "title": "Simple Statistical Gradient-Following Algorithms",
                "authors": ["Ronald Williams"],
                "significance": "REINFORCE algorithm",
                "concepts": ["Policy gradient", "REINFORCE"]
            },
            "Mnih 2015": {
                "title": "Human-level control through deep reinforcement learning",
                "authors": ["Mnih et al."],
                "significance": "DQN - sparked deep RL revolution",
                "concepts": ["Experience replay", "Target network"]
            }
        }
        
    def get_timeline(self) -> str:
        """
        Get a timeline of RL development
        获取RL发展时间线
        """
        timeline = "Reinforcement Learning Timeline\n"
        timeline += "="*60 + "\n\n"
        
        for milestone in self.milestones:
            timeline += f"{milestone.year}: {milestone.event}\n"
            timeline += f"   Contributors: {', '.join(milestone.contributors)}\n"
            timeline += f"   Significance: {milestone.significance}\n\n"
            
        return timeline
        
    def get_era(self, year: int) -> str:
        """
        Get the era of RL development for a given year
        获取给定年份的RL发展时代
        """
        if year < 1950:
            return "Pre-computational Era"
        elif year < 1980:
            return "Classical Era"
        elif year < 2000:
            return "Modern RL Era"
        elif year < 2013:
            return "Convergence Era"
        else:
            return "Deep RL Era"


class KeyConcepts:
    """
    Key concepts in Reinforcement Learning
    强化学习的关键概念
    """
    
    def __init__(self):
        """
        Initialize key concepts
        初始化关键概念
        """
        self.concepts = self._create_concepts()
        
    def _create_concepts(self) -> Dict[str, Dict[str, str]]:
        """
        Create dictionary of key RL concepts
        创建关键RL概念字典
        """
        return {
            "exploration_exploitation": {
                "name": "Exploration vs Exploitation",
                "description": "The fundamental dilemma of trying new actions vs using known good actions",
                "examples": "ε-greedy, UCB, Thompson Sampling",
                "importance": "Central to all RL algorithms"
            },
            "credit_assignment": {
                "name": "Credit Assignment Problem",
                "description": "How to assign credit/blame to actions for delayed rewards",
                "examples": "Temporal difference learning, eligibility traces",
                "importance": "Core challenge in sequential decision making"
            },
            "value_function": {
                "name": "Value Functions",
                "description": "Expected return from states or state-action pairs",
                "examples": "V(s), Q(s,a), advantage functions",
                "importance": "Foundation for many RL algorithms"
            },
            "policy": {
                "name": "Policy",
                "description": "Mapping from states to actions",
                "examples": "Deterministic, stochastic, ε-greedy",
                "importance": "What the agent learns to solve the task"
            },
            "model": {
                "name": "Model",
                "description": "Agent's representation of environment dynamics",
                "examples": "Transition model, reward model",
                "importance": "Enables planning and simulation"
            },
            "bootstrapping": {
                "name": "Bootstrapping",
                "description": "Using estimates to improve other estimates",
                "examples": "TD learning, n-step methods",
                "importance": "Enables learning without complete episodes"
            },
            "on_off_policy": {
                "name": "On-policy vs Off-policy",
                "description": "Learning from behavior policy vs target policy",
                "examples": "SARSA (on), Q-learning (off)",
                "importance": "Determines data efficiency and convergence"
            },
            "function_approximation": {
                "name": "Function Approximation",
                "description": "Using parameterized functions for large/continuous spaces",
                "examples": "Linear FA, neural networks",
                "importance": "Enables RL in complex domains"
            },
            "policy_gradient": {
                "name": "Policy Gradient Methods",
                "description": "Directly optimizing parameterized policies",
                "examples": "REINFORCE, Actor-Critic, PPO",
                "importance": "Natural for continuous actions"
            }
        }
        
    def get_concept_summary(self, concept_key: str) -> str:
        """
        Get a summary of a specific concept
        获取特定概念的摘要
        """
        if concept_key not in self.concepts:
            return f"Concept {concept_key} not found"
            
        concept = self.concepts[concept_key]
        summary = f"{concept['name']}\n"
        summary += "-"*40 + "\n"
        summary += f"Description: {concept['description']}\n"
        summary += f"Examples: {concept['examples']}\n"
        summary += f"Importance: {concept['importance']}\n"
        
        return summary


class EarlyHistory:
    """
    Early history of ideas leading to RL
    导向RL的早期思想史
    """
    
    def __init__(self):
        """
        Initialize early history
        初始化早期历史
        """
        self.animal_learning = self._create_animal_learning()
        self.optimal_control = self._create_optimal_control()
        
    def _create_animal_learning(self) -> Dict[str, str]:
        """
        Create summary of animal learning contributions
        创建动物学习贡献摘要
        """
        return {
            "trial_and_error": "Thorndike's puzzle boxes (1911) - cats learning to escape",
            "law_of_effect": "Responses followed by satisfaction are strengthened",
            "classical_conditioning": "Pavlov's dogs - stimulus-response associations",
            "operant_conditioning": "Skinner's reinforcement schedules",
            "rescorla_wagner": "Mathematical model of conditioning (1972)"
        }
        
    def _create_optimal_control(self) -> Dict[str, str]:
        """
        Create summary of optimal control contributions
        创建最优控制贡献摘要
        """
        return {
            "dynamic_programming": "Bellman's principle of optimality (1957)",
            "pontryagin_maximum": "Necessary conditions for optimal control (1962)",
            "adaptive_control": "Self-tuning controllers",
            "stochastic_control": "Control under uncertainty"
        }


class ModernDevelopments:
    """
    Modern developments in RL
    RL的现代发展
    """
    
    def __init__(self):
        """
        Initialize modern developments
        初始化现代发展
        """
        self.deep_rl = self._create_deep_rl()
        self.applications = self._create_applications()
        
    def _create_deep_rl(self) -> Dict[str, Dict[str, Any]]:
        """
        Create summary of deep RL developments
        创建深度RL发展摘要
        """
        return {
            "dqn": {
                "year": 2013,
                "innovation": "Experience replay + target networks",
                "achievement": "Atari games from pixels"
            },
            "a3c": {
                "year": 2015,
                "innovation": "Asynchronous parallel training",
                "achievement": "Improved sample efficiency"
            },
            "alphago": {
                "year": 2016,
                "innovation": "Monte Carlo tree search + deep networks",
                "achievement": "Defeating world Go champion"
            },
            "ppo": {
                "year": 2017,
                "innovation": "Clipped surrogate objective",
                "achievement": "Stable policy optimization"
            },
            "alphazero": {
                "year": 2017,
                "innovation": "Self-play without human knowledge",
                "achievement": "Mastering multiple board games"
            },
            "gpt_rl": {
                "year": 2023,
                "innovation": "RLHF for language models",
                "achievement": "Aligning AI with human preferences"
            }
        }
        
    def _create_applications(self) -> List[Dict[str, str]]:
        """
        Create list of RL applications
        创建RL应用列表
        """
        return [
            {
                "domain": "Games",
                "examples": "Chess, Go, StarCraft, Dota 2",
                "significance": "Superhuman performance"
            },
            {
                "domain": "Robotics",
                "examples": "Manipulation, locomotion, drones",
                "significance": "Learning complex motor skills"
            },
            {
                "domain": "Finance",
                "examples": "Portfolio management, trading",
                "significance": "Adaptive strategies"
            },
            {
                "domain": "Healthcare",
                "examples": "Treatment planning, drug discovery",
                "significance": "Personalized medicine"
            },
            {
                "domain": "Energy",
                "examples": "Data center cooling, power grids",
                "significance": "Optimization and efficiency"
            },
            {
                "domain": "Transportation",
                "examples": "Autonomous driving, traffic control",
                "significance": "Safety and efficiency"
            },
            {
                "domain": "NLP",
                "examples": "Dialogue systems, RLHF",
                "significance": "Human-aligned AI"
            }
        ]


def demonstrate_history_and_concepts():
    """
    Demonstrate the history and concepts of RL
    演示RL的历史和概念
    """
    print("\n" + "="*80)
    print("Reinforcement Learning: History and Concepts")
    print("强化学习：历史和概念")
    print("="*80)
    
    # Create components
    # 创建组件
    history = RLHistory()
    concepts = KeyConcepts()
    early = EarlyHistory()
    modern = ModernDevelopments()
    
    # 1. Show key milestones
    # 显示关键里程碑
    print("\n1. Key Milestones in RL History RL历史关键里程碑")
    print("-" * 60)
    
    important_years = [1957, 1989, 2013, 2016]
    for year in important_years:
        milestone = next(m for m in history.milestones if m.year == year)
        print(f"\n{milestone.year}: {milestone.event}")
        print(f"   Era: {history.get_era(year)}")
        print(f"   Contributors: {', '.join(milestone.contributors)}")
        print(f"   Significance: {milestone.significance}")
    
    # 2. Show fundamental concepts
    # 显示基本概念
    print("\n2. Fundamental RL Concepts 基本RL概念")
    print("-" * 60)
    
    key_concept_ids = ["exploration_exploitation", "value_function", "policy_gradient"]
    for concept_id in key_concept_ids:
        print(f"\n{concepts.get_concept_summary(concept_id)}")
    
    # 3. Show evolution from animal learning
    # 显示从动物学习的演化
    print("\n3. Roots in Animal Learning 动物学习的根源")
    print("-" * 60)
    
    for key, value in early.animal_learning.items():
        print(f"\n{key.replace('_', ' ').title()}:")
        print(f"   {value}")
    
    # 4. Show modern deep RL achievements
    # 显示现代深度RL成就
    print("\n4. Modern Deep RL Achievements 现代深度RL成就")
    print("-" * 60)
    
    for key, info in modern.deep_rl.items():
        print(f"\n{key.upper()} ({info['year']}):")
        print(f"   Innovation: {info['innovation']}")
        print(f"   Achievement: {info['achievement']}")
    
    # 5. Show application domains
    # 显示应用领域
    print("\n5. RL Application Domains RL应用领域")
    print("-" * 60)
    
    for app in modern.applications[:4]:  # Show first 4
        print(f"\n{app['domain']}:")
        print(f"   Examples: {app['examples']}")
        print(f"   Significance: {app['significance']}")
    
    # 6. Timeline visualization
    # 时间线可视化
    print("\n6. RL Development Timeline RL发展时间线")
    print("-" * 60)
    
    eras = [
        (1950, "Classical Era", "Foundation of core theories"),
        (1980, "Modern RL Era", "TD learning and Q-learning"),
        (2000, "Convergence Era", "Theory meets practice"),
        (2013, "Deep RL Era", "Neural networks + RL"),
        (2020, "Large Scale Era", "RL at unprecedented scale")
    ]
    
    for year, era, description in eras:
        print(f"\n{year}+ : {era}")
        print(f"         {description}")
    
    # 7. Key insights
    # 关键洞察
    print("\n7. Key Insights 关键洞察")
    print("-" * 60)
    
    insights = [
        "RL combines ideas from psychology, control theory, and AI",
        "The exploration-exploitation tradeoff is fundamental",
        "Bootstrapping enables learning without models",
        "Function approximation enables scaling to complex domains",
        "Deep learning revolutionized what's possible with RL"
    ]
    
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    print("\n" + "="*80)
    print("History and Concepts Demonstration Complete!")
    print("历史和概念演示完成！")
    print("="*80)


if __name__ == "__main__":
    # Run the demonstration
    # 运行演示
    demonstrate_history_and_concepts()