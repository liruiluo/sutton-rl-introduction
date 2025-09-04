"""
RL Fundamentals - Core concepts from Chapter 1
RL基础概念 - 第1章核心概念

Implements the fundamental elements of reinforcement learning
实现强化学习的基本要素
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod


class RLProblem:
    """
    The Reinforcement Learning Problem
    强化学习问题
    
    Defines the basic elements and structure of an RL problem
    定义RL问题的基本要素和结构
    """
    
    def __init__(self, name: str = "Generic RL Problem"):
        """
        Initialize RL problem
        初始化RL问题
        """
        self.name = name
        self.agent = None
        self.environment = None
        self.reward_signal = None
        self.value_function = None
        self.policy = None
        self.model = None
        
    def set_components(self,
                      agent: Optional['Agent'] = None,
                      environment: Optional['Environment'] = None,
                      reward_signal: Optional['RewardSignal'] = None,
                      value_function: Optional['ValueFunction'] = None,
                      policy: Optional['Policy'] = None,
                      model: Optional['Model'] = None):
        """
        Set the components of the RL problem
        设置RL问题的组件
        """
        if agent:
            self.agent = agent
        if environment:
            self.environment = environment
        if reward_signal:
            self.reward_signal = reward_signal
        if value_function:
            self.value_function = value_function
        if policy:
            self.policy = policy
        if model:
            self.model = model
            
    def describe(self) -> str:
        """
        Describe the RL problem setup
        描述RL问题设置
        """
        description = f"RL Problem: {self.name}\n"
        description += "=" * 50 + "\n"
        
        components = {
            "Agent": self.agent,
            "Environment": self.environment,
            "Reward Signal": self.reward_signal,
            "Value Function": self.value_function,
            "Policy": self.policy,
            "Model": self.model
        }
        
        for name, component in components.items():
            if component:
                description += f"✓ {name}: {component.__class__.__name__}\n"
            else:
                description += f"✗ {name}: Not set\n"
                
        return description


class Agent(ABC):
    """
    The Agent - the learner and decision maker
    智能体 - 学习者和决策者
    """
    
    def __init__(self, name: str = "Generic Agent"):
        """
        Initialize agent
        初始化智能体
        """
        self.name = name
        self.total_reward = 0.0
        self.episode_count = 0
        
    @abstractmethod
    def select_action(self, state: Any) -> Any:
        """
        Select an action given a state
        给定状态选择动作
        """
        pass
    
    @abstractmethod
    def update(self, state: Any, action: Any, reward: float, next_state: Any):
        """
        Update the agent based on experience
        基于经验更新智能体
        """
        pass
    
    def reset(self):
        """
        Reset the agent for a new episode
        为新回合重置智能体
        """
        self.episode_count += 1


class Environment(ABC):
    """
    The Environment - what the agent interacts with
    环境 - 智能体交互的对象
    """
    
    def __init__(self, name: str = "Generic Environment"):
        """
        Initialize environment
        初始化环境
        """
        self.name = name
        self.current_state = None
        self.step_count = 0
        
    @abstractmethod
    def reset(self) -> Any:
        """
        Reset the environment to initial state
        重置环境到初始状态
        """
        pass
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """
        Take a step in the environment
        在环境中执行一步
        
        Returns:
            next_state: The next state
            reward: The reward received
            done: Whether the episode is done
            info: Additional information
        """
        pass
    
    @abstractmethod
    def render(self):
        """
        Render the environment (optional)
        渲染环境（可选）
        """
        pass


class RewardSignal:
    """
    The Reward Signal - defines the goal
    奖励信号 - 定义目标
    """
    
    def __init__(self, reward_function=None):
        """
        Initialize reward signal
        初始化奖励信号
        
        Args:
            reward_function: Function that computes reward
                           计算奖励的函数
        """
        self.reward_function = reward_function
        self.total_reward = 0.0
        self.episode_rewards = []
        
    def compute_reward(self, state: Any, action: Any, next_state: Any) -> float:
        """
        Compute the reward for a transition
        计算转移的奖励
        """
        if self.reward_function:
            return self.reward_function(state, action, next_state)
        else:
            # Default: -1 per step (encourage reaching goal quickly)
            # 默认：每步-1（鼓励快速达到目标）
            return -1.0
            
    def record_reward(self, reward: float):
        """
        Record a reward
        记录奖励
        """
        self.total_reward += reward
        if not self.episode_rewards or self.episode_rewards[-1] is None:
            self.episode_rewards.append(reward)
        else:
            self.episode_rewards[-1] += reward
            
    def new_episode(self):
        """
        Start recording a new episode
        开始记录新回合
        """
        self.episode_rewards.append(0.0)


class ValueFunction:
    """
    The Value Function - how good is each state/action
    价值函数 - 每个状态/动作有多好
    """
    
    def __init__(self, initial_value: float = 0.0):
        """
        Initialize value function
        初始化价值函数
        """
        self.values = {}  # State/action -> value mapping
        self.initial_value = initial_value
        self.update_count = {}
        
    def get_value(self, state: Any, action: Optional[Any] = None) -> float:
        """
        Get the value of a state or state-action pair
        获取状态或状态-动作对的价值
        """
        key = (state, action) if action is not None else state
        return self.values.get(key, self.initial_value)
        
    def update_value(self, state: Any, new_value: float, 
                     action: Optional[Any] = None, alpha: float = 0.1):
        """
        Update the value of a state or state-action pair
        更新状态或状态-动作对的价值
        """
        key = (state, action) if action is not None else state
        
        if key not in self.values:
            self.values[key] = self.initial_value
            self.update_count[key] = 0
            
        # Incremental update
        # 增量更新
        old_value = self.values[key]
        self.values[key] = old_value + alpha * (new_value - old_value)
        self.update_count[key] += 1
        
    def get_best_action(self, state: Any, actions: List[Any]) -> Any:
        """
        Get the best action for a state based on action values
        基于动作价值获取状态的最佳动作
        """
        if not actions:
            return None
            
        action_values = [(a, self.get_value(state, a)) for a in actions]
        best_action = max(action_values, key=lambda x: x[1])[0]
        return best_action


class Policy:
    """
    The Policy - mapping from states to actions
    策略 - 从状态到动作的映射
    """
    
    def __init__(self, exploration_rate: float = 0.1):
        """
        Initialize policy
        初始化策略
        """
        self.exploration_rate = exploration_rate
        self.action_probabilities = {}
        
    def select_action(self, state: Any, actions: List[Any], 
                     value_function: Optional[ValueFunction] = None) -> Any:
        """
        Select an action according to the policy
        根据策略选择动作
        """
        if not actions:
            return None
            
        # Epsilon-greedy policy
        # ε-贪婪策略
        if np.random.random() < self.exploration_rate:
            # Explore: random action
            # 探索：随机动作
            return np.random.choice(actions)
        else:
            # Exploit: best action according to value function
            # 利用：根据价值函数选择最佳动作
            if value_function:
                return value_function.get_best_action(state, actions)
            else:
                return np.random.choice(actions)
                
    def get_action_probability(self, state: Any, action: Any, 
                              actions: List[Any]) -> float:
        """
        Get the probability of taking an action in a state
        获取在状态中采取动作的概率
        """
        n_actions = len(actions)
        if n_actions == 0:
            return 0.0
            
        # For epsilon-greedy
        # 对于ε-贪婪
        if action in actions:
            # Probability is epsilon/n for exploration, 1-epsilon for best action
            # 探索的概率是epsilon/n，最佳动作是1-epsilon
            return self.exploration_rate / n_actions + \
                   (1 - self.exploration_rate) * (1.0 if action == actions[0] else 0.0)
        else:
            return 0.0


class Model:
    """
    The Model - the agent's representation of the environment
    模型 - 智能体对环境的表示
    """
    
    def __init__(self):
        """
        Initialize model
        初始化模型
        """
        self.transitions = {}  # (state, action) -> (next_state, reward) history
        self.reward_model = {}  # (state, action) -> expected reward
        self.transition_model = {}  # (state, action) -> next_state probabilities
        
    def update(self, state: Any, action: Any, reward: float, next_state: Any):
        """
        Update the model based on experience
        基于经验更新模型
        """
        key = (state, action)
        
        # Record transition
        # 记录转移
        if key not in self.transitions:
            self.transitions[key] = []
        self.transitions[key].append((next_state, reward))
        
        # Update reward model (running average)
        # 更新奖励模型（运行平均）
        if key not in self.reward_model:
            self.reward_model[key] = reward
        else:
            n = len(self.transitions[key])
            self.reward_model[key] = ((n-1) * self.reward_model[key] + reward) / n
            
        # Update transition model (count-based)
        # 更新转移模型（基于计数）
        if key not in self.transition_model:
            self.transition_model[key] = {}
        if next_state not in self.transition_model[key]:
            self.transition_model[key][next_state] = 0
        self.transition_model[key][next_state] += 1
        
    def predict(self, state: Any, action: Any) -> Tuple[Any, float]:
        """
        Predict the next state and reward
        预测下一个状态和奖励
        """
        key = (state, action)
        
        # Predict reward
        # 预测奖励
        predicted_reward = self.reward_model.get(key, 0.0)
        
        # Predict next state (most likely)
        # 预测下一个状态（最可能的）
        if key in self.transition_model:
            next_states = self.transition_model[key]
            predicted_next_state = max(next_states.items(), key=lambda x: x[1])[0]
        else:
            predicted_next_state = state  # Stay in same state if unknown
            
        return predicted_next_state, predicted_reward
        
    def sample(self, state: Any, action: Any) -> Tuple[Any, float]:
        """
        Sample a transition from the model
        从模型中采样转移
        """
        key = (state, action)
        
        if key in self.transitions and self.transitions[key]:
            # Sample from recorded transitions
            # 从记录的转移中采样
            return np.random.choice(self.transitions[key])
        else:
            # Use prediction if no samples
            # 如果没有样本则使用预测
            return self.predict(state, action)


def demonstrate_rl_fundamentals():
    """
    Demonstrate the fundamental concepts of RL
    演示RL的基本概念
    """
    print("\n" + "="*80)
    print("Reinforcement Learning Fundamentals")
    print("强化学习基础概念")
    print("="*80)
    
    # Create an RL problem
    # 创建一个RL问题
    problem = RLProblem("Grid World Navigation")
    
    # Create components
    # 创建组件
    print("\n1. Creating RL Components 创建RL组件")
    print("-" * 40)
    
    # Simple grid world environment
    # 简单网格世界环境
    class SimpleGridWorld(Environment):
        def __init__(self):
            super().__init__("3x3 Grid World")
            self.size = 3
            self.goal = (2, 2)
            self.reset()
            
        def reset(self):
            self.current_state = (0, 0)
            self.step_count = 0
            return self.current_state
            
        def step(self, action):
            x, y = self.current_state
            
            # Actions: 0=up, 1=right, 2=down, 3=left
            if action == 0 and x > 0:
                x -= 1
            elif action == 1 and y < self.size - 1:
                y += 1
            elif action == 2 and x < self.size - 1:
                x += 1
            elif action == 3 and y > 0:
                y -= 1
                
            self.current_state = (x, y)
            self.step_count += 1
            
            # Reward: +1 for goal, -0.1 per step
            # 奖励：目标+1，每步-0.1
            if self.current_state == self.goal:
                reward = 1.0
                done = True
            else:
                reward = -0.1
                done = False
                
            return self.current_state, reward, done, {}
            
        def render(self):
            for i in range(self.size):
                for j in range(self.size):
                    if (i, j) == self.current_state:
                        print("A", end=" ")
                    elif (i, j) == self.goal:
                        print("G", end=" ")
                    else:
                        print(".", end=" ")
                print()
    
    # Simple learning agent
    # 简单学习智能体
    class SimpleLearningAgent(Agent):
        def __init__(self, value_function, policy):
            super().__init__("Simple Q-Learning Agent")
            self.value_function = value_function
            self.policy = policy
            self.last_state = None
            self.last_action = None
            
        def select_action(self, state):
            actions = [0, 1, 2, 3]  # up, right, down, left
            return self.policy.select_action(state, actions, self.value_function)
            
        def update(self, state, action, reward, next_state):
            # Simple TD update
            # 简单TD更新
            old_q = self.value_function.get_value(state, action)
            
            # Get max Q-value for next state
            # 获取下一状态的最大Q值
            next_actions = [0, 1, 2, 3]
            next_values = [self.value_function.get_value(next_state, a) 
                          for a in next_actions]
            max_next_q = max(next_values) if next_values else 0.0
            
            # TD target
            # TD目标
            td_target = reward + 0.9 * max_next_q
            
            # Update Q-value
            # 更新Q值
            self.value_function.update_value(state, td_target, action, alpha=0.1)
            
            self.last_state = state
            self.last_action = action
    
    # Create all components
    # 创建所有组件
    env = SimpleGridWorld()
    reward_signal = RewardSignal()
    value_function = ValueFunction(initial_value=0.0)
    policy = Policy(exploration_rate=0.1)
    model = Model()
    agent = SimpleLearningAgent(value_function, policy)
    
    # Set components in the problem
    # 在问题中设置组件
    problem.set_components(
        agent=agent,
        environment=env,
        reward_signal=reward_signal,
        value_function=value_function,
        policy=policy,
        model=model
    )
    
    print(problem.describe())
    
    # Run a learning episode
    # 运行学习回合
    print("\n2. Running Learning Episodes 运行学习回合")
    print("-" * 40)
    
    n_episodes = 100
    episode_lengths = []
    episode_rewards = []
    
    for episode in range(n_episodes):
        state = env.reset()
        agent.reset()
        reward_signal.new_episode()
        
        episode_reward = 0
        episode_length = 0
        
        while episode_length < 50:  # Max steps per episode
            # Select and execute action
            # 选择并执行动作
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # Update agent
            # 更新智能体
            agent.update(state, action, reward, next_state)
            
            # Update model
            # 更新模型
            model.update(state, action, reward, next_state)
            
            # Record reward
            # 记录奖励
            reward_signal.record_reward(reward)
            episode_reward += reward
            episode_length += 1
            
            state = next_state
            
            if done:
                break
        
        episode_lengths.append(episode_length)
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 20 == 0:
            avg_length = np.mean(episode_lengths[-20:])
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"Episode {episode+1}: Avg Length = {avg_length:.1f}, "
                  f"Avg Reward = {avg_reward:.2f}")
    
    # Show learned values
    # 显示学习到的价值
    print("\n3. Learned Q-Values 学习到的Q值")
    print("-" * 40)
    
    print("Sample Q-values for state (0,0):")
    for action in range(4):
        q_value = value_function.get_value((0, 0), action)
        action_name = ["Up", "Right", "Down", "Left"][action]
        print(f"  {action_name}: {q_value:.3f}")
    
    # Test the learned policy
    # 测试学习到的策略
    print("\n4. Testing Learned Policy 测试学习到的策略")
    print("-" * 40)
    
    # Set exploration to 0 for testing
    # 测试时将探索设为0
    policy.exploration_rate = 0.0
    
    state = env.reset()
    print("\nInitial state 初始状态:")
    env.render()
    
    steps = 0
    total_reward = 0
    
    while steps < 10:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        action_name = ["Up", "Right", "Down", "Left"][action]
        print(f"\nStep {steps+1}: Action = {action_name}, Reward = {reward:.2f}")
        env.render()
        
        total_reward += reward
        state = next_state
        steps += 1
        
        if done:
            print(f"\nGoal reached! Total reward = {total_reward:.2f}")
            break
    
    # Model-based planning
    # 基于模型的规划
    print("\n5. Model-Based Planning 基于模型的规划")
    print("-" * 40)
    
    print("Model predictions from state (1,1):")
    for action in range(4):
        predicted_next, predicted_reward = model.predict((1, 1), action)
        action_name = ["Up", "Right", "Down", "Left"][action]
        print(f"  {action_name} -> State: {predicted_next}, "
              f"Reward: {predicted_reward:.3f}")
    
    print("\n" + "="*80)
    print("RL Fundamentals Demonstration Complete!")
    print("RL基础概念演示完成！")
    print("="*80)


if __name__ == "__main__":
    # Run the demonstration
    # 运行演示
    demonstrate_rl_fundamentals()