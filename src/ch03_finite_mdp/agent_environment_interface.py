"""
================================================================================
第2.2节：智能体-环境接口 - 强化学习的交互协议
Section 2.2: Agent-Environment Interface - The Interaction Protocol of RL
================================================================================

这一节定义了强化学习中最重要的接口！
This section defines the most important interface in RL!

关键概念 Key Concepts:
1. 时间步(Time Step): 离散的决策时刻
2. 轨迹(Trajectory): 状态-动作-奖励序列
3. 回合(Episode): 从开始到结束的完整交互
4. 交互循环(Interaction Loop): RL的核心循环

这个接口定义了所有RL算法的基础结构！
This interface defines the basic structure of all RL algorithms!
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from collections import deque
import logging
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns

from .mdp_framework import State, Action, MDPEnvironment, MDPAgent

# 设置日志
logger = logging.getLogger(__name__)


# ================================================================================
# 第2.2.1节：经验和轨迹
# Section 2.2.1: Experience and Trajectory
# ================================================================================

@dataclass
class Experience:
    """
    单步经验 - RL的基本数据单元
    Single-step experience - Basic data unit of RL
    
    这是学习的原材料！
    This is the raw material for learning!
    
    经验元组 Experience Tuple:
    (S_t, A_t, R_{t+1}, S_{t+1}, done)
    
    深入理解 Deep Understanding:
    - 这个元组包含了一次完整的交互信息
      This tuple contains complete information of one interaction
    - 是所有RL算法更新的基础
      Is the basis for all RL algorithm updates
    - 可以存储并重复使用（经验回放）
      Can be stored and reused (experience replay)
    """
    
    state: State               # S_t: 当前状态
    action: Action             # A_t: 执行的动作
    reward: float              # R_{t+1}: 获得的奖励
    next_state: State          # S_{t+1}: 下一状态
    done: bool                 # 是否终止
    
    # 额外信息（可选）
    info: Dict[str, Any] = field(default_factory=dict)
    
    # 时间信息
    timestep: int = 0
    
    def __repr__(self):
        """友好的表示"""
        return (f"Exp(s={self.state.id}, a={self.action.id}, "
                f"r={self.reward:.2f}, s'={self.next_state.id}, "
                f"done={self.done})")
    
    def to_tuple(self) -> Tuple:
        """转换为元组（便于处理）"""
        return (self.state, self.action, self.reward, 
                self.next_state, self.done)


@dataclass
class Trajectory:
    """
    轨迹 - 经验序列
    Trajectory - Sequence of experiences
    
    轨迹是智能体与环境交互的完整记录！
    Trajectory is the complete record of agent-environment interaction!
    
    数学表示 Mathematical Representation:
    τ = (S_0, A_0, R_1, S_1, A_1, R_2, ..., S_T)
    
    用途 Uses:
    1. 蒙特卡洛学习需要完整轨迹
       Monte Carlo learning needs complete trajectories
    2. 策略梯度使用轨迹估计梯度
       Policy gradient uses trajectories to estimate gradients
    3. 模仿学习从专家轨迹学习
       Imitation learning learns from expert trajectories
    """
    
    experiences: List[Experience] = field(default_factory=list)
    
    # 轨迹元数据
    total_reward: float = 0.0
    length: int = 0
    
    # 轨迹标识
    episode_id: Optional[int] = None
    agent_name: Optional[str] = None
    
    def add_experience(self, exp: Experience):
        """添加经验"""
        self.experiences.append(exp)
        self.total_reward += exp.reward
        self.length += 1
    
    def get_returns(self, gamma: float = 0.99) -> List[float]:
        """
        计算每一步的回报G_t
        Calculate return G_t for each step
        
        G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...
        
        这是价值函数估计的目标！
        This is the target for value function estimation!
        
        Args:
            gamma: 折扣因子 Discount factor
            
        Returns:
            每一步的回报列表
            List of returns for each step
        """
        returns = []
        G = 0
        
        # 从后向前计算（更高效）
        # Calculate backwards (more efficient)
        for exp in reversed(self.experiences):
            G = exp.reward + gamma * G
            returns.insert(0, G)
        
        return returns
    
    def get_advantages(self, values: List[float], 
                       gamma: float = 0.99) -> List[float]:
        """
        计算优势函数A_t
        Calculate advantage function A_t
        
        A_t = G_t - V(S_t)
        
        优势函数衡量动作比平均好多少！
        Advantage measures how much better an action is than average!
        
        Args:
            values: 状态价值估计 State value estimates
            gamma: 折扣因子
            
        Returns:
            优势值列表
            List of advantage values
        """
        returns = self.get_returns(gamma)
        advantages = [G - V for G, V in zip(returns, values)]
        return advantages
    
    def __len__(self):
        """轨迹长度"""
        return self.length
    
    def __getitem__(self, idx):
        """索引访问"""
        return self.experiences[idx]


@dataclass
class Episode:
    """
    回合 - 从初始状态到终止状态的完整交互
    Episode - Complete interaction from initial to terminal state
    
    回合是有明确开始和结束的任务单元！
    Episode is a task unit with clear beginning and end!
    
    分类 Categories:
    1. 回合式任务(Episodic Tasks): 有终止状态
       如：游戏、迷宫导航
       e.g., games, maze navigation
       
    2. 持续式任务(Continuing Tasks): 无终止状态
       如：机器人控制、股票交易
       e.g., robot control, stock trading
    """
    
    trajectory: Trajectory
    
    # 回合统计
    start_time: float = 0.0
    end_time: float = 0.0
    
    # 回合结果
    success: bool = False
    termination_reason: str = ""
    
    @property
    def duration(self) -> float:
        """回合持续时间"""
        return self.end_time - self.start_time
    
    @property
    def return_value(self) -> float:
        """回合总回报"""
        return self.trajectory.total_reward
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取回合统计信息"""
        return {
            'episode_id': self.trajectory.episode_id,
            'length': len(self.trajectory),
            'total_reward': self.return_value,
            'duration': self.duration,
            'success': self.success,
            'termination': self.termination_reason,
            'average_reward': self.return_value / max(1, len(self.trajectory))
        }


# ================================================================================
# 第2.2.2节：智能体-环境接口
# Section 2.2.2: Agent-Environment Interface
# ================================================================================

class AgentEnvironmentInterface:
    """
    智能体-环境接口 - RL的标准交互协议
    Agent-Environment Interface - Standard interaction protocol for RL
    
    这定义了所有RL系统的基本结构！
    This defines the basic structure of all RL systems!
    
    接口规范 Interface Specification:
    1. 环境提供状态和奖励
       Environment provides states and rewards
    2. 智能体选择动作
       Agent selects actions
    3. 循环直到终止
       Loop until termination
    
    设计原则 Design Principles:
    - 解耦：智能体和环境相互独立
      Decoupling: Agent and environment are independent
    - 标准化：统一的接口便于算法比较
      Standardization: Unified interface for algorithm comparison
    - 可扩展：易于添加新功能
      Extensibility: Easy to add new features
    """
    
    def __init__(self, agent: MDPAgent, environment: MDPEnvironment):
        """
        初始化接口
        Initialize interface
        
        Args:
            agent: 智能体实例
            environment: 环境实例
        """
        self.agent = agent
        self.environment = environment
        
        # 交互历史
        self.episodes: List[Episode] = []
        self.current_trajectory: Optional[Trajectory] = None
        
        # 统计信息
        self.total_steps = 0
        self.total_episodes = 0
        
        logger.info(f"初始化智能体-环境接口: {agent.name} <-> {environment.name}")
    
    def run_episode(self, max_steps: int = 1000,
                   render: bool = False) -> Episode:
        """
        运行一个回合
        Run one episode
        
        这是RL的核心循环！
        This is the core loop of RL!
        
        伪代码 Pseudocode:
        ```
        初始化 s_0
        for t = 0, 1, 2, ... do
            选择动作 a_t ~ π(·|s_t)
            执行动作，观察 r_{t+1}, s_{t+1}
            存储经验 (s_t, a_t, r_{t+1}, s_{t+1})
            学习/更新
            if 终止 then break
        end for
        ```
        
        Args:
            max_steps: 最大步数限制
            render: 是否渲染
            
        Returns:
            完成的回合
            Completed episode
        """
        # 开始新回合
        self.current_trajectory = Trajectory(
            episode_id=self.total_episodes,
            agent_name=self.agent.name
        )
        
        # 重置环境和智能体
        state = self.environment.reset()
        self.agent.reset()
        
        # 记录开始时间
        import time
        start_time = time.time()
        
        # 回合主循环
        for step in range(max_steps):
            # 渲染（如果需要）
            if render:
                self.environment.render()
            
            # 智能体选择动作
            action = self.agent.select_action(state)
            
            # 环境执行动作
            next_state, reward, done, info = self.environment.step(action)
            
            # 创建经验
            experience = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                info=info,
                timestep=step
            )
            
            # 存储经验
            self.current_trajectory.add_experience(experience)
            
            # 智能体学习
            self.agent.update(state, action, reward, next_state, done)
            
            # 更新状态
            state = next_state
            self.total_steps += 1
            
            # 检查终止
            if done:
                break
        
        # 创建回合对象
        episode = Episode(
            trajectory=self.current_trajectory,
            start_time=start_time,
            end_time=time.time(),
            success=done,
            termination_reason="terminal" if done else "max_steps"
        )
        
        # 保存回合
        self.episodes.append(episode)
        self.total_episodes += 1
        
        # 记录统计
        stats = episode.get_statistics()
        logger.info(f"Episode {self.total_episodes} finished: "
                   f"Steps={stats['length']}, "
                   f"Reward={stats['total_reward']:.2f}")
        
        return episode
    
    def run_episodes(self, n_episodes: int = 100,
                    max_steps_per_episode: int = 1000,
                    render_freq: Optional[int] = None) -> List[Episode]:
        """
        运行多个回合
        Run multiple episodes
        
        用于训练和评估！
        For training and evaluation!
        
        Args:
            n_episodes: 回合数
            max_steps_per_episode: 每回合最大步数
            render_freq: 渲染频率（None表示不渲染）
            
        Returns:
            回合列表
            List of episodes
        """
        episodes = []
        
        for i in range(n_episodes):
            # 决定是否渲染
            render = render_freq is not None and (i % render_freq == 0)
            
            # 运行回合
            episode = self.run_episode(
                max_steps=max_steps_per_episode,
                render=render
            )
            
            episodes.append(episode)
            
            # 定期报告进度
            if (i + 1) % 10 == 0:
                recent_rewards = [ep.return_value for ep in episodes[-10:]]
                avg_reward = np.mean(recent_rewards)
                logger.info(f"Episodes {i+1}/{n_episodes}, "
                           f"Recent avg reward: {avg_reward:.2f}")
        
        return episodes
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取交互统计
        Get interaction statistics
        
        用于监控学习进度！
        For monitoring learning progress!
        """
        if not self.episodes:
            return {
                'total_steps': 0,
                'total_episodes': 0,
                'average_reward': 0,
                'average_length': 0
            }
        
        rewards = [ep.return_value for ep in self.episodes]
        lengths = [len(ep.trajectory) for ep in self.episodes]
        
        return {
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'average_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'average_length': np.mean(lengths),
            'success_rate': np.mean([ep.success for ep in self.episodes])
        }
    
    def plot_learning_curve(self, window_size: int = 10):
        """
        绘制学习曲线
        Plot learning curve
        
        可视化是理解算法行为的关键！
        Visualization is key to understanding algorithm behavior!
        
        Args:
            window_size: 移动平均窗口大小
        """
        if not self.episodes:
            print("No episodes to plot")
            return
        
        # 提取数据
        rewards = [ep.return_value for ep in self.episodes]
        lengths = [len(ep.trajectory) for ep in self.episodes]
        
        # 计算移动平均
        def moving_average(data, window):
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 1. 回合奖励
        ax1 = axes[0, 0]
        ax1.plot(rewards, alpha=0.3, color='blue', label='Raw')
        if len(rewards) >= window_size:
            ma_rewards = moving_average(rewards, window_size)
            ma_x = range(window_size-1, len(rewards))
            ax1.plot(ma_x, ma_rewards, color='red', linewidth=2,
                    label=f'MA({window_size})')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Episode Rewards / 回合奖励')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 回合长度
        ax2 = axes[0, 1]
        ax2.plot(lengths, alpha=0.3, color='green', label='Raw')
        if len(lengths) >= window_size:
            ma_lengths = moving_average(lengths, window_size)
            ax2.plot(ma_x, ma_lengths, color='red', linewidth=2,
                    label=f'MA({window_size})')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Episode Length')
        ax2.set_title('Episode Lengths / 回合长度')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 奖励分布
        ax3 = axes[1, 0]
        ax3.hist(rewards, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax3.axvline(np.mean(rewards), color='red', linestyle='--',
                   label=f'Mean: {np.mean(rewards):.2f}')
        ax3.set_xlabel('Total Reward')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Reward Distribution / 奖励分布')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. 累积奖励
        ax4 = axes[1, 1]
        cumulative_rewards = np.cumsum(rewards)
        ax4.plot(cumulative_rewards, color='purple', linewidth=2)
        ax4.fill_between(range(len(cumulative_rewards)),
                        0, cumulative_rewards, alpha=0.3, color='purple')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Cumulative Reward')
        ax4.set_title('Cumulative Rewards / 累积奖励')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Learning Progress: {self.agent.name} in {self.environment.name}',
                    fontsize=14)
        plt.tight_layout()
        
        return fig


# ================================================================================
# 第2.2.3节：经验缓冲区
# Section 2.2.3: Experience Buffer
# ================================================================================

class ExperienceBuffer:
    """
    经验缓冲区 - 存储和管理经验
    Experience Buffer - Store and manage experiences
    
    经验回放的基础设施！
    Infrastructure for experience replay!
    
    为什么需要经验缓冲？
    Why need experience buffer?
    1. 打破数据相关性
       Break data correlation
    2. 提高样本效率
       Improve sample efficiency
    3. 实现离策略学习
       Enable off-policy learning
    """
    
    def __init__(self, capacity: int = 10000):
        """
        初始化经验缓冲区
        
        Args:
            capacity: 最大容量
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
        # 统计信息
        self.total_added = 0
        
        logger.info(f"初始化经验缓冲区，容量: {capacity}")
    
    def add(self, experience: Experience):
        """添加经验"""
        self.buffer.append(experience)
        self.total_added += 1
    
    def add_trajectory(self, trajectory: Trajectory):
        """添加整条轨迹"""
        for exp in trajectory.experiences:
            self.add(exp)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """
        随机采样一批经验
        Randomly sample a batch of experiences
        
        用于小批量学习！
        For mini-batch learning!
        
        Args:
            batch_size: 批大小
            
        Returns:
            经验列表
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        indices = np.random.choice(len(self.buffer), 
                                  batch_size, 
                                  replace=False)
        return [self.buffer[i] for i in indices]
    
    def sample_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        采样并整理成批次
        Sample and organize into batch
        
        便于神经网络处理！
        Convenient for neural network processing!
        
        Returns:
            批次数据字典
        """
        experiences = self.sample(batch_size)
        
        # 整理成数组
        batch = {
            'states': [exp.state for exp in experiences],
            'actions': [exp.action for exp in experiences],
            'rewards': np.array([exp.reward for exp in experiences]),
            'next_states': [exp.next_state for exp in experiences],
            'dones': np.array([exp.done for exp in experiences])
        }
        
        return batch
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
    
    def __len__(self):
        """缓冲区大小"""
        return len(self.buffer)
    
    def is_full(self):
        """是否已满"""
        return len(self.buffer) >= self.capacity
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.buffer:
            return {
                'size': 0,
                'capacity': self.capacity,
                'utilization': 0.0
            }
        
        rewards = [exp.reward for exp in self.buffer]
        
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'utilization': len(self.buffer) / self.capacity,
            'total_added': self.total_added,
            'average_reward': np.mean(rewards),
            'reward_std': np.std(rewards)
        }


# ================================================================================
# 第2.2.4节：简单示例智能体
# Section 2.2.4: Simple Example Agent
# ================================================================================

class RandomAgent(MDPAgent):
    """
    随机智能体 - 基准对比
    Random Agent - Baseline for comparison
    
    最简单的智能体，用于测试接口！
    Simplest agent for testing interface!
    """
    
    def __init__(self, action_space: List[Action]):
        """初始化随机智能体"""
        super().__init__(name="Random Agent")
        self.action_space = action_space
    
    def select_action(self, state: State) -> Action:
        """随机选择动作"""
        return np.random.choice(self.action_space)
    
    def update(self, state: State, action: Action,
              reward: float, next_state: State, done: bool):
        """随机智能体不学习"""
        # 只保存经验，不更新
        self.save_experience(state, action, reward, next_state, done)


# ================================================================================
# 示例和测试
# Examples and Tests
# ================================================================================

def demonstrate_interface():
    """
    演示智能体-环境接口
    Demonstrate agent-environment interface
    """
    print("\n" + "="*80)
    print("智能体-环境接口演示")
    print("Agent-Environment Interface Demonstration")
    print("="*80)
    
    # 导入回收机器人环境
    from .mdp_framework import RecyclingRobot
    
    # 创建环境
    env = RecyclingRobot()
    
    # 创建随机智能体
    agent = RandomAgent(env.action_space)
    
    # 创建接口
    interface = AgentEnvironmentInterface(agent, env)
    
    # 运行单个回合
    print("\n运行单个回合：")
    print("Running single episode:")
    episode = interface.run_episode(max_steps=20, render=True)
    
    print(f"\n回合统计：")
    print("Episode statistics:")
    for key, value in episode.get_statistics().items():
        print(f"  {key}: {value}")
    
    # 计算回报
    returns = episode.trajectory.get_returns(gamma=0.9)
    print(f"\n各步回报 Returns: {[f'{r:.2f}' for r in returns[:5]]}...")
    
    # 运行多个回合
    print("\n运行多个回合进行学习：")
    print("Running multiple episodes for learning:")
    episodes = interface.run_episodes(n_episodes=50, max_steps_per_episode=20)
    
    # 显示总体统计
    print("\n总体统计：")
    print("Overall statistics:")
    stats = interface.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # 绘制学习曲线
    print("\n绘制学习曲线...")
    print("Plotting learning curve...")
    fig = interface.plot_learning_curve(window_size=5)
    
    # 测试经验缓冲区
    print("\n" + "="*80)
    print("经验缓冲区测试")
    print("Experience Buffer Test")
    print("="*80)
    
    buffer = ExperienceBuffer(capacity=100)
    
    # 添加所有轨迹
    for ep in episodes[:10]:
        buffer.add_trajectory(ep.trajectory)
    
    print(f"缓冲区状态：")
    print("Buffer status:")
    for key, value in buffer.get_statistics().items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # 采样批次
    batch = buffer.sample_batch(batch_size=5)
    print(f"\n采样批次大小: {len(batch['rewards'])}")
    print(f"批次平均奖励: {np.mean(batch['rewards']):.2f}")
    
    return fig


def main():
    """主函数"""
    print("\n" + "="*80)
    print("第2.2节：智能体-环境接口")
    print("Section 2.2: Agent-Environment Interface")
    print("="*80)
    
    # 运行演示
    fig = demonstrate_interface()
    
    print("\n" + "="*80)
    print("接口演示完成！")
    print("Interface Demo Complete!")
    print("\n这个接口是所有RL算法的基础结构")
    print("This interface is the basic structure of all RL algorithms")
    print("="*80)
    
    # 显示图表
    plt.show()
    
    return fig


if __name__ == "__main__":
    main()