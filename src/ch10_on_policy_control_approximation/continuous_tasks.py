"""
================================================================================
第10.3-10.4节：连续任务与平均奖励
Section 10.3-10.4: Continuing Tasks and Average Reward
================================================================================

无终止的持续学习！
Learning that never ends!

连续任务的挑战 Challenges of Continuing Tasks:
1. 无自然终止点
   No natural termination
2. 折扣可能不合适
   Discounting may be inappropriate
3. 需要新的性能度量
   Need new performance metric

平均奖励设置 Average Reward Setting:
不最大化折扣回报，而是最大化平均奖励率
Instead of discounted return, maximize average reward rate

r(π) = lim_{T→∞} 1/T Σ_{t=1}^T E[R_t|π]

差分价值函数 Differential Value Function:
衡量相对于平均的优势
Measures advantage relative to average

q_π(s,a) = E[Σ_{k=1}^∞ (R_{t+k} - r(π)) | S_t=s, A_t=a]

Access-Control Queuing例子:
- 队列管理问题
  Queue management problem  
- 接受/拒绝决策
  Accept/reject decisions
- 优先级权衡
  Priority tradeoffs
"""

import numpy as np
from typing import List, Tuple, Optional, Any, Dict
from dataclasses import dataclass
from collections import deque
import logging

# 设置日志
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第10.3.1节：平均奖励设置
# Section 10.3.1: Average Reward Setting
# ================================================================================

class AverageRewardSetting:
    """
    平均奖励设置
    Average Reward Setting
    
    适用于连续任务的性能度量
    Performance metric for continuing tasks
    
    核心概念 Core Concepts:
    1. 平均奖励率 r(π)
       Average reward rate
    2. 差分回报
       Differential return
    3. 相对价值函数
       Relative value function
    
    关键性质 Key Properties:
    - 与起始状态无关(遍历性)
      Independent of start state (ergodicity)
    - 长期平均性能
      Long-term average performance
    """
    
    def __init__(self, alpha: float = 0.01):
        """
        初始化平均奖励设置
        Initialize average reward setting
        
        Args:
            alpha: 平均奖励学习率
                  Average reward learning rate
        """
        self.alpha = alpha
        
        # 平均奖励估计
        # Average reward estimate
        self.average_reward = 0.0
        
        # 奖励历史(用于计算真实平均)
        # Reward history (for true average)
        self.reward_history = deque(maxlen=1000)
        self.total_rewards = 0.0
        self.total_steps = 0
        
        logger.info(f"初始化平均奖励设置: α={alpha}")
    
    def update_average(self, reward: float):
        """
        更新平均奖励估计
        Update average reward estimate
        
        增量更新：
        Incremental update:
        r̄ ← r̄ + α(R - r̄)
        
        Args:
            reward: 当前奖励
                   Current reward
        """
        # 增量更新
        # Incremental update
        self.average_reward += self.alpha * (reward - self.average_reward)
        
        # 记录历史
        # Record history
        self.reward_history.append(reward)
        self.total_rewards += reward
        self.total_steps += 1
    
    def get_true_average(self) -> float:
        """
        获取真实平均奖励
        Get true average reward
        
        Returns:
            历史平均奖励
            Historical average reward
        """
        if self.total_steps == 0:
            return 0.0
        return self.total_rewards / self.total_steps
    
    def get_recent_average(self, window: int = 100) -> float:
        """
        获取最近的平均奖励
        Get recent average reward
        
        Args:
            window: 窗口大小
                   Window size
        
        Returns:
            最近窗口的平均
            Recent window average
        """
        if len(self.reward_history) == 0:
            return 0.0
        
        recent = list(self.reward_history)[-window:]
        return np.mean(recent)
    
    def compute_differential_return(self, rewards: List[float]) -> float:
        """
        计算差分回报
        Compute differential return
        
        G = Σ_{k=0}^∞ (R_{t+k+1} - r̄)
        
        Args:
            rewards: 奖励序列
                    Reward sequence
        
        Returns:
            差分回报
            Differential return
        """
        differential_return = 0.0
        for reward in rewards:
            differential_return += reward - self.average_reward
        return differential_return


# ================================================================================
# 第10.3.2节：差分半梯度Sarsa
# Section 10.3.2: Differential Semi-gradient Sarsa
# ================================================================================

class DifferentialSemiGradientSarsa:
    """
    差分半梯度Sarsa
    Differential Semi-gradient Sarsa
    
    平均奖励设置下的控制算法
    Control algorithm for average reward setting
    
    更新规则 Update Rules:
    δ = R - r̄ + q̂(S',A',w) - q̂(S,A,w)
    r̄ ← r̄ + βδ
    w ← w + αδ∇q̂(S,A,w)
    
    关键区别 Key Differences:
    - 无折扣因子
      No discount factor
    - 减去平均奖励
      Subtract average reward
    - 同时学习r̄和q
      Learn both r̄ and q
    """
    
    def __init__(self,
                n_features: int,
                n_actions: int,
                alpha: float = 0.1,
                beta: float = 0.01,
                epsilon: float = 0.1):
        """
        初始化差分Sarsa
        Initialize Differential Sarsa
        
        Args:
            n_features: 特征数
                       Number of features
            n_actions: 动作数
                      Number of actions
            alpha: 权重学习率
                  Weight learning rate
            beta: 平均奖励学习率
                 Average reward learning rate
            epsilon: 探索率
                    Exploration rate
        """
        self.n_features = n_features
        self.n_actions = n_actions
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        
        # 权重向量
        # Weight vectors
        self.weights = np.zeros((n_actions, n_features))
        
        # 平均奖励
        # Average reward
        self.average_reward = 0.0
        
        # 统计
        # Statistics
        self.step_count = 0
        self.td_errors = []
        
        logger.info(f"初始化差分Sarsa: α={alpha}, β={beta}")
    
    def get_features(self, state: Any) -> np.ndarray:
        """获取状态特征"""
        if isinstance(state, np.ndarray):
            return state
        return np.array(state)
    
    def get_q_value(self, state: Any, action: int) -> float:
        """
        获取差分动作价值
        Get differential action value
        
        Args:
            state: 状态
                  State
            action: 动作
                   Action
        
        Returns:
            差分动作价值
            Differential action value
        """
        features = self.get_features(state)
        return np.dot(self.weights[action], features)
    
    def select_action(self, state: Any) -> int:
        """
        ε-贪婪动作选择
        ε-greedy action selection
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            q_values = [self.get_q_value(state, a) for a in range(self.n_actions)]
            # 随机打破平局
            # Random tie-breaking
            max_q = max(q_values)
            max_actions = [a for a, q in enumerate(q_values) if q == max_q]
            return np.random.choice(max_actions)
    
    def update(self, state: Any, action: int, reward: float,
              next_state: Any, next_action: int):
        """
        差分Sarsa更新
        Differential Sarsa update
        
        Args:
            state: 当前状态
                  Current state
            action: 当前动作
                   Current action
            reward: 奖励
                   Reward
            next_state: 下一状态
                       Next state
            next_action: 下一动作
                        Next action
        """
        features = self.get_features(state)
        
        # 计算TD误差(差分形式)
        # Compute TD error (differential form)
        current_q = self.get_q_value(state, action)
        next_q = self.get_q_value(next_state, next_action)
        
        # δ = R - r̄ + Q(S',A') - Q(S,A)
        td_error = reward - self.average_reward + next_q - current_q
        
        self.td_errors.append(td_error)
        
        # 更新平均奖励
        # Update average reward
        self.average_reward += self.beta * td_error
        
        # 更新权重
        # Update weights
        self.weights[action] += self.alpha * td_error * features
        
        self.step_count += 1
    
    def learn_steps(self, env: Any, n_steps: int = 1000) -> Dict[str, float]:
        """
        学习指定步数
        Learn for specified steps
        
        Args:
            env: 环境
                Environment
            n_steps: 步数
                    Number of steps
        
        Returns:
            学习统计
            Learning statistics
        """
        # 初始化
        # Initialize
        state = env.reset()
        action = self.select_action(state)
        
        total_reward = 0.0
        
        for step in range(n_steps):
            # 执行动作
            # Execute action
            next_state, reward, _, _ = env.step(action)
            total_reward += reward
            
            # 选择下一动作
            # Select next action
            next_action = self.select_action(next_state)
            
            # 更新
            # Update
            self.update(state, action, reward, next_state, next_action)
            
            # 前进
            # Advance
            state = next_state
            action = next_action
            
            # 记录进度
            # Log progress
            if (step + 1) % (n_steps // 10) == 0:
                avg_td_error = np.mean(np.abs(self.td_errors[-100:]))
                logger.debug(f"步 {step+1}: r̄={self.average_reward:.3f}, "
                           f"|δ|={avg_td_error:.3f}")
        
        return {
            'total_reward': total_reward,
            'average_reward': self.average_reward,
            'final_td_error': np.mean(np.abs(self.td_errors[-100:]))
        }


# ================================================================================
# 第10.4节：Access-Control队列问题
# Section 10.4: Access-Control Queuing Problem
# ================================================================================

@dataclass
class Customer:
    """
    顾客
    Customer
    
    具有优先级和服务奖励
    Has priority and service reward
    """
    priority: int      # 优先级(1-4)
    reward: float      # 服务奖励
    service_time: int  # 服务时间


class AccessControlQueuing:
    """
    接入控制队列
    Access-Control Queuing
    
    决定是否接受新顾客进入有限容量队列
    Decide whether to accept new customers into limited capacity queue
    
    问题设置 Problem Setup:
    - k个服务器
      k servers
    - n个优先级
      n priorities
    - 有限队列容量
      Limited queue capacity
    - 接受/拒绝决策
      Accept/reject decisions
    
    目标 Goal:
    最大化长期平均奖励
    Maximize long-term average reward
    """
    
    def __init__(self,
                n_servers: int = 10,
                n_priorities: int = 4,
                queue_capacity: int = 10,
                arrival_prob: float = 0.5):
        """
        初始化队列系统
        Initialize queuing system
        
        Args:
            n_servers: 服务器数
                      Number of servers
            n_priorities: 优先级数
                         Number of priorities
            queue_capacity: 队列容量
                           Queue capacity
            arrival_prob: 到达概率
                         Arrival probability
        """
        self.n_servers = n_servers
        self.n_priorities = n_priorities
        self.queue_capacity = queue_capacity
        self.arrival_prob = arrival_prob
        
        # 队列状态
        # Queue state
        self.free_servers = n_servers
        self.queue = deque(maxlen=queue_capacity)
        
        # 优先级奖励(高优先级更高奖励)
        # Priority rewards (higher priority, higher reward)
        self.priority_rewards = {
            1: 1.0,
            2: 2.0,
            3: 4.0,
            4: 8.0
        }
        
        # 统计
        # Statistics
        self.accepted_customers = 0
        self.rejected_customers = 0
        self.total_reward = 0.0
        
        logger.info(f"初始化队列系统: {n_servers}服务器, {n_priorities}优先级")
    
    def reset(self) -> np.ndarray:
        """
        重置队列
        Reset queue
        
        Returns:
            初始状态
            Initial state
        """
        self.free_servers = self.n_servers
        self.queue.clear()
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """
        获取当前状态
        Get current state
        
        状态包含:
        State contains:
        - 空闲服务器数
          Number of free servers
        - 各优先级队列长度
          Queue length per priority
        
        Returns:
            状态向量
            State vector
        """
        # 统计各优先级顾客数
        # Count customers per priority
        priority_counts = {i: 0 for i in range(1, self.n_priorities + 1)}
        for customer in self.queue:
            priority_counts[customer.priority] += 1
        
        # 构造状态向量
        # Construct state vector
        state = [self.free_servers]
        for i in range(1, self.n_priorities + 1):
            state.append(priority_counts[i])
        
        return np.array(state)
    
    def generate_customer(self) -> Optional[Customer]:
        """
        生成新顾客
        Generate new customer
        
        Returns:
            新顾客或None
            New customer or None
        """
        if np.random.random() < self.arrival_prob:
            # 随机优先级(高优先级较少)
            # Random priority (higher priority less common)
            # 根据优先级数量动态生成概率分布
            # Generate probability distribution based on number of priorities
            if self.n_priorities == 2:
                probs = [0.7, 0.3]
            elif self.n_priorities == 3:
                probs = [0.5, 0.3, 0.2]
            elif self.n_priorities == 4:
                probs = [0.5, 0.25, 0.15, 0.1]
            else:
                # 默认均匀分布
                # Default uniform distribution
                probs = [1.0 / self.n_priorities] * self.n_priorities
            
            priority = np.random.choice(
                range(1, self.n_priorities + 1),
                p=probs
            )
            
            reward = self.priority_rewards.get(priority, 1.0)
            service_time = np.random.geometric(0.3)  # 几何分布服务时间
            
            return Customer(priority, reward, service_time)
        
        return None
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步
        Execute one step
        
        Args:
            action: 0=拒绝, 1=接受
                   0=reject, 1=accept
        
        Returns:
            (下一状态, 奖励, done, info)
            (next_state, reward, done, info)
        """
        reward = 0.0
        
        # 生成新顾客
        # Generate new customer
        new_customer = self.generate_customer()
        
        if new_customer is not None:
            if action == 1 and len(self.queue) < self.queue_capacity:
                # 接受顾客
                # Accept customer
                self.queue.append(new_customer)
                self.accepted_customers += 1
            else:
                # 拒绝顾客
                # Reject customer
                self.rejected_customers += 1
        
        # 处理服务完成
        # Process service completions
        if self.free_servers < self.n_servers:
            # 随机一些服务完成
            # Random service completions
            completions = np.random.binomial(
                self.n_servers - self.free_servers, 0.2
            )
            self.free_servers = min(
                self.n_servers,
                self.free_servers + completions
            )
        
        # 分配空闲服务器给队列顾客
        # Assign free servers to queued customers
        while self.free_servers > 0 and len(self.queue) > 0:
            # 服务最高优先级顾客
            # Serve highest priority customer
            customer = self.queue.popleft()
            self.free_servers -= 1
            reward += customer.reward
            self.total_reward += customer.reward
        
        next_state = self.get_state()
        
        # 连续任务，永不结束
        # Continuing task, never ends
        done = False
        
        info = {
            'accepted': self.accepted_customers,
            'rejected': self.rejected_customers,
            'queue_length': len(self.queue),
            'free_servers': self.free_servers
        }
        
        return next_state, reward, done, info


# ================================================================================
# 主函数：演示连续任务控制
# Main Function: Demonstrate Continuing Task Control
# ================================================================================

def demonstrate_continuous_control():
    """
    演示连续任务控制
    Demonstrate continuing task control
    """
    print("\n" + "="*80)
    print("第10.3-10.4节：连续任务与平均奖励")
    print("Section 10.3-10.4: Continuing Tasks and Average Reward")
    print("="*80)
    
    # 1. 测试平均奖励设置
    # 1. Test average reward setting
    print("\n" + "="*60)
    print("1. 平均奖励设置")
    print("1. Average Reward Setting")
    print("="*60)
    
    avg_reward = AverageRewardSetting(alpha=0.01)
    
    print("\n模拟奖励流...")
    # 模拟变化的奖励
    # Simulate varying rewards
    for t in range(100):
        # 正弦变化的奖励
        # Sinusoidal rewards
        reward = 5.0 + 2.0 * np.sin(t * 0.1)
        avg_reward.update_average(reward)
        
        if (t + 1) % 20 == 0:
            print(f"  t={t+1}: r={reward:.2f}, "
                  f"r̄_est={avg_reward.average_reward:.2f}, "
                  f"r̄_true={avg_reward.get_true_average():.2f}")
    
    print(f"\n最终估计: {avg_reward.average_reward:.3f}")
    print(f"真实平均: {avg_reward.get_true_average():.3f}")
    print(f"最近平均: {avg_reward.get_recent_average(50):.3f}")
    
    # 测试差分回报
    # Test differential return
    test_rewards = [6.0, 4.0, 5.0, 7.0, 3.0]
    diff_return = avg_reward.compute_differential_return(test_rewards)
    print(f"\n差分回报示例: G={diff_return:.2f}")
    print(f"  奖励: {test_rewards}")
    print(f"  平均: {avg_reward.average_reward:.2f}")
    
    # 2. 测试差分Sarsa
    # 2. Test Differential Sarsa
    print("\n" + "="*60)
    print("2. 差分半梯度Sarsa")
    print("2. Differential Semi-gradient Sarsa")
    print("="*60)
    
    n_features = 8
    n_actions = 2
    
    diff_sarsa = DifferentialSemiGradientSarsa(
        n_features=n_features,
        n_actions=n_actions,
        alpha=0.1,
        beta=0.01,
        epsilon=0.1
    )
    
    print("\n模拟学习步骤...")
    for step in range(10):
        state = np.random.randn(n_features)
        action = diff_sarsa.select_action(state)
        reward = np.random.randn() + 1.0  # 正偏奖励
        next_state = np.random.randn(n_features)
        next_action = diff_sarsa.select_action(next_state)
        
        diff_sarsa.update(state, action, reward, next_state, next_action)
        
        if (step + 1) % 3 == 0:
            print(f"  步 {step+1}: r̄={diff_sarsa.average_reward:.3f}, "
                  f"Q={diff_sarsa.get_q_value(state, action):.3f}")
    
    # 3. 测试队列系统
    # 3. Test Queuing System
    print("\n" + "="*60)
    print("3. Access-Control队列")
    print("3. Access-Control Queuing")
    print("="*60)
    
    queue_env = AccessControlQueuing(
        n_servers=5,
        n_priorities=4,
        queue_capacity=10,
        arrival_prob=0.6
    )
    
    print(f"\n队列配置:")
    print(f"  服务器: {queue_env.n_servers}")
    print(f"  优先级: {queue_env.n_priorities}")
    print(f"  容量: {queue_env.queue_capacity}")
    
    # 重置并运行几步
    # Reset and run some steps
    state = queue_env.reset()
    print(f"\n初始状态: {state}")
    
    print("\n模拟队列操作...")
    total_reward = 0.0
    
    for step in range(20):
        # 简单策略：队列不满就接受
        # Simple policy: accept if queue not full
        action = 1 if state[0] > 0 else 0
        
        next_state, reward, _, info = queue_env.step(action)
        total_reward += reward
        
        if (step + 1) % 5 == 0:
            print(f"  步 {step+1}: 奖励={reward:.1f}, "
                  f"队列长={info['queue_length']}, "
                  f"空闲={info['free_servers']}")
        
        state = next_state
    
    print(f"\n队列统计:")
    print(f"  接受顾客: {queue_env.accepted_customers}")
    print(f"  拒绝顾客: {queue_env.rejected_customers}")
    print(f"  总奖励: {total_reward:.1f}")
    print(f"  平均奖励: {total_reward/20:.2f}")
    
    # 4. 组合测试
    # 4. Combined test
    print("\n" + "="*60)
    print("4. 差分Sarsa学习队列控制")
    print("4. Differential Sarsa Learning Queue Control")
    print("="*60)
    
    # 创建新队列环境
    # Create new queue environment
    queue_env2 = AccessControlQueuing(
        n_servers=3,
        n_priorities=2,
        queue_capacity=5
    )
    
    # 差分Sarsa学习器
    # Differential Sarsa learner
    state_size = 1 + queue_env2.n_priorities
    learner = DifferentialSemiGradientSarsa(
        n_features=state_size,
        n_actions=2,  # 接受/拒绝
        alpha=0.1,
        beta=0.01,
        epsilon=0.2
    )
    
    print("\n训练差分Sarsa...")
    state = queue_env2.reset()
    action = learner.select_action(state)
    
    for step in range(100):
        next_state, reward, _, _ = queue_env2.step(action)
        next_action = learner.select_action(next_state)
        
        learner.update(state, action, reward, next_state, next_action)
        
        state = next_state
        action = next_action
        
        if (step + 1) % 25 == 0:
            print(f"  步 {step+1}: r̄={learner.average_reward:.3f}")
    
    print(f"\n最终平均奖励: {learner.average_reward:.3f}")
    
    # 总结
    # Summary
    print("\n" + "="*80)
    print("连续任务控制总结")
    print("Continuing Tasks Control Summary")
    print("="*80)
    
    print("""
    关键要点 Key Takeaways:
    =======================
    
    1. 连续任务需要平均奖励
       Continuing tasks need average reward
       
    2. 差分价值函数衡量相对优势
       Differential value measures relative advantage
       
    3. 同时学习r̄和q
       Learn both r̄ and q simultaneously
       
    4. 无折扣因子(γ=1)
       No discount factor
       
    5. 适合长期运行系统
       Suitable for long-running systems
    
    队列控制挑战:
    - 动态负载平衡
      Dynamic load balancing
    - 优先级权衡
      Priority tradeoffs
    - 容量管理
      Capacity management
    - 长期优化
      Long-term optimization
    """)


if __name__ == "__main__":
    demonstrate_continuous_control()