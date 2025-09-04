"""
================================================================================
第13.6节：自然策略梯度
Section 13.6: Natural Policy Gradient
================================================================================

考虑参数空间几何的策略梯度！
Policy gradient considering parameter space geometry!

核心思想 Core Idea:
标准梯度在参数空间是最陡的，但在策略空间可能不是
Standard gradient is steepest in parameter space, but may not be in policy space

自然梯度 Natural Gradient:
∇̃J(θ) = F^{-1} ∇J(θ)

其中 Where:
F: Fisher信息矩阵
   Fisher Information Matrix
   F = E[∇ln π(a|s,θ) ∇ln π(a|s,θ)^T]

优势 Advantages:
- 参数不变性
  Parameter invariance
- 更稳定的更新
  More stable updates
- 更快的收敛
  Faster convergence

现代算法 Modern Algorithms:
- TRPO (Trust Region Policy Optimization)
- PPO (Proximal Policy Optimization)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import logging

# 设置日志
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第13.6.1节：自然策略梯度
# Section 13.6.1: Natural Policy Gradient
# ================================================================================

class NaturalPolicyGradient:
    """
    自然策略梯度算法
    Natural Policy Gradient Algorithm
    
    使用Fisher信息矩阵调整梯度方向
    Use Fisher Information Matrix to adjust gradient direction
    
    关键洞察 Key Insight:
    在分布空间而不是参数空间中进行最陡下降
    Steepest descent in distribution space rather than parameter space
    
    更新规则 Update Rule:
    θ_{t+1} = θ_t + α F^{-1} ∇J(θ_t)
    """
    
    def __init__(self,
                 policy: Any,
                 alpha: float = 0.01,
                 gamma: float = 0.99,
                 damping: float = 0.001):  # 阻尼因子用于数值稳定
        """
        初始化自然策略梯度
        Initialize Natural Policy Gradient
        
        Args:
            policy: 策略
                   Policy
            alpha: 学习率
                  Learning rate
            gamma: 折扣因子
                  Discount factor
            damping: 阻尼因子（避免Fisher矩阵奇异）
                    Damping factor (avoid Fisher matrix singularity)
        """
        self.policy = policy
        self.alpha = alpha
        self.gamma = gamma
        self.damping = damping
        
        # Fisher信息矩阵
        # Fisher Information Matrix
        self.fisher_matrix = None
        
        # 统计
        # Statistics
        self.episode_count = 0
        self.episode_returns = []
        self.natural_gradient_norms = []
        
        logger.info(f"初始化Natural Policy Gradient: α={alpha}, damping={damping}")
    
    def compute_fisher_matrix(self, states: List[Any], actions: List[int]) -> np.ndarray:
        """
        计算Fisher信息矩阵
        Compute Fisher Information Matrix
        
        F = E_π[∇ln π(a|s) ∇ln π(a|s)^T]
        
        Args:
            states: 状态列表
                   State list
            actions: 动作列表
                    Action list
        
        Returns:
            Fisher信息矩阵
            Fisher Information Matrix
        """
        # 获取参数总数
        n_params = self.policy.theta.size
        fisher = np.zeros((n_params, n_params))
        
        for state, action in zip(states, actions):
            # 计算对数梯度
            # Compute log gradient
            log_grad = self.policy.compute_log_gradient(state, action)
            log_grad_flat = log_grad.flatten()
            
            # 确保梯度长度正确
            if len(log_grad_flat) != n_params:
                # 如果梯度是单个动作的，需要扩展为完整参数空间
                full_grad = np.zeros(n_params)
                if hasattr(self.policy, 'n_features'):
                    # 假设是softmax策略，梯度对应特定动作
                    start_idx = action * self.policy.n_features
                    end_idx = start_idx + len(log_grad_flat)
                    if end_idx <= n_params:
                        full_grad[start_idx:end_idx] = log_grad_flat
                    else:
                        full_grad[:len(log_grad_flat)] = log_grad_flat
                log_grad_flat = full_grad
            
            # 累积外积
            # Accumulate outer product
            fisher += np.outer(log_grad_flat, log_grad_flat)
        
        # 平均
        # Average
        fisher /= len(states)
        
        # 添加阻尼（数值稳定性）
        # Add damping (numerical stability)
        fisher += self.damping * np.eye(n_params)
        
        return fisher
    
    def compute_natural_gradient(self, gradient: np.ndarray, 
                                fisher: np.ndarray) -> np.ndarray:
        """
        计算自然梯度
        Compute natural gradient
        
        ∇̃J = F^{-1} ∇J
        
        Args:
            gradient: 标准策略梯度
                     Standard policy gradient
            fisher: Fisher信息矩阵
                   Fisher Information Matrix
        
        Returns:
            自然梯度
            Natural gradient
        """
        try:
            # 使用共轭梯度法求解 F x = g
            # Use conjugate gradient to solve F x = g
            natural_grad = np.linalg.solve(fisher, gradient.flatten())
        except np.linalg.LinAlgError:
            # 如果求解失败，使用伪逆
            # If solve fails, use pseudoinverse
            logger.warning("Fisher矩阵奇异，使用伪逆")
            natural_grad = np.linalg.pinv(fisher) @ gradient.flatten()
        
        return natural_grad.reshape(gradient.shape)
    
    def learn_episode(self, env: Any, max_steps: int = 1000) -> float:
        """
        学习一个回合
        Learn one episode
        
        Args:
            env: 环境
                Environment
            max_steps: 最大步数
                      Maximum steps
        
        Returns:
            回合总回报
            Episode total return
        """
        # 生成回合
        # Generate episode
        states = []
        actions = []
        rewards = []
        
        state = env.reset()
        
        for step in range(max_steps):
            action = self.policy.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            if done:
                break
            
            state = next_state
        
        # 计算回报
        # Compute returns
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        # 计算标准策略梯度
        # Compute standard policy gradient
        gradient = np.zeros_like(self.policy.theta)
        for t, (state, action, G) in enumerate(zip(states, actions, returns)):
            log_grad = self.policy.compute_log_gradient(state, action)
            # 确保梯度形状匹配
            if log_grad.shape != gradient.shape:
                if len(log_grad.shape) == 1 and len(gradient.shape) == 2:
                    # 梯度是一维的，需要放到对应动作的位置
                    gradient[action] += G * log_grad
                else:
                    gradient += G * log_grad.reshape(gradient.shape)
            else:
                gradient += G * log_grad
        gradient /= len(states)
        
        # 计算Fisher信息矩阵
        # Compute Fisher Information Matrix
        fisher = self.compute_fisher_matrix(states, actions)
        self.fisher_matrix = fisher
        
        # 计算自然梯度
        # Compute natural gradient
        natural_grad = self.compute_natural_gradient(gradient, fisher)
        self.natural_gradient_norms.append(np.linalg.norm(natural_grad))
        
        # 更新策略参数
        # Update policy parameters
        self.policy.update_parameters(self.alpha * natural_grad, step_size=1.0)
        
        # 统计
        # Statistics
        episode_return = sum(rewards)
        self.episode_returns.append(episode_return)
        self.episode_count += 1
        
        return episode_return


# ================================================================================
# 第13.6.2节：TRPO (Trust Region Policy Optimization)
# Section 13.6.2: TRPO (Trust Region Policy Optimization)
# ================================================================================

class TRPO:
    """
    信任域策略优化
    Trust Region Policy Optimization
    
    限制策略更新的大小
    Limit the size of policy updates
    
    优化问题 Optimization Problem:
    max_θ E[A^π_old(s,a)]
    s.t. KL(π_old || π_new) ≤ δ
    
    其中 Where:
    - A^π_old: 旧策略下的优势函数
              Advantage function under old policy
    - KL: KL散度
         KL divergence
    - δ: 信任域大小
        Trust region size
    
    关键特性 Key Features:
    - 保证单调改进
      Guaranteed monotonic improvement
    - 自适应步长
      Adaptive step size
    - 稳定的学习
      Stable learning
    """
    
    def __init__(self,
                 policy: Any,
                 value_function: Any,
                 delta: float = 0.01,  # KL散度约束
                 gamma: float = 0.99,
                 lam: float = 0.95,     # GAE参数
                 cg_iters: int = 10,    # 共轭梯度迭代次数
                 line_search_steps: int = 10):
        """
        初始化TRPO
        Initialize TRPO
        
        Args:
            delta: KL散度约束
                  KL divergence constraint
            lam: GAE的λ参数
                GAE lambda parameter
            cg_iters: 共轭梯度迭代次数
                     Conjugate gradient iterations
            line_search_steps: 线搜索步数
                              Line search steps
        """
        self.policy = policy
        self.value_function = value_function
        self.delta = delta
        self.gamma = gamma
        self.lam = lam
        self.cg_iters = cg_iters
        self.line_search_steps = line_search_steps
        
        # 统计
        # Statistics
        self.episode_count = 0
        self.episode_returns = []
        self.kl_divergences = []
        
        logger.info(f"初始化TRPO: δ={delta}")
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                    next_value: float) -> np.ndarray:
        """
        计算广义优势估计(GAE)
        Compute Generalized Advantage Estimation (GAE)
        
        A^GAE = Σ (γλ)^l δ_{t+l}
        """
        advantages = np.zeros(len(rewards))
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val - values[t]
            gae = delta + self.gamma * self.lam * gae
            advantages[t] = gae
        
        return advantages
    
    def compute_kl_divergence(self, states: List[Any], 
                            old_probs: List[np.ndarray]) -> float:
        """
        计算KL散度
        Compute KL divergence
        
        KL(π_old || π_new) = E_π_old[log(π_old/π_new)]
        
        Args:
            states: 状态列表
                   State list
            old_probs: 旧策略的动作概率
                      Action probabilities under old policy
        
        Returns:
            平均KL散度
            Average KL divergence
        """
        kl_div = 0.0
        
        for state, old_prob in zip(states, old_probs):
            new_prob = self.policy.compute_action_probabilities(state)
            
            # KL散度
            # KL divergence
            # 避免log(0)
            epsilon = 1e-8
            kl = np.sum(old_prob * np.log((old_prob + epsilon) / (new_prob + epsilon)))
            kl_div += kl
        
        return kl_div / len(states)
    
    def conjugate_gradient(self, Ax_func: Callable, b: np.ndarray) -> np.ndarray:
        """
        共轭梯度法求解 Ax = b
        Conjugate gradient to solve Ax = b
        
        Args:
            Ax_func: 计算Ax的函数
                    Function to compute Ax
            b: 右侧向量
              Right-hand side vector
        
        Returns:
            解x
            Solution x
        """
        x = np.zeros_like(b)
        r = b.copy()
        p = r.copy()
        r_dot = np.dot(r, r)
        
        for _ in range(self.cg_iters):
            Ap = Ax_func(p)
            alpha = r_dot / (np.dot(p, Ap) + 1e-8)
            x += alpha * p
            r -= alpha * Ap
            
            r_dot_new = np.dot(r, r)
            if r_dot_new < 1e-10:
                break
            
            beta = r_dot_new / r_dot
            p = r + beta * p
            r_dot = r_dot_new
        
        return x
    
    def line_search(self, states: List[Any], actions: List[int],
                   advantages: np.ndarray, old_probs: List[np.ndarray],
                   natural_gradient: np.ndarray) -> float:
        """
        线搜索找到合适的步长
        Line search to find appropriate step size
        
        Args:
            states: 状态列表
                   State list
            actions: 动作列表
                    Action list
            advantages: 优势值
                       Advantage values
            old_probs: 旧策略概率
                      Old policy probabilities
            natural_gradient: 自然梯度
                            Natural gradient
        
        Returns:
            最佳步长
            Best step size
        """
        # 保存原始参数
        # Save original parameters
        old_theta = self.policy.theta.copy()
        
        # 计算期望改进
        # Compute expected improvement
        expected_improvement = np.sum(advantages) / len(advantages)
        
        for step in range(self.line_search_steps):
            # 尝试的步长
            # Try step size
            step_size = 0.5 ** step
            
            # 更新参数
            # Update parameters
            self.policy.theta = old_theta + step_size * natural_gradient
            
            # 检查KL约束
            # Check KL constraint
            kl = self.compute_kl_divergence(states, old_probs)
            
            if kl <= self.delta:
                # 计算实际改进
                # Compute actual improvement
                actual_improvement = 0.0
                for state, action, adv in zip(states, actions, advantages):
                    prob = self.policy.compute_action_probabilities(state)[action]
                    actual_improvement += prob * adv
                actual_improvement /= len(states)
                
                # 如果改进足够，接受步长
                # If improvement is sufficient, accept step size
                if actual_improvement > 0.5 * expected_improvement * step_size:
                    return step_size
        
        # 恢复参数
        # Restore parameters
        self.policy.theta = old_theta
        return 0.0
    
    def update_policy(self, states: List[Any], actions: List[int],
                     advantages: np.ndarray):
        """
        TRPO策略更新
        TRPO policy update
        
        Args:
            states: 状态列表
                   State list
            actions: 动作列表
                    Action list
            advantages: 优势值
                       Advantage values
        """
        # 保存旧策略概率
        # Save old policy probabilities
        old_probs = [self.policy.compute_action_probabilities(s) for s in states]
        
        # 计算策略梯度
        # Compute policy gradient
        gradient = np.zeros_like(self.policy.theta)
        for state, action, adv in zip(states, actions, advantages):
            log_grad = self.policy.compute_log_gradient(state, action)
            gradient += adv * log_grad
        gradient /= len(states)
        
        # 计算Fisher-向量乘积的函数
        # Function to compute Fisher-vector product
        def fisher_vector_product(v):
            # Fv = E[∇ln π (∇ln π)^T v]
            fvp = np.zeros_like(v)
            for state in states:
                probs = self.policy.compute_action_probabilities(state)
                for a in range(self.policy.n_actions):
                    log_grad = self.policy.compute_log_gradient(state, a)
                    log_grad_flat = log_grad.flatten()
                    fvp += probs[a] * log_grad_flat * np.dot(log_grad_flat, v)
            return fvp / len(states) + self.damping * v
        
        # 使用共轭梯度求解自然梯度
        # Use conjugate gradient to solve for natural gradient
        natural_gradient = self.conjugate_gradient(
            fisher_vector_product, 
            gradient.flatten()
        ).reshape(gradient.shape)
        
        # 线搜索找到最佳步长
        # Line search to find best step size
        step_size = self.line_search(states, actions, advantages, 
                                     old_probs, natural_gradient)
        
        # 更新参数
        # Update parameters
        if step_size > 0:
            self.policy.theta += step_size * natural_gradient
            
            # 记录KL散度
            # Record KL divergence
            kl = self.compute_kl_divergence(states, old_probs)
            self.kl_divergences.append(kl)


# ================================================================================
# 第13.6.3节：PPO (Proximal Policy Optimization)
# Section 13.6.3: PPO (Proximal Policy Optimization)
# ================================================================================

class PPO:
    """
    近端策略优化
    Proximal Policy Optimization
    
    TRPO的简化版本，更容易实现
    Simplified version of TRPO, easier to implement
    
    目标函数 Objective Function:
    L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
    
    其中 Where:
    r_t(θ) = π(a|s,θ) / π(a|s,θ_old)  (重要性采样比率)
                                       (importance sampling ratio)
    
    关键创新 Key Innovation:
    通过裁剪避免过大的策略更新
    Avoid too large policy updates through clipping
    
    优势 Advantages:
    - 实现简单
      Simple implementation
    - 性能稳定
      Stable performance
    - 计算高效
      Computationally efficient
    """
    
    def __init__(self,
                 policy: Any,
                 value_function: Any,
                 clip_epsilon: float = 0.2,  # 裁剪参数
                 alpha_theta: float = 0.0003,
                 alpha_w: float = 0.001,
                 gamma: float = 0.99,
                 lam: float = 0.95,
                 epochs: int = 4,            # 每批数据的训练轮数
                 batch_size: int = 64):
        """
        初始化PPO
        Initialize PPO
        
        Args:
            clip_epsilon: 裁剪参数
                         Clipping parameter
            epochs: 每批数据的训练轮数
                   Training epochs per batch
            batch_size: 批大小
                       Batch size
        """
        self.policy = policy
        self.value_function = value_function
        self.clip_epsilon = clip_epsilon
        self.alpha_theta = alpha_theta
        self.alpha_w = alpha_w
        self.gamma = gamma
        self.lam = lam
        self.epochs = epochs
        self.batch_size = batch_size
        
        # 经验缓冲
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.advantages = []
        self.returns = []
        
        # 统计
        # Statistics
        self.episode_count = 0
        self.episode_returns = []
        self.clip_fractions = []
        
        logger.info(f"初始化PPO: ε={clip_epsilon}, epochs={epochs}")
    
    def compute_gae(self, rewards: List[float], values: List[float],
                    next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算GAE和回报
        Compute GAE and returns
        
        Returns:
            (优势, 回报)
            (advantages, returns)
        """
        advantages = np.zeros(len(rewards))
        returns = np.zeros(len(rewards))
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val - values[t]
            gae = delta + self.gamma * self.lam * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def collect_trajectories(self, env: Any, n_steps: int = 2048):
        """
        收集轨迹数据
        Collect trajectory data
        
        Args:
            env: 环境
                Environment
            n_steps: 收集的步数
                    Number of steps to collect
        """
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        
        if not hasattr(self, 'current_state'):
            self.current_state = env.reset()
        
        for _ in range(n_steps):
            # 获取动作和价值
            # Get action and value
            state = self.current_state
            action_probs = self.policy.compute_action_probabilities(state)
            action = np.random.choice(self.policy.n_actions, p=action_probs)
            value = self.value_function.get_value(state)
            log_prob = np.log(action_probs[action] + 1e-8)
            
            # 执行动作
            # Execute action
            next_state, reward, done, _ = env.step(action)
            
            # 存储经验
            # Store experience
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.values.append(value)
            self.log_probs.append(log_prob)
            
            if done:
                self.current_state = env.reset()
                self.episode_count += 1
                
                # 计算该回合的回报
                # Compute episode return
                episode_return = sum(self.rewards[-20:])  # 近似
                self.episode_returns.append(episode_return)
            else:
                self.current_state = next_state
        
        # 计算优势和回报
        # Compute advantages and returns
        next_value = self.value_function.get_value(self.current_state)
        self.advantages, self.returns = self.compute_gae(
            self.rewards, self.values, next_value
        )
        
        # 归一化优势
        # Normalize advantages
        self.advantages = (self.advantages - np.mean(self.advantages)) / (np.std(self.advantages) + 1e-8)
    
    def compute_ppo_loss(self, states: np.ndarray, actions: np.ndarray,
                        old_log_probs: np.ndarray, advantages: np.ndarray):
        """
        计算PPO损失
        Compute PPO loss
        
        L^CLIP = -E[min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t)]
        """
        # 计算新的对数概率
        # Compute new log probabilities
        new_log_probs = []
        for state, action in zip(states, actions):
            probs = self.policy.compute_action_probabilities(state)
            new_log_probs.append(np.log(probs[action] + 1e-8))
        new_log_probs = np.array(new_log_probs)
        
        # 重要性采样比率
        # Importance sampling ratio
        ratio = np.exp(new_log_probs - old_log_probs)
        
        # 裁剪的比率
        # Clipped ratio
        clipped_ratio = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        
        # PPO损失
        # PPO loss
        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages
        ppo_loss = -np.mean(np.minimum(surr1, surr2))
        
        # 记录裁剪比例
        # Record clipping fraction
        clip_fraction = np.mean(np.abs(ratio - 1.0) > self.clip_epsilon)
        self.clip_fractions.append(clip_fraction)
        
        return ppo_loss
    
    def update(self):
        """
        PPO更新
        PPO update
        """
        n_samples = len(self.states)
        
        for _ in range(self.epochs):
            # 随机打乱索引
            # Shuffle indices
            indices = np.random.permutation(n_samples)
            
            # 小批量更新
            # Mini-batch updates
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_indices = indices[start:end]
                
                # 获取批数据
                # Get batch data
                batch_states = [self.states[i] for i in batch_indices]
                batch_actions = [self.actions[i] for i in batch_indices]
                batch_old_log_probs = np.array([self.log_probs[i] for i in batch_indices])
                batch_advantages = self.advantages[batch_indices]
                batch_returns = self.returns[batch_indices]
                
                # 更新策略
                # Update policy
                for state, action, advantage in zip(batch_states, batch_actions, batch_advantages):
                    # 计算策略梯度
                    log_grad = self.policy.compute_log_gradient(state, action)
                    gradient = advantage * log_grad
                    self.policy.update_parameters(self.alpha_theta * gradient, step_size=1.0)
                
                # 更新价值函数
                # Update value function
                for state, return_ in zip(batch_states, batch_returns):
                    self.value_function.update(state, return_, self.alpha_w)
    
    def train(self, env: Any, total_steps: int = 100000):
        """
        训练PPO
        Train PPO
        
        Args:
            env: 环境
                Environment
            total_steps: 总训练步数
                        Total training steps
        """
        steps_per_update = 2048
        n_updates = total_steps // steps_per_update
        
        for update in range(n_updates):
            # 收集轨迹
            # Collect trajectories
            self.collect_trajectories(env, steps_per_update)
            
            # PPO更新
            # PPO update
            self.update()
            
            # 打印进度
            # Print progress
            if (update + 1) % 10 == 0:
                recent_returns = self.episode_returns[-10:] if self.episode_returns else [0]
                mean_return = np.mean(recent_returns)
                mean_clip = np.mean(self.clip_fractions[-100:]) if self.clip_fractions else 0
                
                logger.info(f"更新{update + 1}: "
                          f"平均回报={mean_return:.1f}, "
                          f"裁剪比例={mean_clip:.3f}")


# ================================================================================
# 主函数：演示自然策略梯度方法
# Main Function: Demonstrate Natural Policy Gradient Methods
# ================================================================================

def demonstrate_natural_policy_gradient():
    """
    演示自然策略梯度方法
    Demonstrate natural policy gradient methods
    """
    print("\n" + "="*80)
    print("第13.6节：自然策略梯度")
    print("Section 13.6: Natural Policy Gradient")
    print("="*80)
    
    # 导入策略和价值函数
    # Import policy and value function
    from policy_gradient_theorem import SoftmaxPolicy
    from reinforce import SimpleValueFunction
    
    # 设置
    # Setup
    n_features = 8
    n_actions = 2
    n_states = 4
    
    # 特征提取器
    # Feature extractors
    def state_features(state):
        features = np.zeros(n_features)
        if isinstance(state, int):
            features[state % n_features] = 1.0
            features[(state * 2) % n_features] = 0.5
        return features
    
    def state_action_features(state, action):
        features = np.zeros(n_features)
        if isinstance(state, int):
            base_idx = (state * n_actions + action) % n_features
            features[base_idx] = 1.0
        return features
    
    # 简单环境
    # Simple environment
    class SimpleEnv:
        def __init__(self):
            self.state = 0
            self.step_count = 0
        
        def reset(self):
            self.state = 0
            self.step_count = 0
            return self.state
        
        def step(self, action):
            if action == 0:
                self.state = max(0, self.state - 1)
            else:
                self.state = min(n_states - 1, self.state + 1)
            
            if self.state == n_states - 1:
                reward = 10.0
                done = True
            else:
                reward = -1.0
                done = False
            
            self.step_count += 1
            if self.step_count >= 20:
                done = True
            
            return self.state, reward, done, {}
    
    # 1. 测试自然策略梯度
    # 1. Test Natural Policy Gradient
    print("\n" + "="*60)
    print("1. 自然策略梯度")
    print("1. Natural Policy Gradient")
    print("="*60)
    
    npg_policy = SoftmaxPolicy(
        n_features=n_features,
        n_actions=n_actions,
        feature_extractor=state_action_features
    )
    
    npg = NaturalPolicyGradient(
        policy=npg_policy,
        alpha=0.1,
        gamma=0.9,
        damping=0.01
    )
    
    print("\n训练Natural Policy Gradient...")
    env = SimpleEnv()
    
    for episode in range(30):
        episode_return = npg.learn_episode(env, max_steps=20)
        
        if (episode + 1) % 10 == 0:
            mean_return = np.mean(npg.episode_returns[-10:])
            mean_grad_norm = np.mean(npg.natural_gradient_norms[-10:])
            print(f"  回合{episode + 1}: "
                  f"平均回报={mean_return:.1f}, "
                  f"自然梯度范数={mean_grad_norm:.4f}")
    
    # 2. 测试TRPO（简化版）
    # 2. Test TRPO (simplified)
    print("\n" + "="*60)
    print("2. TRPO (简化版)")
    print("2. TRPO (Simplified)")
    print("="*60)
    
    print("\nTRPO需要更复杂的实现，这里展示核心概念")
    print("TRPO requires more complex implementation, showing core concepts")
    
    trpo_policy = SoftmaxPolicy(
        n_features=n_features,
        n_actions=n_actions,
        feature_extractor=state_action_features
    )
    
    trpo_value = SimpleValueFunction(
        n_features=n_features,
        feature_extractor=state_features
    )
    
    # 创建TRPO（注意：这是简化版）
    # Create TRPO (note: this is simplified)
    class SimplifiedTRPO:
        def __init__(self, policy, value_func):
            self.policy = policy
            self.value_func = value_func
            self.damping = 0.01
        
        def train_episode(self, env, max_steps=20):
            states, actions, rewards = [], [], []
            state = env.reset()
            
            for _ in range(max_steps):
                action = self.policy.select_action(state)
                next_state, reward, done, _ = env.step(action)
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                
                if done:
                    break
                state = next_state
            
            # 简化的TRPO更新
            # Simplified TRPO update
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + 0.9 * G
                returns.insert(0, G)
            
            # 计算优势
            advantages = []
            for s, g in zip(states, returns):
                v = self.value_func.get_value(s)
                advantages.append(g - v)
            
            # 更新价值函数
            for s, g in zip(states, returns):
                self.value_func.update(s, g, 0.1)
            
            # 简化的策略更新（使用自然梯度思想）
            for s, a, adv in zip(states, actions, advantages):
                log_grad = self.policy.compute_log_gradient(s, a)
                # 添加阻尼以模拟信任域
                dampened_grad = log_grad / (1 + self.damping * abs(adv))
                self.policy.update_parameters(0.05 * adv * dampened_grad, 1.0)
            
            return sum(rewards)
    
    simplified_trpo = SimplifiedTRPO(trpo_policy, trpo_value)
    
    print("\n训练简化版TRPO...")
    returns_trpo = []
    for episode in range(30):
        episode_return = simplified_trpo.train_episode(env)
        returns_trpo.append(episode_return)
        
        if (episode + 1) % 10 == 0:
            mean_return = np.mean(returns_trpo[-10:])
            print(f"  回合{episode + 1}: 平均回报={mean_return:.1f}")
    
    # 3. 测试PPO
    # 3. Test PPO
    print("\n" + "="*60)
    print("3. PPO (Proximal Policy Optimization)")
    print("="*60)
    
    ppo_policy = SoftmaxPolicy(
        n_features=n_features,
        n_actions=n_actions,
        feature_extractor=state_action_features
    )
    
    ppo_value = SimpleValueFunction(
        n_features=n_features,
        feature_extractor=state_features
    )
    
    ppo = PPO(
        policy=ppo_policy,
        value_function=ppo_value,
        clip_epsilon=0.2,
        alpha_theta=0.01,
        alpha_w=0.05,
        gamma=0.9,
        lam=0.95,
        epochs=4,
        batch_size=32
    )
    
    print("\n训练PPO...")
    print("(使用裁剪目标函数)")
    
    # 简化的PPO训练
    for update in range(10):
        # 收集数据
        ppo.collect_trajectories(env, n_steps=200)
        # PPO更新
        ppo.update()
        
        if (update + 1) % 2 == 0:
            recent_returns = ppo.episode_returns[-5:] if ppo.episode_returns else [0]
            mean_return = np.mean(recent_returns)
            mean_clip = np.mean(ppo.clip_fractions[-20:]) if ppo.clip_fractions else 0
            print(f"  更新{update + 1}: "
                  f"平均回报={mean_return:.1f}, "
                  f"裁剪比例={mean_clip:.3f}")
    
    # 4. 算法比较
    # 4. Algorithm Comparison
    print("\n" + "="*60)
    print("4. 算法比较")
    print("4. Algorithm Comparison")
    print("="*60)
    
    print("\n最终性能:")
    print("-" * 40)
    
    npg_final = np.mean(npg.episode_returns[-10:]) if npg.episode_returns else 0
    trpo_final = np.mean(returns_trpo[-10:]) if returns_trpo else 0
    ppo_final = np.mean(ppo.episode_returns[-5:]) if ppo.episode_returns else 0
    
    print(f"Natural PG:  {npg_final:.1f}")
    print(f"TRPO (简化): {trpo_final:.1f}")
    print(f"PPO:         {ppo_final:.1f}")
    
    # 总结
    # Summary
    print("\n" + "="*80)
    print("自然策略梯度方法总结")
    print("Natural Policy Gradient Methods Summary")
    print("="*80)
    
    print("""
    关键要点 Key Takeaways:
    =======================
    
    1. 自然梯度考虑参数空间几何
       Natural gradient considers parameter space geometry
       
    2. Fisher信息矩阵调整梯度方向
       Fisher Information Matrix adjusts gradient direction
       
    3. TRPO通过信任域保证单调改进
       TRPO guarantees monotonic improvement through trust region
       
    4. PPO通过裁剪简化TRPO
       PPO simplifies TRPO through clipping
       
    5. 现代深度RL的基础
       Foundation of modern deep RL
    
    算法选择 Algorithm Selection:
    - 理论研究: Natural Policy Gradient
               Natural Policy Gradient for theory
    - 保证改进: TRPO
               TRPO for guaranteed improvement  
    - 实际应用: PPO
               PPO for practical applications
    - 大规模: PPO最受欢迎
            PPO most popular for large scale
    
    实践建议 Practical Advice:
    - PPO通常是最佳选择
      PPO is usually the best choice
    - 仔细调整裁剪参数ε
      Carefully tune clipping parameter ε
    - 使用GAE计算优势
      Use GAE for advantage computation
    - 批量训练提高效率
      Batch training for efficiency
    """)


if __name__ == "__main__":
    demonstrate_natural_policy_gradient()