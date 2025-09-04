"""
================================================================================
第8.5节：期望更新vs样本更新
Section 8.5: Expected vs Sample Updates
================================================================================

两种基本的更新方式！
Two fundamental ways to update!

期望更新 Expected Updates:
- 使用完整的概率分布
  Use full probability distribution
- 公式：Q(s,a) ← Σ_s',r p(s',r|s,a)[r + γ max_a' Q(s',a')]
- 确定性，无采样噪声
  Deterministic, no sampling noise
- 计算量大（遍历所有可能）
  Computationally expensive (iterate all possibilities)

样本更新 Sample Updates:
- 使用采样的转移
  Use sampled transitions
- 公式：Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
- 随机性，有采样噪声
  Stochastic, has sampling noise
- 计算量小（单个样本）
  Computationally cheap (single sample)

权衡 Tradeoff:
准确性 vs 计算成本
Accuracy vs computational cost

分支因子b的影响：
Effect of branching factor b:
- b小：期望更新优势大
  Small b: Expected updates advantageous
- b大：样本更新更实际
  Large b: Sample updates more practical
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import logging
import time
import matplotlib.pyplot as plt

# 导入基础组件
# Import base components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.ch03_finite_mdp.mdp_framework import State, Action, MDPEnvironment
from src.ch03_finite_mdp.policies_and_values import ActionValueFunction
from ch04_monte_carlo.mc_control import EpsilonGreedyPolicy

from .models_and_planning import StochasticModel

# 设置日志
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第8.5.1节：期望更新
# Section 8.5.1: Expected Update
# ================================================================================

class ExpectedUpdate:
    """
    期望更新
    Expected Update
    
    使用环境模型的完整分布进行更新
    Update using full distribution of environment model
    
    动态规划风格的更新 DP-style update:
    Q(s,a) ← Σ_s' p(s'|s,a)[r(s,a,s') + γ max_a' Q(s',a')]
    
    优势 Advantages:
    - 无采样噪声
      No sampling noise
    - 收敛更快（样本数少时）
      Faster convergence (with few samples)
    - 确定性更新
      Deterministic updates
    
    劣势 Disadvantages:
    - 需要完整模型
      Requires complete model
    - 计算成本高（大状态空间）
      High computational cost (large state space)
    - 不适合连续空间
      Not suitable for continuous spaces
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 model: StochasticModel,
                 gamma: float = 0.95,
                 threshold: float = 1e-6):
        """
        初始化期望更新
        Initialize expected update
        
        Args:
            env: 环境
                Environment
            model: 随机模型（需要完整分布）
                  Stochastic model (needs full distribution)
            gamma: 折扣因子
                  Discount factor
            threshold: 收敛阈值
                      Convergence threshold
        """
        self.env = env
        self.model = model
        self.gamma = gamma
        self.threshold = threshold
        
        # Q函数
        # Q function
        self.Q = ActionValueFunction(
            env.state_space,
            env.action_space,
            initial_value=0.0
        )
        
        # 统计
        # Statistics
        self.update_count = 0
        self.total_computation = 0  # 计算的状态-动作对数
        
        logger.info(f"初始化期望更新: γ={gamma}")
    
    def expected_update_step(self, state: State, action: Action):
        """
        执行一步期望更新
        Execute one expected update step
        
        Args:
            state: 状态
                  State
            action: 动作
                   Action
        """
        # 获取所有可能的下一状态
        # Get all possible next states
        next_states = self.model.get_all_next_states(state, action)
        
        if not next_states:
            return
        
        # 计算期望值
        # Compute expected value
        expected_value = 0.0
        
        for next_state in next_states:
            # 获取转移概率
            # Get transition probability
            prob = self.model.get_probability(state, action, next_state)
            
            # 获取最大Q值
            # Get max Q value
            if not next_state.is_terminal:
                max_q = max(
                    self.Q.get_value(next_state, a)
                    for a in self.env.action_space
                )
            else:
                max_q = 0.0
            
            # 累积期望
            # Accumulate expectation
            expected_value += prob * max_q
            
            # 统计计算量
            # Count computation
            self.total_computation += len(self.env.action_space)
        
        # 期望奖励
        # Expected reward
        expected_reward = self.model.get_expected_reward(state, action)
        
        # 更新Q值（直接赋值，不是增量）
        # Update Q value (direct assignment, not incremental)
        new_q = expected_reward + self.gamma * expected_value
        self.Q.set_value(state, action, new_q)
        
        self.update_count += 1
    
    def value_iteration_sweep(self):
        """
        价值迭代扫描
        Value iteration sweep
        
        对所有已知的(s,a)执行期望更新
        Execute expected update for all known (s,a)
        """
        max_delta = 0.0
        
        for state in self.env.state_space:
            if state.is_terminal:
                continue
            
            for action in self.env.action_space:
                if self.model.is_known(state, action):
                    old_q = self.Q.get_value(state, action)
                    self.expected_update_step(state, action)
                    new_q = self.Q.get_value(state, action)
                    
                    delta = abs(new_q - old_q)
                    max_delta = max(max_delta, delta)
        
        return max_delta
    
    def solve(self, max_iterations: int = 100) -> Tuple[ActionValueFunction, int]:
        """
        求解（价值迭代）
        Solve (value iteration)
        
        Args:
            max_iterations: 最大迭代次数
                          Maximum iterations
        
        Returns:
            (Q函数, 迭代次数)
            (Q function, iterations)
        """
        for iteration in range(max_iterations):
            max_delta = self.value_iteration_sweep()
            
            if max_delta < self.threshold:
                logger.info(f"期望更新收敛于{iteration+1}次迭代")
                return self.Q, iteration + 1
        
        logger.warning(f"期望更新未收敛（{max_iterations}次迭代）")
        return self.Q, max_iterations


# ================================================================================
# 第8.5.2节：样本更新
# Section 8.5.2: Sample Update
# ================================================================================

class SampleUpdate:
    """
    样本更新
    Sample Update
    
    使用采样的转移进行更新
    Update using sampled transitions
    
    Q-learning风格的更新：
    Q-learning style update:
    Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    
    优势 Advantages:
    - 计算成本低
      Low computational cost
    - 适合大/连续空间
      Suitable for large/continuous spaces
    - 在线学习
      Online learning
    
    劣势 Disadvantages:
    - 采样噪声
      Sampling noise
    - 需要更多更新
      Needs more updates
    - 收敛较慢
      Slower convergence
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 model: StochasticModel,
                 gamma: float = 0.95,
                 alpha: float = 0.1):
        """
        初始化样本更新
        Initialize sample update
        
        Args:
            env: 环境
                Environment
            model: 模型（用于采样）
                  Model (for sampling)
            gamma: 折扣因子
                  Discount factor
            alpha: 学习率
                  Learning rate
        """
        self.env = env
        self.model = model
        self.gamma = gamma
        self.alpha = alpha
        
        # Q函数
        # Q function
        self.Q = ActionValueFunction(
            env.state_space,
            env.action_space,
            initial_value=0.0
        )
        
        # 统计
        # Statistics
        self.update_count = 0
        self.total_computation = 0
        
        logger.info(f"初始化样本更新: γ={gamma}, α={alpha}")
    
    def sample_update_step(self, state: State, action: Action):
        """
        执行一步样本更新
        Execute one sample update step
        
        Args:
            state: 状态
                  State
            action: 动作
                   Action
        """
        # 从模型采样
        # Sample from model
        next_state, reward = self.model.sample(state, action)
        
        # 计算TD目标
        # Compute TD target
        if not next_state.is_terminal:
            max_q = max(
                self.Q.get_value(next_state, a)
                for a in self.env.action_space
            )
        else:
            max_q = 0.0
        
        td_target = reward + self.gamma * max_q
        
        # 增量更新
        # Incremental update
        old_q = self.Q.get_value(state, action)
        new_q = old_q + self.alpha * (td_target - old_q)
        self.Q.set_value(state, action, new_q)
        
        # 统计
        # Statistics
        self.update_count += 1
        self.total_computation += len(self.env.action_space)
    
    def random_sampling_sweep(self):
        """
        随机采样扫描
        Random sampling sweep
        
        随机选择(s,a)进行样本更新
        Randomly select (s,a) for sample update
        """
        # 获取所有已知的(s,a)
        # Get all known (s,a)
        known_pairs = [
            (s, a)
            for s in self.env.state_space
            for a in self.env.action_space
            if self.model.is_known(s, a) and not s.is_terminal
        ]
        
        if not known_pairs:
            return 0.0
        
        max_delta = 0.0
        
        # 对每个(s,a)执行一次更新
        # Execute one update for each (s,a)
        for state, action in known_pairs:
            old_q = self.Q.get_value(state, action)
            self.sample_update_step(state, action)
            new_q = self.Q.get_value(state, action)
            
            delta = abs(new_q - old_q)
            max_delta = max(max_delta, delta)
        
        return max_delta
    
    def solve(self, n_iterations: int = 1000) -> Tuple[ActionValueFunction, int]:
        """
        求解（通过多次采样更新）
        Solve (through multiple sample updates)
        
        Args:
            n_iterations: 迭代次数
                         Number of iterations
        
        Returns:
            (Q函数, 实际迭代次数)
            (Q function, actual iterations)
        """
        for iteration in range(n_iterations):
            self.random_sampling_sweep()
        
        return self.Q, n_iterations


# ================================================================================
# 第8.5.3节：更新比较器
# Section 8.5.3: Update Comparator
# ================================================================================

@dataclass
class UpdateComparison:
    """
    更新比较结果
    Update Comparison Results
    """
    expected_iterations: int
    expected_computation: int
    expected_time: float
    expected_final_value: float
    
    sample_iterations: int
    sample_computation: int
    sample_time: float
    sample_final_value: float
    
    value_difference: float


class UpdateComparator:
    """
    更新方式比较器
    Update Method Comparator
    
    系统比较期望更新和样本更新
    Systematically compare expected and sample updates
    
    实验维度 Experimental Dimensions:
    1. 分支因子（状态空间大小）
       Branching factor (state space size)
    2. 随机性程度
       Degree of stochasticity
    3. 收敛精度
       Convergence precision
    4. 计算预算
       Computational budget
    """
    
    def __init__(self, env: MDPEnvironment):
        """
        初始化比较器
        Initialize comparator
        
        Args:
            env: 环境
                Environment
        """
        self.env = env
        self.results = []
        
        logger.info("初始化更新比较器")
    
    def build_model_from_experience(self,
                                   n_episodes: int = 100) -> StochasticModel:
        """
        从经验构建模型
        Build model from experience
        
        Args:
            n_episodes: 收集经验的回合数
                       Episodes to collect experience
        
        Returns:
            学习的模型
            Learned model
        """
        model = StochasticModel(self.env.state_space, self.env.action_space)
        
        # 收集经验
        # Collect experience
        for _ in range(n_episodes):
            state = self.env.reset()
            
            while not state.is_terminal:
                # 随机动作
                # Random action
                action = np.random.choice(self.env.action_space)
                next_state, reward, done, _ = self.env.step(action)
                
                # 更新模型
                # Update model
                model.update(state, action, next_state, reward)
                
                state = next_state
                if done:
                    break
        
        return model
    
    def compare_updates(self,
                       model: StochasticModel,
                       expected_iterations: int = 100,
                       sample_iterations: int = 1000,
                       n_runs: int = 5) -> UpdateComparison:
        """
        比较更新方式
        Compare update methods
        
        Args:
            model: 环境模型
                  Environment model
            expected_iterations: 期望更新迭代次数
                               Expected update iterations
            sample_iterations: 样本更新迭代次数
                             Sample update iterations
            n_runs: 运行次数（样本更新）
                   Number of runs (for sample update)
        
        Returns:
            比较结果
            Comparison results
        """
        # 期望更新
        # Expected update
        expected_updater = ExpectedUpdate(self.env, model, gamma=0.95)
        
        start_time = time.time()
        expected_Q, expected_iters = expected_updater.solve(expected_iterations)
        expected_time = time.time() - start_time
        
        expected_comp = expected_updater.total_computation
        
        # 计算期望更新的平均Q值
        # Compute average Q value for expected update
        expected_avg_q = np.mean([
            expected_Q.get_value(s, a)
            for s in self.env.state_space[:10]
            for a in self.env.action_space
            if not s.is_terminal
        ])
        
        # 样本更新（多次运行取平均）
        # Sample update (average over multiple runs)
        sample_times = []
        sample_comps = []
        sample_avg_qs = []
        
        for _ in range(n_runs):
            sample_updater = SampleUpdate(self.env, model, gamma=0.95, alpha=0.1)
            
            start_time = time.time()
            sample_Q, sample_iters = sample_updater.solve(sample_iterations)
            sample_time = time.time() - start_time
            
            sample_times.append(sample_time)
            sample_comps.append(sample_updater.total_computation)
            
            # 计算样本更新的平均Q值
            # Compute average Q value for sample update
            sample_avg_q = np.mean([
                sample_Q.get_value(s, a)
                for s in self.env.state_space[:10]
                for a in self.env.action_space
                if not s.is_terminal
            ])
            sample_avg_qs.append(sample_avg_q)
        
        # 创建比较结果
        # Create comparison result
        result = UpdateComparison(
            expected_iterations=expected_iters,
            expected_computation=expected_comp,
            expected_time=expected_time,
            expected_final_value=expected_avg_q,
            
            sample_iterations=sample_iterations,
            sample_computation=int(np.mean(sample_comps)),
            sample_time=np.mean(sample_times),
            sample_final_value=np.mean(sample_avg_qs),
            
            value_difference=abs(expected_avg_q - np.mean(sample_avg_qs))
        )
        
        return result
    
    def analyze_branching_factor(self,
                                n_values: List[int] = [2, 5, 10],
                                verbose: bool = True):
        """
        分析分支因子的影响
        Analyze effect of branching factor
        
        Args:
            n_values: 要测试的分支因子
                     Branching factors to test
            verbose: 是否输出详细信息
                    Whether to output details
        """
        if verbose:
            print("\n分析分支因子的影响...")
            print("Analyzing effect of branching factor...")
        
        results = {}
        
        for b in n_values:
            if verbose:
                print(f"\n分支因子 b={b}:")
            
            # 这里简化：用动作空间大小模拟分支因子
            # Simplified: use action space size to simulate branching
            # 实际应该创建不同的环境
            # Should actually create different environments
            
            model = self.build_model_from_experience(n_episodes=50)
            
            comparison = self.compare_updates(
                model,
                expected_iterations=50,
                sample_iterations=500,
                n_runs=3
            )
            
            results[b] = comparison
            
            if verbose:
                print(f"  期望更新: {comparison.expected_iterations}次迭代, "
                      f"{comparison.expected_time:.3f}秒")
                print(f"  样本更新: {comparison.sample_iterations}次迭代, "
                      f"{comparison.sample_time:.3f}秒")
                
                if comparison.expected_time < comparison.sample_time:
                    print(f"  → 期望更新更快")
                else:
                    print(f"  → 样本更新更快")
        
        return results


# ================================================================================
# 主函数：演示期望vs样本更新
# Main Function: Demonstrate Expected vs Sample Updates
# ================================================================================

def demonstrate_expected_vs_sample():
    """
    演示期望更新vs样本更新
    Demonstrate expected vs sample updates
    """
    print("\n" + "="*80)
    print("第8.5节：期望更新vs样本更新")
    print("Section 8.5: Expected vs Sample Updates")
    print("="*80)
    
    from src.ch03_finite_mdp.gridworld import GridWorld
    
    # 创建环境
    # Create environment
    env = GridWorld(rows=4, cols=4,
                   start_pos=(0,0),
                   goal_pos=(3,3))
    
    print(f"\n创建4×4 GridWorld")
    print(f"  起点: (0,0)")
    print(f"  终点: (3,3)")
    
    # 1. 构建模型
    # 1. Build model
    print("\n" + "="*60)
    print("1. 从经验构建模型")
    print("1. Build Model from Experience")
    print("="*60)
    
    comparator = UpdateComparator(env)
    model = comparator.build_model_from_experience(n_episodes=100)
    
    # 统计模型信息
    # Count model information
    n_known = sum(
        1
        for s in env.state_space
        for a in env.action_space
        if model.is_known(s, a)
    )
    
    print(f"\n模型统计:")
    print(f"  已知(s,a)对: {n_known}/{len(env.state_space) * len(env.action_space)}")
    
    # 2. 比较期望更新和样本更新
    # 2. Compare expected and sample updates
    print("\n" + "="*60)
    print("2. 比较期望更新vs样本更新")
    print("2. Compare Expected vs Sample Updates")
    print("="*60)
    
    comparison = comparator.compare_updates(
        model,
        expected_iterations=50,
        sample_iterations=500,
        n_runs=5
    )
    
    print("\n比较结果:")
    print(f"{'更新方式':<15} {'迭代次数':<15} {'计算量':<15} {'时间(秒)':<15} {'平均Q值':<15}")
    print("-" * 75)
    
    print(f"{'期望更新':<15} {comparison.expected_iterations:<15} "
          f"{comparison.expected_computation:<15} "
          f"{comparison.expected_time:<15.3f} "
          f"{comparison.expected_final_value:<15.3f}")
    
    print(f"{'样本更新':<15} {comparison.sample_iterations:<15} "
          f"{comparison.sample_computation:<15} "
          f"{comparison.sample_time:<15.3f} "
          f"{comparison.sample_final_value:<15.3f}")
    
    print(f"\nQ值差异: {comparison.value_difference:.4f}")
    
    # 判断哪个更好
    # Determine which is better
    if comparison.expected_time < comparison.sample_time:
        print("→ 在此环境下，期望更新更高效")
    else:
        print("→ 在此环境下，样本更新更高效")
    
    # 3. 详细对比单个状态的更新
    # 3. Detailed comparison of single state update
    print("\n" + "="*60)
    print("3. 单个状态更新对比")
    print("3. Single State Update Comparison")
    print("="*60)
    
    # 选择一个非终止状态
    # Select a non-terminal state
    test_state = env.state_space[5]
    test_action = env.action_space[0]
    
    if not test_state.is_terminal and model.is_known(test_state, test_action):
        print(f"\n测试状态: {test_state.id}, 动作: {test_action.id}")
        
        # 期望更新
        # Expected update
        exp_updater = ExpectedUpdate(env, model, gamma=0.95)
        old_q = exp_updater.Q.get_value(test_state, test_action)
        exp_updater.expected_update_step(test_state, test_action)
        new_q_exp = exp_updater.Q.get_value(test_state, test_action)
        
        print(f"\n期望更新:")
        print(f"  旧Q值: {old_q:.3f}")
        print(f"  新Q值: {new_q_exp:.3f}")
        print(f"  变化: {new_q_exp - old_q:.3f}")
        
        # 样本更新（多次）
        # Sample update (multiple times)
        sample_updater = SampleUpdate(env, model, gamma=0.95, alpha=0.1)
        sample_updater.Q.set_value(test_state, test_action, old_q)  # 重置
        
        print(f"\n样本更新（5次）:")
        for i in range(5):
            old_q_sample = sample_updater.Q.get_value(test_state, test_action)
            sample_updater.sample_update_step(test_state, test_action)
            new_q_sample = sample_updater.Q.get_value(test_state, test_action)
            print(f"  更新{i+1}: {old_q_sample:.3f} → {new_q_sample:.3f} "
                  f"(Δ={new_q_sample - old_q_sample:.3f})")
    
    # 4. 分支因子分析（简化版）
    # 4. Branching factor analysis (simplified)
    print("\n" + "="*60)
    print("4. 分支因子影响分析")
    print("4. Branching Factor Impact Analysis")
    print("="*60)
    
    print("""
    理论分析：
    - 分支因子b = 可能的下一状态数
      Branching factor b = number of possible next states
    
    - 期望更新计算量: O(b)
      Expected update computation: O(b)
    
    - 样本更新计算量: O(1)
      Sample update computation: O(1)
    
    - b小时期望更新有优势
      Expected update advantageous when b is small
    
    - b大时样本更新更实际
      Sample update more practical when b is large
    """)
    
    # 总结
    # Summary
    print("\n" + "="*80)
    print("期望vs样本更新总结")
    print("Expected vs Sample Updates Summary")
    print("="*80)
    
    print("""
    关键要点 Key Takeaways:
    =======================
    
    1. 期望更新：准确但计算密集
       Expected: Accurate but computationally intensive
       
    2. 样本更新：高效但有噪声
       Sample: Efficient but noisy
       
    3. 分支因子是关键
       Branching factor is key
       
    4. 小问题用期望，大问题用样本
       Expected for small problems, sample for large
       
    5. 可以混合使用
       Can be used in combination
    
    实践建议 Practical Tips:
    - 表格型小问题：期望更新
      Tabular small problems: Expected updates
    - 大规模问题：样本更新
      Large-scale problems: Sample updates
    - 考虑计算预算
      Consider computational budget
    """)


if __name__ == "__main__":
    demonstrate_expected_vs_sample()