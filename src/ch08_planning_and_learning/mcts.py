"""
================================================================================
第8.10节：蒙特卡洛树搜索 (MCTS) - 决策时规划
Section 8.10: Monte Carlo Tree Search - Planning at Decision Time
================================================================================

AlphaGo的核心算法！
The core algorithm of AlphaGo!

MCTS的关键思想 Key Ideas of MCTS:
1. 构建搜索树而不是完整树
   Build search tree instead of full tree
2. 使用模拟（rollout）估计叶节点价值
   Use simulation (rollout) to estimate leaf value
3. 通过UCT平衡探索和利用
   Balance exploration and exploitation via UCT
4. 逐步扩展有前景的分支
   Incrementally expand promising branches

四个阶段 Four Phases:
1. 选择 Selection:
   从根到叶，使用UCT选择动作
   From root to leaf, select actions using UCT
   
2. 扩展 Expansion:
   添加一个新节点到树
   Add one new node to tree
   
3. 模拟 Simulation:
   从新节点开始随机模拟到终止
   Random simulation from new node to terminal
   
4. 反向传播 Backpropagation:
   更新路径上所有节点的统计
   Update statistics of all nodes on path

UCT (Upper Confidence bounds for Trees):
UCT(s,a) = Q(s,a) + c√(ln N(s) / N(s,a))

其中 where:
- Q(s,a): 平均回报
         Average return
- N(s): 状态访问次数
       State visit count
- N(s,a): 动作选择次数
         Action selection count
- c: 探索常数
    Exploration constant
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import time

# 导入基础组件
# Import base components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.ch03_finite_mdp.mdp_framework import State, Action, MDPEnvironment

# 设置日志
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第8.10.1节：MCTS节点
# Section 8.10.1: MCTS Node
# ================================================================================

@dataclass
class MCTSNode:
    """
    MCTS树节点
    MCTS Tree Node
    
    存储状态的统计信息
    Stores statistics for a state
    """
    state: State
    parent: Optional['MCTSNode'] = None
    parent_action: Optional[Action] = None
    
    # 统计信息
    # Statistics
    visit_count: int = 0
    total_value: float = 0.0
    
    # 子节点
    # Children
    children: Dict[Action, 'MCTSNode'] = field(default_factory=dict)
    untried_actions: List[Action] = field(default_factory=list)
    
    # 标记
    # Flags
    is_terminal: bool = False
    is_fully_expanded: bool = False
    
    @property
    def average_value(self) -> float:
        """
        平均价值
        Average value
        """
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count
    
    @property
    def uct_value(self) -> float:
        """
        UCT值（用于父节点选择此节点）
        UCT value (for parent to select this node)
        """
        if self.parent is None or self.visit_count == 0:
            return float('inf')
        
        exploitation = self.average_value
        exploration = math.sqrt(2.0 * math.log(self.parent.visit_count) / self.visit_count)
        
        return exploitation + exploration
    
    def best_child(self, c: float = 1.41421356237) -> Optional['MCTSNode']:
        """
        选择最佳子节点（UCT）
        Select best child (UCT)
        
        Args:
            c: 探索常数
               Exploration constant
        
        Returns:
            最佳子节点
            Best child node
        """
        if not self.children:
            return None
        
        def uct_score(child: MCTSNode) -> float:
            if child.visit_count == 0:
                return float('inf')
            exploitation = child.average_value
            exploration = c * math.sqrt(math.log(self.visit_count) / child.visit_count)
            return exploitation + exploration
        
        return max(self.children.values(), key=uct_score)
    
    def best_action(self) -> Optional[Action]:
        """
        选择最佳动作（最多访问）
        Select best action (most visited)
        
        Returns:
            最佳动作
            Best action
        """
        if not self.children:
            return None
        
        return max(self.children.items(),
                  key=lambda x: x[1].visit_count)[0]
    
    def update(self, value: float):
        """
        更新节点统计
        Update node statistics
        
        Args:
            value: 回报值
                  Return value
        """
        self.visit_count += 1
        self.total_value += value
    
    def add_child(self, action: Action, child_state: State) -> 'MCTSNode':
        """
        添加子节点
        Add child node
        
        Args:
            action: 动作
                   Action
            child_state: 子状态
                        Child state
        
        Returns:
            新子节点
            New child node
        """
        child = MCTSNode(
            state=child_state,
            parent=self,
            parent_action=action
        )
        self.children[action] = child
        
        # 从未尝试动作中移除
        # Remove from untried actions
        if action in self.untried_actions:
            self.untried_actions.remove(action)
        
        # 检查是否完全扩展
        # Check if fully expanded
        if not self.untried_actions:
            self.is_fully_expanded = True
        
        return child


# ================================================================================
# 第8.10.2节：UCT选择策略
# Section 8.10.2: UCT Selection Strategy
# ================================================================================

class UCTSelection:
    """
    UCT选择策略
    UCT Selection Strategy
    
    Upper Confidence bounds applied to Trees
    树的置信上界
    
    平衡探索和利用 Balance exploration and exploitation:
    - 高价值节点（利用）
      High value nodes (exploitation)
    - 少访问节点（探索）
      Less visited nodes (exploration)
    """
    
    def __init__(self, c: float = 1.41421356237):
        """
        初始化UCT选择
        Initialize UCT selection
        
        Args:
            c: 探索常数（√2是理论最优）
               Exploration constant (√2 is theoretically optimal)
        """
        self.c = c
        
        logger.info(f"初始化UCT选择: c={c}")
    
    def select_action(self, node: MCTSNode) -> Action:
        """
        选择动作
        Select action
        
        Args:
            node: 当前节点
                 Current node
        
        Returns:
            选择的动作
            Selected action
        """
        # 如果有未尝试的动作，随机选择一个
        # If there are untried actions, randomly select one
        if node.untried_actions:
            return np.random.choice(node.untried_actions)
        
        # 否则用UCT选择
        # Otherwise use UCT selection
        best_child = node.best_child(self.c)
        if best_child is None:
            return None
        
        return best_child.parent_action
    
    def compute_uct_values(self, node: MCTSNode) -> Dict[Action, float]:
        """
        计算所有动作的UCT值
        Compute UCT values for all actions
        
        Args:
            node: 节点
                 Node
        
        Returns:
            动作->UCT值映射
            Action->UCT value mapping
        """
        uct_values = {}
        
        for action, child in node.children.items():
            if child.visit_count == 0:
                uct_values[action] = float('inf')
            else:
                exploitation = child.average_value
                exploration = self.c * math.sqrt(
                    math.log(node.visit_count) / child.visit_count
                )
                uct_values[action] = exploitation + exploration
        
        return uct_values


# ================================================================================
# 第8.10.3节：蒙特卡洛树搜索算法
# Section 8.10.3: Monte Carlo Tree Search Algorithm
# ================================================================================

class MonteCarloTreeSearch:
    """
    蒙特卡洛树搜索
    Monte Carlo Tree Search
    
    在线规划算法，逐步构建搜索树
    Online planning algorithm, incrementally build search tree
    
    关键特性 Key Features:
    1. 异步规划
       Anytime planning
    2. 聚焦有前景区域
       Focus on promising regions
    3. 适合大动作空间
       Suitable for large action spaces
    4. 不需要完整模型
       Doesn't need complete model
    """
    
    def __init__(self,
                 env: MDPEnvironment,
                 c: float = 1.41421356237,
                 gamma: float = 0.99):
        """
        初始化MCTS
        Initialize MCTS
        
        Args:
            env: 环境
                Environment
            c: UCT探索常数
               UCT exploration constant
            gamma: 折扣因子
                  Discount factor
        """
        self.env = env
        self.c = c
        self.gamma = gamma
        
        # UCT选择器
        # UCT selector
        self.uct_selector = UCTSelection(c)
        
        # 根节点（每次搜索重置）
        # Root node (reset for each search)
        self.root: Optional[MCTSNode] = None
        
        # 统计
        # Statistics
        self.total_simulations = 0
        self.tree_size = 0
        
        logger.info(f"初始化MCTS: c={c}, γ={gamma}")
    
    def search(self,
              initial_state: State,
              n_simulations: int = 1000,
              max_depth: int = 100) -> Action:
        """
        执行MCTS搜索
        Execute MCTS search
        
        Args:
            initial_state: 初始状态
                         Initial state
            n_simulations: 模拟次数
                          Number of simulations
            max_depth: 最大深度
                      Maximum depth
        
        Returns:
            最佳动作
            Best action
        """
        # 初始化根节点
        # Initialize root node
        self.root = MCTSNode(
            state=initial_state,
            untried_actions=list(self.env.action_space)
        )
        self.root.is_terminal = initial_state.is_terminal
        self.tree_size = 1
        
        # 执行模拟
        # Execute simulations
        for sim in range(n_simulations):
            self._simulate(max_depth)
            self.total_simulations += 1
        
        # 返回最佳动作
        # Return best action
        return self.root.best_action()
    
    def _simulate(self, max_depth: int):
        """
        执行一次MCTS模拟
        Execute one MCTS simulation
        
        四个阶段：选择、扩展、模拟、反向传播
        Four phases: Selection, Expansion, Simulation, Backpropagation
        
        Args:
            max_depth: 最大深度
                      Maximum depth
        """
        # 1. 选择 Selection
        node, path = self._select(self.root, max_depth)
        
        # 2. 扩展 Expansion
        if not node.is_terminal and node.untried_actions:
            node = self._expand(node)
            path.append(node)
        
        # 3. 模拟 Simulation (rollout)
        value = self._rollout(node.state, max_depth - len(path))
        
        # 4. 反向传播 Backpropagation
        self._backpropagate(path, value)
    
    def _select(self, node: MCTSNode, max_depth: int) -> Tuple[MCTSNode, List[MCTSNode]]:
        """
        选择阶段
        Selection phase
        
        使用UCT向下选择直到叶节点
        Use UCT to select down to leaf node
        
        Args:
            node: 起始节点
                 Starting node
            max_depth: 最大深度
                      Maximum depth
        
        Returns:
            (叶节点, 路径)
            (leaf node, path)
        """
        path = [node]
        depth = 0
        
        while depth < max_depth:
            # 如果是终止节点或未完全扩展，停止
            # If terminal or not fully expanded, stop
            if node.is_terminal or not node.is_fully_expanded:
                break
            
            # UCT选择最佳子节点
            # UCT select best child
            best_child = node.best_child(self.c)
            if best_child is None:
                break
            
            node = best_child
            path.append(node)
            depth += 1
        
        return node, path
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        扩展阶段
        Expansion phase
        
        添加一个新子节点
        Add one new child node
        
        Args:
            node: 要扩展的节点
                 Node to expand
        
        Returns:
            新子节点
            New child node
        """
        # 随机选择未尝试的动作
        # Randomly select untried action
        action = np.random.choice(node.untried_actions)
        
        # 执行动作获取下一状态
        # Execute action to get next state
        # 需要从当前节点的状态开始模拟
        # Need to simulate from current node's state
        # 简化：随机生成下一状态（实际应该使用模型）
        # Simplified: randomly generate next state (should use model)
        next_state = np.random.choice(self.env.state_space)
        done = next_state.is_terminal
        
        # 创建新节点
        # Create new node
        child = node.add_child(action, next_state)
        child.is_terminal = done
        child.untried_actions = list(self.env.action_space) if not done else []
        
        self.tree_size += 1
        
        return child
    
    def _rollout(self, state: State, max_steps: int) -> float:
        """
        模拟阶段（rollout）
        Simulation phase (rollout)
        
        随机模拟到终止或最大步数
        Random simulation to terminal or max steps
        
        Args:
            state: 起始状态
                  Starting state
            max_steps: 最大步数
                      Maximum steps
        
        Returns:
            回报值
            Return value
        """
        if state.is_terminal:
            return 0.0
        
        total_return = 0.0
        discount = 1.0
        current_state = state
        
        for step in range(max_steps):
            if current_state.is_terminal:
                break
            
            # 随机动作
            # Random action
            action = np.random.choice(self.env.action_space)
            
            # 模拟执行动作（简化：随机下一状态和固定奖励）
            # Simulate action execution (simplified: random next state and fixed reward)
            next_state = np.random.choice(self.env.state_space)
            # 简化奖励：终止状态+1，其他-1
            # Simplified reward: terminal +1, others -1
            reward = 1.0 if next_state.is_terminal else -1.0
            done = next_state.is_terminal
            
            # 累积回报
            # Accumulate return
            total_return += discount * reward
            discount *= self.gamma
            
            current_state = next_state
            
            if done:
                break
        
        return total_return
    
    def _backpropagate(self, path: List[MCTSNode], value: float):
        """
        反向传播阶段
        Backpropagation phase
        
        更新路径上所有节点的统计
        Update statistics of all nodes on path
        
        Args:
            path: 节点路径
                 Node path
            value: 回报值
                  Return value
        """
        # 从叶到根更新
        # Update from leaf to root
        for node in reversed(path):
            node.update(value)
            # 折扣值传递给父节点
            # Discount value for parent
            value *= self.gamma
    
    def get_tree_statistics(self) -> Dict[str, Any]:
        """
        获取树统计信息
        Get tree statistics
        
        Returns:
            统计信息
            Statistics
        """
        if self.root is None:
            return {}
        
        # 计算树的深度
        # Calculate tree depth
        def get_depth(node: MCTSNode, current_depth: int = 0) -> int:
            if not node.children:
                return current_depth
            return max(get_depth(child, current_depth + 1)
                      for child in node.children.values())
        
        max_depth = get_depth(self.root)
        
        # 计算分支因子
        # Calculate branching factor
        total_children = 0
        total_nodes = 0
        
        def count_nodes(node: MCTSNode):
            nonlocal total_children, total_nodes
            total_nodes += 1
            total_children += len(node.children)
            for child in node.children.values():
                count_nodes(child)
        
        count_nodes(self.root)
        
        avg_branching = total_children / max(1, total_nodes)
        
        return {
            'tree_size': self.tree_size,
            'max_depth': max_depth,
            'avg_branching_factor': avg_branching,
            'root_visit_count': self.root.visit_count,
            'root_value': self.root.average_value,
            'total_simulations': self.total_simulations
        }


# ================================================================================
# 第8.10.4节：MCTS可视化
# Section 8.10.4: MCTS Visualization
# ================================================================================

class MCTSVisualizer:
    """
    MCTS可视化器
    MCTS Visualizer
    
    可视化搜索树结构
    Visualize search tree structure
    """
    
    @staticmethod
    def print_tree(node: MCTSNode, depth: int = 0, max_depth: int = 3):
        """
        打印树结构
        Print tree structure
        
        Args:
            node: 节点
                 Node
            depth: 当前深度
                  Current depth
            max_depth: 最大打印深度
                      Maximum print depth
        """
        if depth > max_depth:
            return
        
        indent = "  " * depth
        
        # 打印节点信息
        # Print node info
        if node.parent_action is not None:
            action_str = f"--{node.parent_action.id}--> "
        else:
            action_str = ""
        
        print(f"{indent}{action_str}State: {node.state.id}, "
              f"Visits: {node.visit_count}, "
              f"Value: {node.average_value:.3f}")
        
        # 打印子节点
        # Print children
        for action, child in sorted(node.children.items(),
                                  key=lambda x: x[1].visit_count,
                                  reverse=True):
            MCTSVisualizer.print_tree(child, depth + 1, max_depth)
    
    @staticmethod
    def print_action_statistics(node: MCTSNode):
        """
        打印动作统计
        Print action statistics
        
        Args:
            node: 节点
                 Node
        """
        print("\n动作统计 Action Statistics:")
        print(f"{'动作':<10} {'访问次数':<15} {'平均价值':<15} {'UCT值':<15}")
        print("-" * 55)
        
        uct_selector = UCTSelection()
        uct_values = uct_selector.compute_uct_values(node)
        
        for action in node.untried_actions:
            print(f"{action.id:<10} {'未尝试':<15} {'-':<15} {'∞':<15}")
        
        for action, child in sorted(node.children.items(),
                                   key=lambda x: x[1].visit_count,
                                   reverse=True):
            uct = uct_values.get(action, 0.0)
            print(f"{action.id:<10} {child.visit_count:<15} "
                  f"{child.average_value:<15.3f} {uct:<15.3f}")


# ================================================================================
# 主函数：演示MCTS
# Main Function: Demonstrate MCTS
# ================================================================================

def demonstrate_mcts():
    """
    演示蒙特卡洛树搜索
    Demonstrate Monte Carlo Tree Search
    """
    print("\n" + "="*80)
    print("第8.10节：蒙特卡洛树搜索 (MCTS)")
    print("Section 8.10: Monte Carlo Tree Search")
    print("="*80)
    
    from src.ch03_finite_mdp.gridworld import GridWorld
    
    # 创建环境
    # Create environment
    env = GridWorld(rows=4, cols=4,
                   start_pos=(0,0),
                   goal_pos=(3,3),
                   obstacles=[(1,1)])
    
    print(f"\n创建4×4 GridWorld")
    print(f"  起点: (0,0)")
    print(f"  终点: (3,3)")
    print(f"  障碍: (1,1)")
    
    # 1. 演示MCTS搜索
    # 1. Demonstrate MCTS search
    print("\n" + "="*60)
    print("1. MCTS搜索演示")
    print("1. MCTS Search Demo")
    print("="*60)
    
    mcts = MonteCarloTreeSearch(env, c=1.41421356237, gamma=0.95)
    
    # 从起始状态搜索
    # Search from start state
    start_state = env.state_space[0]
    
    print(f"\n从状态{start_state.id}开始搜索...")
    print("执行100次模拟...")
    
    best_action = mcts.search(start_state, n_simulations=100, max_depth=20)
    
    print(f"\n最佳动作: {best_action.id if best_action else 'None'}")
    
    # 打印树统计
    # Print tree statistics
    stats = mcts.get_tree_statistics()
    print("\n搜索树统计:")
    print(f"  树大小: {stats['tree_size']}")
    print(f"  最大深度: {stats['max_depth']}")
    print(f"  平均分支因子: {stats['avg_branching_factor']:.2f}")
    print(f"  根节点访问: {stats['root_visit_count']}")
    print(f"  根节点价值: {stats['root_value']:.3f}")
    
    # 2. 显示搜索树结构
    # 2. Show search tree structure
    print("\n" + "="*60)
    print("2. 搜索树结构")
    print("2. Search Tree Structure")
    print("="*60)
    
    print("\n搜索树（深度限制=2）:")
    MCTSVisualizer.print_tree(mcts.root, max_depth=2)
    
    # 显示根节点的动作统计
    # Show root node action statistics
    MCTSVisualizer.print_action_statistics(mcts.root)
    
    # 3. 比较不同模拟次数
    # 3. Compare different simulation counts
    print("\n" + "="*60)
    print("3. 不同模拟次数比较")
    print("3. Different Simulation Counts Comparison")
    print("="*60)
    
    sim_counts = [10, 50, 100, 500]
    
    print(f"\n{'模拟次数':<15} {'最佳动作':<15} {'树大小':<15} {'根节点价值':<15}")
    print("-" * 60)
    
    for n_sim in sim_counts:
        mcts_test = MonteCarloTreeSearch(env, c=1.41421356237, gamma=0.95)
        best_action = mcts_test.search(start_state, n_simulations=n_sim)
        stats = mcts_test.get_tree_statistics()
        
        print(f"{n_sim:<15} {best_action.id if best_action else 'None':<15} "
              f"{stats['tree_size']:<15} {stats['root_value']:<15.3f}")
    
    # 4. UCT探索参数影响
    # 4. UCT exploration parameter impact
    print("\n" + "="*60)
    print("4. UCT探索参数c的影响")
    print("4. UCT Exploration Parameter c Impact")
    print("="*60)
    
    c_values = [0.0, 0.5, 1.41421356237, 2.0, 5.0]
    
    print(f"\n{'c值':<15} {'最佳动作':<15} {'树大小':<15} {'解释':<25}")
    print("-" * 70)
    
    for c_val in c_values:
        mcts_c = MonteCarloTreeSearch(env, c=c_val, gamma=0.95)
        best_action = mcts_c.search(start_state, n_simulations=100)
        stats = mcts_c.get_tree_statistics()
        
        # 解释
        # Explanation
        if c_val == 0.0:
            explanation = "纯利用 (Pure exploitation)"
        elif c_val < 1.0:
            explanation = "偏利用 (Exploit-biased)"
        elif c_val == 1.41421356237:
            explanation = "理论最优 (Theoretical optimal)"
        elif c_val < 3.0:
            explanation = "偏探索 (Explore-biased)"
        else:
            explanation = "纯探索 (Pure exploration)"
        
        print(f"{c_val:<15.2f} {best_action.id if best_action else 'None':<15} "
              f"{stats['tree_size']:<15} {explanation:<25}")
    
    # 5. MCTS用于规划
    # 5. MCTS for planning
    print("\n" + "="*60)
    print("5. MCTS规划演示")
    print("5. MCTS Planning Demo")
    print("="*60)
    
    print("\n使用MCTS规划完整路径...")
    
    current_state = env.reset()
    path = []
    max_steps = 20
    
    for step in range(max_steps):
        if current_state.is_terminal:
            print("  到达目标!")
            break
        
        # MCTS选择动作
        # MCTS select action
        mcts_planner = MonteCarloTreeSearch(env, c=1.41421356237, gamma=0.95)
        action = mcts_planner.search(current_state, n_simulations=50)
        
        if action is None:
            print("  无可用动作!")
            break
        
        # 执行动作
        # Execute action
        next_state, reward, done, _ = env.step(action)
        
        path.append((current_state.id, action.id, reward))
        current_state = next_state
        
        if done:
            print("  到达终止状态!")
            break
    
    # 显示路径
    # Show path
    print("\n规划的路径:")
    for i, (state, action, reward) in enumerate(path[:10]):
        print(f"  步{i+1}: {state} --{action}--> r={reward:.1f}")
    
    # 总结
    # Summary
    print("\n" + "="*80)
    print("MCTS总结")
    print("MCTS Summary")
    print("="*80)
    
    print("""
    关键要点 Key Takeaways:
    =======================
    
    1. MCTS逐步构建搜索树
       MCTS incrementally builds search tree
       
    2. UCT平衡探索和利用
       UCT balances exploration and exploitation
       
    3. 不需要完整模型
       Doesn't need complete model
       
    4. 适合大动作空间
       Suitable for large action spaces
       
    5. AlphaGo的核心算法
       Core algorithm of AlphaGo
    
    四个阶段 Four Phases:
    1. 选择 Selection - UCT选择
    2. 扩展 Expansion - 添加节点
    3. 模拟 Simulation - Rollout
    4. 反向传播 Backpropagation - 更新统计
    
    应用领域 Applications:
    - 棋类游戏 Board games
    - 实时策略 Real-time strategy
    - 机器人规划 Robot planning
    - 组合优化 Combinatorial optimization
    """)


if __name__ == "__main__":
    demonstrate_mcts()