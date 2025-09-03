# Sutton & Barto 强化学习导论 - 超细粒度代码实现

## 项目原则
- 使用hydra和uv管理这个项目
- 超级细粒度且详细的写代码
- 每一章每一节每一个概念都要做到解释清晰
- 需要以教学的风格解释清楚,我是萌新

## 项目简介

本项目使用超细粒度的代码实现来教授 Sutton & Barto 的《强化学习导论》（第二版）。每个概念都通过详细注释的代码来解释，帮助深入理解强化学习的核心思想。

## 项目结构

```
sutton-rl-introduction/
├── src/
│   ├── configs/                 # Hydra配置文件
│   │   ├── config.yaml          # 主配置
│   │   └── experiment/          # 实验配置
│   │       └── preface.yaml     # 前言实验配置
│   ├── preface/                 # 前言：强化学习基础
│   │   ├── core_concepts.py     # 核心概念实现
│   │   ├── tictactoe.py         # 井字棋TD学习
│   │   └── run_preface.py       # 运行前言演示
│   ├── ch01_bandits/            # 第1章：多臂赌博机（待实现）
│   ├── ch02_mdp/                # 第2章：马尔可夫决策过程（待实现）
│   ├── ch03_dp/                 # 第3章：动态规划（待实现）
│   ├── ch04_mc/                 # 第4章：蒙特卡洛方法（待实现）
│   ├── ch05_td/                 # 第5章：时序差分学习（待实现）
│   └── utils/                   # 工具函数
├── outputs/                     # 输出目录
│   ├── plots/                   # 图表
│   └── models/                  # 保存的模型
└── pyproject.toml               # 项目配置
```

## 快速开始

### 安装依赖
```bash
uv sync
```

### 运行前言演示
```bash
# 完整演示（包括1000回合井字棋训练）
uv run python src/preface/run_preface.py experiment=preface

# 快速演示（100回合）
uv run python src/preface/run_preface.py experiment=preface preface.tictactoe.episodes=100
```

## 已实现内容

### 前言：强化学习基础 ✅
1. **核心概念** (`core_concepts.py`)
   - 强化学习8大基本元素
   - 经验元组 (S, A, R, S')
   - Agent-Environment交互循环
   - 奖励假设及示例
   - 与其他学习范式的比较

2. **井字棋示例** (`tictactoe.py`)
   - 完整游戏环境实现
   - TD(0)学习算法
   - 价值函数近似
   - ε-贪婪探索
   - 自我对弈训练
   - 人机对战模式

## 核心算法实现

### 时序差分学习 (TD Learning)
```python
# TD(0)价值更新
V(S_t) ← V(S_t) + α[R_{t+1} + γV(S_{t+1}) - V(S_t)]

# Q-learning更新  
Q(S_t,A_t) ← Q(S_t,A_t) + α[R_{t+1} + γ max_a Q(S_{t+1},a) - Q(S_t,A_t)]
```

### 探索策略
```python
# ε-贪婪策略实现
if random() < ε:
    action = random_action()  # 探索：随机选择
else:
    action = argmax(Q[state])  # 利用：选择最佳
```

## Hydra参数管理

支持灵活的参数配置：
```bash
# 修改训练参数
uv run python src/preface/run_preface.py \
    experiment=preface \
    preface.tictactoe.episodes=5000 \
    preface.tictactoe.alpha=0.3 \
    preface.tictactoe.epsilon=0.2
```

## 学习建议

1. **阅读代码注释**：每个函数都有详细的中英双语注释
2. **运行交互演示**：通过人机对战理解算法效果
3. **修改参数实验**：调整α、ε、γ观察学习行为变化
4. **跟踪价值函数**：观察V(s)和Q(s,a)的演化过程

## 下一步计划

- [ ] 第1章：多臂赌博机问题
- [ ] 第2章：马尔可夫决策过程
- [ ] 第3章：动态规划
- [ ] 第4章：蒙特卡洛方法
- [ ] 第5章：时序差分学习深入