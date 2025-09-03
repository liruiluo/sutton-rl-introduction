# 🎓 Sutton & Barto《强化学习导论》超细粒度代码实现

> **通过代码学习强化学习** - 每一行代码都是一堂课

## 📖 项目介绍

这是 Sutton & Barto《强化学习导论》第二版的**超细粒度代码实现**。项目的目标是：

**即使你没有读过原书，仅通过阅读代码和注释，也能完全掌握强化学习的核心概念和算法。**

### ✨ 特色

- 📝 **教科书级注释**：每个概念都有5层解释（理论→比喻→数学→代码→例子）
- 🌏 **中英双语教学**：专业术语对照，便于理解和记忆
- 🔬 **超细粒度实现**：每个算法步骤都有详细解释
- 🎮 **可交互式学习**：包含可运行的演示和游戏
- 📊 **参数化配置**：使用Hydra管理实验参数

## 🚀 快速开始

### 环境要求

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (推荐的Python包管理器)

### 安装

```bash
# 安装uv（如果还没有）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装依赖
uv sync
```

### 运行示例

```bash
# 1. 学习基础概念（前言理论部分）
uv run python src/preface/rl_fundamentals.py

# 2. 运行井字棋TD学习（快速版，10回合）
uv run python src/preface/run_preface.py experiment=preface preface.tictactoe.episodes=10

# 3. 完整训练（1000回合）
uv run python src/preface/run_preface.py experiment=preface preface.tictactoe.episodes=1000
```

## 📚 已实现内容

### ✅ 前言：强化学习基础（100%完成）

| 文件 | 内容 | 代码行数 | 注释密度 |
|------|------|----------|----------|
| `rl_fundamentals.py` | RL基础理论详解 | 1200+ | >70% |
| `tictactoe.py` | 井字棋TD学习实战 | 700+ | >60% |
| `core_concepts.py` | 核心概念实现 | 500+ | >60% |
| `run_preface.py` | 集成演示脚本 | 500+ | >50% |

## 📖 学习建议

1. **从理论开始**：运行 `rl_fundamentals.py` 理解基础概念
2. **实践演示**：运行井字棋训练，观察AI学习过程
3. **人机对战**：与训练好的AI对战，体验学习效果
4. **阅读源码**：仔细阅读代码注释，理解每个细节
5. **参数实验**：修改α、ε、γ等参数，观察影响

---

**记住：强化学习是一个旅程，不是目的地。享受学习的过程！**

*Happy Learning! 学习愉快！* 🚀
