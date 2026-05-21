# 差分隐私训练 (DP-SGD) + 成员推理攻击防御评估

## 项目简介

本项目从零实现 **DP-SGD (Differentially Private Stochastic Gradient Descent)**，并通过**成员推理攻击 (Membership Inference Attack, MIA)** 评估其隐私保护效果，展示完整的"攻击→防御"闭环。

### 核心故事线

1. 训练标准模型（无隐私保护）→ 发起成员推理攻击 → 观察隐私泄露
2. 训练 DP-SGD 模型（不同隐私强度）→ 发起同样攻击 → 观察防御效果
3. 可视化"隐私预算 - 模型精度 - 攻击成功率"三角权衡

---

## 需求分析

### 问题背景

机器学习模型在训练过程中会"记忆"训练数据，攻击者可以通过成员推理攻击判断某个数据样本是否被用于训练，导致隐私泄露。在医疗、金融等敏感领域，这种隐私风险尤为严重。

### 解决思路

DP-SGD 在模型训练过程中引入差分隐私机制：
- **逐样本梯度裁剪**：限制单个样本对模型更新的影响
- **高斯噪声注入**：模糊单个样本的贡献
- **形式化隐私保证**：通过 Renyi 差分隐私理论计算隐私预算 epsilon

### 创新点

不是单独实现 DP-SGD，而是构建**攻击-防御完整链路**，直观展示"为什么需要差分隐私"以及"差分隐私如何起作用"。

---

## 方案设计

### 技术架构

```
数据集 (sklearn digits, 8x8 手写数字, 内置无需下载)
    │
    ├── 标准训练 (Adam, 500 epochs)
    │       │
    │       └── 成员推理攻击 → AUC > 0.5 (隐私泄露)
    │
    └── DP-SGD 训练 (不同 sigma)
            │
            ├── sigma=1.0 → 弱隐私
            ├── sigma=3.0 → 中隐私
            └── sigma=5.0 → 强隐私 → AUC ≈ 0.5 (攻击失败)
```

### DP-SGD 核心算法

```
for each batch:
    1. 逐样本计算梯度 g_i
    2. 裁剪: g_i ← g_i * min(1, C/||g_i||)
    3. 聚合: g = (1/B) * (Σ g_i + N(0, σ²C²I))
    4. 更新: θ ← θ - η * g
```

### 成员推理攻击 (阈值法)

基于 Yeom et al. (2018): 过拟合模型对训练数据的损失更低，攻击者利用损失值区分成员/非成员。用 AUC 衡量攻击成功率。

---

## 参考文献

- Abadi et al., "Deep Learning with Differential Privacy", CCS 2016
- Yeom et al., "Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting", CSF 2018
- Mironov, "Renyi Differential Privacy", CSF 2017
- Shokri et al., "Membership Inference Attacks Against Machine Learning Models", S&P 2017

---

## 环境要求

- Python >= 3.8
- PyTorch >= 1.9
- scikit-learn >= 0.24
- numpy >= 1.19
- matplotlib >= 3.3

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 快速开始

### 运行 Python 脚本

```bash
cd DifferentialPrivacy
python dp_sgd.py
```

### 运行 Jupyter Notebook

```bash
jupyter notebook dp_sgd_demo.ipynb
```

---

## 实验结果

| 训练方式 | 模型准确率 | MIA AUC | epsilon | 说明 |
|---------|-----------|---------|---------|------|
| 标准训练 | ~97% | ~0.53 | ∞ | 隐私泄露 |
| DP-SGD (sigma=1.0) | ~84% | ~0.53 | ~9.6 | 弱隐私 |
| DP-SGD (sigma=3.0) | ~76% | ~0.52 | ~0.78 | 中等隐私 |
| DP-SGD (sigma=5.0) | ~62% | ~0.51 | ~0.25 | 强隐私 |

**关键发现**：随着噪声水平增大，MIA AUC 逐渐接近 0.5（随机猜测），说明 DP-SGD 有效防御了成员推理攻击。但同时模型准确率下降，体现了隐私-效用权衡。

> 注：digits 数据集类别区分度高，MIA 效果相对温和。在更复杂的真实数据集上，攻击效果和防御效果的对比会更加显著。

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `dp_sgd.py` | 核心实现脚本，从零实现 DP-SGD + MIA 评估 |
| `dp_sgd_demo.ipynb` | 交互式教学笔记本，含步骤解释和可视化 |
| `requirements.txt` | Python 依赖列表 |
| `README.md` | 项目文档（本文件） |
