#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DP-SGD: 差分隐私随机梯度下降 + 成员推理攻击防御评估

基于 Abadi et al. (CCS 2016) 从零实现 DP-SGD，
并通过成员推理攻击 (Yeom et al., CSF 2018) 评估其隐私保护效果。

数据集: sklearn 内置 digits (8x8 手写数字, 1797 样本, 无需下载)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# 第 1 部分: 模型定义
# ============================================================

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=64, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ============================================================
# 第 2 部分: DP-SGD 核心机制
# ============================================================

def compute_per_sample_grads(model, loss_fn, data, targets):
    """逐样本计算梯度 (教学写法: 显式循环)"""
    per_sample_grads = []
    for x_i, y_i in zip(data, targets):
        model.zero_grad()
        output = model(x_i.unsqueeze(0))
        loss = loss_fn(output, y_i.unsqueeze(0))
        loss.backward()
        sample_grad = []
        for param in model.parameters():
            sample_grad.append(param.grad.detach().clone().flatten())
        per_sample_grads.append(torch.cat(sample_grad))
    return torch.stack(per_sample_grads)


def clip_gradients(per_sample_grads, max_norm):
    """L2 范数裁剪: g_i * min(1, C / ||g_i||)"""
    norms = torch.norm(per_sample_grads, dim=1)
    clip_factors = torch.clamp(max_norm / (norms + 1e-8), max=1.0)
    return per_sample_grads * clip_factors.unsqueeze(1)


def aggregate_and_noise(clipped_grads, noise_multiplier, max_norm, batch_size):
    """聚合裁剪梯度并添加高斯噪声"""
    aggregated = clipped_grads.sum(dim=0)
    noise = torch.randn_like(aggregated) * (noise_multiplier * max_norm)
    return (aggregated + noise) / batch_size


def apply_gradient(model, flat_grad, lr):
    """将扁平梯度向量写回模型参数"""
    offset = 0
    with torch.no_grad():
        for param in model.parameters():
            numel = param.numel()
            param.data -= lr * flat_grad[offset:offset + numel].reshape(param.shape)
            offset += numel


# ============================================================
# 第 3 部分: 隐私预算计算 (RDP)
# ============================================================

def compute_rdp(q, noise_multiplier, steps, orders):
    """计算 Renyi 差分隐私 (Mironov 2017)"""
    rdp = []
    for alpha in orders:
        rdp_single = alpha / (2.0 * noise_multiplier ** 2)
        log_term = math.log1p(q * q * (math.exp(min(rdp_single, 500)) - 1))
        rdp_composed = steps * min(rdp_single, log_term / max(alpha - 1, 1e-10))
        rdp.append(rdp_composed)
    return rdp


def rdp_to_epsilon(rdp_values, orders, delta):
    """RDP -> (epsilon, delta)-DP 转换"""
    eps_list = []
    for rdp_val, alpha in zip(rdp_values, orders):
        eps = rdp_val - math.log(delta) / (alpha - 1)
        eps_list.append(eps)
    return min(eps_list)


def compute_epsilon(batch_size, dataset_size, noise_multiplier, epochs, delta=1e-5):
    """给定训练参数, 计算最终 epsilon"""
    q = batch_size / dataset_size
    steps = epochs * math.ceil(dataset_size / batch_size)
    orders = [1.5, 2, 2.5, 3, 4, 5, 6, 8, 16, 32, 64]
    rdp = compute_rdp(q, noise_multiplier, steps, orders)
    return rdp_to_epsilon(rdp, orders, delta)


# ============================================================
# 第 4 部分: 训练函数
# ============================================================

def make_model():
    """创建模型"""
    return SimpleMLP()


def run_evaluate(model, X, y):
    """评估模型准确率"""
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        preds = outputs.argmax(dim=1)
        return (preds == y).float().mean().item()


def train_standard(model, X_train, y_train, X_test, y_test,
                    epochs=100, lr=0.001, batch_size=64):
    """标准训练 (无隐私保护, 使用 Adam + mini-batch 促进过拟合)"""
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset_size = len(X_train)

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(dataset_size)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, dataset_size, batch_size):
            end = min(start + batch_size, dataset_size)
            batch_idx = indices[start:end]
            optimizer.zero_grad()
            output = model(X_train[batch_idx])
            loss = loss_fn(output, y_train[batch_idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % (epochs // 5) == 0:
            train_acc = run_evaluate(model, X_train, y_train)
            test_acc = run_evaluate(model, X_test, y_test)
            print(f"  [标准训练] Epoch {epoch+1}/{epochs}  "
                  f"Loss: {epoch_loss/n_batches:.4f}  "
                  f"Train Acc: {train_acc:.4f}  Test Acc: {test_acc:.4f}")

    return run_evaluate(model, X_test, y_test)


def train_dpsgd(model, X_train, y_train, X_test, y_test,
                epochs=30, lr=0.01, max_norm=1.0, noise_multiplier=1.0, batch_size=64):
    """DP-SGD 训练 (带差分隐私保护)"""
    loss_fn = nn.CrossEntropyLoss()
    dataset_size = len(X_train)
    delta = 1.0 / dataset_size

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(dataset_size)

        for start in range(0, dataset_size, batch_size):
            end = min(start + batch_size, dataset_size)
            batch_idx = indices[start:end]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]

            per_grads = compute_per_sample_grads(model, loss_fn, X_batch, y_batch)
            clipped = clip_gradients(per_grads, max_norm)
            noised = aggregate_and_noise(clipped, noise_multiplier, max_norm, len(X_batch))
            apply_gradient(model, noised, lr)

        if (epoch + 1) % 10 == 0:
            acc = run_evaluate(model, X_test, y_test)
            eps = compute_epsilon(batch_size, dataset_size, noise_multiplier, epoch + 1, delta)
            print(f"  [DP-SGD sigma={noise_multiplier}] Epoch {epoch+1}/{epochs}  "
                  f"Acc: {acc:.4f}  epsilon={eps:.2f}")

    final_eps = compute_epsilon(batch_size, dataset_size, noise_multiplier, epochs, delta)
    final_acc = run_evaluate(model, X_test, y_test)
    return final_acc, final_eps


# ============================================================
# 第 5 部分: 成员推理攻击 (阈值攻击法)
# ============================================================

def get_per_sample_loss(model, X, y):
    """获取模型对每个样本的交叉熵损失"""
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        losses = F.cross_entropy(outputs, y, reduction='none')
    return losses.numpy()


def membership_inference_attack(model, X_members, y_members, X_nonmembers, y_nonmembers):
    """
    成员推理攻击 (Yeom et al. 2018):
    利用模型对训练数据的过拟合 - 成员样本通常获得更低的损失值。
    损失越低说明模型对该样本"记忆"越深, 越可能是训练集成员。
    用 AUC 衡量攻击成功率: 0.5=随机猜测(攻击失败), >0.5=隐私泄露。
    """
    loss_members = get_per_sample_loss(model, X_members, y_members)
    loss_nonmembers = get_per_sample_loss(model, X_nonmembers, y_nonmembers)

    labels = np.concatenate([np.ones(len(loss_members)), np.zeros(len(loss_nonmembers))])
    # 损失越低越可能是成员, 所以用负损失作为 score
    scores = np.concatenate([-loss_members, -loss_nonmembers])

    auc = roc_auc_score(labels, scores)
    return auc, loss_members, loss_nonmembers


# ============================================================
# 第 6 部分: 可视化
# ============================================================

def generate_plots(results, loss_m_std, loss_nm_std, dp_losses):
    """生成对比图表"""
    plt.rcParams['font.size'] = 11

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 统一两张直方图的 x 轴范围, 便于对比
    all_losses = np.concatenate([loss_m_std, loss_nm_std])
    best_sigma = max(dp_losses.keys())
    loss_m_dp, loss_nm_dp = dp_losses[best_sigma]
    all_losses_dp = np.concatenate([loss_m_dp, loss_nm_dp])
    x_max = max(np.percentile(all_losses, 99), np.percentile(all_losses_dp, 99))
    bins = np.linspace(0, x_max, 30)

    # 图 1: 损失分布 (标准模型) — 成员损失明显低于非成员
    ax1 = axes[0]
    ax1.hist(loss_m_std, bins=bins, alpha=0.6, label='Members', color='#e74c3c', density=True)
    ax1.hist(loss_nm_std, bins=bins, alpha=0.6, label='Non-members', color='#3498db', density=True)
    ax1.set_title('Standard Training\n(No Privacy)')
    ax1.set_xlabel('Per-sample Loss')
    ax1.set_ylabel('Density')
    ax1.set_xlim(0, x_max)
    ax1.legend()

    # 图 2: 损失分布 (最强 DP 模型) — 成员和非成员分布重叠
    ax2 = axes[1]
    ax2.hist(loss_m_dp, bins=bins, alpha=0.6, label='Members', color='#e74c3c', density=True)
    ax2.hist(loss_nm_dp, bins=bins, alpha=0.6, label='Non-members', color='#3498db', density=True)
    ax2.set_title(f'DP-SGD (sigma={best_sigma})\n(Strong Privacy)')
    ax2.set_xlabel('Per-sample Loss')
    ax2.set_ylabel('Density')
    ax2.set_xlim(0, x_max)
    ax2.legend()

    # 图 3: 隐私-效用-攻击 权衡
    ax3 = axes[2]
    dp_results = [(name, acc, auc, eps) for name, acc, auc, eps, sigma in results if sigma > 0]
    if dp_results:
        epsilons = [r[3] for r in dp_results]
        accs = [r[1] for r in dp_results]
        aucs = [r[2] for r in dp_results]

        ax3_twin = ax3.twinx()
        l1, = ax3.plot(epsilons, accs, 'o-', color='#2ecc71', linewidth=2, markersize=8,
                       label='Model Accuracy')
        l2, = ax3_twin.plot(epsilons, aucs, 's-', color='#e74c3c', linewidth=2, markersize=8,
                            label='MIA AUC')
        ax3_twin.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

        ax3.set_xlabel('Privacy Budget (epsilon)')
        ax3.set_ylabel('Model Accuracy', color='#2ecc71')
        ax3_twin.set_ylabel('MIA AUC', color='#e74c3c')
        ax3.set_title('Privacy-Utility-Attack\nTradeoff')

        lines = [l1, l2]
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='center right')

    plt.tight_layout()
    plt.savefig('dp_sgd_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[图表已保存: dp_sgd_results.png]")


# ============================================================
# 第 7 部分: 主实验
# ============================================================

def run_experiment():
    print("=" * 60)
    print("DP-SGD 差分隐私训练 + 成员推理攻击防御评估")
    print("=" * 60)

    # 加载数据 (sklearn 内置, 无需下载)
    digits = load_digits()
    X = torch.tensor(digits.data, dtype=torch.float32)
    y = torch.tensor(digits.target, dtype=torch.long)
    X = X / 16.0

    # 较小的训练集使模型更容易过拟合, 从而让 MIA 效果更明显
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.7, random_state=42, stratify=y
    )
    print(f"\n数据集: sklearn digits (8x8 手写数字)")
    print(f"训练集: {len(X_train)} 样本 | 测试集(非成员): {len(X_test)} 样本")

    n_mia = min(250, len(X_train), len(X_test))
    X_mia_members = X_train[:n_mia]
    y_mia_members = y_train[:n_mia]
    X_mia_nonmembers = X_test[:n_mia]
    y_mia_nonmembers = y_test[:n_mia]

    results = []

    # --- 实验 1: 标准训练 (无隐私保护) ---
    print("\n" + "-" * 40)
    print("实验 1: 标准训练 (无隐私保护)")
    print("-" * 40)

    torch.manual_seed(42)
    model_std = make_model()
    acc_std = train_standard(model_std, X_train, y_train, X_test, y_test,
                             epochs=500, lr=0.001, batch_size=32)
    auc_std, loss_m_std, loss_nm_std = membership_inference_attack(
        model_std, X_mia_members, y_mia_members, X_mia_nonmembers, y_mia_nonmembers
    )
    print(f"  最终准确率: {acc_std:.4f}")
    print(f"  MIA AUC: {auc_std:.4f} (>0.5 表示隐私泄露)")
    results.append(("标准训练", acc_std, auc_std, float('inf'), 0))

    # --- 实验 2: DP-SGD 训练 (不同噪声水平) ---
    noise_levels = [1.0, 3.0, 5.0]
    dp_losses = {}

    for sigma in noise_levels:
        print(f"\n" + "-" * 40)
        print(f"实验: DP-SGD 训练 (sigma={sigma})")
        print("-" * 40)

        torch.manual_seed(42)
        model_dp = make_model()
        acc_dp, eps_dp = train_dpsgd(
            model_dp, X_train, y_train, X_test, y_test,
            epochs=30, lr=0.05, max_norm=1.0,
            noise_multiplier=sigma, batch_size=64
        )
        auc_dp, loss_m_dp, loss_nm_dp = membership_inference_attack(
            model_dp, X_mia_members, y_mia_members, X_mia_nonmembers, y_mia_nonmembers
        )
        print(f"  最终准确率: {acc_dp:.4f}")
        print(f"  隐私预算 epsilon: {eps_dp:.2f}")
        print(f"  MIA AUC: {auc_dp:.4f}")
        results.append((f"DP-SGD(sigma={sigma})", acc_dp, auc_dp, eps_dp, sigma))
        dp_losses[sigma] = (loss_m_dp, loss_nm_dp)

    # --- 结果汇总 ---
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)
    print(f"{'训练方式':<20} {'准确率':>8} {'MIA AUC':>10} {'epsilon':>10}")
    print("-" * 52)
    for name, acc, auc, eps, _ in results:
        eps_str = "inf" if eps == float('inf') else f"{eps:.2f}"
        print(f"{name:<20} {acc:>8.4f} {auc:>10.4f} {eps_str:>10}")

    generate_plots(results, loss_m_std, loss_nm_std, dp_losses)


if __name__ == '__main__':
    run_experiment()
