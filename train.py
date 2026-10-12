"""根据 Config 训练具有层内抑制连接的 Rockpool SNN。"""

import random
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from model import GasSNN, gas_snn
import encoder


def seed_everything(seed: int) -> None:
    """固定模型初始化、批次打乱等训练阶段使用的随机数种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_inputs(features: torch.Tensor, cfg: Config) -> torch.Tensor:
    """将 [B, time_all * sensor_num] 编码为 [B, num_timesteps, sensor_num]。

    先沿时间轴线性插值到单次推理的总时间步数，再执行累积率编码。
    累加器沿整条样本的时间轴连续运行，每个传感器通道独立编码。
    """
    if cfg.num_timesteps <= 0:
        raise ValueError("num_timesteps 必须为正数")
    expected_shape = (cfg.time_all, cfg.sensor_num)
    if features.ndim == 3 and tuple(features.shape[1:]) == expected_shape:
        features = features.flatten(start_dim=1)
    if features.ndim != 2 or features.shape[1] != cfg.time_all * cfg.sensor_num:
        raise ValueError(
            f"每条样本应有 time_all * sensor_num = {cfg.time_all * cfg.sensor_num} 个数值，"
            f"实际批次形状为 {tuple(features.shape)}"
        )
    responses = features.float().reshape(-1, cfg.time_all, cfg.sensor_num)
    rates = F.interpolate(
        responses.transpose(1, 2), size=cfg.num_timesteps,
        mode="linear", align_corners=True,
    ).transpose(1, 2).clamp(0.0, 1.0)
    # 累计积分每越过一个整数阈值就产生一次脉冲；用累计计数的差分实现。
    spike_counts = rates.cumsum(dim=1).floor()
    return torch.diff(spike_counts, dim=1, prepend=torch.zeros_like(spike_counts[:, :1]))


def _batch_loss(scores: torch.Tensor, labels: torch.Tensor, cfg: Config) -> torch.Tensor:
    """每条样本仅用最终类别分数计算一次交叉熵，批次内取平均。"""
    if scores.ndim != 2 or scores.shape != (labels.numel(), cfg.category_num):
        raise ValueError(f"类别分数应为 [B, {cfg.category_num}]，实际为 {tuple(scores.shape)}")
    # scores 是原始 logits，cross_entropy 内部完成 log_softmax。
    return F.cross_entropy(scores, labels)


def _run_epoch(
    model: GasSNN,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(training):
        for features, labels in loader:
            features = features.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long).reshape(-1)
            inputs = encoder.min_rate_encoder(input=features, cfg=cfg, device=device)

            # 不同批次是独立样本，膜电位、突触电流和历史脉冲不能跨批次传递。
            model.reset_state()
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            output, _, _ = model(inputs, record=False)
            scores = model.classification_scores(output)
            loss = _batch_loss(scores, labels, cfg)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

            batch_size = labels.numel()
            total_loss += loss.detach().item() * batch_size
            predictions = scores.detach().argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size

    model.reset_state()
    if total_samples == 0:
        raise ValueError("DataLoader 没有提供样本，请检查数据集划分或 drop_last 设置")
    return {
        "loss": total_loss / total_samples,
        "accuracy": 100.0 * total_correct / total_samples,
    }


def train_one_epoch(model, loader, optimizer, cfg, device):
    """训练一个 epoch，返回按样本数统计的平均损失和准确率（百分比）。"""
    return _run_epoch(model, loader, cfg, torch.device(device), optimizer)


def evaluate(model, loader, cfg, device):
    """评估验证集或测试集，关闭梯度计算并重置批次间的神经元状态。"""
    return _run_epoch(model, loader, cfg, torch.device(device))


def fit(
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    test_loader: DataLoader | None = None,
    device: torch.device | str | None = None,
):
    """创建模型和 Adam，输出每个 epoch 的结果；测试集只在训练结束后评估。"""
    if cfg.num_epochs <= 0:
        raise ValueError("num_epochs 必须为正数")
    seed_everything(cfg.seed)
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = cast(GasSNN, gas_snn(cfg).to(device))
    # 使用 torch 的参数接口，等价于 Rockpool 的 model.parameters().astorch()。
    optimizer = torch.optim.Adam(torch.nn.Module.parameters(model), lr=cfg.learning_rate)
    history = []
    print(
        f"Device: {device} | Epochs: {cfg.num_epochs} | Batch size: {cfg.batch_size} | "
        f"Learning rate: {cfg.learning_rate} | Timesteps/sample: {cfg.num_timesteps}",
        flush=True,
    )
    for epoch in range(1, cfg.num_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, cfg, device)
        val_metrics = evaluate(model, val_loader, cfg, device)
        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        })
        print(
            f"Epoch {epoch:03d}/{cfg.num_epochs:03d} | "
            f"Train Loss: {train_metrics['loss']:.6f} | Train Acc: {train_metrics['accuracy']:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.6f} | Val Acc: {val_metrics['accuracy']:.2f}%",
            flush=True,
        )

    test_metrics = None
    if test_loader is not None:
        test_metrics = evaluate(model, test_loader, cfg, device)
        print(
            f"Test Loss: {test_metrics['loss']:.6f} | Test Acc: {test_metrics['accuracy']:.2f}%",
            flush=True,
        )
    return model, history, test_metrics
