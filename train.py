"""Train a Rockpool SNN with intralayer inhibitory connections using Config."""

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
    """Set random seeds for model initialization, batch shuffling, and other training steps."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_inputs(features: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Encode [B, time_all * sensor_num] as [B, num_timesteps, sensor_num].

    Linearly interpolate along time to the total inference step count, then apply accumulating rate encoding.
    Run the accumulator continuously over each sample's time axis, encoding each sensor channel independently.
    """
    if cfg.num_timesteps <= 0:
        raise ValueError("num_timesteps must be positive")
    expected_shape = (cfg.time_all, cfg.sensor_num)
    if features.ndim == 3 and tuple(features.shape[1:]) == expected_shape:
        features = features.flatten(start_dim=1)
    if features.ndim != 2 or features.shape[1] != cfg.time_all * cfg.sensor_num:
        raise ValueError(
            f"Each sample must contain time_all * sensor_num = {cfg.time_all * cfg.sensor_num} values; "
            f"got batch shape {tuple(features.shape)}"
        )
    responses = features.float().reshape(-1, cfg.time_all, cfg.sensor_num)
    rates = F.interpolate(
        responses.transpose(1, 2), size=cfg.num_timesteps,
        mode="linear", align_corners=True,
    ).transpose(1, 2).clamp(0.0, 1.0)
    # Emit a spike at each integer threshold crossing by differencing cumulative counts.
    spike_counts = rates.cumsum(dim=1).floor()
    return torch.diff(spike_counts, dim=1, prepend=torch.zeros_like(spike_counts[:, :1]))


def _batch_loss(scores: torch.Tensor, labels: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Compute cross entropy once per sample from final class scores and average over the batch."""
    if scores.ndim != 2 or scores.shape != (labels.numel(), cfg.category_num):
        raise ValueError(f"Class scores must have shape [B, {cfg.category_num}]; got {tuple(scores.shape)}")
    # scores contains raw logits; cross_entropy applies log_softmax internally.
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
            inputs_b1 = encoder.min_rate_encoder(input=features, cfg=cfg, device=device) 
            inputs_b2 = encoder.increase_delta_encode(features)
            inputs_b2 = torch.stack([torch.stack(row) for row in zip(*inputs_b2)])# Rate encoding uses [batch, time_steps, sensor_num], while delta encoding uses [time_steps, batch_size, sensor_num]; transpose the delta list with zip.
            # Batches contain independent samples; do not carry membrane potentials, synaptic currents, or spike history across batches.
            model.reset_state()
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            output, _, _ = model(inputs_b1, inputs_b2, record=False)
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
        raise ValueError("DataLoader yielded no samples; check the dataset split or drop_last setting")
    return {
        "loss": total_loss / total_samples,
        "accuracy": 100.0 * total_correct / total_samples,
    }


def train_one_epoch(model, loader, optimizer, cfg, device):
    """Train one epoch and return loss and accuracy (percent), averaged by sample count."""
    return _run_epoch(model, loader, cfg, torch.device(device), optimizer)


def evaluate(model, loader, cfg, device):
    """Evaluate validation or test data without gradients, resetting neuron states between batches."""
    return _run_epoch(model, loader, cfg, torch.device(device))


def fit(
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    test_loader: DataLoader | None = None,
    device: torch.device | str | None = None,
):
    """Create the model and Adam, report each epoch, and evaluate test data only after training."""
    if cfg.num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    seed_everything(cfg.seed)
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = cast(GasSNN, gas_snn(cfg).to(device))
    # Use the torch parameter interface, equivalent to Rockpool's model.parameters().astorch().
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
