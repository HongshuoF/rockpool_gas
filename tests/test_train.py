import contextlib
from dataclasses import replace
import io
import math
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from rockpool.nn.modules import LIFTorch
from rockpool.nn.modules.torch.torch_module import TorchModule
from rockpool.parameters import Constant
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from model import ReadoutLIFTorch, gas_snn
from train import _batch_loss, encode_inputs, evaluate, fit, train_one_epoch


class SimpleReadout(TorchModule):
    """可计算精确损失的读出，用于检查统计方式和批次状态重置。"""

    def __init__(self):
        super().__init__(shape=(2, 2))
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.reset_calls = 0

    def forward(self, data):
        return data * self.scale

    def classification_scores(self, spikes):
        return spikes.sum(dim=1)

    def reset_state(self):
        self.reset_calls += 1
        return self


def small_config(**kwargs):
    cfg = replace(
        Config(), sensor_num=2, time_all=1, num_timesteps=4,
        labellist={"a": 0, "b": 1}, inhibit_num_per_neuron=1,
        num_epochs=2, batch_size=2, **kwargs,
    )
    cfg.hidden_dim = [2, 3, 2]
    return cfg


class TrainingTests(unittest.TestCase):
    def test_total_timesteps_and_accumulating_rates(self):
        cfg = small_config()
        # 半幅输入每两步发放一次，满幅输入每步发放一次。
        encoded = encode_inputs(torch.tensor([[0.5, 1.0]]), cfg)
        self.assertEqual(tuple(encoded.shape), (1, 4, 2))
        torch.testing.assert_close(encoded[0, :, 0], torch.tensor([0.0, 1.0, 0.0, 1.0]))
        torch.testing.assert_close(encoded[0, :, 1], torch.ones(4))

    def test_time_order_is_preserved(self):
        cfg = small_config()
        cfg.time_all = 2
        # 通道 0 从 0 升到 1，通道 1 一直为 0；时间与通道不能互换。
        features = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
        encoded = encode_inputs(features, cfg)
        torch.testing.assert_close(encoded[0, :, 0], torch.tensor([0.0, 0.0, 1.0, 1.0]))
        self.assertTrue(torch.all(encoded[0, :, 1] == 0))
        torch.testing.assert_close(encoded, encode_inputs(features.reshape(1, 2, 2), cfg))
        with self.assertRaises(ValueError):
            encode_inputs(torch.zeros(1, 3), cfg)

    def test_metrics_weight_last_batch_by_sample_count(self):
        cfg = small_config()
        loader = DataLoader(TensorDataset(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
            torch.tensor([0, 1, 1]),
        ), batch_size=2)
        model = SimpleReadout()
        # 固定编码结果，使损失和准确率能独立于传感器预处理精确计算。
        with patch("train.encoder.min_rate_encoder", side_effect=lambda input, cfg, device: encode_inputs(input, cfg)):
            metrics = evaluate(model, loader, cfg, "cpu")
        expected_scores = torch.tensor([[4.0, 0.0], [0.0, 4.0], [4.0, 0.0]])
        expected_loss = F.cross_entropy(expected_scores, torch.tensor([0, 1, 1])).item()
        self.assertAlmostEqual(metrics["loss"], expected_loss, places=6)
        self.assertAlmostEqual(metrics["accuracy"], 200.0 / 3.0)
        self.assertEqual(model.reset_calls, 3)
        self.assertIsNone(model.scale.grad)
        self.assertFalse(model.training)
        optimizer = torch.optim.Adam(torch.nn.Module.parameters(model), lr=cfg.learning_rate)
        original_scale = model.scale.detach().clone()
        with patch("train.encoder.min_rate_encoder", side_effect=lambda input, cfg, device: encode_inputs(input, cfg)):
            train_one_epoch(model, loader, optimizer, cfg, "cpu")
        self.assertFalse(torch.equal(model.scale.detach(), original_scale))
        self.assertTrue(model.training)

    def test_readout_preserves_dynamics_and_post_reset_membrane(self):
        kwargs = dict(
            tau_mem=Constant(0.1), tau_syn=Constant(0.05),
            threshold=Constant(2.0), bias=Constant(0.0), dt=0.01,
        )
        reference = LIFTorch(2, **kwargs)
        readout = ReadoutLIFTorch(2, **kwargs)
        data = torch.tensor([
            [[3.0, 0.1], [1.2, 0.2], [0.0, 0.4]],
            [[0.1, 0.2], [0.0, 0.1], [0.2, 0.0]],
        ], requires_grad=True)
        reference_spikes, _, records = reference(data, record=True)
        readout_spikes, _, _ = readout(data, record=False)
        torch.testing.assert_close(readout_spikes, reference_spikes)
        torch.testing.assert_close(readout.final_vmem, records["vmem"][:, -1])
        self.assertEqual(tuple(readout.final_vmem.shape), (2, 2))
        # 第二条样本没有发放，剩余膜电位仍应保留到输入的梯度路径。
        readout.final_vmem[1].sum().backward()
        self.assertGreater(data.grad[1].abs().sum().item(), 0.0)
        readout.reset_state()
        self.assertIsNone(readout.final_vmem)

    def test_classification_uses_spike_charge_and_residual_for_entire_batch(self):
        cfg = small_config()
        cfg.hidden_dim = [2]
        cfg.threshold = 2.0
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                model = gas_snn(cfg).to(device)
                # 模拟脉冲数并列，最后剩余膜电位使类别 1 获胜。
                spikes = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]], device=device)
                output_lif = model[-2]._mod
                residual = torch.tensor([[0.1, 0.8, 99.0, 99.0]], device=device, requires_grad=True)
                output_lif.final_vmem = residual
                scores = model.classification_scores(spikes)
                torch.testing.assert_close(scores, torch.tensor([[2.1, 2.8]], device=device))
                self.assertEqual(scores.argmax(dim=1).item(), 1)
                _batch_loss(scores, torch.tensor([1], device=device), cfg).backward()
                self.assertGreater(residual.grad[:, :2].abs().sum().item(), 0.0)
                self.assertTrue(torch.all(residual.grad[:, 2:] == 0))
                # 检查完整批次和不足配置大小的尾批次，直接验证实际前向。
                for batch_size in (16, 14, 1):
                    model.reset_state()
                    output, _, _ = model(torch.full((batch_size, 4, 2), 0.05, device=device))
                    scores = model.classification_scores(output)
                    self.assertEqual(tuple(scores.shape), (batch_size, 2))
                    expected = output.sum(dim=1) * cfg.threshold + output_lif.final_vmem[:, :2]
                    torch.testing.assert_close(scores, expected)
                model.reset_state()
                with self.assertRaises(RuntimeError):
                    model.classification_scores(output)

    def test_cross_entropy_uses_final_logits_once(self):
        cfg = small_config()
        scores = torch.tensor([[1.5, -0.2], [-1.0, 0.7]], requires_grad=True)
        labels = torch.tensor([0, 1])
        loss = _batch_loss(scores, labels, cfg)
        expected = (torch.logsumexp(scores, dim=1) - scores[torch.arange(2), labels]).mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_real_snn_fit_and_epoch_output(self):
        cfg = small_config()
        loader = DataLoader(TensorDataset(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            torch.tensor([0, 1, 1]),
        ), batch_size=cfg.batch_size)
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    model, history, test_metrics = fit(loader, loader, cfg, loader, device)
                self.assertEqual(len(history), cfg.num_epochs)
                self.assertIn("Epoch 001/002", buffer.getvalue())
                self.assertIn("Epoch 002/002", buffer.getvalue())
                self.assertIn("Test Loss:", buffer.getvalue())
                self.assertTrue(math.isfinite(test_metrics["loss"]))
                for row in history:
                    self.assertTrue(math.isfinite(row["train_loss"]))
                    self.assertTrue(math.isfinite(row["val_loss"]))
                # 重复评估应一致，验证批次之间和两次评估之间均无状态残留。
                self.assertEqual(evaluate(model, loader, cfg, device), evaluate(model, loader, cfg, device))


if __name__ == "__main__":
    unittest.main()
