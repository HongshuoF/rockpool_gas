from typing import Callable, cast

import torch

from rockpool.nn.combinators.sequential import TorchSequential
from rockpool.nn.modules import LIFTorch, LinearTorch
from rockpool.nn.modules.torch.torch_module import TorchModule
from rockpool.parameters import Constant
from rockpool.transform.torch_transform import TWrapper

import config


cfg = config.Config()


def weight_mask_generate(input_dim, output_dim, recrrument_dim, input_feed_dim=None):
    """行是源神经元，列是目标神经元；每层排列为 [主神经元, 抑制神经元]。

    output_dim 是目标层主神经元数，recrrument_dim 是目标层抑制神经元数。
    input_feed_dim 用于屏蔽上一层抑制神经元的层间输出；传感器输入无需设置。
    """
    if input_feed_dim is None:
        input_feed_dim = input_dim
    if not 0 < input_feed_dim <= input_dim:
        raise ValueError("需要满足 0 < input_feed_dim <= input_dim")
    if output_dim <= 0 or recrrument_dim < 0:
        raise ValueError("output_dim 必须为正数，recrrument_dim 不能为负数")

    mask = torch.zeros((input_dim, output_dim + recrrument_dim))
    mask[:input_feed_dim, :output_dim] = 1.0
    return mask


def random_recrument_mask_generate(
    recrrument_dim, feed_dim, inhibit_dim, p_inhib, inhibit_num_per_neuron=None
):
    """生成层内循环连接掩码，形状为 (本层总神经元数, 本层总神经元数)。"""
    if feed_dim + inhibit_dim != recrrument_dim:
        raise ValueError("feed_dim + inhibit_dim must be equal to recrrument_dim")
    if not 0 <= p_inhib <= 1:
        raise ValueError("p_inhib 必须在 [0, 1] 之间")
    k = cfg.inhibit_num_per_neuron if inhibit_num_per_neuron is None else inhibit_num_per_neuron
    if feed_dim <= 0 or k <= 0 or inhibit_dim != feed_dim * k:
        raise ValueError("需要满足 inhibit_dim == feed_dim * k，且 feed_dim、k > 0")

    mask = torch.zeros((recrrument_dim, recrrument_dim))
    # 主神经元 -> 抑制神经元：随机连接。
    mask[:feed_dim, feed_dim:] = (torch.rand((feed_dim, inhibit_dim)) < p_inhib).float()
    # 第 i 个抑制神经元 -> 第 i // k 个主神经元：固定连接拓扑。
    inhibit_idx = torch.arange(inhibit_dim)
    mask[feed_dim + inhibit_idx, inhibit_idx // k] = 1.0
    return mask


class MaskedWeights(TWrapper):
    """使用 Rockpool 的参数变换，在每次前向计算中应用掩码。

    原始可训练参数保存在 _mod 中；effective_weight() 返回实际参与计算的权重。
    掩码注册为 buffer，会随 .to(device) 移动并写入 state_dict()。
    """

    def __init__(self, mod, parameter_name, mask, sign=None):
        super().__init__(mod)
        if tuple(mask.shape) != tuple(getattr(mod, parameter_name).shape):
            raise ValueError("掩码形状必须与权重形状一致")
        self.parameter_name = parameter_name
        # 使用 torch 的 buffer 注册：掩码不是会被 reset_state() 重置的神经元状态。
        torch.nn.Module.register_buffer(self, "weight_mask", mask.detach().clone())
        if sign is not None:
            if sign.shape != mask.shape:
                raise ValueError("符号矩阵形状必须与掩码形状一致")
            torch.nn.Module.register_buffer(self, "weight_sign", sign.detach().clone())
        else:
            self.weight_sign = None

    def effective_weight(self):
        weight = getattr(self._mod, self.parameter_name)
        if self.weight_sign is not None:
            # 循环连接的强度可训练，但兴奋/抑制的符号不随训练改变。
            weight = weight.abs() * self.weight_sign
        return weight * self.weight_mask

    def _T(self):
        return {self.parameter_name: self.effective_weight()}


class ReadoutLIFTorch(LIFTorch):
    """保留输出层最后一次减去发放阈值后的膜电位及其计算图。"""

    final_vmem: torch.Tensor | None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_vmem = None

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        input_data, _ = self._auto_batch(input_data)
        steps_left = input_data.shape[1]
        self.final_vmem = None
        original_spike_fn = cast(Callable[..., torch.Tensor], self.spike_generation_fn)

        def capture_final_vmem(vmem, threshold, *args):
            nonlocal steps_left
            spikes = original_spike_fn(vmem, threshold, *args)
            steps_left -= 1
            if steps_left == 0:
                # 与 LIFTorch 的减阈值重置完全一致；保留整个批次，不能 detach。
                self.final_vmem = vmem - spikes * threshold
            return spikes

        # 沿用 Rockpool 的神经元更新，只在最后一次发放时截取剩余膜电位。
        self.spike_generation_fn = capture_final_vmem
        try:
            return super().forward(input_data)
        finally:
            self.spike_generation_fn = original_spike_fn

    def reset_state(self):
        self.final_vmem = None
        return super().reset_state()


class FeedNeuronReadout(TorchModule):
    """只读取最后一层的主神经元，避免将抑制神经元当成分类输出。"""

    def __init__(self, total_dim, feed_dim):
        super().__init__(shape=(total_dim, feed_dim), spiking_input=True, spiking_output=True)

    def forward(self, data):
        data, _ = self._auto_batch(data)
        return data[..., :self.size_out]


class GasSNN(TorchSequential):
    """保留脉冲序列接口，并提供用于损失和分类的最终类别分数。"""

    def classification_scores(self, spikes: torch.Tensor) -> torch.Tensor:
        """score = 累计脉冲数 * 发放阈值 + 最后重置后的剩余膜电位。"""
        if spikes.ndim != 3 or spikes.shape[-1] != self.size_out:
            raise ValueError(f"脉冲输出应为 [B, T, {self.size_out}]")
        output_layer = cast(MaskedWeights, self[-2])
        readout = cast(ReadoutLIFTorch, output_layer._mod)
        final_vmem = readout.final_vmem
        if final_vmem is None:
            raise RuntimeError("请先执行模型前向计算，再读取类别分数；reset_state 后需重新计算")
        if final_vmem.shape[0] != spikes.shape[0]:
            raise ValueError("脉冲输出与最后膜电位的批次大小不一致")
        # 同时支持标量阈值和逐神经元阈值，抑制神经元不参加分类。
        threshold = cast(torch.Tensor, readout.threshold).expand(readout.size_out)[:self.size_out]
        return spikes.sum(dim=1) * threshold + final_vmem[:, :self.size_out]


def gas_snn(model_cfg=None) -> GasSNN:
    """构建具有层内抑制回路的 SNN；hidden_dim 只填写各层主神经元数。"""
    model_cfg = cfg if model_cfg is None else model_cfg
    if not model_cfg.hidden_dim or any(n <= 0 for n in model_cfg.hidden_dim):
        raise ValueError("hidden_dim 必须包含正数层宽")
    if model_cfg.inhibit_num_per_neuron <= 0:
        raise ValueError("inhibit_num_per_neuron 必须为正数")
    if model_cfg.w_inhib >= 0:
        raise ValueError("w_inhib 必须为负数")

    layers = []
    input_dim = model_cfg.sensor_num
    input_feed_dim = input_dim
    for layer_index, feed_dim in enumerate(model_cfg.hidden_dim):
        inhibit_dim = feed_dim * model_cfg.inhibit_num_per_neuron
        total_dim = feed_dim + inhibit_dim
        input_mask = weight_mask_generate(input_dim, feed_dim, inhibit_dim, input_feed_dim)
        recurrent_mask = random_recrument_mask_generate(
            total_dim, feed_dim, inhibit_dim, model_cfg.p_inhib,
            model_cfg.inhibit_num_per_neuron,
        )

        linear = LinearTorch((input_dim, total_dim))
        # 初始权重也应用掩码；持续约束由 MaskedWeights 的前向变换实现。
        with torch.no_grad():
            linear.weight.mul_(input_mask)
        layers.append(MaskedWeights(linear, "weight", input_mask))

        # Rockpool 使用 spikes @ w_rec，行是源、列是目标。
        recurrent_sign = torch.ones((total_dim, total_dim))
        recurrent_sign[feed_dim:, :] = -1.0
        recurrent_weight = torch.rand((total_dim, total_dim))
        recurrent_weight[feed_dim:, :] = abs(model_cfg.w_inhib)
        recurrent_weight = recurrent_weight * recurrent_sign * recurrent_mask
        lif_type = ReadoutLIFTorch if layer_index == len(model_cfg.hidden_dim) - 1 else LIFTorch
        lif = lif_type(
            total_dim,
            has_rec=True,
            w_rec=recurrent_weight,
            tau_syn=Constant(model_cfg.tau_syn),
            tau_mem=Constant(model_cfg.tau_mem),
            threshold=Constant(model_cfg.threshold),
            bias=Constant(model_cfg.bias),
            dt=model_cfg.dt,
        )
        layers.append(MaskedWeights(lif, "w_rec", recurrent_mask, recurrent_sign))
        input_dim = total_dim
        input_feed_dim = feed_dim

    layers.append(FeedNeuronReadout(input_dim, input_feed_dim))
    return GasSNN(*layers)


def test_main():
    from dataclasses import replace

    torch.manual_seed(42)
    test1 = weight_mask_generate(4, 5, 25)
    assert test1.shape == (4, 30)
    assert torch.all(test1[:, :5] == 1)
    assert torch.all(test1[:, 5:] == 0)

    test2 = random_recrument_mask_generate(30, 5, 25, 0.2, 5)
    assert test2.shape == (30, 30)
    assert torch.all((test2 == 0) | (test2 == 1))
    assert torch.all(test2[5:, :5].sum(dim=1) == 1)

    # 小网络验证前向、反向和优化后连接拓扑。
    test_cfg = replace(cfg, sensor_num=2, inhibit_num_per_neuron=2, p_inhib=1.0)
    test_cfg.hidden_dim = [3, 2]
    snn = gas_snn(test_cfg)
    optimizer = torch.optim.Adam(snn.parameters().astorch(), lr=1e-2)
    data = torch.ones((2, 12, 2))
    for _ in range(3):
        snn.reset_state()
        optimizer.zero_grad()
        output, _, _ = snn(data, record=True)
        assert output.shape == (2, 12, 2)
        loss = (output - 1).square().mean()
        loss.backward()
        for layer in snn:
            if isinstance(layer, MaskedWeights):
                weight = getattr(layer._mod, layer.parameter_name)
                assert weight.grad is not None
                assert torch.all(weight.grad[layer.weight_mask == 0] == 0)
        optimizer.step()
        for layer in snn:
            if isinstance(layer, MaskedWeights):
                weight = layer.effective_weight()
                assert torch.all(weight[layer.weight_mask == 0] == 0)
                if layer.weight_sign is not None:
                    assert torch.all(weight[layer.weight_sign > 0] >= 0)
                    assert torch.all(weight[layer.weight_sign < 0] <= 0)
    # 第二层不能接收上一层抑制神经元，也不能直接输入本层抑制神经元。
    assert torch.all(snn[2].weight_mask[3:, :] == 0)
    assert torch.all(snn[2].weight_mask[:, 2:] == 0)

    # 掩码跟随权重转换 dtype；重新构造网络后加载 state_dict 可恢复随机拓扑。
    # replace() 会重新计算 hidden_dim，因此恢复为被测试的层宽。
    restore_cfg = replace(test_cfg, p_inhib=0.0)
    restore_cfg.hidden_dim = test_cfg.hidden_dim.copy()
    restored = gas_snn(restore_cfg)
    restored.load_state_dict(snn.state_dict())
    snn.reset_state()
    restored.reset_state()
    for i in range(4):
        torch.testing.assert_close(snn[i].effective_weight(), restored[i].effective_weight())
    torch.testing.assert_close(snn(data)[0], restored(data)[0])
    restored.to(dtype=torch.float64)
    assert restored[0].weight_mask.dtype == torch.float64
    assert restored[1].weight_sign.dtype == torch.float64
    # 本机 Rockpool 的脉冲生成使用 float32，前向计算继续使用该 dtype。
    restored.to(dtype=torch.float32)
    if torch.cuda.is_available():
        restored.to("cuda")
        assert restored[0].weight_mask.device.type == "cuda"
        assert restored(data.cuda())[0].shape == (2, 12, 2)
    print("掩码、前向/反向、优化后的连接约束及保存恢复测试通过")


if __name__ == "__main__":
    test_main()
