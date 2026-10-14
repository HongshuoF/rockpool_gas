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
    """Rows are source neurons and columns are targets; each layer is ordered as [principal, inhibitory neurons].

    output_dim and recrrument_dim count principal and inhibitory neurons in the target layer, respectively.
    input_feed_dim masks interlayer outputs from the previous layer's inhibitory neurons; omit it for sensor inputs.
    """
    if input_feed_dim is None:
        input_feed_dim = input_dim
    if not 0 < input_feed_dim <= input_dim:
        raise ValueError("Expected 0 < input_feed_dim <= input_dim")
    if output_dim <= 0 or recrrument_dim < 0:
        raise ValueError("output_dim must be positive and recrrument_dim must be non-negative")

    mask = torch.zeros((input_dim, output_dim + recrrument_dim))
    mask[:input_feed_dim, :output_dim] = 1.0
    return mask


def random_recrument_mask_generate(
    recrrument_dim, feed_dim, inhibit_dim, p_inhib, inhibit_num_per_neuron=None
):
    """Generate an intralayer recurrent mask with shape (total neurons in this layer, total neurons in this layer)."""
    if feed_dim + inhibit_dim != recrrument_dim:
        raise ValueError("feed_dim + inhibit_dim must be equal to recrrument_dim")
    if not 0 <= p_inhib <= 1:
        raise ValueError("p_inhib must be in [0, 1]")
    k = cfg.inhibit_num_per_neuron if inhibit_num_per_neuron is None else inhibit_num_per_neuron
    if feed_dim <= 0 or k <= 0 or inhibit_dim != feed_dim * k:
        raise ValueError("Expected inhibit_dim == feed_dim * k with feed_dim > 0 and k > 0")

    mask = torch.zeros((recrrument_dim, recrrument_dim))
    # Principal neurons -> inhibitory neurons: random connections.
    mask[:feed_dim, feed_dim:] = (torch.rand((feed_dim, inhibit_dim)) < p_inhib).float()
    # Inhibitory neuron i -> principal neuron i // k: fixed connection topology.
    inhibit_idx = torch.arange(inhibit_dim)
    mask[feed_dim + inhibit_idx, inhibit_idx // k] = 1.0
    return mask


class MaskedWeights(TWrapper):
    """Apply masks on every forward pass using Rockpool parameter transformations.

    Raw trainable parameters are stored in _mod; effective_weight() returns the weights used in computation.
    Masks are registered as buffers, moved by .to(device), and included in state_dict().
    """

    def __init__(self, mod, parameter_name, mask, sign=None):
        super().__init__(mod)
        if tuple(mask.shape) != tuple(getattr(mod, parameter_name).shape):
            raise ValueError("Mask shape must match weight shape")
        self.parameter_name = parameter_name
        # Register as a torch buffer so reset_state() does not reset the mask as neuron state.
        torch.nn.Module.register_buffer(self, "weight_mask", mask.detach().clone())
        if sign is not None:
            if sign.shape != mask.shape:
                raise ValueError("Sign matrix shape must match mask shape")
            torch.nn.Module.register_buffer(self, "weight_sign", sign.detach().clone())
        else:
            self.weight_sign = None

    def effective_weight(self):
        weight = getattr(self._mod, self.parameter_name)
        if self.weight_sign is not None:
            # Recurrent connection strengths are trainable; excitatory/inhibitory signs stay fixed.
            weight = weight.abs() * self.weight_sign
        return weight * self.weight_mask

    def _T(self):
        return {self.parameter_name: self.effective_weight()}


class ReadoutLIFTorch(LIFTorch):
    """Retain the output layer's final membrane potential after threshold subtraction and its computation graph."""

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
                # Match LIFTorch's subtractive threshold reset; retain the entire batch without detaching.
                self.final_vmem = vmem - spikes * threshold
            return spikes

        # Keep Rockpool's neuron updates and capture the residual membrane potential only at the final spiking step.
        self.spike_generation_fn = capture_final_vmem
        try:
            return super().forward(input_data)
        finally:
            self.spike_generation_fn = original_spike_fn

    def reset_state(self):
        self.final_vmem = None
        return super().reset_state()


class FeedNeuronReadout(TorchModule):
    """Read only the final layer's principal neurons so inhibitory neurons are excluded from class outputs."""

    def __init__(self, total_dim, feed_dim):
        super().__init__(shape=(total_dim, feed_dim), spiking_input=True, spiking_output=True)

    def forward(self, data):
        data, _ = self._auto_batch(data)
        return data[..., :self.size_out]


class GasSNN(TorchModule):
    """Concatenate two independent first-layer branches, then use shared second and third (output) layers.

    input_b1 and input_b2 have shape [B, T, sensor_num] (also accepting [T, sensor_num]);
    their batch sizes and time step counts must match. Calls return (spikes, state, records).
    """

    def __init__(self, branch_b1, branch_b2, shared_layers):
        super().__init__(
            shape=(branch_b1.size_in + branch_b2.size_in, shared_layers.size_out),
            spiking_input=True,
            spiking_output=True,
        )
        self.branch_b1 = branch_b1
        self.branch_b2 = branch_b2
        self.shared_layers = shared_layers

    def __call__(self, input_b1, input_b2, record=False):
        # Rockpool's evolve accepts a single input object; pack both branch inputs into a pair.
        return super().__call__((input_b1, input_b2), record=record)

    def forward(self, input_b1, input_b2=None, record=None):
        if record is not None:
            self._record = record
        if input_b2 is None:
            if not isinstance(input_b1, tuple) or len(input_b1) != 2:
                raise ValueError("Both input_b1 and input_b2 are required")
            input_b1, input_b2 = input_b1
        self._record_dict = {}
        inputs = []
        for name, data, branch in (
            ("input_b1", input_b1, self.branch_b1),
            ("input_b2", input_b2, self.branch_b2),
        ):
            if not isinstance(data, torch.Tensor) or data.ndim not in (2, 3):
                raise ValueError(f"{name} must have shape [B, T, {branch.size_in}] or [T, {branch.size_in}]")
            if data.shape[-1] != branch.size_in:
                raise ValueError(f"{name} must have {branch.size_in} sensor channels")
            inputs.append(data.unsqueeze(0) if data.ndim == 2 else data)
        if inputs[0].shape[:2] != inputs[1].shape[:2]:
            raise ValueError("input_b1 and input_b2 must have matching batch sizes and time step counts")

        output_b1, _, records_b1 = self.branch_b1(inputs[0], record=self._record)
        output_b2, _, records_b2 = self.branch_b2(inputs[1], record=self._record)
        # Both branches exclude inhibitory neurons; concatenate principal neuron spikes along the feature axis.
        merged = torch.cat((output_b1, output_b2), dim=-1)
        output, _, records_shared = self.shared_layers(merged, record=self._record)
        if self._record:
            self._record_dict = {
                "branch_b1": records_b1,
                "branch_b2": records_b2,
                "branch_b1_output": output_b1,
                "branch_b2_output": output_b2,
                "merged_output": merged,
                "shared_layers": records_shared,
            }
        return output

    def reset_state(self):
        self._record_dict = {}
        return super().reset_state()

    def classification_scores(self, spikes: torch.Tensor) -> torch.Tensor:
        """score = cumulative spike count * firing threshold + residual membrane potential after the final reset."""
        if spikes.ndim != 3 or spikes.shape[-1] != self.size_out:
            raise ValueError(f"Spike output must have shape [B, T, {self.size_out}]")
        output_layer = cast(MaskedWeights, self.shared_layers[-2])
        readout = cast(ReadoutLIFTorch, output_layer._mod)
        final_vmem = readout.final_vmem
        if final_vmem is None:
            raise RuntimeError("请先执行模型前向计算，再读取类别分数；reset_state 后需重新计算")
        if final_vmem.shape[0] != spikes.shape[0]:
            raise ValueError("Spike output and final membrane potential must have matching batch sizes")
        # Support scalar and per-neuron thresholds; exclude inhibitory neurons from classification.
        threshold = cast(torch.Tensor, readout.threshold).expand(readout.size_out)[:self.size_out]
        return spikes.sum(dim=1) * threshold + final_vmem[:, :self.size_out]


def _make_snn_layer(model_cfg, input_dim, feed_dim, input_feed_dim=None, readout=False, branch_num = 0):
    """Build a Linear + LIF layer with interlayer masks and intralayer excitatory/inhibitory circuits."""
    inhibit_dim = feed_dim * model_cfg.inhibit_num_per_neuron
    total_dim = feed_dim + inhibit_dim
    input_mask = weight_mask_generate(input_dim, feed_dim, inhibit_dim, input_feed_dim)
    recurrent_mask = random_recrument_mask_generate(
        total_dim, feed_dim, inhibit_dim, model_cfg.p_inhib,
        model_cfg.inhibit_num_per_neuron,
    )

    linear = LinearTorch((input_dim, total_dim))
    # Mask the initial weights too; MaskedWeights enforces constraints on every forward pass.
    with torch.no_grad():
        linear.weight.mul_(input_mask)

    # Rockpool uses spikes @ w_rec, with source rows and target columns.
    recurrent_sign = torch.ones((total_dim, total_dim))
    recurrent_sign[feed_dim:, :] = -1.0
    recurrent_weight = torch.rand((total_dim, total_dim))
    recurrent_weight[feed_dim:, :] = abs(model_cfg.w_inhib)
    recurrent_weight = recurrent_weight * recurrent_sign * recurrent_mask
    lif_type = ReadoutLIFTorch if readout else LIFTorch
    if branch_num == 1:
        lif = lif_type(
            total_dim,
            has_rec=True,
            w_rec=recurrent_weight,
            tau_syn=Constant(model_cfg.tau_syn[0]),
            tau_mem=Constant(model_cfg.tau_mem[0]),
            threshold=Constant(model_cfg.threshold[0]),
            bias=Constant(model_cfg.bias[0]),
            dt=model_cfg.dt,
        )
    elif branch_num == 2:
        lif = lif_type(
            total_dim,
            has_rec=True,
            w_rec=recurrent_weight,
            tau_syn=Constant(model_cfg.tau_syn[1]),
            tau_mem=Constant(model_cfg.tau_mem[1]),
            threshold=Constant(model_cfg.threshold[1]),
            bias=Constant(model_cfg.bias[1]),
            dt=model_cfg.dt,
        )
    else: 
        lif = lif_type(
            total_dim,
            has_rec=True,
            w_rec=recurrent_weight,
            tau_syn=Constant(model_cfg.tau_syn[2]),
            tau_mem=Constant(model_cfg.tau_mem[2]),
            threshold=Constant(model_cfg.threshold[2]),
            bias=Constant(model_cfg.bias[2]),
            dt=model_cfg.dt,
        )
    return (
        MaskedWeights(linear, "weight", input_mask),
        MaskedWeights(lif, "w_rec", recurrent_mask, recurrent_sign),
    )


def gas_snn(model_cfg=None) -> GasSNN:
    """hidden_dim = [width of each first-layer branch, shared second-layer width, output-layer width].

    Widths count principal neurons only; each input branch has sensor_num channels.
    """
    model_cfg = cfg if model_cfg is None else model_cfg
    if len(model_cfg.hidden_dim) != 3 or any(n <= 0 for n in model_cfg.hidden_dim):
        raise ValueError("hidden_dim must contain three positive values: first-layer branch, second-layer, and output-layer widths")
    if model_cfg.sensor_num <= 0:
        raise ValueError("sensor_num must be positive")
    if model_cfg.inhibit_num_per_neuron <= 0:
        raise ValueError("inhibit_num_per_neuron must be positive")
    if model_cfg.w_inhib >= 0:
        raise ValueError("w_inhib must be negative")

    branch_dim, hidden_dim, output_dim = model_cfg.hidden_dim
    branch_total_dim = branch_dim * (1 + model_cfg.inhibit_num_per_neuron)
    # Build the two first layers separately so they do not share weights or neuron states.
    branch_b1 = TorchSequential(
        *_make_snn_layer(model_cfg, model_cfg.sensor_num, branch_dim, branch_num=1),
        FeedNeuronReadout(branch_total_dim, branch_dim),
    )
    branch_b2 = TorchSequential(
        *_make_snn_layer(model_cfg, model_cfg.sensor_num, branch_dim, branch_num=2),
        FeedNeuronReadout(branch_total_dim, branch_dim),
    )
    hidden_total_dim = hidden_dim * (1 + model_cfg.inhibit_num_per_neuron)
    output_total_dim = output_dim * (1 + model_cfg.inhibit_num_per_neuron)
    shared_layers = TorchSequential(
        *_make_snn_layer(model_cfg, 2 * branch_dim, hidden_dim, branch_num=0),
        *_make_snn_layer(model_cfg, hidden_total_dim, output_dim, hidden_dim, readout=True, branch_num=0),
        FeedNeuronReadout(output_total_dim, output_dim),
    )
    return GasSNN(branch_b1, branch_b2, shared_layers)


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

    # Use a small network to check forward/backward passes and connection topology after optimization.
    test_cfg = replace(cfg, sensor_num=2, inhibit_num_per_neuron=2, p_inhib=1.0)
    test_cfg.hidden_dim = [3, 4, 2]
    snn = gas_snn(test_cfg)
    optimizer = torch.optim.Adam(snn.parameters().astorch(), lr=1e-2)
    input_b1 = torch.ones((2, 12, 2), requires_grad=True)
    input_b2 = torch.full((2, 12, 2), 0.5, requires_grad=True)
    assert snn.branch_b1[0]._mod.weight is not snn.branch_b2[0]._mod.weight
    masked_layers = [layer for layer in torch.nn.Module.modules(snn) if isinstance(layer, MaskedWeights)]
    for _ in range(3):
        snn.reset_state()
        optimizer.zero_grad()
        input_b1.grad = input_b2.grad = None
        output, states, records = snn(input_b1=input_b1, input_b2=input_b2, record=True)
        assert output.shape == (2, 12, 2)
        assert set(states) == {"branch_b1", "branch_b2", "shared_layers"}
        torch.testing.assert_close(
            records["merged_output"],
            torch.cat((records["branch_b1_output"], records["branch_b2_output"]), dim=-1),
        )
        assert records["merged_output"].shape == (2, 12, 6)
        assert snn.classification_scores(output).shape == (2, 2)
        loss = (output - 1).square().mean()
        loss.backward()
        assert input_b1.grad is not None and torch.isfinite(input_b1.grad).all()
        assert input_b2.grad is not None and torch.isfinite(input_b2.grad).all()
        for layer in masked_layers:
            weight = getattr(layer._mod, layer.parameter_name)
            assert weight.grad is not None
            assert torch.all(weight.grad[layer.weight_mask == 0] == 0)
        optimizer.step()
        for layer in masked_layers:
            weight = layer.effective_weight()
            assert torch.all(weight[layer.weight_mask == 0] == 0)
            if layer.weight_sign is not None:
                assert torch.all(weight[layer.weight_sign > 0] >= 0)
                assert torch.all(weight[layer.weight_sign < 0] <= 0)
    # Layer 2 receives concatenated principal neurons from both branches; layer 3 masks layer 2's inhibitory neurons.
    assert snn.shared_layers[0].weight_mask.shape == (6, 12)
    assert torch.all(snn.shared_layers[0].weight_mask[:, 4:] == 0)
    assert torch.all(snn.shared_layers[2].weight_mask[4:, :] == 0)
    assert torch.all(snn.shared_layers[2].weight_mask[:, 2:] == 0)

    # Masks follow weight dtype conversions; loading state_dict into a rebuilt network restores the random topology.
    # replace() recalculates hidden_dim, so restore the layer widths used in this test.
    restore_cfg = replace(test_cfg, p_inhib=0.0)
    restore_cfg.hidden_dim = test_cfg.hidden_dim.copy()
    restored = gas_snn(restore_cfg)
    restored.load_state_dict(snn.state_dict())
    snn.reset_state()
    restored.reset_state()
    restored_layers = [layer for layer in torch.nn.Module.modules(restored) if isinstance(layer, MaskedWeights)]
    for original_layer, restored_layer in zip(masked_layers, restored_layers):
        torch.testing.assert_close(original_layer.effective_weight(), restored_layer.effective_weight())
    torch.testing.assert_close(snn(input_b1, input_b2)[0], restored(input_b1, input_b2)[0])
    # Changing the second input leaves the first branch's output unchanged; resetting clears both branches.
    snn.reset_state()
    _, _, original_records = snn(input_b1, input_b2, record=True)
    snn.reset_state()
    _, _, changed_records = snn(input_b1, torch.zeros_like(input_b2), record=True)
    torch.testing.assert_close(original_records["branch_b1_output"], changed_records["branch_b1_output"])
    snn.reset_state()
    output, _, records = snn(input_b1[0], input_b2[0])
    assert output.shape == (1, 12, 2) and records == {}
    snn.reset_state()
    try:
        snn.classification_scores(output)
    except RuntimeError:
        pass
    else:
        raise AssertionError("reset_state 必须清空最后膜电位")
    for invalid_b2 in (torch.zeros(1, 12, 2), torch.zeros(2, 11, 2), torch.zeros(2, 12, 3)):
        try:
            snn(input_b1, invalid_b2)
        except ValueError:
            pass
        else:
            raise AssertionError("两个输入的批次、时间或通道维度不合法时应报错")
    restored.to(dtype=torch.float64)
    assert restored.branch_b1[0].weight_mask.dtype == torch.float64
    assert restored.branch_b2[1].weight_sign.dtype == torch.float64
    # The local Rockpool spike generator uses float32; keep this dtype for forward passes.
    restored.to(dtype=torch.float32)
    if torch.cuda.is_available():
        restored.to("cuda")
        assert restored.branch_b1[0].weight_mask.device.type == "cuda"
        assert restored(input_b1.cuda(), input_b2.cuda())[0].shape == (2, 12, 2)
    print("Tests passed for dual-branch concatenation, forward/backward passes, state resets, connection constraints, and save/restore.")


if __name__ == "__main__":
    test_main()
