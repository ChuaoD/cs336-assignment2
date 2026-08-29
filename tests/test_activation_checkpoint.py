"""Correctness tests for the standalone activation-checkpoint operation."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
import torch.nn as nn

from cs336_systems.activation_checkpoint import (
    activation_checkpoint,
    is_checkpoint_recomputing,
)


class _ToyBlock(nn.Module):
    """Small deterministic block with parameters and a non-differentiable input."""

    def __init__(self, d_model: int, d_hidden: int) -> None:
        super().__init__()
        self.linear_in = nn.Linear(d_model, d_hidden)
        self.linear_out = nn.Linear(d_hidden, d_model, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        position_bias = positions.to(hidden.dtype)[None, :, None] / 17.0
        return self.linear_out(torch.tanh(self.linear_in(hidden))) + position_bias


class _ObservedBlock(_ToyBlock):
    """Record execution count, gradient mode, and recomputation context."""

    def __init__(self, d_model: int, d_hidden: int) -> None:
        super().__init__(d_model, d_hidden)
        self.grad_modes: list[bool] = []
        self.recomputation_modes: list[bool] = []

    @property
    def call_count(self) -> int:
        return len(self.grad_modes)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        self.grad_modes.append(torch.is_grad_enabled())
        self.recomputation_modes.append(is_checkpoint_recomputing())
        return super().forward(hidden, positions)


class _FailDuringRecomputation(_ToyBlock):
    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if is_checkpoint_recomputing():
            raise RuntimeError("intentional recompute failure")
        return super().forward(hidden, positions)


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float64:
        return 1e-10, 1e-12
    return 1e-5, 1e-6


def _assert_parameter_gradients_match(
    actual: nn.Module,
    expected: nn.Module,
    *,
    rtol: float,
    atol: float,
) -> None:
    actual_parameters = dict(actual.named_parameters())
    expected_parameters = dict(expected.named_parameters())
    assert actual_parameters.keys() == expected_parameters.keys()

    for name, expected_parameter in expected_parameters.items():
        actual_gradient = actual_parameters[name].grad
        expected_gradient = expected_parameter.grad
        assert actual_gradient is not None, f"Checkpointed parameter {name!r} has no gradient"
        assert expected_gradient is not None, f"Reference parameter {name!r} has no gradient"
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=rtol,
            atol=atol,
            msg=lambda message: f"Gradient mismatch for parameter {name!r}: {message}",
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("upstream_scale", [1.0, 0.37, -1.25])
def test_checkpoint_forward_and_backward_match_reference(
    dtype: torch.dtype,
    upstream_scale: float,
) -> None:
    """Compare output, input gradient, and parameter gradients."""

    torch.manual_seed(11)
    expected_block = _ToyBlock(d_model=5, d_hidden=9).to(dtype=dtype)
    actual_block = deepcopy(expected_block)

    expected_hidden = torch.randn(2, 4, 5, dtype=dtype, requires_grad=True)
    actual_hidden = expected_hidden.detach().clone().requires_grad_(True)
    positions = torch.tensor([0, 2, 5, 9], dtype=torch.int64)

    expected_output = expected_block(expected_hidden, positions)
    actual_output = activation_checkpoint(actual_block, actual_hidden, positions)

    rtol, atol = _tolerances(dtype)
    torch.testing.assert_close(actual_output, expected_output, rtol=rtol, atol=atol)

    torch.manual_seed(29)
    upstream_gradient = torch.randn_like(expected_output) * upstream_scale
    torch.autograd.backward(expected_output, upstream_gradient)
    torch.autograd.backward(actual_output, upstream_gradient)

    assert expected_hidden.grad is not None
    assert actual_hidden.grad is not None
    torch.testing.assert_close(
        actual_hidden.grad,
        expected_hidden.grad,
        rtol=rtol,
        atol=atol,
    )
    _assert_parameter_gradients_match(
        actual_block,
        expected_block,
        rtol=rtol,
        atol=atol,
    )


def test_checkpoint_returns_gradients_for_multiple_differentiable_inputs() -> None:
    """Backward must return one correctly positioned gradient per input."""

    def run_function(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.sin(left * right) + left.square() * 0.2

    torch.manual_seed(31)
    expected_left = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    expected_right = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    actual_left = expected_left.detach().clone().requires_grad_(True)
    actual_right = expected_right.detach().clone().requires_grad_(True)

    expected_output = run_function(expected_left, expected_right)
    actual_output = activation_checkpoint(run_function, actual_left, actual_right)
    upstream_gradient = torch.randn_like(expected_output)

    torch.autograd.backward(expected_output, upstream_gradient)
    torch.autograd.backward(actual_output, upstream_gradient)

    assert expected_left.grad is not None
    assert expected_right.grad is not None
    assert actual_left.grad is not None
    assert actual_right.grad is not None
    torch.testing.assert_close(
        actual_left.grad,
        expected_left.grad,
        rtol=1e-10,
        atol=1e-12,
    )
    torch.testing.assert_close(
        actual_right.grad,
        expected_right.grad,
        rtol=1e-10,
        atol=1e-12,
    )


def test_checkpoint_runs_forward_once_and_recomputes_once() -> None:
    """The original execution is graphless; recomputation is gradient-enabled."""

    torch.manual_seed(3)
    block = _ObservedBlock(d_model=4, d_hidden=7)
    hidden = torch.randn(2, 3, 4, requires_grad=True)
    positions = torch.arange(3)

    output = activation_checkpoint(block, hidden, positions)
    assert block.call_count == 1
    assert block.grad_modes == [False]
    assert block.recomputation_modes == [False]
    assert not is_checkpoint_recomputing()

    output.square().sum().backward()

    assert block.call_count == 2
    assert block.grad_modes == [False, True]
    assert block.recomputation_modes == [False, True]
    assert not is_checkpoint_recomputing()


def test_checkpoint_bypasses_custom_operation_when_gradients_are_disabled() -> None:
    torch.manual_seed(5)
    block = _ObservedBlock(d_model=4, d_hidden=6)
    hidden = torch.randn(2, 3, 4, requires_grad=True)
    positions = torch.arange(3)

    with torch.no_grad():
        output = activation_checkpoint(block, hidden, positions)

    assert block.call_count == 1
    assert block.grad_modes == [False]
    assert block.recomputation_modes == [False]
    assert not output.requires_grad
    assert not is_checkpoint_recomputing()


def test_checkpoint_bypasses_when_no_input_requires_gradient() -> None:
    """Direct execution preserves parameter gradients in this edge case."""

    torch.manual_seed(7)
    block = _ObservedBlock(d_model=4, d_hidden=6)
    hidden = torch.randn(2, 3, 4)
    positions = torch.arange(3)

    output = activation_checkpoint(block, hidden, positions)
    output.sum().backward()

    assert block.call_count == 1
    assert block.grad_modes == [True]
    assert block.recomputation_modes == [False]
    assert block.linear_in.weight.grad is not None
    assert block.linear_out.weight.grad is not None
    assert not is_checkpoint_recomputing()


def test_checkpoint_restores_recomputation_context_after_error() -> None:
    torch.manual_seed(13)
    block = _FailDuringRecomputation(d_model=4, d_hidden=6)
    hidden = torch.randn(2, 3, 4, requires_grad=True)
    positions = torch.arange(3)

    output = activation_checkpoint(block, hidden, positions)
    with pytest.raises(RuntimeError, match="intentional recompute failure"):
        output.sum().backward()

    assert not is_checkpoint_recomputing()


def test_checkpoint_saves_only_tensor_inputs_during_original_forward() -> None:
    """Internal block activations must not be retained by the custom operation."""

    torch.manual_seed(17)
    block = _ToyBlock(d_model=4, d_hidden=8)
    hidden = torch.randn(2, 3, 4, requires_grad=True)
    positions = torch.arange(3)
    saved_tensors: list[tuple[torch.Size, torch.dtype, bool]] = []

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        saved_tensors.append((tensor.shape, tensor.dtype, tensor.requires_grad))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        activation_checkpoint(block, hidden, positions)

    assert saved_tensors == [
        (hidden.shape, hidden.dtype, True),
        (positions.shape, positions.dtype, False),
    ]


def test_checkpoint_matches_reference_across_optimizer_steps() -> None:
    """Catch stale gradients or context state that appears after the first step."""

    torch.manual_seed(23)
    expected_block = _ToyBlock(d_model=4, d_hidden=7).to(dtype=torch.float64)
    actual_block = deepcopy(expected_block)
    expected_optimizer = torch.optim.SGD(expected_block.parameters(), lr=0.03)
    actual_optimizer = torch.optim.SGD(actual_block.parameters(), lr=0.03)
    positions = torch.tensor([0, 1, 4], dtype=torch.int64)

    for step in range(3):
        expected_optimizer.zero_grad(set_to_none=True)
        actual_optimizer.zero_grad(set_to_none=True)

        torch.manual_seed(100 + step)
        base_hidden = torch.randn(2, 3, 4, dtype=torch.float64)
        expected_hidden = base_hidden.detach().clone().requires_grad_(True)
        actual_hidden = base_hidden.detach().clone().requires_grad_(True)

        expected_output = expected_block(expected_hidden, positions)
        actual_output = activation_checkpoint(actual_block, actual_hidden, positions)
        coefficients = torch.linspace(
            -0.8,
            1.1,
            expected_output.numel(),
            dtype=expected_output.dtype,
        ).reshape_as(expected_output)
        expected_loss = (expected_output * coefficients).sum()
        actual_loss = (actual_output * coefficients).sum()

        expected_loss.backward()
        actual_loss.backward()
        expected_optimizer.step()
        actual_optimizer.step()

        torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-10, atol=1e-12)
        assert expected_hidden.grad is not None
        assert actual_hidden.grad is not None
        torch.testing.assert_close(
            actual_hidden.grad,
            expected_hidden.grad,
            rtol=1e-10,
            atol=1e-12,
        )
        for actual_parameter, expected_parameter in zip(
            actual_block.parameters(),
            expected_block.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(
                actual_parameter,
                expected_parameter,
                rtol=1e-10,
                atol=1e-12,
            )
        assert not is_checkpoint_recomputing()
