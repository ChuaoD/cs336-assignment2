from __future__ import annotations

import copy
import math

import pytest
import torch

from cs336_systems.fused_adamw import FusedAdamW, fused_adamw_update_


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="A CUDA GPU is required for the fused Triton AdamW tests",
)


def _make_tensor(
    shape: tuple[int, ...],
    *,
    seed: int,
    scale: float = 1.0,
) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return torch.randn(
        shape,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    ) * scale


@torch.no_grad()
def _reference_adamw_update_(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    *,
    step: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
) -> None:
    """Independent reference matching the Assignment 1 update order."""
    parameter.sub_(lr * weight_decay * parameter)
    exp_avg.copy_(beta1 * exp_avg + (1.0 - beta1) * gradient)
    exp_avg_sq.copy_(
        beta2 * exp_avg_sq + (1.0 - beta2) * gradient.square()
    )
    step_size = lr * math.sqrt(1.0 - beta2**step) / (1.0 - beta1**step)
    parameter.sub_(step_size * exp_avg / (exp_avg_sq.sqrt() + eps))


def _assert_update_matches_reference(
    shape: tuple[int, ...],
    *,
    step: int = 1,
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
    block_size: int = 256,
) -> None:
    parameter = _make_tensor(shape, seed=1)
    gradient = _make_tensor(shape, seed=2, scale=0.25)
    exp_avg = _make_tensor(shape, seed=3, scale=0.1)
    exp_avg_sq = _make_tensor(shape, seed=4, scale=0.1).abs()

    expected_parameter = parameter.clone()
    expected_exp_avg = exp_avg.clone()
    expected_exp_avg_sq = exp_avg_sq.clone()

    _reference_adamw_update_(
        expected_parameter,
        gradient,
        expected_exp_avg,
        expected_exp_avg_sq,
        step=step,
        lr=lr,
        beta1=betas[0],
        beta2=betas[1],
        eps=eps,
        weight_decay=weight_decay,
    )
    fused_adamw_update_(
        parameter,
        gradient,
        exp_avg,
        exp_avg_sq,
        step,
        lr,
        betas[0],
        betas[1],
        eps,
        weight_decay,
        block_size,
    )

    torch.testing.assert_close(parameter, expected_parameter, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(exp_avg, expected_exp_avg, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(exp_avg_sq, expected_exp_avg_sq, rtol=2e-5, atol=2e-6)
    assert torch.isfinite(parameter).all()
    assert torch.isfinite(exp_avg).all()
    assert torch.isfinite(exp_avg_sq).all()


@requires_cuda
@pytest.mark.parametrize(
    ("shape", "block_size"),
    [
        ((1,), 256),
        ((7,), 256),
        ((255,), 256),
        ((256,), 256),
        ((257,), 256),
        ((4097,), 256),
        ((17, 31), 128),
        ((33, 65), 512),
    ],
)
def test_fused_adamw_update_matches_reference_for_awkward_shapes(
    shape: tuple[int, ...],
    block_size: int,
) -> None:
    _assert_update_matches_reference(shape, block_size=block_size)


@requires_cuda
@pytest.mark.parametrize(
    ("step", "lr", "betas", "eps", "weight_decay"),
    [
        (1, 1e-3, (0.9, 0.999), 1e-8, 1e-2),
        (7, 3e-4, (0.8, 0.95), 1e-6, 0.0),
        (19, 2e-3, (0.0, 0.0), 1e-4, 0.1),
    ],
)
def test_fused_adamw_update_matches_reference_for_hyperparameters(
    step: int,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
) -> None:
    _assert_update_matches_reference(
        (1025,),
        step=step,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )


@requires_cuda
def test_fused_adamw_mask_does_not_modify_guard_elements() -> None:
    numel = 257
    guard = 19
    sentinel = 12_345.0

    storages = [
        torch.full((numel + 2 * guard,), sentinel, device="cuda")
        for _ in range(4)
    ]
    parameter, gradient, exp_avg, exp_avg_sq = (
        storage[guard : guard + numel] for storage in storages
    )
    parameter.copy_(_make_tensor((numel,), seed=10))
    gradient.copy_(_make_tensor((numel,), seed=11))
    exp_avg.copy_(_make_tensor((numel,), seed=12))
    exp_avg_sq.copy_(_make_tensor((numel,), seed=13).abs())
    original_gradient = gradient.clone()

    fused_adamw_update_(
        parameter,
        gradient,
        exp_avg,
        exp_avg_sq,
        1,
        1e-3,
        0.9,
        0.999,
        1e-8,
        1e-2,
        256,
    )

    torch.testing.assert_close(gradient, original_gradient, rtol=0, atol=0)
    for storage in storages:
        torch.testing.assert_close(
            storage[:guard],
            torch.full_like(storage[:guard], sentinel),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            storage[-guard:],
            torch.full_like(storage[-guard:], sentinel),
            rtol=0,
            atol=0,
        )


@requires_cuda
def test_fused_adamw_optimizer_matches_reference_across_steps() -> None:
    parameter = torch.nn.Parameter(_make_tensor((37, 19), seed=20))
    expected_parameter = parameter.detach().clone()
    expected_exp_avg = torch.zeros_like(expected_parameter)
    expected_exp_avg_sq = torch.zeros_like(expected_parameter)
    optimizer = FusedAdamW(
        [parameter],
        lr=3e-4,
        betas=(0.85, 0.97),
        eps=1e-6,
        weight_decay=0.03,
        block_size=256,
    )

    exp_avg_data_ptr = None
    exp_avg_sq_data_ptr = None
    for step in range(1, 11):
        gradient = _make_tensor(parameter.shape, seed=100 + step, scale=0.2)
        parameter.grad = gradient.clone()
        _reference_adamw_update_(
            expected_parameter,
            gradient,
            expected_exp_avg,
            expected_exp_avg_sq,
            step=step,
            lr=3e-4,
            beta1=0.85,
            beta2=0.97,
            eps=1e-6,
            weight_decay=0.03,
        )
        optimizer.step()

        state = optimizer.state[parameter]
        assert state["t"] == step + 1
        torch.testing.assert_close(parameter, expected_parameter, rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(state["m"], expected_exp_avg, rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(state["v"], expected_exp_avg_sq, rtol=3e-5, atol=3e-6)

        if exp_avg_data_ptr is None:
            exp_avg_data_ptr = state["m"].data_ptr()
            exp_avg_sq_data_ptr = state["v"].data_ptr()
        else:
            assert state["m"].data_ptr() == exp_avg_data_ptr
            assert state["v"].data_ptr() == exp_avg_sq_data_ptr


@requires_cuda
def test_fused_adamw_skips_parameter_without_gradient() -> None:
    updated = torch.nn.Parameter(_make_tensor((257,), seed=30))
    skipped = torch.nn.Parameter(_make_tensor((257,), seed=31))
    skipped_before = skipped.detach().clone()
    optimizer = FusedAdamW([updated, skipped], weight_decay=0.2)

    updated.grad = _make_tensor(updated.shape, seed=32)
    skipped.grad = None
    optimizer.step()

    assert updated in optimizer.state
    assert skipped not in optimizer.state
    torch.testing.assert_close(skipped, skipped_before, rtol=0, atol=0)


@requires_cuda
def test_fused_adamw_respects_parameter_groups() -> None:
    parameters = [
        torch.nn.Parameter(_make_tensor((513,), seed=40)),
        torch.nn.Parameter(_make_tensor((513,), seed=41)),
    ]
    expected_parameters = [parameter.detach().clone() for parameter in parameters]
    expected_m = [torch.zeros_like(parameter) for parameter in parameters]
    expected_v = [torch.zeros_like(parameter) for parameter in parameters]
    groups = [
        {"params": [parameters[0]], "lr": 1e-3, "weight_decay": 0.0},
        {
            "params": [parameters[1]],
            "lr": 2e-4,
            "betas": (0.7, 0.9),
            "eps": 1e-5,
            "weight_decay": 0.2,
        },
    ]
    optimizer = FusedAdamW(groups, betas=(0.9, 0.999), eps=1e-8)

    gradients = [
        _make_tensor((513,), seed=42),
        _make_tensor((513,), seed=43),
    ]
    for parameter, gradient in zip(parameters, gradients, strict=True):
        parameter.grad = gradient.clone()

    _reference_adamw_update_(
        expected_parameters[0],
        gradients[0],
        expected_m[0],
        expected_v[0],
        step=1,
        lr=1e-3,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    )
    _reference_adamw_update_(
        expected_parameters[1],
        gradients[1],
        expected_m[1],
        expected_v[1],
        step=1,
        lr=2e-4,
        beta1=0.7,
        beta2=0.9,
        eps=1e-5,
        weight_decay=0.2,
    )
    optimizer.step()

    for index, parameter in enumerate(parameters):
        torch.testing.assert_close(
            parameter,
            expected_parameters[index],
            rtol=2e-5,
            atol=2e-6,
        )


@requires_cuda
def test_fused_adamw_closure_runs_with_gradients_enabled() -> None:
    parameter = torch.nn.Parameter(_make_tensor((257,), seed=50))
    target = _make_tensor((257,), seed=51)
    initial_parameter = parameter.detach().clone()
    expected_parameter = initial_parameter.clone()
    expected_m = torch.zeros_like(expected_parameter)
    expected_v = torch.zeros_like(expected_parameter)
    optimizer = FusedAdamW([parameter], lr=1e-3)

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = (parameter - target).square().sum()
        loss.backward()
        return loss

    expected_loss = (initial_parameter - target).square().sum()
    expected_gradient = 2.0 * (initial_parameter - target)
    _reference_adamw_update_(
        expected_parameter,
        expected_gradient,
        expected_m,
        expected_v,
        step=1,
        lr=1e-3,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=1e-2,
    )

    returned_loss = optimizer.step(closure)

    assert returned_loss is not None
    torch.testing.assert_close(returned_loss, expected_loss, rtol=0, atol=0)
    torch.testing.assert_close(parameter, expected_parameter, rtol=2e-5, atol=2e-6)


@requires_cuda
def test_fused_adamw_state_dict_round_trip() -> None:
    parameter = torch.nn.Parameter(_make_tensor((513,), seed=60))
    optimizer = FusedAdamW([parameter], lr=7e-4, betas=(0.8, 0.95))

    for step in range(3):
        parameter.grad = _make_tensor(parameter.shape, seed=61 + step)
        optimizer.step()

    resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
    resumed_optimizer = FusedAdamW(
        [resumed_parameter],
        lr=7e-4,
        betas=(0.8, 0.95),
    )
    resumed_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))

    next_gradient = _make_tensor(parameter.shape, seed=70)
    parameter.grad = next_gradient.clone()
    resumed_parameter.grad = next_gradient.clone()
    optimizer.step()
    resumed_optimizer.step()

    torch.testing.assert_close(resumed_parameter, parameter, rtol=0, atol=0)
    original_state = optimizer.state[parameter]
    resumed_state = resumed_optimizer.state[resumed_parameter]
    assert resumed_state["t"] == original_state["t"]
    torch.testing.assert_close(resumed_state["m"], original_state["m"], rtol=0, atol=0)
    torch.testing.assert_close(resumed_state["v"], original_state["v"], rtol=0, atol=0)
