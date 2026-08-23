from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from cs336_systems.fused_lm_head_loss import (
    FusedLMHeadCrossEntropyFunction,
    fused_lm_head_loss_forward_torch
)


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="A CUDA GPU is required for the fused Triton autograd tests",
)


def _make_inputs(
    Nt: int,
    D: int,
    V: int,
    scale: float = 0.5,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #torch.random.manual_seed(0)
    hidden = torch.randn(Nt, D, dtype=dtype, device=device) * scale
    weight = torch.randn(V, D, dtype=dtype, device=device) * scale
    targets = torch.randint(V, (Nt,), dtype=torch.int64, device=device)

    if Nt > 1:
        targets[0] = 0
        targets[-1] = V - 1

    hidden.requires_grad_(requires_grad)
    weight.requires_grad_(requires_grad)
    return hidden, weight, targets


def _make_model_inputs(
    B: int,
    S: int,
    D: int,
    V: int,
    scale: float = 0.5,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create model-facing inputs with hidden [B, S, D] and targets [B, S]."""
    #torch.random.manual_seed(0)
    hidden = torch.randn(B, S, D, dtype=dtype, device=device) * scale
    weight = torch.randn(V, D, dtype=dtype, device=device) * scale
    targets = torch.randint(V, (B, S), dtype=torch.int64, device=device)

    if B * S > 1:
        targets.reshape(-1)[0] = 0
        targets.reshape(-1)[-1] = V - 1

    hidden.requires_grad_(requires_grad)
    weight.requires_grad_(requires_grad)
    return hidden, weight, targets


def _reference_forward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = F.linear(hidden, weight).float()
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    loss = F.cross_entropy(flat_logits, flat_targets, reduction="sum")
    logsumexp = torch.logsumexp(flat_logits, dim=-1)
    return loss, logsumexp


def _reference_loss_and_gradients(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    upstream_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reference_hidden = hidden.detach().clone().requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()
    loss, _ = _reference_forward(reference_hidden, reference_weight, targets)
    grad_hidden, grad_weight = torch.autograd.grad(
        loss,
        (reference_hidden, reference_weight),
        grad_outputs=loss.new_tensor(upstream_scale),
    )
    return loss.detach(), grad_hidden, grad_weight


def _assert_fused_backward_matches_reference(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    upstream_scale: float,
    loss_rtol: float = 1e-3,
    loss_atol: float = 1e-3,
    grad_rtol: float = 1e-3,
    grad_atol: float = 1e-3,
) -> None:
    """Compare the fused autograd path directly with the PyTorch oracle."""

    expected_loss, expected_grad_hidden, expected_grad_weight = _reference_loss_and_gradients(
        hidden,
        weight,
        targets,
        upstream_scale=upstream_scale,
    )

    actual_hidden = hidden.detach().clone().requires_grad_()
    actual_weight = torch.nn.Parameter(weight.detach().clone())
    loss = FusedLMHeadCrossEntropyFunction.apply(
        actual_hidden,
        actual_weight,
        targets,
    )
    loss.backward(gradient=loss.new_tensor(upstream_scale))

    assert actual_hidden.grad is not None
    assert actual_weight.grad is not None

    assert actual_hidden.grad.shape == hidden.shape
    assert actual_weight.grad.shape == weight.shape
    assert actual_hidden.grad.dtype == hidden.dtype
    assert actual_weight.grad.dtype == weight.dtype

    torch.testing.assert_close(loss, expected_loss, rtol=loss_rtol, atol=loss_atol)
    torch.testing.assert_close(
        actual_hidden.grad,
        expected_grad_hidden,
        rtol=grad_rtol,
        atol=grad_atol,
    )
    torch.testing.assert_close(
        actual_weight.grad,
        expected_grad_weight,
        rtol=grad_rtol,
        atol=grad_atol,
    )
   
    if upstream_scale == 0.0:
        assert torch.count_nonzero(actual_hidden.grad) == 0
        assert torch.count_nonzero(actual_weight.grad) == 0


@pytest.mark.parametrize(
    ("Nt", "D", "V", "Bt", "Bv"),
    [
        (1, 16, 7, 1, 4),
        (7, 16, 13, 3, 5),
        (9, 16, 23, 32, 64),
        (23, 32, 71, 32, 128),
        (23, 32, 71, 23, 71),
        (31, 32, 85, 23, 71),
        (65, 64, 129, 16, 32),
    ],
)
def test_fused_loss_torch_forward(Nt: int, D: int, V: int, Bt: int, Bv: int):
    hidden, weight, targets = _make_inputs(Nt, D, V)
    expected_loss, expected_lse = _reference_forward(hidden, weight, targets)

    actual_loss, actual_lse = fused_lm_head_loss_forward_torch(
        hidden,
        weight,
        targets,
        Bt=Bt,
        Bv=Bv,
    )

    assert actual_loss.shape == torch.Size([])
    assert actual_lse.shape == (Nt,)
    assert actual_loss.dtype == torch.float32
    assert actual_lse.dtype == torch.float32
    assert actual_loss.device == hidden.device
    assert actual_lse.device == hidden.device
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-5, atol=1e-5)


def test_fused_loss_torch_forward_large_logits():
    hidden, weight, targets = _make_inputs(13, 32, 47, scale=30.0)
    expected_loss, expected_lse = _reference_forward(hidden, weight, targets)

    actual_loss, actual_lse = fused_lm_head_loss_forward_torch(
        hidden,
        weight,
        targets,
        Bt=5,
        Bv=11,
    )

    assert torch.isfinite(actual_loss)
    assert torch.isfinite(actual_lse).all()
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the BF16 reference test")
def test_fused_loss_torch_forward_bf16():
    hidden, weight, targets = _make_inputs(33, 64, 131, dtype=torch.bfloat16, device="cuda")
    expected_loss, expected_lse = _reference_forward(hidden, weight, targets)

    actual_loss, actual_lse = fused_lm_head_loss_forward_torch(
        hidden,
        weight,
        targets,
        Bt=8,
        Bv=32,
    )

    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-2, atol=1e-2)


@requires_cuda
def test_fused_loss_triton_autograd():
    implementation = FusedLMHeadCrossEntropyFunction
    assert isinstance(implementation, type)
    assert issubclass(implementation, torch.autograd.Function)


@requires_cuda
@pytest.mark.parametrize(
    ("B", "S", "D", "V", "scale"),
    [
        # One valid token/vocabulary row inside padded output tiles.
        (1, 1, 32, 7, 1.0),
        # Incomplete final token and vocabulary tiles.
        (2, 5, 64, 67, 0.7),
        # Multiple token, vocabulary, and hidden-dimension tiles.
        (3, 11, 128, 131, 0.25),
        (4, 8, 128, 64, -0.5),
        (4, 16, 256, 1000, 9.0),
        (4, 256, 256, 1000, -6.7)
    ],
)
def test_fused_loss_triton_forward(
    B: int,
    S: int,
    D: int,
    V: int,
    scale: float
):
    hidden, weight, targets = _make_model_inputs(
        B,
        S,
        D,
        V,
        scale=scale,
        device="cuda",
        requires_grad=True,
    )
    expected_loss, _ = _reference_forward(hidden, weight, targets)

    actual_loss = FusedLMHeadCrossEntropyFunction.apply(
        hidden,
        weight,
        targets,
    )

    assert actual_loss.shape == torch.Size([])
    assert actual_loss.dtype == torch.float32
    assert actual_loss.device == hidden.device
    assert actual_loss.requires_grad
    assert actual_loss.grad_fn is not None
    assert torch.isfinite(actual_loss)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-3, atol=1e-3)

@requires_cuda
@pytest.mark.parametrize(
    ("B", "S", "D", "V", "scale"),
    [
        # One valid token/vocabulary row inside padded output tiles.
        (1, 1, 32, 7, 0.76),
        # Incomplete final token and vocabulary tiles.
        (2, 5, 64, 67, -0.91),
        # Multiple token, vocabulary, and hidden-dimension tiles.
        (3, 11, 128, 131, 0.5),
        (4, 8, 128, 64, 0.05),
        (4, 16, 256, 1000, 10.2),
        (4, 256, 1024, 1000, -5.8)
    ],
)
def test_fused_loss_triton_forward_bf16(
    B: int,
    S: int,
    D: int,
    V: int,
    scale: float
):
    hidden, weight, targets = _make_model_inputs(
        B,
        S,
        D,
        V,
        scale=scale,
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    expected_loss, _ = _reference_forward(hidden, weight, targets)

    actual_loss = FusedLMHeadCrossEntropyFunction.apply(
        hidden,
        weight,
        targets,
    )

    assert actual_loss.shape == torch.Size([])
    assert actual_loss.dtype == torch.float32
    assert actual_loss.device == hidden.device
    assert actual_loss.requires_grad
    assert actual_loss.grad_fn is not None
    assert torch.isfinite(actual_loss)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-3, atol=1e-3)


@requires_cuda
@pytest.mark.parametrize(
    ("B", "S", "D", "V"),
    [
        (1, 1, 32, 7),
        (2, 9, 64, 79),
        (3, 11, 64, 79),
        (3, 22, 128, 131),
        (4, 8, 256, 32),
        (4, 16, 1024, 64),
        (4, 16, 2048, 1000)
    ],
)
@pytest.mark.parametrize("upstream_scale", [1.0, 0.37, -1.25, 0.0])
def test_fused_loss_triton_backward(
    B: int,
    S: int,
    D: int,
    V: int,
    upstream_scale: float,
):
    hidden, weight, targets = _make_model_inputs(
        B,
        S,
        D,
        V,
        device="cuda",
    )
    _assert_fused_backward_matches_reference(
        hidden,
        weight,
        targets,
        upstream_scale=upstream_scale
    )


@requires_cuda
def test_fused_loss_triton_backward_targets_cross_tile_boundaries():
    B, S, D, V = 2, 4, 64, 97
    hidden, weight, _ = _make_model_inputs(B, S, D, V, device="cuda")
    targets = torch.tensor(
        [[0, 31, 32, 63], [64, 95, 96, 7]],
        dtype=torch.int64,
        device="cuda",
    )
    _assert_fused_backward_matches_reference(
        hidden,
        weight,
        targets,
        upstream_scale=0.37
    )


@requires_cuda
def test_fused_loss_triton_backward_large_logits():
    hidden, weight, targets = _make_model_inputs(
        3,
        11,
        64,
        79,
        scale=10.0,
        device="cuda",
    )
    _assert_fused_backward_matches_reference(
        hidden,
        weight,
        targets,
        upstream_scale=1.0,
        loss_rtol=5e-3,
        loss_atol=5e-3,
        grad_rtol=5e-3,
        grad_atol=5e-3,
    )


@requires_cuda
@pytest.mark.parametrize(
    ("B", "S", "D", "V"),
    [
        (2, 9, 64, 79),
        (3, 11, 128, 131),
        (16, 256, 1024, 1000)
    ],
)
def test_fused_loss_triton_backward_bf16(
    B: int,
    S: int,
    D: int,
    V: int,
):
    hidden, weight, targets = _make_model_inputs(
        B,
        S,
        D,
        V,
        dtype=torch.bfloat16,
        device="cuda",
    )
    _assert_fused_backward_matches_reference(
        hidden,
        weight,
        targets,
        upstream_scale=0.5,
        loss_rtol=1e-2,
        loss_atol=1e-2,
        grad_rtol=1e-2,
        grad_atol=1e-2,
    )
