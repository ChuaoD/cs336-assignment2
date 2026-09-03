from collections.abc import Callable
from typing import Any

import torch
import triton
import triton.language as tl
import math

@triton.jit
def _fused_adamw_kernel(
    param_ptr,
    grad_ptr,
    m_ptr,
    v_ptr,
    n_elements,
    lr,
    step_size,
    beta1,
    beta2,
    eps,
    weight_decay,
    BLOCK_SIZE: tl.constexpr,
):
    """Update one flattened parameter and its two moment tensors in place."""
    # Program indices
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    param = tl.load(param_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    m = tl.load(m_ptr + offsets, mask=mask)
    v = tl.load(v_ptr + offsets, mask=mask)


    param -= lr * weight_decay * param # Apply weight decay
    m = beta1 * m + (1 - beta1) * grad  # Update the first moment
    v = beta2 * v + (1 - beta2) * (grad * grad) 
    param -= step_size * m / (tl.sqrt(v) + eps)

    tl.store(param_ptr + offsets, param, mask=mask)
    tl.store(m_ptr + offsets, m, mask=mask)
    tl.store(v_ptr + offsets, v, mask=mask)


def fused_adamw_update_(
    param: torch.Tensor,
    grad: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    step: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    block_size
) -> None:
    """Update one contiguous FP32 CUDA parameter and its state in place."""
    assert step >= 1
    step_size = lr * math.sqrt(1 - beta2 ** step) / (1 - beta1 ** step)
    n_elements = param.numel()
    Kt = math.ceil(n_elements / block_size)
    _fused_adamw_kernel[(Kt,)](
        param,
        grad,
        m,
        v,
        n_elements,
        lr,
        step_size,
        beta1,
        beta2,
        eps,
        weight_decay,
        block_size
    )

class FusedAdamW(torch.optim.Optimizer):
    """AdamW using one fused Triton update per parameter."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        block_size: int = 256,
    ) -> None:
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)
        self.block_size = block_size

    def _initialize_state(
        self,
        param: torch.Tensor,
    ) -> dict[str, Any]:
        """Lazily create FP32 moment tensors and initialize step to one."""
        if param.grad is None:
            return
        state = self.state[param]
        state["t"] = 1
        state["m"] = torch.zeros_like(param, dtype=torch.float32)
        state["v"] = torch.zeros_like(param, dtype=torch.float32)

    def step(
        self,
        closure: Callable[[], torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        """Update every parameter with a non-None gradient."""
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if not self.state[p]:
                    self._initialize_state(p)
                state = self.state[p]
                t = state["t"]  # Get iteration number from the state
                m = state["m"]  # Get the first moment
                v = state["v"]  # Get the second moment
                fused_adamw_update_(p, p.grad, m, v, t, lr, beta1, beta2, eps, weight_decay, self.block_size)
                state["t"] = t + 1
        return loss
        