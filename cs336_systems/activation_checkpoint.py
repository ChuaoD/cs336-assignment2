# cs336_systems/activation_checkpoint.py

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar

import torch
import torch.cuda.nvtx as nvtx
from contextlib import nullcontext


_RECOMPUTING: ContextVar[bool] = ContextVar(
    "activation_checkpoint_recomputing",
    default=False,
)


def is_checkpoint_recomputing() -> bool:
    """Return whether the current execution is checkpoint recomputation."""
    return _RECOMPUTING.get()


@contextmanager
def _recomputation_context():
    """Mark recomputation and reliably restore the previous state."""
    token = _RECOMPUTING.set(True)
    try:
        yield
    finally:
        _RECOMPUTING.reset(token)

# For a custom autograd function, ordinary gradient recording is disabled inside its forward() method, and how gradients are calculated is promised to define in its own backward() method.
class ActivationCheckpointFunction(torch.autograd.Function):
    """Recompute a callable during backward instead of saving its activations.

    Initial restrictions:
        - all callable arguments are tensors;
        - exactly one tensor is returned;
        - at least one input requires gradients;
        - only first-order gradients are supported;
        - stochastic operations are not yet supported.
    """

    @staticmethod
    def forward(
        ctx,
        run_function: Callable[..., torch.Tensor],
        *args: torch.Tensor,
    ) -> torch.Tensor:
        """Run the callable without recording its internal autograd graph.

        Responsibilities:
            1. Save run_function on ctx.
            2. Save the tensor arguments with ctx.save_for_backward().
            3. Save which arguments require gradients.
            4. Execute run_function(*args).
            5. Return its tensor output.
        """

        ctx.run_function = run_function
        ctx.save_for_backward(*args)
        ctx.input_requires_grad = tuple(i.requires_grad for i in args)
        output = run_function(*args)
        return output


    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        """Recompute the callable and return gradients for its inputs.

        Responsibilities:
            1. Recover the saved inputs.
            2. Create detached inputs with their original requires_grad flags.
            3. Enable autograd.
            4. Recompute inside _recomputation_context().
            5. Exit the recomputation context.
            6. Backpropagate grad_output through the recomputed output.
        """
        saved_input = ctx.saved_tensors
        run_function = ctx.run_function
        input_requires_grad = ctx.input_requires_grad
        recomputation_input = []
        for i, inp in enumerate(saved_input):
            detached_input = inp.detach()
            if input_requires_grad[i]:
                detached_input.requires_grad_()
            recomputation_input.append(detached_input)
        nvtx_context = (
            nvtx.range(f"activation checkpoint recomputation")
            if saved_input[0].is_cuda
            else nullcontext()
        )
        with torch.enable_grad():
            with nvtx_context:
                with _recomputation_context():
                    recomputed_output = run_function(*recomputation_input)
        recomputed_output.backward(gradient=grad_output)
        input_grads = tuple(i.grad if i.requires_grad else None for i in recomputation_input)
        return None, *input_grads


def activation_checkpoint(
    run_function: Callable[..., torch.Tensor],
    *args: torch.Tensor,
) -> torch.Tensor:
    """Public activation-checkpointing interface.

    If gradients are disabled, execute run_function directly.

    If no tensor input requires gradients, execute directly. A custom autograd
    Function would otherwise have no differentiable input connecting its output
    to backward.

    Otherwise invoke ActivationCheckpointFunction.apply(run_function, *args).
    """
    grad_enabled = torch.is_grad_enabled()
    input_requires_grads = any(i.requires_grad for i in args)
    if not grad_enabled or not input_requires_grads:
        return run_function(*args)
    else:
        return ActivationCheckpointFunction.apply(run_function, *args)