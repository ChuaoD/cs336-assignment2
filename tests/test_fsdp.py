"""Tests for FullyShardedDataParallel (FSDP) implementation."""

from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from .adapters import (
    fsdp_gather_full_params,
    fsdp_on_after_backward,
    get_fsdp,
)
from .common import (
    _cleanup_process_group,
    _setup_process_group,
)

import torch.nn.functional as F
from cs336_systems.model_flash import TransformerLM as FlashTransformerLM

class ToyFSDPModel(nn.Module):
    """Simple model using cs336_basics Linear and Embedding for FSDP testing."""

    def __init__(self, vocab_size=100, d_model=64, d_ff=128):
        super().__init__()
        from cs336_basics.model import Embedding, Linear, RMSNorm

        self.embedding = Embedding(vocab_size, d_model)
        self.norm1 = RMSNorm(d_model)
        self.linear1 = Linear(d_model, d_ff)
        self.norm2 = RMSNorm(d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.norm1(x)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.norm2(x)
        x = self.linear2(x)
        x = self.lm_head(x)
        return x


def _apply_mixed_precision_hooks(model, compute_dtype):
    """
    Apply hooks to a non-parallel model that replicate FSDP's mixed-precision
    behavior: cast Linear/Embedding weights to compute_dtype for
    forward/backward, keep master weights and optimizer updates in fp32.
    """
    from cs336_basics.model import Embedding, Linear

    for mod in model.modules():
        if not isinstance(mod, (Linear, Embedding)):
            continue

        # Forward: cast weight to compute_dtype, restore fp32 after
        def make_fwd_pre(dt):
            def hook(m, inp):
                m._saved_fp32 = m.weight.data
                m.weight.data = m.weight.data.to(dt)

            return hook

        def make_fwd_post():
            def hook(m, inp, out):
                m.weight.data = m._saved_fp32
                del m._saved_fp32
                m.weight.grad = None

            return hook

        mod.register_forward_pre_hook(make_fwd_pre(compute_dtype))
        mod.register_forward_hook(make_fwd_post())

        # Linear backward needs the weight in compute_dtype for grad_input
        if isinstance(mod, Linear):

            def make_bwd_pre(dt):
                def hook(m, grad_output):
                    m._saved_fp32_bwd = m.weight.data
                    m.weight.data = m.weight.data.to(dt)
                    m.weight.grad = None

                return hook

            mod.register_full_backward_pre_hook(make_bwd_pre(compute_dtype))

        # After gradient is computed, restore fp32 weight and cast grad to fp32
        def make_grad_hook(m, is_linear):
            def hook(param):
                if is_linear and hasattr(m, "_saved_fp32_bwd"):
                    m.weight.data = m._saved_fp32_bwd
                    del m._saved_fp32_bwd
                if param.grad is not None:
                    param.grad = param.grad.to(torch.float32)

            return hook

        mod.weight.register_post_accumulate_grad_hook(make_grad_hook(mod, isinstance(mod, Linear)))


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("compute_dtype", [None, torch.float16], ids=["fp32", "fp16"])
def test_fsdp_correctness(compute_dtype):
    """Test that FSDP produces the same results as non-parallel training."""
    world_size = 2
    mp.spawn(
        _test_fsdp_correctness,
        args=(world_size, compute_dtype),
        nprocs=world_size,
        join=True,
    )


def _test_fsdp_correctness(rank: int, world_size: int, compute_dtype):
    torch.use_deterministic_algorithms(True)
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    dist.barrier()

    torch.manual_seed(42)
    base_model = ToyFSDPModel(vocab_size=100, d_model=64, d_ff=128).to(device)

    # Non-parallel baseline (with matching mixed-precision if needed)
    non_parallel_model = deepcopy(base_model)
    if compute_dtype is not None:
        _apply_mixed_precision_hooks(non_parallel_model, compute_dtype)

    # FSDP model
    fsdp_model = get_fsdp(deepcopy(base_model), compute_dtype=compute_dtype)

    loss_fn = nn.CrossEntropyLoss()
    fsdp_optimizer = torch.optim.SGD(fsdp_model.parameters(), lr=0.01)
    non_parallel_optimizer = torch.optim.SGD(non_parallel_model.parameters(), lr=0.01)

    # Generate data
    torch.manual_seed(123)
    batch_size = 20
    seq_len = 8
    all_input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)
    all_labels = torch.randint(0, 100, (batch_size,), device=device)

    local_bs = batch_size // world_size

    for step in range(3):
        fsdp_optimizer.zero_grad(set_to_none=True)
        non_parallel_optimizer.zero_grad(set_to_none=True)

        # Non-parallel: forward on all data
        non_parallel_out = non_parallel_model(all_input_ids)
        non_parallel_loss = loss_fn(non_parallel_out[:, -1, :].float(), all_labels)
        non_parallel_loss.backward()
        non_parallel_optimizer.step()

        # FSDP: each rank sees a different subset
        offset = rank * local_bs
        local_input = all_input_ids[offset : offset + local_bs]
        local_labels = all_labels[offset : offset + local_bs]
        fsdp_out = fsdp_model(local_input)
        fsdp_loss = loss_fn(fsdp_out[:, -1, :].float(), local_labels)
        fsdp_loss.backward()

        fsdp_on_after_backward(fsdp_model, fsdp_optimizer)
        fsdp_optimizer.step()

        # Compare all parameters
        full_params = fsdp_gather_full_params(fsdp_model)
        for name, np_param in non_parallel_model.named_parameters():
            fsdp_full = full_params[name]
            if compute_dtype is None:
                assert torch.allclose(np_param.data, fsdp_full, atol=1e-6, rtol=1e-4), (
                    f"Step {step}: Parameter {name} mismatch. Max diff: {(np_param.data - fsdp_full).abs().max().item()}"
                )
            else:
                assert torch.allclose(np_param.data, fsdp_full, atol=1e-4, rtol=1e-4), (
                    f"Step {step}: Parameter {name} mismatch. Max diff: {(np_param.data - fsdp_full).abs().max().item()}"
                )

        # Shuffle data
        torch.manual_seed(42 + step)
        perm = torch.randperm(batch_size)
        all_input_ids = all_input_ids[perm]
        all_labels = all_labels[perm]

    _cleanup_process_group()


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("compute_dtype", [None, torch.float16], ids=["fp32", "fp16"])
def test_fsdp_gradient_sync(compute_dtype):
    """Test that gradients are properly synchronized and correctly shaped."""
    world_size = 2
    mp.spawn(
        _test_fsdp_gradient_sync,
        args=(world_size, compute_dtype),
        nprocs=world_size,
        join=True,
    )


def _test_fsdp_gradient_sync(rank: int, world_size: int, compute_dtype):
    from cs336_basics.model import Embedding, Linear

    torch.use_deterministic_algorithms(True)
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    dist.barrier()

    torch.manual_seed(42)
    model = ToyFSDPModel(vocab_size=100, d_model=64, d_ff=128).to(device)
    fsdp_model = get_fsdp(model, compute_dtype=compute_dtype)

    # Each rank gets different data
    torch.manual_seed(rank)
    input_ids = torch.randint(0, 100, (4, 8), device=device)

    fsdp_optimizer = torch.optim.SGD(fsdp_model.parameters(), lr=0.01)

    out = fsdp_model(input_ids)
    loss = out.sum()
    loss.backward()

    fsdp_on_after_backward(fsdp_model, fsdp_optimizer)

    # After sync, every parameter must have a gradient matching its data shape
    # and in the master weight dtype (fp32), regardless of compute_dtype
    for name, param in fsdp_model.module.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient is None for {name}"
            assert param.grad.shape == param.data.shape, f"Gradient shape {param.grad.shape} != data shape {param.data.shape} for {name}"
            assert param.grad.dtype == param.data.dtype, f"Gradient dtype {param.grad.dtype} != data dtype {param.data.dtype} for {name}"

    # Replicated (non-FSDP) parameter gradients must be identical across ranks
    for name, param in fsdp_model.module.named_parameters():
        if not param.requires_grad:
            continue
        parts = name.rsplit(".", 1)
        mod = dict(fsdp_model.module.named_modules())[parts[0]] if len(parts) == 2 else fsdp_model.module
        if isinstance(mod, (Linear, Embedding)):
            continue
        gathered = [torch.zeros_like(param.grad) for _ in range(world_size)]
        dist.all_gather(gathered, param.grad)
        for r in range(1, world_size):
            assert torch.allclose(gathered[0], gathered[r], atol=1e-4, rtol=1e-4), (
                f"Replicated gradient for {name} differs between rank 0 and rank {r}. Max diff: {(gathered[0] - gathered[r]).abs().max().item()}"
            )

    _cleanup_process_group()


# Test for FSDP on Transfomer LM with Flash-attention and Fused-loss
@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Requires two CUDA GPUs",
)
@pytest.mark.parametrize(
    "compute_dtype",
    [None, torch.bfloat16],
    ids=["fp32", "bf16"],
)
def test_fsdp_fused_loss_two_ranks(compute_dtype):
    world_size = 2
    mp.spawn(
        _test_fsdp_fused_loss_two_ranks,
        args=(world_size, compute_dtype, False),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Requires two CUDA GPUs",
)
@pytest.mark.parametrize(
    "compute_dtype",
    [None, torch.bfloat16],
    ids=["fp32", "bf16"],
)
def test_fsdp_checkpointed_fused_loss_two_ranks(compute_dtype):
    """Compare checkpointed FSDP training with an uncheckpointed reference."""
    world_size = 2
    mp.spawn(
        _test_fsdp_fused_loss_two_ranks,
        args=(world_size, compute_dtype, True),
        nprocs=world_size,
        join=True,
    )


def _test_fsdp_fused_loss_two_ranks(
    rank: int,
    world_size: int,
    compute_dtype: torch.dtype | None,
    activation_checkpointing: bool,
):
    device = torch.device(
        _setup_process_group(
            rank=rank,
            world_size=world_size,
            backend="nccl",
        )
    )

    try:
        torch.manual_seed(42)

        base_model = FlashTransformerLM(
            vocab_size=67,
            context_length=7,
            num_layers=4,
            d_model=64,
            num_heads=4,
            d_ff=128,
            activation_checkpointing=activation_checkpointing,
            theta=10_000.0,
            device=device,
        )

        reference_model = deepcopy(base_model)
        reference_model.activation_checkpointing = False
        if compute_dtype is not None:
            _apply_mixed_precision_hooks(reference_model, compute_dtype)

        fsdp_model = get_fsdp(
            deepcopy(base_model),
            compute_dtype=compute_dtype,
        )

        reference_optimizer = torch.optim.SGD(
            reference_model.parameters(),
            lr=1e-3,
        )

        # Count every block invocation. The expected count is derived from the
        # model's checkpoint_blocks() policy, so this remains valid if that
        # policy changes.
        block_forward_counts = [0 for _ in fsdp_model.module.layers]
        block_count_handles = []

        if activation_checkpointing:
            checkpointed_blocks = tuple(
                fsdp_model.module.checkpoint_blocks()
            )
            checkpointed_block_ids = {
                id(block) for block in checkpointed_blocks
            }
            assert checkpointed_blocks, (
                "The checkpointed test must exercise at least one block."
            )

            for block_index, block in enumerate(fsdp_model.module.layers):
                def count_block_forward(
                    module,
                    args,
                    output,
                    *,
                    index=block_index,
                ):
                    block_forward_counts[index] += 1

                block_count_handles.append(
                    block.register_forward_hook(count_block_forward)
                )
        fsdp_optimizer = torch.optim.SGD(
            fsdp_model.parameters(),
            lr=1e-3,
        )

        # Equal local batch sizes are important because FSDP averages gradients.
        global_batch_size = 10
        local_batch_size = global_batch_size // world_size
        sequence_length = 7
        vocab_size = 67

        torch.manual_seed(123)
        all_input_ids = torch.randint(
            0,
            vocab_size,
            (global_batch_size, sequence_length),
            device=device,
        )
        all_targets = torch.randint(
            0,
            vocab_size,
            (global_batch_size, sequence_length),
            device=device,
        )
        all_targets.reshape(-1)[0] = 0
        all_targets.reshape(-1)[-1] = vocab_size - 1

        token_positions = torch.arange(sequence_length, device=device)

        start = rank * local_batch_size
        end = start + local_batch_size
        local_input_ids = all_input_ids[start:end]
        local_targets = all_targets[start:end]

        # Multiple iterations check weight restoration and repeated gathers.
        for step in range(3):
            reference_optimizer.zero_grad(set_to_none=True)
            fsdp_optimizer.zero_grad(set_to_none=True)

            # Full-batch materialized reference.
            reference_logits = reference_model(
                all_input_ids,
                token_positions,
            )
            reference_loss_sum = F.cross_entropy(
                reference_logits.float().reshape(-1, vocab_size),
                all_targets.reshape(-1),
                reduction="sum",
            )
            reference_loss = reference_loss_sum / all_targets.numel()

            # Each rank uses its local batch and the fused-loss path.
            local_loss_sum = fsdp_model(
                local_input_ids,
                token_positions,
                targets=local_targets,
            )
            local_loss = local_loss_sum / local_targets.numel()

            if activation_checkpointing:
                # Every block has completed this step's original forward.
                # Checkpointed blocks were also recomputed once in each prior
                # completed step; ordinary blocks were not.
                expected_forward_counts = [
                    2 * step + 1
                    if id(block) in checkpointed_block_ids
                    else step + 1
                    for block in fsdp_model.module.layers
                ]
                assert block_forward_counts == expected_forward_counts

                # The first forward must have recorded a non-empty local order
                # for exactly the blocks selected by the model's policy.
                assert set(fsdp_model._checkpoint_block_order) == (
                    checkpointed_block_ids
                )
                assert all(
                    len(order) > 0
                    for order in fsdp_model._checkpoint_block_order.values()
                )

            # The sum of all local fused losses should equal the global loss.
            distributed_loss_sum = local_loss_sum.detach().clone()
            dist.all_reduce(distributed_loss_sum, op=dist.ReduceOp.SUM)

            torch.testing.assert_close(
                distributed_loss_sum,
                reference_loss_sum.detach(),
                atol=1e-4 if compute_dtype is None else 1e-3,
                rtol=1e-4 if compute_dtype is None else 1e-3,
            )

            reference_loss.backward()
            local_loss.backward()
            fsdp_on_after_backward(fsdp_model, fsdp_optimizer)

            if activation_checkpointing:
                # Backward replays each selected block exactly once. Blocks not
                # selected by the policy still run only in the original
                # forward.
                expected_forward_counts = [
                    2 * (step + 1)
                    if id(block) in checkpointed_block_ids
                    else step + 1
                    for block in fsdp_model.module.layers
                ]
                assert block_forward_counts == expected_forward_counts

            # Every execution path must finish with no active recomputation or
            # outstanding all-gather and with all parameters resharded.
            assert fsdp_model._recomputation is None
            assert not fsdp_model._pending_gathers
            assert all(
                metadata.state.name == "SHARDED"
                for metadata in fsdp_model._shards.values()
            )

            # Directly compare synchronized gradient shards.
            reference_parameters = dict(reference_model.named_parameters())

            for name, fsdp_parameter in fsdp_model.module.named_parameters():
                reference_gradient = reference_parameters[name].grad
                assert reference_gradient is not None
                assert fsdp_parameter.grad is not None

                parameter_id = id(fsdp_parameter)

                if parameter_id in fsdp_model._shards:
                    metadata = fsdp_model._shards[parameter_id]
                    padded_reference_gradient = F.pad(
                        reference_gradient.flatten(),
                        (0, metadata.padding),
                    )
                    expected_gradient = padded_reference_gradient.chunk(
                        world_size
                    )[rank].reshape_as(fsdp_parameter.grad)
                else:
                    expected_gradient = reference_gradient

                torch.testing.assert_close(
                    fsdp_parameter.grad,
                    expected_gradient,
                    atol=1e-3 if compute_dtype is None else 5e-3,
                    rtol=1e-3 if compute_dtype is None else 5e-3,
                    msg=lambda message: (
                        f"Step {step}, parameter {name!r}:\n{message}"
                    ),
                )

            reference_optimizer.step()
            fsdp_optimizer.step()

            # Check complete parameters after the optimizer update.
            full_fsdp_parameters = fsdp_gather_full_params(fsdp_model)

            for name, reference_parameter in reference_model.named_parameters():
                torch.testing.assert_close(
                    full_fsdp_parameters[name],
                    reference_parameter.detach(),
                    atol=1e-4 if compute_dtype is None else 1e-3,
                    rtol=1e-4 if compute_dtype is None else 1e-3,
                )

        for handle in block_count_handles:
            handle.remove()
    finally:
        _cleanup_process_group()
