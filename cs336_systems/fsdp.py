from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
from cs336_basics.model import Linear, Embedding
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum, auto
from cs336_systems.activation_checkpoint import is_checkpoint_recomputing

class NaiveFSDP(nn.Module):

    def __init__(
        self, 
        module: torch.nn.Module, 
        compute_dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.module = module
        self.parameters
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.compute_dtype = compute_dtype
        self._shardable_params = {}
        self._grad_sync_handles = []
        self._shardable_grad_sync_handles = []
        def grad_sync_hook(p):
            if p.grad is not None:
                comm_handle = dist.all_reduce(
                    p.grad, 
                    op=dist.ReduceOp.AVG, 
                    async_op=True
                )
                self._grad_sync_handles.append(comm_handle)

        # Broadcast model parameters and buffers from rank 0 to all other ranks
        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param, src=0)
            for buf in self.module.buffers():
                dist.broadcast(buf, src=0)
            for submodule in self.module.modules():
                # Only shard weights of Linear and Embedding
                if isinstance(submodule, (Linear, Embedding)):
                    param = submodule.weight
                    param_id = id(param)
                    full_numel = param.numel()
                    full_shape = param.shape
                    flattened_param = torch.flatten(param)
                    padding = (-full_numel) % self.world_size
                    padded_param = F.pad(flattened_param, (0, padding))
                    shard_size = padded_param.numel() // self.world_size
                    param_shards = padded_param.chunk(self.world_size)
                    submodule.weight.data = param_shards[self.rank].clone().to(dtype=torch.float32)
                    self._shardable_params[param_id] = {
                        "param": submodule.weight,
                        "full_numel": full_numel,
                        "full_shape": full_shape,
                        "shard_size": shard_size,
                        "shard_data": submodule.weight.data
                    }
                    def make_forward_pre_hook(param_id):
                        def sync_weight(module, args):
                            self._gather_weight(param_id)
                        return sync_weight

                    def make_forward_post_hook(param_id):
                        def restore_local_shard(module, args, output):
                            if output.requires_grad:
                                output.register_hook(lambda x: self._gather_weight(param_id))
                            self._restore_local_weight(param_id)
                        return restore_local_shard


                    submodule.register_forward_pre_hook(make_forward_pre_hook(param_id))
                    submodule.register_forward_hook(make_forward_post_hook(param_id))

                    def make_shardable_grad_sync_hook(p_name, padding, shard_size):
                        def sync_shardable_grad(p):
                            self._restore_local_weight(p_name)
                            if p.grad is not None:
                                flattened_grad = torch.flatten(p.grad)
                                padded_grad = F.pad(flattened_grad, (0, padding))
                                grad_shard = torch.empty(
                                    shard_size, 
                                    dtype=p.grad.dtype, device=p.device
                                )
                                p.grad = None
                                comm_handle = dist.reduce_scatter_tensor(
                                    grad_shard, padded_grad,
                                    op=dist.ReduceOp.AVG,
                                    async_op=True
                                )
                                self._shardable_grad_sync_handles.append({
                                    "comm_handle": comm_handle,
                                    "param": p,
                                    "padded_grad": padded_grad,
                                    "grad_shard": grad_shard
                                })
                        return sync_shardable_grad

                    param.register_post_accumulate_grad_hook(make_shardable_grad_sync_hook(param_id, padding, shard_size))
            for p in self.module.parameters():
                if id(p) not in self._shardable_params and p.requires_grad:
                    p.register_post_accumulate_grad_hook(grad_sync_hook)
                

    def _gather_weight(self, param_id: int):
        compute_dtype = self.compute_dtype if self.compute_dtype is not None else torch.float32
        shard_metadata = self._shardable_params[param_id]
        param = shard_metadata["param"]
        full_numel = shard_metadata["full_numel"]
        full_shape = shard_metadata["full_shape"]
        shard_size = shard_metadata["shard_size"]
        shard_data = shard_metadata["shard_data"]
        gathered_param_data = torch.empty(int(shard_size * self.world_size), dtype=compute_dtype, device=shard_data.device)
        dist.all_gather_into_tensor(gathered_param_data, shard_data.to(compute_dtype), async_op=False)
        gathered_param_data = gathered_param_data[:full_numel]
        param.data = gathered_param_data.reshape(full_shape)


    def _restore_local_weight(self, param_id: int):
        shard_metadata = self._shardable_params[param_id]
        param = shard_metadata["param"]
        param.data = shard_metadata["shard_data"]


    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for h in self._grad_sync_handles:
            h.wait()
        self._grad_sync_handles.clear()
        while self._shardable_grad_sync_handles:
            record = self._shardable_grad_sync_handles.pop(0)
            h = record["comm_handle"]
            p = record["param"]
            grad_shard = record["grad_shard"]
            h.wait()
            p.grad = grad_shard.to(torch.float32).reshape(p.data.shape)
        self._shardable_grad_sync_handles.clear()

class _WeightState(Enum):
    SHARDED = auto()
    GATHERING = auto()
    FULL = auto()


@dataclass
class _ShardMetadata:
    param: nn.Parameter
    full_numel: int
    full_shape: torch.Size
    padding: int
    shard_size: int
    shard_data: torch.Tensor
    state: _WeightState = _WeightState.SHARDED


@dataclass
class _PendingGather:
    handle: dist.Work
    gathered_data: torch.Tensor
    local_data: torch.Tensor

@dataclass
class _Recomputation:
    block_id: int
    order: list[int]
    cursor: int = 0


class FSDP(nn.Module):

    def __init__(
        self, 
        module: torch.nn.Module, 
        compute_dtype: torch.dtype | None = None,
        enable_nvtx: bool = False
    ):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.compute_dtype = compute_dtype
        self.enable_nvtx = enable_nvtx
        self._shards: dict[int, _ShardMetadata] = {}
        self._pending_gathers: dict[int, _PendingGather] = {}
        self._grad_sync_handles = []
        self._shardable_grad_sync_handles = []
        self._init_forward = True  # indicator for initial order tracing
        self._layer_order = []  # execution order of Linear or Embedding layers
        self._inspect_hook_handles = []
        self._param_names = {
            id(param): name
            for name, param in self.module.named_parameters()
        }
        get_blocks = getattr(self.module, "checkpoint_blocks", None)
        if callable(get_blocks):
            checkpoint_blocks = get_blocks()
        else:
            checkpoint_blocks = ()
        self._checkpoint_block_order: dict[int, list[int]] = {}
        self._recomputation = None
        for block in checkpoint_blocks:
            block_id = id(block)
            self._checkpoint_block_order[block_id] = []
            block.register_forward_pre_hook(self._make_block_pre_hook(block_id))
            block.register_forward_hook(self._make_block_post_hook(block_id))

        # Broadcast model parameters and buffers from rank 0 to all other ranks
        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param, src=0)
            for buf in self.module.buffers():
                dist.broadcast(buf, src=0)
            for submodule in self.module.modules():
                # Only shard weights of Linear and Embedding
                if isinstance(submodule, (Linear, Embedding)):
                    param = submodule.weight
                    param_id = id(param)
                    full_numel = param.numel()
                    full_shape = param.shape
                    flattened_param = torch.flatten(param)
                    padding = (-full_numel) % self.world_size
                    padded_param = F.pad(flattened_param, (0, padding))
                    shard_size = padded_param.numel() // self.world_size
                    param_shards = padded_param.chunk(self.world_size)
                    submodule.weight.data = param_shards[self.rank].clone().to(dtype=torch.float32)
                    self._shards[param_id] = _ShardMetadata(
                        param=submodule.weight,
                        full_numel=full_numel,
                        full_shape=full_shape,
                        padding=padding,
                        shard_size=shard_size,
                        shard_data=submodule.weight.data,
                        state = _WeightState.SHARDED
                    )
                    self._inspect_hook_handles.append(submodule.register_forward_pre_hook(self._inspect_execution_order(param_id)))
                    submodule.register_forward_pre_hook(self._make_forward_pre_hook(param_id))
                    submodule.register_forward_hook(self._make_forward_post_hook(param_id))
                    # Assume all sharded weights are trainable
                    param.register_post_accumulate_grad_hook(self._make_post_grad_hook(param_id, is_shardable=True))
            for p in self.module.parameters():
                param_id = id(p)
                if param_id not in self._shards and p.requires_grad:
                    p.register_post_accumulate_grad_hook(self._make_post_grad_hook(param_id, is_shardable=False))
    def _inspect_execution_order(self, param_id):
        def inspect_hook(module, args):
            self._layer_order.append(param_id)
            if self._curr_block_id is not None:
                self._checkpoint_block_order[self._curr_block_id].append(param_id)
        return inspect_hook
    
    def _make_forward_pre_hook(self, param_id):
        def sync_weight(module, args):
            self._complete_gather(param_id)
            param = self._shards[param_id].param
            param_name = self._param_names[param_id]
            if self.enable_nvtx and param.is_cuda:
                nvtx.range_push(f"layer forward: {param_name}")
        return sync_weight

    def _make_forward_post_hook(self, param_id):
        def restore_local_shard(module, args, output):
            param = self._shards[param_id].param
            if self.enable_nvtx and param.is_cuda:
                nvtx.range_pop()
            if output.requires_grad:
                output.register_hook(
                    self._make_backward_pre_hook(param_id)
                )
            self._restore_local_weight(param_id)
            if self._init_forward:
                return 
            if is_checkpoint_recomputing():
                assert self._recomputation is not None
                cursor_in_block = self._recomputation.cursor
                order_in_block = self._recomputation.order
                assert cursor_in_block < len(order_in_block)
                assert param_id == order_in_block[cursor_in_block]
                self._recomputation.cursor += 1
                return
            current_layer_id = self._forward_cursor
            self._forward_cursor += 1
            expected_param_id = self._layer_order[current_layer_id]
            if expected_param_id != param_id:
                raise RuntimeError(
                    "Forward execution-order mismatch."
                    f"Execution-order mismatch at position {current_layer_id}: "
                    f"expected parameter {expected_param_id}, got {param_id}"
                )
            if current_layer_id < len(self._layer_order) - 2:
                self._start_gather(self._layer_order[current_layer_id + 2])
        return restore_local_shard

    def _make_backward_pre_hook(self, param_id):
        def sync_weight(grad):
            if self._shards[param_id].state is _WeightState.SHARDED:
                self._start_gather(param_id)
                if len(self._layer_order) > 1 and param_id == self._layer_order[-1]:
                    prefecth_param_id = self._layer_order[-2]
                    self._start_gather(prefecth_param_id)
            self._complete_gather(param_id)
        return sync_weight

    def _make_post_grad_hook(self, param_id, is_shardable):
        def sync_grad_and_prefetch_weight(p):
            if not is_shardable:
                # All-reduce gradients
                if p.grad is not None:
                    comm_handle = dist.all_reduce(
                        p.grad, 
                        op=dist.ReduceOp.AVG, 
                        async_op=True
                    )
                    self._grad_sync_handles.append(comm_handle)
                return
            # Reduce-scatter shardable gradients
            if p.grad is not None:
                self._restore_local_weight(param_id)
                shard_metadata = self._shards[param_id]
                shard_size = shard_metadata.shard_size
                padding = shard_metadata.padding
                flattened_grad = torch.flatten(p.grad)
                padded_grad = F.pad(flattened_grad, (0, padding))
                grad_shard = torch.empty(
                    shard_size, 
                    dtype=p.grad.dtype, device=p.device
                )
                p.grad = None
                comm_handle = dist.reduce_scatter_tensor(
                    grad_shard, padded_grad,
                    op=dist.ReduceOp.AVG,
                    async_op=True
                )
                self._shardable_grad_sync_handles.append({
                    "comm_handle": comm_handle,
                    "param": p,
                    "padded_grad": padded_grad,
                    "grad_shard": grad_shard
                })
            # Prefetch weights   
            current_layer_id = self._backward_cursor
            self._backward_cursor -= 1
            expected_param_id = self._layer_order[current_layer_id]
            if expected_param_id != param_id:
                raise RuntimeError(
                    "Backward execution-order mismatch."
                    f"Execution-order mismatch at position {current_layer_id}: "
                    f"expected parameter {expected_param_id}, got {param_id}"
                )
            if current_layer_id >= 2:
                self._start_gather(self._layer_order[current_layer_id - 2])
            
        return sync_grad_and_prefetch_weight

    def _make_block_pre_hook(self, block_id):
        def enter_block(module, args):
            # Initial trace execution order inside the block
            if self._init_forward:
                self._curr_block_id = block_id
            # Recomputation
            if not self._init_forward and is_checkpoint_recomputing():
                assert self._recomputation is None
                self._recomputation = _Recomputation(
                    block_id=block_id,
                    order=self._checkpoint_block_order[block_id],
                    cursor=0
                )

        return enter_block
    def _make_block_post_hook(self, block_id):
        def exit_block(module, args, output):
            if self._init_forward:
                assert self._curr_block_id == block_id
                self._curr_block_id = None
                return
            if is_checkpoint_recomputing():
                assert self._recomputation is not None
                assert self._recomputation.block_id == block_id
                assert self._recomputation.cursor == len(self._recomputation.order)
                self._recomputation = None
                return
        return exit_block

    def _start_gather(self, param_id: int):
        ''' SHARDED -> GATHERING '''
        shard_metadata = self._shards.get(param_id)
        if shard_metadata is None or shard_metadata.state is not _WeightState.SHARDED:
            return
        compute_dtype = self.compute_dtype if self.compute_dtype is not None else torch.float32
        shard_size = shard_metadata.shard_size
        shard_data = shard_metadata.shard_data
        gathered_data = torch.empty(
            int(shard_size * self.world_size),
            dtype=compute_dtype, 
            device=shard_data.device
        )
        local_data = shard_data.to(compute_dtype)
        param_name = self._param_names[param_id]
        context = (
            nvtx.range(f"all-gather launch: {param_name}")
            if self.enable_nvtx and shard_data.is_cuda
            else nullcontext()
        )
        with context:
            handle = dist.all_gather_into_tensor(
                gathered_data, 
                local_data, 
                async_op=True
            )
        self._pending_gathers[param_id] = _PendingGather(
            handle=handle,
            gathered_data=gathered_data,
            local_data=local_data
        )
        shard_metadata.state = _WeightState.GATHERING

    def _complete_gather(self, param_id: int):
        ''' SHARDED/GATHERING -> FULL '''
        shard_metadata = self._shards[param_id]
        if shard_metadata.state is _WeightState.FULL:
            return
        if shard_metadata.state is _WeightState.SHARDED:
            self._start_gather(param_id)
        pending = self._pending_gathers[param_id]
        param = shard_metadata.param
        param_name = self._param_names[param_id]
        context = (
            nvtx.range(f"all-gather wait: {param_name}")
            if self.enable_nvtx and param.is_cuda
            else nullcontext()
        )
        with context:
            pending.handle.wait()
        full_numel = shard_metadata.full_numel
        full_shape = shard_metadata.full_shape
        full_data = pending.gathered_data[:full_numel]
        shard_metadata.param.data = full_data.reshape(full_shape)
        shard_metadata.state = _WeightState.FULL
        self._pending_gathers.pop(param_id)


    def _restore_local_weight(self, param_id: int):
        '''
        FULL -> SHARDED
        SHARDED -> no-op
        GATHERING -> error
        '''
        shard_metadata = self._shards[param_id]
        if shard_metadata.state is _WeightState.FULL:
            shard_metadata.param.data = shard_metadata.shard_data
            shard_metadata.state = _WeightState.SHARDED
        elif shard_metadata.state is _WeightState.GATHERING:
            raise RuntimeError("Cannot reshard an unconsumed gather.")


    def forward(self, *args, **kwargs):
        self._curr_block_id = None
        self._forward_cursor = 0
        if not self._init_forward:
            for param_id in self._layer_order[:2]:
                self._start_gather(param_id)
        output = self.module(*args, **kwargs)
        if self._init_forward:
            for h in self._inspect_hook_handles:
                h.remove()
            del self._inspect_hook_handles
            self._init_forward = False
        self._backward_cursor = len(self._layer_order) - 1
        assert not self._pending_gathers
        return output

    def finish_gradient_synchronization(self):
        for h in self._grad_sync_handles:
            h.wait()
        self._grad_sync_handles.clear()
        while self._shardable_grad_sync_handles:
            record = self._shardable_grad_sync_handles.pop(0)
            h = record["comm_handle"]
            p = record["param"]
            grad_shard = record["grad_shard"]
            h.wait()
            p.grad = grad_shard.to(torch.float32).reshape(p.data.shape)
        self._shardable_grad_sync_handles.clear()
        assert len(self._pending_gathers) == 0
        for p in self._shards:
            assert self._shards[p].state is _WeightState.SHARDED