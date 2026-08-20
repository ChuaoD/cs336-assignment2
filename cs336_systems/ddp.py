import torch
import torch.distributed as dist
import torch.nn as nn


class NaiveDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        # Broadcast model parameters and buffers from rank 0 to all other ranks
        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param, src=0)
            for buf in self.module.buffers():
                dist.broadcast(buf, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        world_size = dist.get_world_size()
        for param in self.module.parameters():
            if param.grad is None:
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size

class FlatDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        # Broadcast model parameters and buffers from rank 0 to all other ranks
        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param, src=0)
            for buf in self.module.buffers():
                dist.broadcast(buf, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        world_size = dist.get_world_size()
        param_with_grads = []
        grads = []
        for param in self.module.parameters():
            if param.grad is None:
                continue
            param_with_grads.append(param)
            grads.append(param.grad)
        flattened_grads = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flattened_grads, op=dist.ReduceOp.SUM)
        flattened_grads /= world_size
        synced_grads = torch._utils._unflatten_dense_tensors(flattened_grads, grads)
        for new_grad, param in zip(synced_grads, param_with_grads):
            param.grad.copy_(new_grad)


class OverlapDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.world_size = dist.get_world_size()
        self.module = module
        self.comm_handels = []
        # Broadcast model parameters and buffers from rank 0 to all other ranks
        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param, src=0)
            for buf in self.module.buffers():
                dist.broadcast(buf, src=0)
        def hook(param):
            if param.grad is not None:
                param.grad /= self.world_size
                comm_handle = dist.all_reduce(
                    param.grad, 
                    op=dist.ReduceOp.SUM, 
                    async_op=True
                )
                self.comm_handels.append(comm_handle)
        # Register one hook on each trainable parameter
        for param in self.module.parameters():
            if param.requires_grad == True:
                param.register_post_accumulate_grad_hook(hook)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        for comm_handel in self.comm_handels:
            comm_handel.wait()
        self.comm_handels.clear()
