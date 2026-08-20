from __future__ import annotations

from typing import Any, Optional, Type
from collections.abc import Callable
import torch
from torch.optim import Optimizer
import torch.distributed as dist

class ShardedOptimizer(Optimizer):
    def __init__(
        self, 
        params,
        optimizer_cls: Type[Optimizer], 
        **kwargs: Any
    ):

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self._param_owners = []  # list[tuple[param, param_owner]]
        self._init_local_groups = []
        self._next_global_param_id = 0
        self.inner = None
        super().__init__(params, dict(kwargs))
        self.inner = optimizer_cls(self._init_local_groups, **kwargs)

    def step(self, closure: Optional[Callable] = None, **kwargs):
        # Update the local shard
        result = self.inner.step(closure, **kwargs)
        with torch.no_grad():
            for p, p_owner in self._param_owners:
                dist.broadcast(p, src=p_owner)
        return result

    def add_param_group(self, param_group) -> None:
        full_group = dict(param_group)
        full_group["params"] = list(full_group["params"])
        super().add_param_group(full_group)
        local_group = dict(full_group)
        local_params = []
        for param in local_group["params"]:
            param_owner = self._next_global_param_id % self.world_size
            self._param_owners.append((param, param_owner))
            self._next_global_param_id += 1
            if param_owner == self.rank:
                local_params.append(param)
        local_group["params"] = local_params
        if self.inner is None:
            self._init_local_groups.append(local_group)
        else: 
            self.inner.add_param_group(local_group)
