import typing
from typing import Union

import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric import is_compiling
from torch_geometric.utils import is_sparse
from torch_geometric.typing import Size, SparseTensor

from dynamicmpnn.modules.layers.layers import *


from typing import List, NamedTuple, Optional, Union

import torch
from torch import Tensor

from torch_geometric import EdgeIndex
from torch_geometric.index import ptr2index
from torch_geometric.utils import is_torch_sparse_tensor
from torch_geometric.typing import SparseTensor


class CollectArgs(NamedTuple):
    s_i: torch.Tensor
    v_i: torch.Tensor
    s_j: torch.Tensor
    v_j: torch.Tensor
    edge_attr: torch.Tensor
    index: torch.Tensor
    ptr: typing.Optional[torch.Tensor]
    dim_size: typing.Optional[int]


def collect(
    self,
    edge_index: Union[Tensor, SparseTensor],
    s: torch.Tensor,
    v: torch.Tensor,
    edge_attr: torch.Tensor,
    size: List[Optional[int]],
) -> CollectArgs:

    i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)

    # Collect special arguments:
    if isinstance(edge_index, Tensor):
        if is_torch_sparse_tensor(edge_index):
            adj_t = edge_index
            if adj_t.layout == torch.sparse_coo:
                edge_index_i = adj_t.indices()[0]
                edge_index_j = adj_t.indices()[1]
                ptr = None
            elif adj_t.layout == torch.sparse_csr:
                ptr = adj_t.crow_indices()
                edge_index_j = adj_t.col_indices()
                edge_index_i = ptr2index(ptr, output_size=edge_index_j.numel())
            else:
                raise ValueError(f"Received invalid layout '{adj_t.layout}'")
            if edge_attr is None:
                _value = adj_t.values()
                edge_attr = None if _value.dim() == 1 else _value

        else:
            edge_index_i = edge_index[i]
            edge_index_j = edge_index[j]

            ptr = None
            if not torch.jit.is_scripting() and isinstance(edge_index, EdgeIndex):
                if i == 0 and edge_index.is_sorted_by_row:
                  (ptr, _), _ = edge_index.get_csr()
                elif i == 1 and edge_index.is_sorted_by_col:
                  (ptr, _), _ = edge_index.get_csc()

    elif isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        edge_index_i, edge_index_j, _value = adj_t.coo()
        ptr, _, _ = adj_t.csr()
        if edge_attr is None:
            edge_attr = None if _value is None or _value.dim() == 1 else _value

    else:
        raise NotImplementedError
    if torch.jit.is_scripting():
        assert edge_attr is not None

    # Collect user-defined arguments:
    # (1) - Collect `s_i`:
    if isinstance(s, (tuple, list)):
        assert len(s) == 2
        _s_0, _s_1 = s[0], s[1]
        if isinstance(_s_0, Tensor):
            self._set_size(size, 0, _s_0)
        if isinstance(_s_1, Tensor):
            self._set_size(size, 1, _s_1)
            s_i = self._index_select(_s_1, edge_index_i)
        else:
            s_i = None
    elif isinstance(s, Tensor):
        self._set_size(size, i, s)
        s_i = self._index_select(s, edge_index_i)
    else:
        s_i = None
    # (2) - Collect `v_i`:
    if isinstance(v, (tuple, list)):
        assert len(v) == 2
        _v_0, _v_1 = v[0], v[1]
        if isinstance(_v_0, Tensor):
            self._set_size(size, 0, _v_0)
        if isinstance(_v_1, Tensor):
            self._set_size(size, 1, _v_1)
            v_i = self._index_select(_v_1, edge_index_i)
        else:
            v_i = None
    elif isinstance(v, Tensor):
        self._set_size(size, i, v)
        v_i = self._index_select(v, edge_index_i)
    else:
        v_i = None
    # (3) - Collect `s_j`:
    if isinstance(s, (tuple, list)):
        assert len(s) == 2
        _s_0, _s_1 = s[0], s[1]
        if isinstance(_s_0, Tensor):
            self._set_size(size, 0, _s_0)
            s_j = self._index_select(_s_0, edge_index_j)
        else:
            s_j = None
        if isinstance(_s_1, Tensor):
            self._set_size(size, 1, _s_1)
    elif isinstance(s, Tensor):
        self._set_size(size, j, s)
        s_j = self._index_select(s, edge_index_j)
    else:
        s_j = None
    # (4) - Collect `v_j`:
    if isinstance(v, (tuple, list)):
        assert len(v) == 2
        _v_0, _v_1 = v[0], v[1]
        if isinstance(_v_0, Tensor):
            self._set_size(size, 0, _v_0)
            v_j = self._index_select(_v_0, edge_index_j)
        else:
            v_j = None
        if isinstance(_v_1, Tensor):
            self._set_size(size, 1, _v_1)
    elif isinstance(v, Tensor):
        self._set_size(size, j, v)
        v_j = self._index_select(v, edge_index_j)
    else:
        v_j = None

    # Collect default arguments:

    index = edge_index_i
    size_i = size[i] if size[i] is not None else size[j]
    size_j = size[j] if size[j] is not None else size[i]
    dim_size = size_i

    return CollectArgs(
        s_i,
        v_i,
        s_j,
        v_j,
        edge_attr,
        index,
        ptr,
        dim_size,
    )


def propagate(
    self,
    edge_index: Union[Tensor, SparseTensor],
    s: torch.Tensor,
    v: torch.Tensor,
    edge_attr: torch.Tensor,
    size: Size = None,
) -> torch.Tensor:

    # Begin Propagate Forward Pre Hook #########################################
    if not torch.jit.is_scripting() and not is_compiling():
        for hook in self._propagate_forward_pre_hooks.values():
            hook_kwargs = dict(
                s=s,
                v=v,
                edge_attr=edge_attr,
            )
            res = hook(self, (edge_index, size, hook_kwargs))
            if res is not None:
                edge_index, size, hook_kwargs = res
                s = hook_kwargs['s']
                v = hook_kwargs['v']
                edge_attr = hook_kwargs['edge_attr']
    # End Propagate Forward Pre Hook ###########################################

    mutable_size = self._check_input(edge_index, size)

    # Run "fused" message and aggregation (if applicable).
    fuse = False
    if self.fuse:
        if is_sparse(edge_index):
            fuse = True
        elif not torch.jit.is_scripting() and isinstance(edge_index, EdgeIndex):
            if self.SUPPORTS_FUSED_EDGE_INDEX and edge_index.is_sorted_by_col:
                fuse = True

    if fuse:
        raise NotImplementedError("'message_and_aggregate' not implemented")

    else:

        kwargs = self.collect(
            edge_index,
            s,
            v,
            edge_attr,
            mutable_size,
        )

        # Begin Message Forward Pre Hook #######################################
        if not torch.jit.is_scripting() and not is_compiling():
            for hook in self._message_forward_pre_hooks.values():
                hook_kwargs = dict(
                    s_i=kwargs.s_i,
                    v_i=kwargs.v_i,
                    s_j=kwargs.s_j,
                    v_j=kwargs.v_j,
                    edge_attr=kwargs.edge_attr,
                )
                res = hook(self, (hook_kwargs, ))
                hook_kwargs = res[0] if isinstance(res, tuple) else res
                if res is not None:
                    kwargs = CollectArgs(
                        s_i=hook_kwargs['s_i'],
                        v_i=hook_kwargs['v_i'],
                        s_j=hook_kwargs['s_j'],
                        v_j=hook_kwargs['v_j'],
                        edge_attr=hook_kwargs['edge_attr'],
                        index=kwargs.index,
                        ptr=kwargs.ptr,
                        dim_size=kwargs.dim_size,
                    )
        # End Message Forward Pre Hook #########################################

        out = self.message(
            s_i=kwargs.s_i,
            v_i=kwargs.v_i,
            s_j=kwargs.s_j,
            v_j=kwargs.v_j,
            edge_attr=kwargs.edge_attr,
        )

        # Begin Message Forward Hook ###########################################
        if not torch.jit.is_scripting() and not is_compiling():
            for hook in self._message_forward_hooks.values():
                hook_kwargs = dict(
                    s_i=kwargs.s_i,
                    v_i=kwargs.v_i,
                    s_j=kwargs.s_j,
                    v_j=kwargs.v_j,
                    edge_attr=kwargs.edge_attr,
                )
                res = hook(self, (hook_kwargs, ), out)
                out = res if res is not None else out
        # End Message Forward Hook #############################################

        # Begin Aggregate Forward Pre Hook #####################################
        if not torch.jit.is_scripting() and not is_compiling():
            for hook in self._aggregate_forward_pre_hooks.values():
                hook_kwargs = dict(
                    index=kwargs.index,
                    ptr=kwargs.ptr,
                    dim_size=kwargs.dim_size,
                )
                res = hook(self, (hook_kwargs, ))
                hook_kwargs = res[0] if isinstance(res, tuple) else res
                if res is not None:
                    kwargs = CollectArgs(
                        s_i=kwargs.s_i,
                        v_i=kwargs.v_i,
                        s_j=kwargs.s_j,
                        v_j=kwargs.v_j,
                        edge_attr=kwargs.edge_attr,
                        index=hook_kwargs['index'],
                        ptr=hook_kwargs['ptr'],
                        dim_size=hook_kwargs['dim_size'],
                    )
        # End Aggregate Forward Pre Hook #######################################

        out = self.aggregate(
            out,
            index=kwargs.index,
            ptr=kwargs.ptr,
            dim_size=kwargs.dim_size,
        )

        # Begin Aggregate Forward Hook #########################################
        if not torch.jit.is_scripting() and not is_compiling():
            for hook in self._aggregate_forward_hooks.values():
                hook_kwargs = dict(
                    index=kwargs.index,
                    ptr=kwargs.ptr,
                    dim_size=kwargs.dim_size,
                )
                res = hook(self, (hook_kwargs, ), out)
                out = res if res is not None else out
        # End Aggregate Forward Hook ###########################################

        out = self.update(
            out,
        )

    # Begin Propagate Forward Hook ############################################
    if not torch.jit.is_scripting() and not is_compiling():
        for hook in self._propagate_forward_hooks.values():
            hook_kwargs = dict(
                s=s,
                v=v,
                edge_attr=edge_attr,
            )
            res = hook(self, (edge_index, mutable_size, hook_kwargs), out)
            out = res if res is not None else out
    # End Propagate Forward Hook ##############################################

    return out