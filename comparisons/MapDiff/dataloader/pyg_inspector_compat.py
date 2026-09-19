"""Let upstream MapDiff's hand-rolled `propagate()` run on torch_geometric >= 2.5.

`model/egnn_pytorch/{egnn_pyg,egnn_pytorch_geometric}.py` reimplement
`MessagePassing.propagate()` so the EGNN layer can reuse the collected message
kwargs for its coordinate update. They copy PyG 2.4's internals, including

    msg_kwargs = self.inspector.distribute('message', coll_dict)

PyG 2.5 renamed `Inspector.distribute(func_name, kwargs)` to
`Inspector.collect_param_data(func, kwargs)` -- same semantics (pull each of the
function's parameters out of the blob, falling back to its default, raise if
neither exists), new name -- so on the 2.6.1 the image installs every EGNN
forward dies with "'Inspector' object has no attribute 'distribute'".

Upstream already guards the `__collect__` -> `_collect` rename of the same
release with a try/except but not this one, and we keep `MapDiff/` pristine, so
restore the old name as an alias instead of editing (or vendoring) those two
files. Harmless on PyG < 2.5, where the real method is already there.
"""
from torch_geometric.inspector import Inspector


def patch_inspector_distribute():
    if not hasattr(Inspector, 'distribute'):
        Inspector.distribute = Inspector.collect_param_data
