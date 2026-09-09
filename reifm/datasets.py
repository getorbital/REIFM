"""Dataset loading, reusing ULTRA's dataset classes (ULTRA submodule).

We instantiate ULTRA datasets with pre_transform=None (no relation graph) and
a separate root so processed caches never clash with ULTRA's own runs.
Each split is a PyG Data with: edge_index/edge_type (fact graph),
target_edge_index/target_edge_type (supervision edges), num_relations.
"""

import os
import sys
import functools

import torch

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ULTRA_PATH = os.path.join(_HERE, "vendor", "ULTRA")
if ULTRA_PATH not in sys.path:
    sys.path.insert(0, ULTRA_PATH)

# ULTRA pickles full PyG Data objects; torch>=2.6 defaults weights_only=True
# which refuses them. These are locally-built trusted caches.
if not isinstance(torch.load, functools.partial):
    torch.load = functools.partial(torch.load, weights_only=False)

# ULTRA imports torch_scatter, which has no macOS wheel for recent torch.
# Provide a shim backed by PyG's pure-torch scatter.
try:
    import torch_scatter  # noqa: F401
except ImportError:
    import types

    from torch_geometric.utils import scatter as _pyg_scatter

    def _scatter_add(src, index, dim=0, out=None, dim_size=None):
        assert out is None, "shim does not support out="
        return _pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce="sum")

    def _scatter(src, index, dim=-1, out=None, dim_size=None, reduce="sum"):
        assert out is None, "shim does not support out="
        return _pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce=reduce)

    def _scatter_mean(src, index, dim=-1, out=None, dim_size=None):
        return _scatter(src, index, dim, out, dim_size, "mean")

    def _scatter_max(src, index, dim=-1, out=None, dim_size=None):
        # torch_scatter returns (values, argmax); callers usually take [0]
        return (_scatter(src, index, dim, out, dim_size, "max"), None)

    _shim = types.ModuleType("torch_scatter")
    _shim.scatter_add = _scatter_add
    _shim.scatter = _scatter
    _shim.scatter_mean = _scatter_mean
    _shim.scatter_max = _scatter_max
    sys.modules["torch_scatter"] = _shim

from ultra import datasets as ultra_datasets  # noqa: E402

DATA_ROOT = os.path.join(_HERE, "data", "kg-datasets")


def load_dataset(name: str, version: str | None = None, root: str = DATA_ROOT):
    """Return (train_data, valid_data, test_data, num_relations).

    `name` is an ULTRA dataset class name, e.g. FB15k237Inductive (versions
    v1..v4), WN18RRInductive, NELLInductive, FB15k237, WN18RR, CoDExMedium,
    or InGram ones (NLIngram/FBIngram/WKIngram with versions 25/50/75/100).
    """
    import inspect

    obj = getattr(ultra_datasets, name)
    if inspect.isclass(obj):
        kwargs = {"root": root}
        if version is not None:
            kwargs["version"] = version
        try:
            ds = obj(**kwargs, pre_transform=None)
        except TypeError:  # some classes don't expose pre_transform
            ds = obj(**kwargs)
    else:
        # transductive FB15k237 / WN18RR are factory functions (not classes);
        # they bake in build_relation_graph (we ignore the .relation_graph attr)
        ds = obj(root=root)
    train_data, valid_data, test_data = ds[0], ds[1], ds[2]
    num_relations = ds.num_relations
    # ULTRA datasets double relations with inverses (r and r+R). The reified
    # representation does not need inverse relations: drop them from fact
    # graphs (they are exact duplicates flipped) and from targets if present.
    return train_data, valid_data, test_data, num_relations


def strip_inverse_facts(data, num_relations_with_inv: int):
    """ULTRA fact graphs contain each triple plus its flipped copy with
    relation id r+R. Keep only the originals (r < R) for reification, where
    direction is encoded structurally via HAS_SUBJECT/HAS_OBJECT."""
    R = num_relations_with_inv // 2
    mask = data.edge_type < R
    return data.edge_index[:, mask], data.edge_type[mask], R
