"""Reusable building blocks for RelBench entity-task models on the reified graph,
shared by the single-DB harness and the multi-DB foundation-model pretrainer.

A DB is reified on the fly (OrigGraph, never materialised) with schema-agnostic
semantic features; a task is a set of (entity, label, time) examples. The model
is a SHARED stack — semantic-feature projector + reified GNN + attention pool —
so one model spans many DBs (the reified vocabulary + MiniLM column names are
dataset-agnostic). Only the tiny per-task output head is task-specific.
"""
import random

import torch
from torch_geometric.utils import scatter

from .models import ReiFM, SEED_SUBJECT
from .onthefly import OrigGraph, _sample_one
from .relbench_reify import build_rdb_kg
from .rdb_features import build_entity_features


class AttnPool(torch.nn.Module):
    """Query-conditioned multi-head attention pooling (feature-aware aggregation):
    the seed entity attends over its ego-subgraph nodes, focusing on the relevant
    feature-bearing neighbours instead of mean/max diluting them."""

    def __init__(self, d, heads=4):
        super().__init__()
        self.h, self.dh, self.d = heads, d // heads, d
        self.q = torch.nn.Linear(d, d)
        self.k = torch.nn.Linear(d, d)
        self.v = torch.nn.Linear(d, d)

    def forward(self, x, batch, seed_local, B):
        N = x.size(0)
        q = self.q(x[seed_local]).view(B, self.h, self.dh)
        k = self.k(x).view(N, self.h, self.dh)
        v = self.v(x).view(N, self.h, self.dh)
        score = (q[batch] * k).sum(-1) / (self.dh ** 0.5)
        m = scatter(score, batch, dim=0, dim_size=B, reduce="max")
        s = (score - m[batch]).exp()
        denom = scatter(s, batch, dim=0, dim_size=B, reduce="sum")[batch]
        attn = (s / denom.clamp(min=1e-9)).unsqueeze(-1)
        pooled = scatter(attn * v, batch, dim=0, dim_size=B, reduce="sum")
        return pooled.reshape(B, self.d)


class RFM(torch.nn.Module):
    """Shared relational foundation backbone: semantic-feature projector +
    reified GNN + attention pool. Produces a per-query [B, 2*dim] embedding
    ([seed ; attention-pooled]) consumed by per-task heads."""

    def __init__(self, dim=64, layers=4, heads=4, backbone="ginemax",
                 no_features=False):
        super().__init__()
        self.dim = dim
        self.gnn = ReiFM(backbone=backbone, dim=dim, num_layers=layers)
        self.feat_proj = torch.nn.Linear(384, dim)
        self.pool = AttnPool(dim, heads=heads)
        # featureless variant (M-2 arm b): zero the node_feat injection, keep
        # feat_proj in the module so state_dicts stay shape-compatible.
        self.no_features = no_features

    def forward(self, sub, feat_matrix):
        # feat_matrix lives in CPU RAM (big DBs); gather only this batch's entity
        # rows on CPU then move to device → keeps GPU memory bounded by the batch.
        device = sub["x_kind"].device
        is_ent = sub["x_kind"] == 0
        nf = torch.zeros(sub["x_kind"].size(0), self.dim, device=device)
        if getattr(self, "no_features", False):
            is_ent = torch.zeros_like(is_ent)
        if is_ent.any():
            ng_ent = sub["node_global"][is_ent.cpu()]
            rows = feat_matrix[ng_ent].float().to(device)
            nf[is_ent] = self.feat_proj(rows)
        sub["node_feat"] = nf
        emb = self.gnn.embed_subgraphs(sub)
        seed = emb[sub["seed_ent_local"]]
        B = sub["seed_ent_local"].size(0)
        pooled = self.pool(emb, sub["batch"], sub["seed_ent_local"], B)
        return torch.cat([seed, pooled], dim=-1)


try:
    import reifm_sampler as _rust
except ImportError:
    _rust = None
import os as _os
_USE_RUST = _rust is not None and _os.environ.get("REIFM_NO_RUST") != "1"
_GIDX = [0]


def _ensure_rust(orig):
    """Register an OrigGraph in the Rust on-the-fly sampler once; cache its id."""
    if not hasattr(orig, "_rust_gidx"):
        from .onthefly import register_rust
        gidx = _GIDX[0]
        _GIDX[0] += 1
        register_rust(gidx, orig)
        orig._rust_gidx = gidx
    return orig._rust_gidx


def sample_entity_batch(orig, ent_ids, num_hops, fanout, device, rng, t_max=None):
    """Ego-subgraph per seed entity (no relation seed) → batch dict for RFM.

    Uses the RUST on-the-fly sampler (sample_entity_batch_original) when
    available and no per-example temporal cut is needed — the Python BFS below
    is the multi-DB bottleneck on big graphs."""
    if _USE_RUST and t_max is None:
        gidx = _ensure_rust(orig)
        r = _rust.sample_entity_batch_original(
            gidx, list(ent_ids), num_hops, fanout, rng.getrandbits(48), 0)
        esrc, edst = r["edge_src"], r["edge_dst"]
        ei = (torch.tensor([esrc, edst], device=device) if esrc
              else torch.zeros(2, 0, dtype=torch.long, device=device))
        return {
            "x_kind": torch.tensor(r["x_kind"], device=device),
            "edge_index": ei,
            "edge_type": (torch.tensor(r["edge_type"], device=device) if r["edge_type"]
                          else torch.zeros(0, dtype=torch.long, device=device)),
            "batch": torch.tensor(r["batch"], device=device),
            "seed_ent_local": torch.tensor(r["seed_ent_local"], device=device),
            "seed_role": torch.full((len(ent_ids),), SEED_SUBJECT, device=device),
            "node_global": torch.tensor(r["node_global"]),
        }
    x_kind, esrc, edst, etype, batch = [], [], [], [], []
    seed_local, seed_role, node_global = [], [], []
    offset = 0
    for b, e in enumerate(ent_ids):
        tm = None if t_max is None else t_max[b]
        nodes, edges = _sample_one(orig, [e], num_hops, fanout, -1, rng, t_max=tm)
        g2l = {g: i for i, g in enumerate(nodes)}
        x_kind.extend(orig.kind(n) for n in nodes)
        node_global.extend(nodes)
        batch.extend([b] * len(nodes))
        for s, d, et in edges:
            esrc.append(g2l[s] + offset)
            edst.append(g2l[d] + offset)
            etype.append(et)
        seed_local.append(g2l[e] + offset)
        seed_role.append(SEED_SUBJECT)
        offset += len(nodes)
    ei = (torch.tensor([esrc, edst], device=device) if esrc
          else torch.zeros(2, 0, dtype=torch.long, device=device))
    return {
        "x_kind": torch.tensor(x_kind, device=device),
        "edge_index": ei,
        "edge_type": (torch.tensor(etype, device=device) if etype
                      else torch.zeros(0, dtype=torch.long, device=device)),
        "batch": torch.tensor(batch, device=device),
        "seed_ent_local": torch.tensor(seed_local, device=device),
        "seed_role": torch.tensor(seed_role, device=device),
        "node_global": torch.tensor(node_global),
    }


def _to_float(v):
    """Coerce a RelBench target to float (handles bool, postgres 't'/'f', etc.)."""
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower()
    if s in ("t", "true", "yes"):
        return 1.0
    if s in ("f", "false", "no"):
        return 0.0
    return float(s)


def make_examples(task_table, entity_col, target_col, pkey_to_global, entity_table):
    """([entity_global], [label]) for rows whose entity exists in the graph."""
    df = task_table.df
    ents, labels = [], []
    for ev, yv in zip(df[entity_col].tolist(), df[target_col].tolist()):
        g = pkey_to_global.get((entity_table, ev))
        if g is None:
            continue
        ents.append(g)
        labels.append(_to_float(yv))
    return ents, labels


class DB:
    """A reified RelBench DB snapshot + semantic features, shared across all
    that DB's tasks. Built once.

    `upto` selects the temporal snapshot: "val" (rows ≤ val_timestamp — the
    leak-safe input graph for TRAIN and VAL examples, as in
    train_relbench_entity.py) or "test" (rows ≤ test_timestamp — for TEST-time
    inputs only). Feeding the test snapshot to train/val examples leaks rows
    posterior to their timestamps into message passing and features."""

    def __init__(self, name, device, feat_dtype=torch.float16, upto="test"):
        from relbench.datasets import get_dataset
        self.name = name
        self.ds = get_dataset(name, download=True)
        ts = self.ds.val_timestamp if upto == "val" else self.ds.test_timestamp
        db = self.ds.get_db().upto(ts)
        kg = build_rdb_kg(db)
        self.graph = OrigGraph(kg["edge_index"], kg["edge_type"],
                               kg["num_entities"], kg["num_relations"])
        self.pk = kg["pkey_to_global"]
        feat, _ = build_entity_features(db, device=device)  # MiniLM on `device`
        self.feat = feat.to(feat_dtype)  # [num_entities, 384] kept in CPU RAM
        del db  # free the pandas Database (unused after graph+features built)
        # register the graph in Rust now and drop the Python CSR lists → keeps
        # multi-DB RAM bounded (the Rust sampler holds the only adjacency copy).
        if _USE_RUST:
            _ensure_rust(self.graph)
            self.graph.free_python_csr()
