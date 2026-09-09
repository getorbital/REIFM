"""Neighbor sampling on the reified graph.

The dense full-graph forward (models.py fast path) replicates the whole graph
B times per batch, so its cost grows with the graph size — 133 min/epoch on
WN18RR v3. This module bounds the per-query receptive field instead: for each
query we sample a small ego-subgraph around its seeds {anchor entity, relation
type}, then the GNN runs ALL its layers within that fixed subgraph.

Key point: sampling depth (`num_hops`, e.g. 3) is NOT the number of GNN layers
(e.g. 12). We sample a bounded subgraph once, then propagate many times inside
it. Subgraph size depends only on (num_hops, fanout), never on |graph| — this
is what makes pretraining scale to arbitrarily large KGs.

BFS follows OUTGOING edges of the reified graph. From an entity, outgoing edges
are SUBJECT_OF/OBJECT_OF (→ its facts); from a fact, HAS_SUBJECT/HAS_OBJECT
(→ its entities) and HAS_TYPE (→ its relation type); from a relation type,
HAS_INSTANCE (→ its facts). So seeding from {h, r} and expanding reaches h's
neighbourhood, r's support facts, and candidate tails — exactly the context
needed to score the query. Both edge directions exist as distinct meta-relations
in the reified graph, so they are collected naturally as the BFS expands.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import torch

from .reify import ENTITY

try:
    import reifm_sampler as _rust  # Rust BFS sampler (optional, for GPU/L4)
except ImportError:
    _rust = None

USE_RUST = _rust is not None and os.environ.get("REIFM_NO_RUST") != "1"

# Registry of reified graphs, indexed by an int id, used by sampler worker
# processes. With the 'fork' start method the workers inherit this (and the
# graphs' cached CSR) copy-on-write — no pickling of the big adjacency lists.
_GRAPH_REGISTRY = {}
# Raw Rust-registration payloads, kept so 'spawn' workers (which do NOT inherit
# the Rust global registry) can replay register_graph in their own process.
# fork inherits this COW; spawn re-serialises it once at worker startup.
_RUST_PAYLOADS = {}


# Graph ids registered via the on-the-fly (compact original) path. The sampler
# dispatches on membership here so engine.py is untouched: a graph is either a
# reified Data (materialised CSR) or an OrigGraph (lazy reified BFS).
_OTF_IDS = set()


def register_graph(idx, graph):
    from .onthefly import OrigGraph, register_rust  # local import (avoid cycle)
    if isinstance(graph, OrigGraph):
        _OTF_IDS.add(idx)
        _GRAPH_REGISTRY[idx] = graph
        if USE_RUST:
            register_rust(idx, graph)
            # stash the Rust-registration args so 'spawn' workers can replay them
            _RUST_PAYLOADS[idx] = ("otf", idx, graph)
        return
    _OTF_IDS.discard(idx)
    rowptr, col, etype = build_csr(graph)  # built+cached before forking
    _GRAPH_REGISTRY[idx] = graph
    if USE_RUST:
        payload = (idx, rowptr, col, etype, graph.node_kind.tolist(),
                   int(graph.num_entities), int(graph.rel_offset))
        _RUST_PAYLOADS[idx] = payload
        _rust.register_graph(*payload)


def _worker_init(rust_payloads):
    """ProcessPoolExecutor initializer for the 'spawn' start method: re-register
    every graph in the fresh worker's Rust module (spawn doesn't inherit the
    parent's Rust global state the way fork does). Handles both reified and
    on-the-fly ('otf') payloads."""
    if USE_RUST and rust_payloads:
        from .onthefly import register_rust
        for payload in rust_payloads.values():
            if isinstance(payload, tuple) and payload and payload[0] == "otf":
                _, idx, graph = payload
                _OTF_IDS.add(idx)
                register_rust(idx, graph)
            else:
                _rust.register_graph(*payload)


# Per-subgraph node budget (0 = unbounded). Caps worst-case ego-graph size so a
# single high-degree anchor can't blow up GPU memory at deep hops — set via the
# REIFM_MAX_NODES env var so it survives 'spawn' workers. See --max_nodes.
def _max_nodes():
    try:
        return int(os.environ.get("REIFM_MAX_NODES", "0"))
    except ValueError:
        return 0


def _rust_sample(gidx, anchor, rel, gold, drop, num_hops, fanout,
                 train_only_reached, otf=False):
    """Call the Rust BFS sampler and assemble the same dict of CPU tensors
    that sample_batch() returns. `otf` selects the on-the-fly (compact original
    graph) backend, which produces an identical subgraph dict."""
    if drop is None:
        drop = torch.full_like(anchor, -1)
    seed = (int(anchor.sum()) ^ (int(gold.sum()) << 1)) & 0xFFFFFFFFFFFF
    fn = _rust.sample_batch_original if otf else _rust.sample_batch
    r = fn(gidx, anchor.tolist(), rel.tolist(), gold.tolist(),
           drop.tolist(), num_hops, fanout, train_only_reached,
           seed, _max_nodes())
    t = lambda k: torch.tensor(r[k], dtype=torch.long)  # noqa: E731
    esrc, edst = r["edge_src"], r["edge_dst"]
    edge_index = (torch.tensor([esrc, edst], dtype=torch.long)
                  if esrc else torch.zeros(2, 0, dtype=torch.long))
    return {
        "x_kind": t("x_kind"),
        "edge_index": edge_index,
        "edge_type": t("edge_type"),
        "batch": t("batch"),
        "num_nodes": len(r["x_kind"]),
        "seed_ent_local": t("seed_ent_local"),
        "seed_rel_local": t("seed_rel_local"),
        "cand_local": t("cand_local"),
        "cand_query": t("cand_query"),
        "cand_global": t("cand_global"),
        "gold_pos": t("gold_pos"),
        "gold_global": gold.clone(),
        "kept": t("kept"),
        "B": r["B"],
        "B_total": r["B_total"],
    }


def _worker_sample(gidx, anchor, rel, gold, drop, num_hops, fanout,
                   train_only_reached=False):
    otf = gidx in _OTF_IDS
    if USE_RUST:
        return _rust_sample(gidx, anchor, rel, gold, drop, num_hops, fanout,
                            train_only_reached, otf=otf)
    g = _GRAPH_REGISTRY[gidx]
    if otf:
        from .onthefly import sample_batch_onthefly
        return sample_batch_onthefly(g, anchor, rel, gold, num_hops, fanout,
                                     torch.device("cpu"), drop_facts=drop,
                                     train_only_reached=train_only_reached)
    return sample_batch(g, anchor, rel, gold, num_hops, fanout,
                        torch.device("cpu"), drop_facts=drop,
                        train_only_reached=train_only_reached)


def build_csr(graph):
    """Outgoing-edge CSR of the reified graph as plain python lists (fast BFS).

    Cached on the graph object as `graph._csr`.
    """
    if hasattr(graph, "_csr"):
        return graph._csr
    src = graph.edge_index[0]
    order = torch.argsort(src)
    col = graph.edge_index[1][order]
    etype = graph.edge_type[order]
    N = graph.num_nodes
    counts = torch.bincount(src[order], minlength=N)
    rowptr = torch.zeros(N + 1, dtype=torch.long)
    rowptr[1:] = counts.cumsum(0)
    csr = (rowptr.tolist(), col.tolist(), etype.tolist())
    graph._csr = csr
    return csr


def _sample_one(csr, seeds, num_hops, fanout, drop_node=-1):
    """BFS from `seeds` (global node ids) along outgoing edges, fan-out capped.

    `drop_node` (the reified fact node of the queried triple) is excluded
    entirely — never traversed nor linked — to prevent the model from reading
    the answer off the very fact it must predict (anti-leak, mirrors the dense
    path's drop_facts).

    Returns (nodes, edges, hops) where nodes is an ordered list of global ids
    (seeds first), edges is a list of (src_global, dst_global, etype), and hops
    is the BFS distance of each node from the seed set (seeds=0). `hops` enables
    RelGT-style hop-distance tokenization on the reified ego-subgraph.
    """
    rowptr, col, etype = csr
    nodes = list(dict.fromkeys(seeds))  # dedup, keep order (seeds first)
    seen = set(nodes)
    hop_of = {n: 0 for n in nodes}      # seeds at hop 0
    edges = []
    frontier = list(nodes)
    for h in range(num_hops):
        nxt = []
        for u in frontier:
            if u == drop_node:
                continue
            start, end = rowptr[u], rowptr[u + 1]
            deg = end - start
            if deg == 0:
                continue
            if deg <= fanout:
                picks = range(start, end)
            else:
                perm = torch.randperm(deg)[:fanout].tolist()
                picks = (start + p for p in perm)
            for i in picks:
                v = col[i]
                if v == drop_node:
                    continue
                edges.append((u, v, etype[i]))
                if v not in seen:
                    seen.add(v)
                    nodes.append(v)
                    hop_of[v] = h + 1
                    nxt.append(v)
        frontier = nxt
    hops = [hop_of[n] for n in nodes]
    return nodes, edges, hops


def sample_batch(graph, anchor_ent, rel_seed, gold, num_hops, fanout, device,
                 drop_facts=None, train_only_reached=False):
    """Sample one ego-subgraph per query and assemble a disjoint-union batch.

    anchor_ent: [B] global entity ids (the known endpoint of each query)
    rel_seed:   [B] relation-type ids (0-based; rel_offset added here)
    gold:       [B] global entity ids of the true answer (for CE target)
    drop_facts: [B] fact index (0-based) of the queried triple to exclude from
                its subgraph (-1 to keep all); anti-leak. Converted to the
                global fact-node id (num_entities + fact_index) internally.

    Returns a dict with the flat batched subgraph and, per query, the local
    positions of its candidate entity nodes and the gold among them
    (gold_pos = -1 if the gold was not reached → caller skips it in the loss).

    If `graph` is an OrigGraph, reification is simulated on the fly (no
    materialised reified graph) — the emitted dict is identical.
    """
    from .onthefly import OrigGraph, sample_batch_onthefly
    if isinstance(graph, OrigGraph):
        return sample_batch_onthefly(graph, anchor_ent, rel_seed, gold,
                                     num_hops, fanout, device,
                                     drop_facts=drop_facts,
                                     train_only_reached=train_only_reached)
    csr = build_csr(graph)
    node_kind = graph.node_kind.tolist()
    rel_off = int(graph.rel_offset)

    x_kind, edge_src, edge_dst, edge_etype, batch_vec = [], [], [], [], []
    node_hop = []
    seed_ent_local, seed_rel_local = [], []
    cand_local, cand_query, cand_global, gold_pos = [], [], [], []
    kept = []  # original query indices actually emitted
    offset = 0
    B = anchor_ent.size(0)
    a_list = anchor_ent.tolist()
    r_list = (rel_seed + rel_off).tolist()
    g_list = gold.tolist()
    num_ent = int(graph.num_entities)
    if drop_facts is not None:
        drop_list = [(num_ent + f) if f >= 0 else -1 for f in drop_facts.tolist()]
    else:
        drop_list = [-1] * B

    for b in range(B):
        seeds = [a_list[b], r_list[b]]
        nodes, edges, hops = _sample_one(csr, seeds, num_hops, fanout, drop_list[b])
        g2l = {g: i for i, g in enumerate(nodes)}
        ents = [i for i, n in enumerate(nodes) if node_kind[n] == ENTITY]
        gp = -1
        if g_list[b] in g2l:
            gl = g2l[g_list[b]]
            if gl not in ents:  # gold reached but as non-entity? shouldn't happen
                ents.append(gl)
            gp = ents.index(gl)
        # at train time, an unreached gold yields no loss → skip its whole
        # subgraph BEFORE the forward (it is ~45% of queries → ~1.8x less
        # forward work). At eval, keep it (unreached gold = miss).
        if train_only_reached and gp < 0:
            continue
        bc = len(kept)  # compacted query index
        kept.append(b)
        x_kind.extend(node_kind[n] for n in nodes)
        node_hop.extend(hops)
        for s, d, et in edges:
            edge_src.append(g2l[s] + offset)
            edge_dst.append(g2l[d] + offset)
            edge_etype.append(et)
        batch_vec.extend([bc] * len(nodes))
        seed_ent_local.append(g2l[a_list[b]] + offset)
        seed_rel_local.append(g2l[r_list[b]] + offset)
        cand_local.extend(e + offset for e in ents)
        cand_query.extend([bc] * len(ents))
        cand_global.extend(nodes[e] for e in ents)
        gold_pos.append(gp)
        offset += len(nodes)

    return {
        "x_kind": torch.tensor(x_kind, device=device),
        "edge_index": torch.tensor([edge_src, edge_dst], device=device)
        if edge_src else torch.zeros(2, 0, dtype=torch.long, device=device),
        "edge_type": torch.tensor(edge_etype, device=device)
        if edge_etype else torch.zeros(0, dtype=torch.long, device=device),
        "batch": torch.tensor(batch_vec, device=device),
        "node_hop": torch.tensor(node_hop, device=device)
        if node_hop else torch.zeros(0, dtype=torch.long, device=device),
        "num_nodes": offset,
        "seed_ent_local": torch.tensor(seed_ent_local, device=device),
        "seed_rel_local": torch.tensor(seed_rel_local, device=device),
        "cand_local": torch.tensor(cand_local, device=device),
        "cand_query": torch.tensor(cand_query, device=device),
        "cand_global": torch.tensor(cand_global),  # cpu; entity global ids
        "gold_pos": torch.tensor(gold_pos),  # cpu; -1 = gold unreached
        "gold_global": gold.clone(),  # cpu; [B]
        "kept": torch.tensor(kept, dtype=torch.long),  # cpu; original idx emitted
        "B": len(kept),       # emitted queries
        "B_total": B,         # original batch size (for reach rate)
    }


_DEVICE_KEYS = ("x_kind", "edge_index", "edge_type", "batch",
                "seed_ent_local", "seed_rel_local", "cand_local", "cand_query")


def sub_to_device(sub, device):
    """Move the device-bound tensors of a sampled batch to `device`
    (CPU-built in a worker → relocated in the main process)."""
    for k in _DEVICE_KEYS:
        sub[k] = sub[k].to(device)
    return sub


def parallel_sampled_batches(jobs, num_hops, fanout, device, num_workers,
                             prefetch=4, train_only_reached=False):
    """Yield sampled batches for `jobs` = [(gidx, anchor, rel, gold, drop), ...],
    sampling in `num_workers` fork processes with bounded prefetch. Order is
    preserved. Falls back to inline sampling if num_workers <= 0.
    """
    if num_workers <= 0:
        for gidx, anchor, rel, gold, drop in jobs:
            yield sub_to_device(
                _worker_sample(gidx, anchor, rel, gold, drop, num_hops, fanout,
                               train_only_reached),
                device)
        return

    import multiprocessing as mp
    # 'spawn' (not 'fork'): forking after the parent has initialised CUDA
    # deadlocks the workers on some CUDA stacks (the sampler only needs the CPU
    # Rust registry, which spawn workers rebuild via _worker_init). Slightly
    # slower startup (payloads re-serialised) but stable on the L4.
    start_method = os.environ.get("REIFM_MP_START", "spawn")
    ctx = mp.get_context(start_method)
    init_kwargs = {}
    if start_method != "fork":
        init_kwargs = dict(initializer=_worker_init, initargs=(_RUST_PAYLOADS,))
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx,
                             **init_kwargs) as ex:
        futures = []
        it = iter(jobs)

        def submit(job):
            gidx, anchor, rel, gold, drop = job
            return ex.submit(_worker_sample, gidx, anchor, rel, gold, drop,
                             num_hops, fanout, train_only_reached)

        for _ in range(num_workers * prefetch):
            try:
                futures.append(submit(next(it)))
            except StopIteration:
                break
        i = 0
        while i < len(futures):
            sub = futures[i].result()
            i += 1
            try:  # keep the pipeline full
                futures.append(submit(next(it)))
            except StopIteration:
                pass
            yield sub_to_device(sub, device)
