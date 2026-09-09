"""On-the-fly reification during neighbour sampling.

The sampled training/eval path (`reifm/sampling.py`) currently *materialises*
the full reified graph first — E entity nodes + F fact nodes + R relation nodes,
with 6·F meta-edges — then builds its CSR and BFS-samples ego-subgraphs from it.
On RelBench-scale graphs (F = 13M…80M facts) that materialisation is the OOM
point: `reify()` alone allocates edge_index [2, 6F] + edge_type [6F] +
node_kind [E+F+R] (~19·F ints), and `build_csr()` then another ~12·F.

But reification is **deterministic and local**: the reified graph's structure is
a pure function of the original multigraph. So we never need to build it. We keep
only the compact original graph (E entities, F facts-as-edges, R relations) and
*simulate* the reification lazily inside the BFS — generating each reified node's
outgoing neighbours on demand:

    entity e  --SUBJECT_OF / OBJECT_OF-->  facts where e is head / tail
    fact   f  --HAS_SUBJECT/OBJECT/TYPE->  head(f), tail(f), reltype(f)
    rel    r  --HAS_INSTANCE-->            facts of type r

Reified node id layout is **identical to reify.py**, so the emitted ego-subgraph
dict is byte-for-byte interchangeable with `sampling.sample_batch()`:

    entity e : id = e            in [0, E)
    fact   f : id = E + f        in [E, E+F)
    rel    r : id = E + F + r     in [E+F, E+F+R)

Registered memory is ~6·F ints (3 adjacency-CSR cols + the original
edge_index/edge_type) versus ~12·F for the reified CSR *plus* the ~19·F reified
Data tensors that are never built at all — a ~5× smaller footprint and, crucially,
no giant up-front allocation. The receptive field per query is unchanged, so the
model sees exactly the same subgraphs.
"""

import random

import torch

from .reify import (ENTITY, FACT, REL_TYPE, HAS_SUBJECT, SUBJECT_OF,
                    HAS_OBJECT, OBJECT_OF, HAS_TYPE, HAS_INSTANCE)


def _csr(key, n):
    """Group item ids 0..len(key)-1 by their bucket key in [0, n).

    Returns (rowptr [n+1] python list, col python list of item ids ordered by
    key). For head/tail this buckets fact ids by their head/tail entity; for the
    relation index it buckets fact ids by their relation type.
    """
    order = torch.argsort(key, stable=True)
    col = order.tolist()                       # item (fact) ids grouped by key
    counts = torch.bincount(key, minlength=n)
    rowptr = torch.zeros(n + 1, dtype=torch.long)
    rowptr[1:] = counts.cumsum(0)
    return rowptr.tolist(), col


class OrigGraph:
    """Compact original multigraph + lazy reified-neighbour generation.

    Holds only O(6F) ints (3 CSR cols + edge_index/edge_type as lists). Exposes
    the handful of attributes the sampler reads (`num_entities`, `rel_offset`)
    so it is a drop-in for the reified `Data` in the sampling functions.
    """

    def __init__(self, fact_index, fact_type, num_entities, num_relations,
                 fact_time=None):
        E = int(num_entities)
        F = int(fact_index.size(1))
        R = int(num_relations)
        h = fact_index[0].contiguous()
        t = fact_index[1].contiguous()
        r = fact_type.contiguous()
        self.head_rowptr, self.head_col = _csr(h, E)   # entity -> incident facts (as head)
        self.tail_rowptr, self.tail_col = _csr(t, E)   # entity -> incident facts (as tail)
        self.rel_rowptr, self.rel_col = _csr(r, R)     # relation -> instance facts
        self.fact_head = h.tolist()
        self.fact_tail = t.tolist()
        self.fact_type = r.tolist()
        # OPTIMISATION #2 (same family): lazy temporal filtering. Keep ONE graph
        # with per-fact timestamps and apply the `<= t` cut DURING the BFS,
        # instead of materialising a separate snapshot graph per prediction time
        # (RelBench's core is temporal: every entity example is queried at a time
        # t and may only see rows <= t). One graph + per-query t_max replaces N
        # snapshot builds. None = static graph (no filtering).
        self.fact_time = (fact_time.tolist() if torch.is_tensor(fact_time)
                          else list(fact_time)) if fact_time is not None else None
        self.num_entities = E
        self.num_facts = F
        self.num_relations = R
        self.num_rel_types = R           # alias, matches reified Data
        self.rel_offset = E + F          # reified id of relation 0
        self.fact_offset = E             # reified id of fact 0
        self.num_nodes = E + F + R

    def to(self, *args, **kwargs):
        # the sampler runs on CPU python lists and moves each ego-subgraph to
        # the device per batch, so the graph itself stays put — no-op for parity
        # with the reified Data's .to(device) in train.py.
        return self

    def free_python_csr(self):
        """Drop the Python CSR lists once the graph is registered in the Rust
        sampler (which holds its own copy). Frees ~6F ints of Python-list
        overhead per graph — significant for multi-DB pretraining. The Python
        BFS fallback is unavailable afterwards (Rust-only)."""
        for a in ("head_rowptr", "head_col", "tail_rowptr", "tail_col",
                  "rel_rowptr", "rel_col", "fact_head", "fact_tail", "fact_type"):
            if hasattr(self, a):
                delattr(self, a)
        self._freed = True

    # --- lazy reified adjacency ------------------------------------------------
    def kind(self, u):
        if u < self.num_entities:
            return ENTITY
        if u < self.rel_offset:
            return FACT
        return REL_TYPE

    def degree(self, u):
        E = self.num_entities
        if u < E:                                   # entity
            return ((self.head_rowptr[u + 1] - self.head_rowptr[u]) +
                    (self.tail_rowptr[u + 1] - self.tail_rowptr[u]))
        if u < self.rel_offset:                     # fact
            return 3
        r = u - self.rel_offset                     # relation
        return self.rel_rowptr[r + 1] - self.rel_rowptr[r]

    def neighbors(self, u):
        """All outgoing reified (neighbour_global, etype) pairs of node u."""
        E = self.num_entities
        if u < E:                                   # entity
            hs, he = self.head_rowptr[u], self.head_rowptr[u + 1]
            for i in range(hs, he):
                yield (E + self.head_col[i], SUBJECT_OF)
            ts, te = self.tail_rowptr[u], self.tail_rowptr[u + 1]
            for i in range(ts, te):
                yield (E + self.tail_col[i], OBJECT_OF)
        elif u < self.rel_offset:                   # fact
            f = u - E
            yield (self.fact_head[f], HAS_SUBJECT)
            yield (self.fact_tail[f], HAS_OBJECT)
            yield (self.rel_offset + self.fact_type[f], HAS_TYPE)
        else:                                        # relation
            r = u - self.rel_offset
            rs, re = self.rel_rowptr[r], self.rel_rowptr[r + 1]
            for i in range(rs, re):
                yield (E + self.rel_col[i], HAS_INSTANCE)

    def neighbor_at(self, u, i):
        """The i-th outgoing neighbour of u (0<=i<degree(u)) — for fan-out
        subsampling without materialising the whole neighbour list."""
        E = self.num_entities
        if u < E:                                   # entity
            hd = self.head_rowptr[u + 1] - self.head_rowptr[u]
            if i < hd:
                return (E + self.head_col[self.head_rowptr[u] + i], SUBJECT_OF)
            i -= hd
            return (E + self.tail_col[self.tail_rowptr[u] + i], OBJECT_OF)
        if u < self.rel_offset:                     # fact
            f = u - E
            if i == 0:
                return (self.fact_head[f], HAS_SUBJECT)
            if i == 1:
                return (self.fact_tail[f], HAS_OBJECT)
            return (self.rel_offset + self.fact_type[f], HAS_TYPE)
        r = u - self.rel_offset                     # relation
        return (E + self.rel_col[self.rel_rowptr[r] + i], HAS_INSTANCE)


def register_rust(idx, orig):
    """Register a compact original graph in the Rust on-the-fly sampler."""
    import reifm_sampler as _rust
    _rust.register_graph_original(
        idx, orig.head_rowptr, orig.head_col, orig.tail_rowptr, orig.tail_col,
        orig.rel_rowptr, orig.rel_col, orig.fact_head, orig.fact_tail,
        orig.fact_type, orig.num_entities, orig.num_facts)


def _sample_one(orig, seeds, num_hops, fanout, drop_node, rng, t_max=None):
    """BFS from `seeds` (reified global ids) over lazily-generated outgoing
    edges, fan-out capped, excluding `drop_node` (the queried fact). Mirrors
    `sampling._sample_one` exactly but generates neighbours on the fly.

    If `t_max` is given (and the graph carries fact timestamps), fact nodes with
    timestamp > t_max are skipped — the lazy temporal cut: future rows become
    unreachable (their facts are filtered), so the ego-graph equals the one from
    a snapshot built up to t_max, without building that snapshot."""
    ft = orig.fact_time if t_max is not None else None
    E, rel_off = orig.num_entities, orig.rel_offset

    def future_fact(v):  # v is a fact node whose row postdates the query time
        return ft is not None and E <= v < rel_off and ft[v - E] > t_max

    nodes = list(dict.fromkeys(seeds))
    seen = set(nodes)
    edges = []
    frontier = list(nodes)
    for _ in range(num_hops):
        nxt = []
        for u in frontier:
            if u == drop_node:
                continue
            deg = orig.degree(u)
            if deg == 0:
                continue
            if deg <= fanout:
                picks = orig.neighbors(u)
            else:
                # distinct indices, O(fanout) (random.sample never materialises
                # the deg-long range) → safe on million-degree relation hubs
                picks = (orig.neighbor_at(u, i)
                         for i in rng.sample(range(deg), fanout))
            for v, et in picks:
                if v == drop_node or future_fact(v):
                    continue
                edges.append((u, v, et))
                if v not in seen:
                    seen.add(v)
                    nodes.append(v)
                    nxt.append(v)
        frontier = nxt
    return nodes, edges


def sample_batch_onthefly(orig, anchor_ent, rel_seed, gold, num_hops, fanout,
                          device, drop_facts=None, train_only_reached=False,
                          seed=None, t_max=None):
    """On-the-fly-reification analogue of `sampling.sample_batch`.

    Produces the identical batched-subgraph dict (same keys, same semantics)
    from the compact `OrigGraph`, without ever materialising the reified graph.
    """
    rng = random.Random(0 if seed is None else seed)
    rel_off = orig.rel_offset
    num_ent = orig.num_entities

    x_kind, edge_src, edge_dst, edge_etype, batch_vec = [], [], [], [], []
    seed_ent_local, seed_rel_local = [], []
    cand_local, cand_query, cand_global, gold_pos = [], [], [], []
    kept = []
    offset = 0
    B = anchor_ent.size(0)
    a_list = anchor_ent.tolist()
    r_list = (rel_seed + rel_off).tolist()
    g_list = gold.tolist()
    if drop_facts is not None:
        drop_list = [(num_ent + f) if f >= 0 else -1 for f in drop_facts.tolist()]
    else:
        drop_list = [-1] * B

    if t_max is None or not hasattr(t_max, "__len__"):
        tmax_list = [t_max] * B
    else:
        tmax_list = list(t_max)
    for b in range(B):
        seeds = [a_list[b], r_list[b]]
        nodes, edges = _sample_one(orig, seeds, num_hops, fanout, drop_list[b],
                                   rng, t_max=tmax_list[b])
        g2l = {g: i for i, g in enumerate(nodes)}
        ents = [i for i, n in enumerate(nodes) if n < num_ent]  # ENTITY kind
        gp = -1
        if g_list[b] in g2l:
            gl = g2l[g_list[b]]
            if gl not in ents:
                ents.append(gl)
            gp = ents.index(gl)
        if train_only_reached and gp < 0:
            continue
        bc = len(kept)
        kept.append(b)
        x_kind.extend(orig.kind(n) for n in nodes)
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
        "num_nodes": offset,
        "seed_ent_local": torch.tensor(seed_ent_local, device=device),
        "seed_rel_local": torch.tensor(seed_rel_local, device=device),
        "cand_local": torch.tensor(cand_local, device=device),
        "cand_query": torch.tensor(cand_query, device=device),
        "cand_global": torch.tensor(cand_global),
        "gold_pos": torch.tensor(gold_pos),
        "gold_global": gold.clone(),
        "kept": torch.tensor(kept, dtype=torch.long),
        "B": len(kept),
        "B_total": B,
    }
