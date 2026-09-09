"""ReiFM: a query-conditioned GNN over the reified graph.

The model is deliberately generic. Node states start from a per-kind embedding
(entity / fact / relation type), the query is injected by two additive markers
(a role marker on the known endpoint entity, a relation marker on the queried
relation-type node), any node-inductive message-passing backbone propagates,
and an MLP scores every entity from its final state (plain readout). The
generic corpus-pretrained model of the KG->RDB probe adds a query-conditioned
readout (`qc_readout`, `qc_complex`).

Backbones (`make_conv`): `gat`, `gine` (sum aggregation), `sage`, `rgcn` are
PyTorch Geometric convolutions as shipped; `ginemax` is a GINE-style message
with PyG's mean+max multi-aggregation; `ginemaxlm` is the same convolution in a
memory-lean full-graph implementation (the dense path, used for large graphs).
"""

import torch
from torch import nn
from torch_geometric.nn import GATConv, GINEConv, MessagePassing, RGCNConv, SAGEConv

from .reify import NUM_META_RELATIONS

SEED_SUBJECT, SEED_OBJECT, SEED_RELATION = 0, 1, 2

# Backbones that run the batched dense full-graph path in ReiFM.forward
# (states [B, N, d] over a shared edge structure). The others run the
# disjoint-union path (B copies of the graph through the PyG convolution).
DENSE_PATH_BACKBONES = ("ginemaxlm",)


class ReifiedMeanMaxConv(MessagePassing):
    """GINE-style message passing with mean+max aggregation.

    GINE's SUM aggregation explodes at full-graph evaluation on the huge-degree
    relation-type hubs; plain MEAN is degree-invariant but loses the ability to
    count. Concatenating the MEAN and MAX aggregators (both degree-invariant)
    keeps a "is there a strong neighbour" signal. Consumes the 6 meta-relation
    types as edge features; no dataset-specific parameters.
    """

    def __init__(self, dim):
        super().__init__(aggr=["mean", "max"])
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(3 * dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, dim))

    def forward(self, x, edge_index, edge_attr):
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)  # [N, 2*dim]
        return self.mlp(torch.cat([(1 + self.eps) * x, agg], dim=-1))

    def message(self, x_j, edge_attr):
        return torch.relu(x_j + edge_attr)


class _ReifiedMeanMaxMsg(torch.autograd.Function):
    """Full-graph reified mean+max message (the `ginemax` equivalent on batched
    dense states [B, N, d]), recomputing per-edge messages in backward instead
    of storing them. Mean grad is distributed by 1/in-degree; max grad is routed
    to the winning edge via an exact equality test (the recompute is
    bit-identical to the forward)."""

    @staticmethod
    def forward(ctx, x, edge_emb, src, dst, num_facts, inv_indeg, drop_mask):
        B, N, d = x.shape
        F_ = num_facts
        sum_agg = x.new_zeros(B, N, d)
        max_agg = x.new_full((B, N, d), float("-inf"))
        for k in range(NUM_META_RELATIONS):
            sl = slice(k * F_, (k + 1) * F_)
            sk, dk = src[sl], dst[sl]
            m = torch.relu(x[:, sk] + edge_emb[k])
            if drop_mask is not None:
                m = m * drop_mask.unsqueeze(-1)
            sum_agg.index_add_(1, dk, m)
            max_agg.index_reduce_(1, dk, m, "amax", include_self=True)
        mean_agg = sum_agg * inv_indeg.view(1, -1, 1)
        max_agg = max_agg.masked_fill(max_agg == float("-inf"), 0.0)
        ctx.save_for_backward(x, edge_emb, src, dst, inv_indeg, drop_mask, max_agg)
        ctx.num_facts = F_
        return mean_agg, max_agg

    @staticmethod
    def backward(ctx, g_mean, g_max):
        x, edge_emb, src, dst, inv_indeg, drop_mask, max_agg = ctx.saved_tensors
        F_ = ctx.num_facts
        g_sum = g_mean * inv_indeg.view(1, -1, 1)
        # pass 1: count, per destination, how many incoming edges attain the max,
        # so the max gradient is split equally among tied winners
        tie = torch.zeros_like(max_agg)
        for k in range(NUM_META_RELATIONS):
            sl = slice(k * F_, (k + 1) * F_)
            sk, dk = src[sl], dst[sl]
            m = torch.relu(x[:, sk] + edge_emb[k])
            m_eff = m * drop_mask.unsqueeze(-1) if drop_mask is not None else m
            tie.index_add_(1, dk, (m_eff == max_agg[:, dk]).to(max_agg.dtype))
        g_max_split = g_max / tie.clamp(min=1)
        # pass 2: route mean grad (all edges) + split max grad (winners) back
        g_x = torch.zeros_like(x)
        g_emb = torch.zeros_like(edge_emb)
        for k in range(NUM_META_RELATIONS):
            sl = slice(k * F_, (k + 1) * F_)
            sk, dk = src[sl], dst[sl]
            m = torch.relu(x[:, sk] + edge_emb[k])            # recompute [B,F,d]
            m_eff = m * drop_mask.unsqueeze(-1) if drop_mask is not None else m
            is_max = m_eff == max_agg[:, dk]
            g_m = g_sum[:, dk] + g_max_split[:, dk] * is_max
            g_pre = g_m * (m > 0)                              # relu grad
            if drop_mask is not None:
                g_pre = g_pre * drop_mask.unsqueeze(-1)
            g_x.index_add_(1, sk, g_pre)
            g_emb[k] = g_pre.sum(dim=(0, 1))
        return g_x, g_emb, None, None, None, None, None


class FastReifiedGINEMaxConvLowMem(nn.Module):
    """Memory-lean full-graph mean+max conv — the `ginemax` equivalent for the
    dense path (states [B, N, d]). Same mathematics as ReifiedMeanMaxConv; the
    only structural difference is a per-layer copy of the meta-relation
    embedding table (`edge_emb`) instead of the model-level shared one.

    Note: uses Tensor.index_reduce_(reduce='amax'); if that op is unsupported on
    a backend (some MPS builds), this conv will fail there."""

    def __init__(self, dim):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.edge_emb = nn.Embedding(NUM_META_RELATIONS, dim)
        self.mlp = nn.Sequential(
            nn.Linear(3 * dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, dim))

    def forward(self, x, graph, drop_mask=None):
        if not hasattr(graph, "inv_indeg") or graph.inv_indeg.dtype != x.dtype:
            deg = torch.zeros(graph.num_nodes, device=x.device, dtype=x.dtype)
            deg.index_add_(0, graph.edge_index[1],
                           torch.ones(graph.edge_index.size(1), device=x.device,
                                      dtype=x.dtype))
            graph.inv_indeg = 1.0 / deg.clamp(min=1)
        mean_agg, max_agg = _ReifiedMeanMaxMsg.apply(
            x, self.edge_emb.weight, graph.edge_index[0], graph.edge_index[1],
            graph.num_facts, graph.inv_indeg, drop_mask)
        return self.mlp(torch.cat([(1 + self.eps) * x, mean_agg, max_agg], dim=-1))


def make_conv(name: str, dim: int, agg: str = "sum"):
    if name == "rgcn":
        return RGCNConv(dim, dim, NUM_META_RELATIONS)
    if name == "sage":
        return SAGEConv(dim, dim, aggr=agg)
    if name == "gat":
        return GATConv(dim, dim, heads=4, concat=False, edge_dim=dim)
    if name == "gine":
        return GINEConv(
            nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)),
            edge_dim=dim, aggr=agg,
        )
    if name == "ginemax":
        return ReifiedMeanMaxConv(dim)
    if name == "ginemaxlm":
        return FastReifiedGINEMaxConvLowMem(dim)
    raise ValueError(f"unknown backbone {name}")


class ReiFM(nn.Module):

    def __init__(self, backbone: str = "gat", dim: int = 64, num_layers: int = 12,
                 dropout: float = 0.0, agg: str = "sum", qc_readout: bool = False,
                 qc_complex: bool = False, edge_dropout: float = 0.0):
        super().__init__()
        self.backbone = backbone
        self.dim = dim
        self.num_layers = num_layers
        self.agg = agg
        # Query-conditioned readout (generic model of the probe only): score
        # each candidate t from [x_t ; x_t*x_rel ; x_t*x_head ; complex(h, r, t)]
        # where x_rel / x_head are the final states of the queried relation node
        # and of the known endpoint. Anonymous-node states only.
        self.qc_readout = qc_readout
        self.qc_complex = qc_complex and qc_readout
        # Structural fact dropout at training time (dense path only): a random
        # fraction of facts is removed from each query's graph copy.
        self.edge_dropout = edge_dropout
        self.grad_checkpoint = False  # set True to trade compute for memory
        self.kind_emb = nn.Embedding(3, dim)       # entity / fact / rel_type
        self.seed_emb = nn.Embedding(3, dim)       # subject / object / relation
        # edge-type features for convs that consume edge_attr instead of types
        self.edge_emb = nn.Embedding(NUM_META_RELATIONS, dim)
        self.convs = nn.ModuleList([make_conv(backbone, dim, agg) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1),
        )
        if qc_readout:
            n_feats = 3 + int(self.qc_complex)
            self.qc_head = nn.Sequential(
                nn.Linear(n_feats * dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def _edge_attr(self, edge_type):
        return self.edge_emb(edge_type)

    def _qc_score(self, ent_states, rel_state, head_state):
        """Query-conditioned scoring: ent_states [B,Ne,d], rel_state/head_state
        [B,d] (final states of the query relation node and head entity)."""
        r = rel_state.unsqueeze(1)
        h = head_state.unsqueeze(1)
        parts = [ent_states, ent_states * r, ent_states * h]
        if self.qc_complex:  # ComplEx-style asymmetric term: (h *_C r) *_C conj(t)
            d2 = self.dim // 2
            h_re, h_im = h[..., :d2], h[..., d2:]
            r_re, r_im = r[..., :d2], r[..., d2:]
            t_re, t_im = ent_states[..., :d2], ent_states[..., d2:]
            hr_re = h_re * r_re - h_im * r_im   # complex product h *_C r
            hr_im = h_re * r_im + h_im * r_re
            out_re = hr_re * t_re + hr_im * t_im   # (hr) *_C conj(t), real part
            out_im = hr_im * t_re - hr_re * t_im   # imaginary part (asymmetric)
            parts.append(torch.cat([out_re, out_im], dim=-1))
        feats = torch.cat(parts, dim=-1)
        return self.qc_head(feats).squeeze(-1)

    def conv_forward(self, conv, x, edge_index, edge_type):
        if self.backbone == "rgcn":
            return conv(x, edge_index, edge_type)
        if self.backbone == "sage":
            return conv(x, edge_index)
        return conv(x, edge_index, edge_attr=self._edge_attr(edge_type))

    def forward_subgraphs(self, sub):
        """Forward on a batch of pre-sampled ego-subgraphs (sampled path).

        `sub` is the dict returned by sampling.sample_batch: a disjoint union
        of B small subgraphs. Returns per-node logits [num_nodes]; the caller
        gathers candidates.
        """
        x = self.kind_emb(sub["x_kind"]).clone()
        x[sub["seed_ent_local"]] += self.seed_emb(sub["seed_role"])
        x[sub["seed_rel_local"]] += self.seed_emb.weight[SEED_RELATION]
        edge_index, edge_type = sub["edge_index"], sub["edge_type"]
        for conv, norm in zip(self.convs, self.norms):
            h = self.conv_forward(conv, x, edge_index, edge_type)
            x = norm(x + self.dropout(torch.relu(h)))
        return self.readout(x).squeeze(-1)

    def forward(self, graph, ent_seed, rel_seed, seed_role, drop_facts=None):
        """Score all entities for a batch of B queries on one reified graph.

        graph: reified Data (on device)
        ent_seed: [B] entity node ids (the known endpoint of each query)
        rel_seed: [B] relation-type ids (0-based, offset added here)
        seed_role: [B] SEED_SUBJECT for (h,r,?) or SEED_OBJECT for (?,r,t)
        drop_facts: optional [B] fact ids whose 6 reified edges are removed
            from that query's graph copy (-1 = keep all). Used at training
            time to avoid leaking the queried fact itself.
        returns: [B, num_entities] scores
        """
        B = ent_seed.size(0)
        N = graph.num_nodes
        device = ent_seed.device

        x = self.kind_emb(graph.node_kind).unsqueeze(0).expand(B, N, -1).contiguous()
        batch_idx = torch.arange(B, device=device)
        x[batch_idx, ent_seed] = x[batch_idx, ent_seed] + self.seed_emb(seed_role)
        rel_nodes = graph.rel_offset + rel_seed
        rel_vec = self.seed_emb.weight[SEED_RELATION]
        x[batch_idx, rel_nodes] = x[batch_idx, rel_nodes] + rel_vec

        if self.backbone in DENSE_PATH_BACKBONES:
            # batched dense states over a shared edge structure
            drop_mask = None
            if drop_facts is not None and (drop_facts >= 0).any():
                drop_mask = torch.ones(B, graph.num_facts, device=device)
                b_ids = (drop_facts >= 0).nonzero(as_tuple=True)[0]
                drop_mask[b_ids, drop_facts[b_ids]] = 0.0
            if self.training and self.edge_dropout > 0:  # structural regularization
                if drop_mask is None:
                    drop_mask = torch.ones(B, graph.num_facts, device=device)
                keep = (torch.rand(B, graph.num_facts, device=device)
                        >= self.edge_dropout).to(drop_mask.dtype)
                drop_mask = drop_mask * keep
            use_ckpt = self.training and self.grad_checkpoint

            def layer(conv, norm, x):
                # whole-layer block so checkpointing stores only the layer
                # boundary x (not the conv/relu/dropout/norm activations)
                h = conv(x, graph, drop_mask)
                return norm(x + self.dropout(torch.relu(h)))

            for conv, norm in zip(self.convs, self.norms):
                if use_ckpt:
                    x = torch.utils.checkpoint.checkpoint(
                        layer, conv, norm, x, use_reentrant=False)
                else:
                    x = layer(conv, norm, x)
            ent_states = x[:, : graph.num_entities]
            if self.qc_readout:
                rel_state = x[batch_idx, rel_nodes]
                head_state = x[batch_idx, ent_seed]
                return self._qc_score(ent_states, rel_state, head_state)
            return self.readout(ent_states).squeeze(-1)

        x = x.view(B * N, -1)
        # disjoint union of B copies of the graph
        offsets = (torch.arange(B, device=device) * N).view(B, 1, 1)
        edge_index = (graph.edge_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
        edge_type = graph.edge_type.repeat(B)
        if drop_facts is not None and (drop_facts >= 0).any():
            # reify() lays edges out as 6 blocks of num_facts, so fact f of
            # copy b sits at positions b*|E| + k*F + f, k in [0, 6)
            F_ = graph.num_facts
            E_ = graph.edge_index.size(1)
            b_ids = (drop_facts >= 0).nonzero(as_tuple=True)[0]
            f_ids = drop_facts[b_ids]
            pos = (b_ids.unsqueeze(1) * E_
                   + torch.arange(6, device=device).unsqueeze(0) * F_
                   + f_ids.unsqueeze(1)).reshape(-1)
            keep = torch.ones(edge_type.size(0), dtype=torch.bool, device=device)
            keep[pos] = False
            edge_index = edge_index[:, keep]
            edge_type = edge_type[keep]

        def union_layer(conv, norm, x):
            h = self.conv_forward(conv, x, edge_index, edge_type)
            return norm(x + self.dropout(torch.relu(h)))

        use_ckpt = self.training and self.grad_checkpoint
        for conv, norm in zip(self.convs, self.norms):
            if use_ckpt:
                x = torch.utils.checkpoint.checkpoint(
                    union_layer, conv, norm, x, use_reentrant=False)
            else:
                x = union_layer(conv, norm, x)

        x = x.view(B, N, -1)
        ent_states = x[:, : graph.num_entities]
        return self.readout(ent_states).squeeze(-1)
