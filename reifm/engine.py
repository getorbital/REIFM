"""Training and filtered-ranking evaluation for ReiFM."""

from collections import defaultdict

import torch
from torch.nn import functional as F

from .models import SEED_OBJECT, SEED_SUBJECT


def build_queries(target_edge_index, target_edge_type):
    """Each triple (h, r, t) yields a tail query (h, r, ?)->t and a head
    query (?, r, t)->h. Returns (ent_seed, rel_seed, role, gold), each [2T]."""
    h, t = target_edge_index[0], target_edge_index[1]
    r = target_edge_type
    ent_seed = torch.cat([h, t])
    rel_seed = torch.cat([r, r])
    role = torch.cat([
        torch.full_like(h, SEED_SUBJECT),
        torch.full_like(t, SEED_OBJECT),
    ])
    gold = torch.cat([t, h])
    return ent_seed, rel_seed, role, gold


def train_epoch_mixed(model, optimizer, train_sets, batch_size, device,
                      batches_per_graph=None, accum_steps=1, grad_clip=0.0):
    """One epoch over several graphs with their batches shuffled together,
    so no graph is left stale at epoch end.

    `accum_steps`>1 accumulates gradients over that many (micro-)batches before
    stepping. For the full-graph path memory scales with the micro-batch B in
    [B,N,d], so a small bs + accumulation gives a large effective batch without
    the [B,N,d] blow-up."""
    model.train()
    schedule = []
    for si, ts in enumerate(train_sets):
        n = ts["queries"][0].size(0)
        perm = torch.randperm(n)
        n_batches = (n + batch_size - 1) // batch_size
        if batches_per_graph is not None:
            cap = (batches_per_graph[si] if isinstance(batches_per_graph, (list, tuple))
                   else batches_per_graph)
            n_batches = min(n_batches, cap)
        for b in range(n_batches):
            schedule.append((si, perm[b * batch_size:(b + 1) * batch_size]))
    schedule = [schedule[i] for i in torch.randperm(len(schedule)).tolist()]

    totals = defaultdict(float)
    counts = defaultdict(int)
    optimizer.zero_grad()
    # Interleaving graphs of different sizes fragments the MPS caching
    # allocator; release the cache when the graph changes.
    mps = device.type == "mps"
    prev_si = None
    for step, (si, idx) in enumerate(schedule):
        if mps and si != prev_si:
            torch.mps.empty_cache()
        prev_si = si
        ts = train_sets[si]
        ent_seed, rel_seed, role, gold = ts["queries"]
        gold_b = gold[idx].to(device)
        drop = ts["qfacts"][idx].to(device) if ts["qfacts"] is not None else None
        scores = model(ts["graph"], ent_seed[idx].to(device),
                       rel_seed[idx].to(device), role[idx].to(device),
                       drop_facts=drop)
        pos_filter = ts.get("pos_filter")
        if pos_filter is not None:
            # don't treat other known-true answers of the same (seed, rel)
            # query as negatives: mask them out of the cross-entropy
            rows, cols = [], []
            for i, q in enumerate(idx.tolist()):
                filt = pos_filter[role[q].item()].get(
                    (ent_seed[q].item(), rel_seed[q].item()))
                if filt is not None and filt.numel() > 1:
                    rows.extend([i] * filt.numel())
                    cols.append(filt)
            if rows:
                gold_scores = scores.gather(1, gold_b.unsqueeze(1))
                scores = scores.index_put(
                    (torch.tensor(rows, device=device), torch.cat(cols).to(device)),
                    torch.tensor(float("-inf"), device=device))
                scores = scores.scatter(1, gold_b.unsqueeze(1), gold_scores)
        loss = F.cross_entropy(scores, gold_b)
        (loss / accum_steps).backward()
        if (step + 1) % accum_steps == 0 or step + 1 == len(schedule):
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
        totals[ts["spec"]] += loss.item()
        counts[ts["spec"]] += 1
    return {spec: totals[spec] / counts[spec] for spec in totals}


def train_epoch_sampled(model, optimizer, train_sets, batch_size, device,
                        num_hops, fanout, batches_per_graph=None, num_workers=0):
    """One epoch using neighbor-sampled ego-subgraphs instead of the full
    replicated graph (the sampled regime of the paper's Appendix C.3). Cost per
    query is bounded by (num_hops, fanout). Returns (losses_per_spec, reach_rate)."""
    from .sampling import parallel_sampled_batches, register_graph

    model.train()
    schedule = []
    for si, ts in enumerate(train_sets):
        register_graph(si, ts["graph"])
        ent_seed, rel_seed, role, gold = ts["queries"]
        n = ent_seed.size(0)
        perm = torch.randperm(n)
        nb = (n + batch_size - 1) // batch_size
        if batches_per_graph is not None:
            cap = (batches_per_graph[si] if isinstance(batches_per_graph, (list, tuple))
                   else batches_per_graph)
            nb = min(nb, cap)
        for b in range(nb):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            qf = ts["qfacts"][idx] if ts.get("qfacts") is not None else None
            schedule.append((si, idx, (si, ent_seed[idx], rel_seed[idx],
                                       gold[idx], qf)))
    order = torch.randperm(len(schedule)).tolist()
    schedule = [schedule[i] for i in order]
    jobs = [s[2] for s in schedule]

    totals, counts = defaultdict(float), defaultdict(int)
    reached, total_q = 0, 0
    mps = device.type == "mps"
    batch_iter = parallel_sampled_batches(jobs, num_hops, fanout, device,
                                          num_workers, train_only_reached=True)
    for step, ((si, idx, _), sub) in enumerate(zip(schedule, batch_iter)):
        ts = train_sets[si]
        ent_seed, rel_seed, role, gold = ts["queries"]
        total_q += sub["B_total"]
        reached += sub["B"]
        if sub["B"] == 0:
            continue  # no reachable gold in this batch
        sub["seed_role"] = role[idx][sub["kept"]].to(device)
        logits = model.forward_subgraphs(sub)

        gold_pos = sub["gold_pos"]
        cand_local, cand_query = sub["cand_local"], sub["cand_query"]
        losses = []
        for b in range(sub["B"]):
            mask = cand_query == b
            cand_logits = logits[cand_local[mask]]
            target = torch.tensor(gold_pos[b].item(), device=device)
            losses.append(F.cross_entropy(cand_logits.unsqueeze(0),
                                          target.unsqueeze(0)))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        totals[ts["spec"]] += loss.item()
        counts[ts["spec"]] += 1
        if mps and (step + 1) % 50 == 0:
            torch.mps.empty_cache()

    out = {spec: totals[spec] / max(counts[spec], 1) for spec in totals}
    return out, (reached / max(total_q, 1))


@torch.no_grad()
def evaluate_sampled(model, graph, queries, filter_index, batch_size, device,
                     num_hops, fanout, return_ranks=False):
    """Filtered ranking with neighbor-sampled subgraphs (same bounded-degree
    regime as sampled training). Gold entities not reached within the sampled
    receptive field count as misses (rank = num_entities)."""
    from .sampling import sample_batch

    model.eval()
    ent_seed, rel_seed, role, gold = queries
    num_ent = int(graph.num_entities)
    mps = device.type == "mps"
    ranks = []
    n_reached = 0
    for start in range(0, ent_seed.size(0), batch_size):
        if mps and (start // batch_size + 1) % 50 == 0:
            torch.mps.empty_cache()
        sl = slice(start, start + batch_size)
        sub = sample_batch(graph, ent_seed[sl], rel_seed[sl], gold[sl],
                           num_hops, fanout, device)
        sub["seed_role"] = role[sl].to(device)
        logits = model.forward_subgraphs(sub).cpu()
        cand_local = sub["cand_local"].cpu()
        cand_query = sub["cand_query"].cpu()
        cand_global = sub["cand_global"]
        gold_pos = sub["gold_pos"]
        for b in range(sub["B"]):
            q = start + b
            if gold_pos[b].item() < 0:
                ranks.append(num_ent)  # gold unreachable -> miss
                continue
            n_reached += 1
            mask = cand_query == b
            cs = logits[cand_local[mask]].clone()
            cg = cand_global[mask]
            gpos = (cg == gold[q]).nonzero(as_tuple=True)[0].item()
            gold_score = cs[gpos].clone()
            key = (ent_seed[q].item(), rel_seed[q].item())
            filt = filter_index[role[q].item()].get(key)
            if filt is not None:
                fset = set(filt.tolist())
                drop = torch.tensor([g.item() in fset for g in cg])
                cs[drop] = float("-inf")
            cs[gpos] = float("-inf")
            ranks.append(1 + int((cs >= gold_score).sum().item()))
    ranks = torch.tensor(ranks, dtype=torch.float)
    out = {
        "mrr": (1.0 / ranks).mean().item(),
        "hits@1": (ranks <= 1).float().mean().item(),
        "hits@3": (ranks <= 3).float().mean().item(),
        "hits@10": (ranks <= 10).float().mean().item(),
        "mr": ranks.mean().item(),
        "num_queries": ranks.numel(),
        "reach": n_reached / max(ranks.numel(), 1),
    }
    if return_ranks:
        out["ranks"] = [int(r) for r in ranks.tolist()]
    return out


def build_filter_index(filter_edge_index, filter_edge_type):
    """Map (seed_entity, relation, role) -> tensor of entities to filter."""
    tails = defaultdict(list)  # (h, r) -> [t]
    heads = defaultdict(list)  # (t, r) -> [h]
    h_list = filter_edge_index[0].tolist()
    t_list = filter_edge_index[1].tolist()
    r_list = filter_edge_type.tolist()
    for h, t, r in zip(h_list, t_list, r_list):
        tails[(h, r)].append(t)
        heads[(t, r)].append(h)
    return {
        SEED_SUBJECT: {k: torch.tensor(v) for k, v in tails.items()},
        SEED_OBJECT: {k: torch.tensor(v) for k, v in heads.items()},
    }


@torch.no_grad()
def evaluate(model, graph, queries, filter_index, batch_size, device,
             return_ranks=False):
    """Filtered ranking over all entities; pessimistic tie-breaking (a
    candidate scoring equal to the gold counts against it), as in ULTRA."""
    model.eval()
    ent_seed, rel_seed, role, gold = queries
    ranks = []
    for start in range(0, ent_seed.size(0), batch_size):
        sl = slice(start, start + batch_size)
        scores = model(
            graph,
            ent_seed[sl].to(device),
            rel_seed[sl].to(device),
            role[sl].to(device),
        ).float().cpu()
        for i in range(scores.size(0)):
            q = start + i
            s = scores[i]
            gold_score = s[gold[q]].clone()
            key = (ent_seed[q].item(), rel_seed[q].item())
            filt = filter_index[role[q].item()].get(key)
            if filt is not None:
                s[filt] = float("-inf")
            s[gold[q]] = float("-inf")
            ranks.append(1 + (s >= gold_score).sum().item())
    ranks = torch.tensor(ranks, dtype=torch.float)
    out = {
        "mrr": (1.0 / ranks).mean().item(),
        "hits@1": (ranks <= 1).float().mean().item(),
        "hits@3": (ranks <= 3).float().mean().item(),
        "hits@10": (ranks <= 10).float().mean().item(),
        "mr": ranks.mean().item(),
        "num_queries": ranks.numel(),
    }
    if return_ranks:  # per-query ranks for offline bootstrap CIs
        out["ranks"] = [int(r) for r in ranks.tolist()]
    return out
