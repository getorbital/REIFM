"""P0-7 — Consolidate the KG->RDB zero-shot structural probe (signature result).

Full-graph, featureless, class-constrained FK prediction on a reified RelBench
DB, using a KG-pretrained ReiFM checkpoint that NEVER saw an RDB. For every
foreign-key relation of a DB we report, over held-out FK facts:
  (a) KG-pretrained checkpoint  (b) random-init same arch (structure-only ctrl)
  (c) degree/frequency heuristic (rank candidates by in-degree, featureless)
  (d) random baseline (1 / #class candidates)
with per-query bootstrap 95% CIs. Runs on CPU (leaves the GPU to P0-4).

Usage:
  uv run python scripts/probe_rdb.py --dataset rel-f1 --targets all \
     --ckpt checkpoints/generic3kg_seedA/best.pt --ckpt_tag generic3kg_seedA \
     --backbone ginemaxlm --qc_readout --qc_complex --device cuda \
     --out results/probe/probe_rel-f1_generic3kg_seedA.csv
"""
import argparse, csv, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from relbench.datasets import get_dataset
from reifm.engine import build_filter_index
from reifm.models import SEED_SUBJECT, ReiFM
from reifm.reify import reify
from reifm.relbench_reify import build_rdb_kg
from reifm.relbench_reify import split_target


def build_model(a, device):
    return ReiFM(backbone=a.backbone, dim=a.dim, num_layers=a.layers, agg=a.agg,
                 qc_readout=a.qc_readout, qc_complex=a.qc_complex).to(device)


@torch.no_grad()
def rank_model(model, graph, ent_seed, rel_seed, role, gold, filt, class_of,
               ref_class, device, bs=2):
    model.eval()
    ranks = []
    for s in range(0, ent_seed.numel(), bs):
        sl = slice(s, s + bs)
        scores = model(graph, ent_seed[sl].to(device), rel_seed[sl].to(device),
                       role[sl].to(device)).cpu()
        for i in range(scores.size(0)):
            q = s + i
            cs = scores[i].clone()
            cs[class_of != ref_class] = float("-inf")
            g = int(gold[q]); gs = cs[g].clone()
            fl = filt[SEED_SUBJECT].get((int(ent_seed[q]), int(rel_seed[q])))
            if fl is not None:
                cs[fl] = float("-inf")
            cs[g] = float("-inf")
            ranks.append(1 + int((cs >= gs).sum()))
    return np.array(ranks, dtype=np.float64)


def rank_degree(kg, target_rid, ent_seed, gold, filt, class_of, ref_class):
    """Featureless frequency heuristic: score each candidate by how often it is
    the OBJECT of the target FK relation (in-degree under that relation).
    Same filtered-ranking protocol as the model path (pessimistic ties)."""
    ei, et = kg["edge_index"], kg["edge_type"]
    obj = ei[1, et == target_rid]
    deg = torch.zeros(kg["num_entities"], dtype=torch.float64)
    deg.scatter_add_(0, obj, torch.ones(obj.numel(), dtype=torch.float64))
    ranks = []
    cand_mask = class_of == ref_class
    for q in range(ent_seed.numel()):
        cs = deg.clone()
        cs[~cand_mask] = float("-inf")
        g = int(gold[q]); gs = cs[g].item()
        fl = filt[SEED_SUBJECT].get((int(ent_seed[q]), int(target_rid)))
        if fl is not None:
            cs[fl] = float("-inf")
        cs[g] = float("-inf")
        ranks.append(1 + int((cs >= gs).sum().item()))
    return np.array(ranks, dtype=np.float64)


def metrics(r):
    return dict(mrr=float(np.mean(1/r)), h1=float(np.mean(r <= 1)),
                h10=float(np.mean(r <= 10)), n=int(r.size))


def ci(r, stat, nboot=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [stat(r[rng.integers(0, r.size, r.size)]) for _ in range(nboot)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="rel-f1")
    p.add_argument("--targets", nargs="*", default=["all"],
                   help="FK relation names, or 'all'")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--ckpt_tag", default="ckpt")
    p.add_argument("--backbone", default="ginemax")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--agg", default="sum")
    p.add_argument("--qc_readout", action="store_true")
    p.add_argument("--qc_complex", action="store_true")
    p.add_argument("--max_queries", type=int, default=1500)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    device = torch.device(a.device)

    db = get_dataset(a.dataset, download=True).get_db()
    kg = build_rdb_kg(db)
    fks = kg["rel_names"] if a.targets == ["all"] else a.targets
    fks = [f for f in fks if f in kg["rel_names"]]
    print(f"[probe] {a.dataset}: {kg['num_entities']} ent, {len(kg['rel_names'])} FKs; "
          f"probing {len(fks)}; ckpt={a.ckpt_tag}")

    sd = torch.load(a.ckpt, map_location=device, weights_only=True)
    rows = []
    for target in fks:
        target_rid = kg["rel_names"].index(target)
        subj_table, fk_col = target.split(".")
        ref_table = db.table_dict[subj_table].fkey_col_to_pkey_table[fk_col]
        ref_class = kg["class_names"].index(ref_table)
        class_of = kg["entity_class"]
        class_size = int((class_of == ref_class).sum())
        fact_index, fact_type, q_sub, q_obj = split_target(kg, target_rid)
        graph = reify(fact_index, fact_type, kg["num_entities"], kg["num_relations"]).to(device)
        if a.max_queries and q_sub.numel() > a.max_queries:
            sel = torch.randperm(q_sub.numel())[:a.max_queries]
            q_sub, q_obj = q_sub[sel], q_obj[sel]
        ent_seed = q_sub
        rel_seed = torch.full_like(q_sub, target_rid)
        role = torch.full_like(q_sub, SEED_SUBJECT)
        gold = q_obj
        all_t = kg["edge_type"] == target_rid
        filt = build_filter_index(kg["edge_index"][:, all_t], kg["edge_type"][all_t])
        if q_sub.numel() < 5:
            print(f"[probe] {target}: only {q_sub.numel()} held-out — skip")
            continue

        # (a) pretrained, (b) random-init same arch
        torch.manual_seed(0)
        m_rand = build_model(a, device)
        m_pre = build_model(a, device)
        try:
            m_pre.load_state_dict(sd)
        except Exception as e:
            print(f"[probe] LOAD FAILED for {a.ckpt_tag} on arch: {e}")
            return
        r_pre = rank_model(m_pre, graph, ent_seed, rel_seed, role, gold, filt,
                           class_of, ref_class, device)
        r_rand = rank_model(m_rand, graph, ent_seed, rel_seed, role, gold, filt,
                            class_of, ref_class, device)
        r_deg = rank_degree(kg, target_rid, ent_seed, gold, filt, class_of, ref_class)
        mp, mr, md = metrics(r_pre), metrics(r_rand), metrics(r_deg)
        lo, hi = ci(r_pre, lambda x: float(np.mean(x <= 10)))
        mlo, mhi = ci(r_pre, lambda x: float(np.mean(1/x)))
        print(f"[probe] {target:28s} cands={class_size:5d} n={mp['n']:5d} | "
              f"PRE mrr={mp['mrr']:.4f}[{mlo:.4f},{mhi:.4f}] h10={mp['h10']:.4f}[{lo:.4f},{hi:.4f}] "
              f"| RAND h10={mr['h10']:.4f} | DEG h10={md['h10']:.4f} | RANDBASE={1/class_size:.4f}",
              flush=True)
        rows.append(dict(dataset=a.dataset, fk=target, ckpt=a.ckpt_tag,
                         candidates=class_size, n=mp['n'],
                         pre_mrr=round(mp['mrr'],4), pre_mrr_lo=round(mlo,4), pre_mrr_hi=round(mhi,4),
                         pre_h1=round(mp['h1'],4), pre_h10=round(mp['h10'],4),
                         pre_h10_lo=round(lo,4), pre_h10_hi=round(hi,4),
                         rand_mrr=round(mr['mrr'],4), rand_h10=round(mr['h10'],4),
                         deg_mrr=round(md['mrr'],4), deg_h10=round(md['h10'],4),
                         randbase=round(1/class_size,4)))
        # incremental write so a mid-run crash keeps completed FKs
        if a.out:
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            with open(a.out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
    if a.out and rows:
        print(f"[probe] wrote {a.out} ({len(rows)} FKs)", flush=True)


if __name__ == "__main__":
    main()
