"""S-7 (B-49) — ULTRA-3g baseline on the KG->RDB probe.

Runbook + preregistered reading: journal/runs/ar_short_S-7.md.

Zero-shot FK-target ranking on the NON-reified foreign-key graph of a
RelBench DB (rows = nodes, FK columns = relation types), with the released
ULTRA 3-graph checkpoint. Protocol paired with probe_rdb.py: same
DB cache, same split_target (frac_support 0.5, seed 0), same
class-constrained candidates, same filtered pessimistic-tie ranking, same
metrics and bootstrap CIs. Arms:
  (a) ultra_3g pretrained            -> pre_*  columns, row ckpt=ultra_3g
  (b) ULTRA same arch, random init   -> rand_* columns of the same row
  (c) degree heuristic + random baseline (model-independent; compared
      against the reference CSV as the split-identity gate G3)
  (d) --paired_ckpt: re-rank the reified reference checkpoint (run48
      recipe) on the SAME queries -> a second row, ckpt=<tag>_paired.

Why arm (d): the historical probe subsampled queries with an UNSEEDED
randperm (before its torch.manual_seed(0)), so the exact query subsets of
the reference CSVs are not reconstructible where subsampling occurred.
Here subsampling is seeded (--qseed, fresh generator per FK), and the
paired arm re-ranks our reference model on the identical queries, so the
ULTRA-vs-reference comparison is exact-paired regardless. Where the
held-out pool <= --max_queries there is no subsampling, queries match the
reference run exactly, and deg_*/randbase must match the reference CSV
(gate G3 strict).

Usage (A100, runbook S-7):
  uv run python scripts/probe_rdb_ultra.py --dataset rel-f1 \
    --device cuda --bs 4 --max_queries 60 \
    --ref_csv results/probe/probe_rel-f1_run48.csv \
    --paired_ckpt results/_ref_run48/best.pt --paired_tag run48 \
    --dump_ranks results/_s7_ranks \
    --out results/probe/probe_rel-f1_ultra3g.csv
  uv run python scripts/probe_rdb_ultra.py --dataset rel-stack \
    --device cuda --bs 1 --max_queries 60 --timebox_s 10800 \
    --ref_csv results/probe/probe_rel-stack_run48.csv \
    --paired_ckpt results/_ref_run48/best.pt --paired_tag run48 \
    --dump_ranks results/_s7_ranks \
    --out results/probe/probe_rel-stack_ultra3g.csv
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "vendor", "ULTRA"))

import reifm.datasets  # noqa: F401  (torch.load + torch_scatter shims, as in ultra_baseline.py)

from relbench.datasets import get_dataset
from torch_geometric.data import Data

from reifm.engine import build_filter_index
from reifm.models import SEED_SUBJECT
from reifm.reify import reify
from reifm.relbench_reify import build_rdb_kg
from reifm.relbench_reify import split_target
from probe_rdb import build_model as build_reifm
from probe_rdb import ci, metrics, rank_degree
from probe_rdb import rank_model as rank_reifm

from ultra import tasks as ultra_tasks
from ultra.models import Ultra

# ultra_3g architecture, verbatim from vendor/ULTRA config/inductive/inference.yaml
ULTRA_NBF = dict(input_dim=64, hidden_dims=[64, 64, 64, 64, 64, 64],
                 message_func="distmult", aggregate_func="sum",
                 short_cut=True, layer_norm=True)

# recipes for the paired arm (amendment 8: several reified checkpoints can be
# re-ranked on the same queries).
#  run48  : verbatim from probe_all.sh ($A48 + the probe parser's
#           defaults) — the generic 3-graph reified reference of the probe
#  gat_s1 : the short paper's flagship vanilla GAT (S-1 matrix, FB15k-237-
#           Inductive v1 only, plain readout), args read from
#           results/_p0_3/gat_seed*/results.json
PAIRED_RECIPES = {
    "run48": argparse.Namespace(backbone="ginemaxlm", dim=64, layers=12,
                                agg="sum", rel_graph_layers=-1, cofact_layers=2,
                                rel_pe=0, qc_readout=True, qc_complex=True,
                                use_cofact=False),
    "gat_s1": argparse.Namespace(backbone="gat", dim=64, layers=12, agg="sum",
                                 rel_graph_layers=-1, cofact_layers=2, rel_pe=0,
                                 qc_readout=False, qc_complex=False,
                                 use_cofact=False),
    # S-8: GINE mean+max, paper (S-1) recipe, plain readout. `ginemax` = PyG
    # path (the S-1 checkpoints, FB-v1 only); `ginemaxlm` = low-memory path
    # (the S-8 3-graph trainings), mathematically the same conv.
    "ginemax_s1": argparse.Namespace(backbone="ginemax", dim=64, layers=12,
                                     agg="sum", rel_graph_layers=-1,
                                     cofact_layers=2, rel_pe=0,
                                     qc_readout=False, qc_complex=False,
                                     use_cofact=False),
    "ginemaxlm_s1": argparse.Namespace(backbone="ginemaxlm", dim=64,
                                       layers=12, agg="sum",
                                       rel_graph_layers=-1, cofact_layers=2,
                                       rel_pe=0, qc_readout=False,
                                       qc_complex=False, use_cofact=False),
}

CSV_FIELDS = ["dataset", "fk", "ckpt", "candidates", "n",
              "pre_mrr", "pre_mrr_lo", "pre_mrr_hi", "pre_h1", "pre_h10",
              "pre_h10_lo", "pre_h10_hi", "rand_mrr", "rand_h10",
              "deg_mrr", "deg_h10", "randbase"]


def build_ultra(device):
    return Ultra(rel_model_cfg={"class": "RelNBFNet", **ULTRA_NBF},
                 entity_model_cfg={"class": "EntityNBFNet", **ULTRA_NBF}).to(device)


def build_ultra_data(fact_index, fact_type, num_entities, num_rels, device,
                     union=False):
    """FK support graph -> ULTRA input: inverse edges appended (vendor
    convention: inverse of r is r + num_rels), relation graph attached.

    union=True (amendment 9, HYPER-style test): the raw FK edges PLUS the
    reified structure of the same facts — one fact node per FK fact, one
    node per FK relation, meta-edges fact->subject (type R), fact->object
    (R+1), fact->relation-node (R+2); their reverses come from ULTRA's own
    inverse convention. The query stays (u, raw type r, ?): a KGFM cannot
    pose the two-anchor reified query (entity, relation node), so this is
    NOT "the reified graph instead of the raw one" — it measures whether the
    added reified structure changes what ULTRA extracts zero-shot."""
    ei, et, num_nodes, R = fact_index, fact_type, num_entities, num_rels
    if union:
        g = reify(fact_index, fact_type, num_entities, num_rels)
        nf = g.num_facts
        # reify() blocks: 0 HAS_SUBJECT (fact->head), 2 HAS_OBJECT
        # (fact->tail), 4 HAS_TYPE (fact->relation node); 1/3/5 are their
        # reverses, regenerated below by the ULTRA inverse convention.
        blocks = [g.edge_index[:, k * nf:(k + 1) * nf] for k in (0, 2, 4)]
        ei = torch.cat([fact_index] + blocks, dim=1)
        et = torch.cat([fact_type] + [torch.full((nf,), R + k)
                                      for k in range(3)])
        num_nodes, R = g.num_nodes, R + 3
    ei = torch.cat([ei, ei.flip(0)], dim=1)
    et = torch.cat([et, et + R])
    data = Data(edge_index=ei, edge_type=et, num_nodes=num_nodes,
                num_relations=2 * R)
    data = ultra_tasks.build_relation_graph(data)
    return data.to(device)


@torch.no_grad()
def rank_ultra(model, data, ent_seed, rel_seed, gold, filt, class_of,
               ref_class, device, bs, max_s_per_query=None):
    """Filtered class-constrained ranking, pessimistic ties — the exact
    counterpart of probe_rdb.rank_model for the ULTRA API."""
    model.eval()
    ranks = []
    off_class = class_of != ref_class
    for s in range(0, ent_seed.numel(), bs):
        sl = slice(s, min(s + bs, ent_seed.numel()))
        t0 = time.time()
        batch = torch.stack([ent_seed[sl], gold[sl], rel_seed[sl]],
                            dim=-1).to(device)
        t_batch, _ = ultra_tasks.all_negative(data, batch)
        scores = model(data, t_batch).cpu()
        # G5 on the SECOND batch: the first one pays the one-off rspmm
        # extension load and would fail the gate spuriously
        if s == bs and max_s_per_query:
            spq = (time.time() - t0) / scores.size(0)
            print(f"[s7] steady-state: {spq:.1f} s/query", flush=True)
            if spq > max_s_per_query:
                raise RuntimeError(
                    f"G5 FAIL: {spq:.1f} s/query > {max_s_per_query} "
                    f"(runbook feasibility gate)")
        for i in range(scores.size(0)):
            q = s + i
            cs = scores[i].clone()
            cs[off_class] = float("-inf")
            g = int(gold[q]); gs = cs[g].clone()
            fl = filt[SEED_SUBJECT].get((int(ent_seed[q]), int(rel_seed[q])))
            if fl is not None:
                cs[fl] = float("-inf")
            cs[g] = float("-inf")
            ranks.append(1 + int((cs >= gs).sum()))
    return np.array(ranks, dtype=np.float64)


def row_from_ranks(dataset, fk, tag, class_size, r_pre, r_rand, r_deg):
    mp = metrics(r_pre)
    lo, hi = ci(r_pre, lambda x: float(np.mean(x <= 10)))
    mlo, mhi = ci(r_pre, lambda x: float(np.mean(1 / x)))
    row = dict(dataset=dataset, fk=fk, ckpt=tag,
               candidates=class_size, n=mp["n"],
               pre_mrr=round(mp["mrr"], 4), pre_mrr_lo=round(mlo, 4),
               pre_mrr_hi=round(mhi, 4), pre_h1=round(mp["h1"], 4),
               pre_h10=round(mp["h10"], 4), pre_h10_lo=round(lo, 4),
               pre_h10_hi=round(hi, 4),
               rand_mrr="", rand_h10="",
               deg_mrr="", deg_h10="", randbase=round(1 / class_size, 4))
    if r_rand is not None:
        mr = metrics(r_rand)
        row.update(rand_mrr=round(mr["mrr"], 4), rand_h10=round(mr["h10"], 4))
    if r_deg is not None:
        md = metrics(r_deg)
        row.update(deg_mrr=round(md["mrr"], 4), deg_h10=round(md["h10"], 4))
    return row


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def load_ref(path):
    if not path:
        return {}
    with open(path, newline="") as f:
        return {r["fk"]: r for r in csv.DictReader(f)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="rel-f1")
    p.add_argument("--targets", nargs="*", default=["all"])
    p.add_argument("--ultra_ckpt",
                   default=os.path.join(REPO, "vendor", "ULTRA", "ckpts",
                                        "ultra_3g.pth"))
    p.add_argument("--ultra_tag", default="ultra_3g")
    p.add_argument("--paired_ckpt", nargs="*", default=[],
                   help="ReiFM checkpoint(s) re-ranked on the same queries; "
                        "one <paired_tag>_paired row each")
    p.add_argument("--paired_tag", nargs="*", default=[])
    p.add_argument("--paired_recipe", default="run48",
                   choices=sorted(PAIRED_RECIPES),
                   help="architecture of the paired checkpoint(s)")
    p.add_argument("--no_ultra", action="store_true",
                   help="skip the ULTRA arms (paired arm + controls only)")
    p.add_argument("--ultra_graph", default="raw", choices=["raw", "union"],
                   help="graph fed to ULTRA: raw FK graph (protocol) or raw "
                        "+ reified structure (amendment 9, HYPER-style)")
    p.add_argument("--max_queries", type=int, default=60)
    p.add_argument("--qseed", type=int, default=0,
                   help="seed of the per-FK query-subsampling generator")
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ref_csv", default=None,
                   help="reference probe CSV for the G3 split-identity gate")
    p.add_argument("--max_s_per_query", type=float, default=60.0,
                   help="G5 feasibility gate, measured on the first batch")
    p.add_argument("--timebox_s", type=float, default=0,
                   help="stop starting new FKs after this many seconds")
    p.add_argument("--dump_ranks", default=None,
                   help="directory for per-query rank dumps (JSON per FK)")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    device = torch.device(a.device)

    db = get_dataset(a.dataset, download=True).get_db()
    kg = build_rdb_kg(db)
    num_rels = kg["num_relations"]
    fks = kg["rel_names"] if a.targets == ["all"] else a.targets
    fks = [f for f in fks if f in kg["rel_names"]]
    # G1 — graph stats, to check against the P0-7 sheet before reading results
    print(f"[s7][G1] {a.dataset}: {kg['num_entities']} rows, "
          f"{len(kg['rel_names'])} FK relations, {len(kg['class_names'])} "
          f"classes, {kg['edge_type'].numel()} FK facts; probing {len(fks)}",
          flush=True)

    m_pre = m_rand = None
    if not a.no_ultra:
        state = torch.load(a.ultra_ckpt, map_location="cpu")
        torch.manual_seed(0)
        m_rand = build_ultra(device)
        m_pre = build_ultra(device)
        m_pre.load_state_dict(state["model"])
        print(f"[s7][G4] ultra checkpoint loaded strict from {a.ultra_ckpt}",
              flush=True)

    if a.paired_ckpt and len(a.paired_tag) != len(a.paired_ckpt):
        raise SystemExit("--paired_tag must have one entry per --paired_ckpt")
    paired = []  # (tag, model)
    for ck, tag in zip(a.paired_ckpt, a.paired_tag):
        sd = torch.load(ck, map_location=device, weights_only=True)
        m = build_reifm(PAIRED_RECIPES[a.paired_recipe], device)
        m.load_state_dict(sd)
        paired.append((tag, m))
        print(f"[s7] paired arm ({a.paired_recipe}): {tag} loaded strict "
              f"from {ck}", flush=True)

    ref = load_ref(a.ref_csv)
    rows, paired_rows = [], []
    t_start = time.time()
    for target in fks:
        if a.timebox_s and time.time() - t_start > a.timebox_s:
            print(f"[s7] TIMEBOX hit before {target} — stopping; completed "
                  f"FKs are in the CSV", flush=True)
            break
        target_rid = kg["rel_names"].index(target)
        subj_table, fk_col = target.split(".")
        ref_table = db.table_dict[subj_table].fkey_col_to_pkey_table[fk_col]
        ref_class = kg["class_names"].index(ref_table)
        class_of = kg["entity_class"]
        class_size = int((class_of == ref_class).sum())
        fact_index, fact_type, q_sub, q_obj = split_target(kg, target_rid)
        subsampled = False
        if a.max_queries and q_sub.numel() > a.max_queries:
            g = torch.Generator().manual_seed(a.qseed)
            sel = torch.randperm(q_sub.numel(), generator=g)[:a.max_queries]
            q_sub, q_obj = q_sub[sel], q_obj[sel]
            subsampled = True
        if q_sub.numel() < 5:
            print(f"[s7] {target}: only {q_sub.numel()} held-out — skip",
                  flush=True)
            continue
        ent_seed, gold = q_sub, q_obj
        rel_seed = torch.full_like(q_sub, target_rid)
        all_t = kg["edge_type"] == target_rid
        filt = build_filter_index(kg["edge_index"][:, all_t],
                                  kg["edge_type"][all_t])

        r_deg = rank_degree(kg, target_rid, ent_seed, gold, filt, class_of,
                            ref_class)
        deg_h10 = round(metrics(r_deg)["h10"], 4)
        randbase = round(1 / class_size, 4)
        r_pre = r_rand = None
        if not a.no_ultra:
            data = build_ultra_data(fact_index, fact_type, kg["num_entities"],
                                    num_rels, device,
                                    union=(a.ultra_graph == "union"))
            # non-entity nodes (fact / relation nodes of the union graph) are
            # never candidates: class -1
            class_of_u = torch.cat([class_of, torch.full(
                (data.num_nodes - kg["num_entities"],), -1,
                dtype=class_of.dtype)])
            r_pre = rank_ultra(m_pre, data, ent_seed, rel_seed, gold, filt,
                               class_of_u, ref_class, device, a.bs,
                               a.max_s_per_query)
            r_rand = rank_ultra(m_rand, data, ent_seed, rel_seed, gold, filt,
                                class_of_u, ref_class, device, a.bs)
            row = row_from_ranks(a.dataset, target, a.ultra_tag, class_size,
                                 r_pre, r_rand, r_deg)
            rows.append(row)
            print(f"[s7] {target:28s} cands={class_size:7d} n={row['n']:4d} | "
                  f"ULTRA mrr={row['pre_mrr']:.4f} h10={row['pre_h10']:.4f}"
                  f"[{row['pre_h10_lo']:.4f},{row['pre_h10_hi']:.4f}] | "
                  f"RAND h10={row['rand_h10']:.4f} | DEG h10={row['deg_h10']:.4f}"
                  f" | RANDBASE={row['randbase']:.4f}", flush=True)
        else:
            print(f"[s7] {target:28s} cands={class_size:7d} "
                  f"n={ent_seed.numel():4d} | DEG h10={deg_h10:.4f} | "
                  f"RANDBASE={randbase:.4f}", flush=True)

        # G3 — split identity vs the reference CSV (deg/randbase are
        # model-independent; strict only where no subsampling occurred)
        if target in ref:
            rr = ref[target]
            same_n = int(rr["n"]) == ent_seed.numel()
            exact = (not subsampled) and same_n
            d_deg = abs(float(rr["deg_h10"]) - deg_h10)
            d_rb = abs(float(rr["randbase"]) - randbase)
            status = "STRICT" if exact else "info (subsampled queries differ)"
            print(f"[s7][G3:{status}] {target}: deg_h10 ref="
                  f"{rr['deg_h10']} here={deg_h10} | randbase ref="
                  f"{rr['randbase']} here={randbase}", flush=True)
            if exact and (d_deg > 1e-4 or d_rb > 1e-4):
                raise RuntimeError(
                    f"G3 FAIL on {target}: unsubsampled queries should "
                    f"reproduce the reference controls exactly")
            if d_rb > 1e-4:
                raise RuntimeError(
                    f"G3 FAIL on {target}: randbase mismatch — candidate "
                    f"set differs from the reference run")

        r_paired = {}
        if paired:
            graph = reify(fact_index, fact_type, kg["num_entities"],
                          num_rels).to(device)
        for tag, m_paired in paired:
            r = rank_reifm(m_paired, graph, ent_seed, rel_seed,
                           torch.full_like(q_sub, SEED_SUBJECT), gold,
                           filt, class_of, ref_class, device)
            r_paired[tag] = r
            prow = row_from_ranks(a.dataset, target, f"{tag}_paired",
                                  class_size, r, None, r_deg)
            paired_rows.append(prow)
            print(f"[s7] {target:28s} paired {tag}: "
                  f"mrr={prow['pre_mrr']:.4f} h10={prow['pre_h10']:.4f}"
                  f"[{prow['pre_h10_lo']:.4f},{prow['pre_h10_hi']:.4f}]",
                  flush=True)

        if a.dump_ranks:
            os.makedirs(a.dump_ranks, exist_ok=True)
            fn = os.path.join(a.dump_ranks,
                              f"{a.dataset}_{target.replace('.', '_')}.json")
            with open(fn, "w") as f:
                json.dump(dict(
                    dataset=a.dataset, fk=target, qseed=a.qseed,
                    subsampled=subsampled,
                    queries=torch.stack([ent_seed, gold]).t().tolist(),
                    ranks_ultra=(r_pre.tolist() if r_pre is not None
                                 else None),
                    ranks_ultra_randinit=(r_rand.tolist()
                                          if r_rand is not None else None),
                    ranks_degree=r_deg.tolist(),
                    # dict tag -> ranks (the 3g runs of 2026-09-02 wrote a
                    # bare list for the single run48 arm)
                    ranks_paired={t: r.tolist() for t, r in r_paired.items()}
                    or None), f)

        if a.out:  # incremental write, a mid-run crash keeps completed FKs
            write_csv(a.out, rows + paired_rows)

    if a.out and (rows or paired_rows):
        write_csv(a.out, rows + paired_rows)
        print(f"[s7] wrote {a.out} ({len(rows)} ULTRA rows, "
              f"{len(paired_rows)} paired rows)", flush=True)


if __name__ == "__main__":
    main()
