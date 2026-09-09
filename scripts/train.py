"""Train ReiFM on one or more KGs and evaluate (incl. zero-shot) on others.

The 15 KG models of the paper (configs/kg_models_frozen_recipe.json):
  uv run python scripts/train.py --train FB15k237Inductive:v1 --backbone gat --seed 0 \
      --dim 64 --layers 12 --dropout 0.2 --cosine --lr 5e-4 --bs 16 --epochs 20 \
      --max_train_secs 1800 --valid_queries 1000 --select_eval FB15k237Inductive:v1 \
      --eval FB15k237Inductive:v1 --out gat_seed0
The generic 3-KG model of the probe: configs/generic_3kg_model.md.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reifm.datasets import load_dataset, strip_inverse_facts  # noqa: E402
from reifm.engine import (build_filter_index, build_queries, evaluate,  # noqa: E402
                          evaluate_sampled, train_epoch_mixed, train_epoch_sampled)
from reifm.models import ReiFM  # noqa: E402
from reifm.reify import reify  # noqa: E402


def parse_spec(spec):
    name, _, version = spec.partition(":")
    return name, version or None


def dedup_edges(edge_index, edge_type):
    key = torch.stack([edge_index[0], edge_index[1], edge_type])
    uniq = torch.unique(key, dim=1)
    return uniq[:2], uniq[2]


def test_filter_edges(name, num_relations_with_inv, va, te, fi, ft):
    """Filtered-ranking edge set for a TEST eval, matching ULTRA's protocol
    (vendor/ULTRA/script/run.py): inference-graph facts + test targets, PLUS
    the valid split for datasets whose valid lives on the inference graph
    (InGram / ILPC — GraIL valids live on the train graph and must NOT be
    added)."""
    parts_i = [fi, te.target_edge_index]
    parts_t = [ft, te.target_edge_type]
    if "Ingram" in name or "ILPC" in name:
        vfi, vft, _ = strip_inverse_facts(va, num_relations_with_inv)
        parts_i += [vfi, va.target_edge_index]
        parts_t += [vft, va.target_edge_type]
    return dedup_edges(torch.cat(parts_i, dim=1), torch.cat(parts_t))


def prepare_split(data, num_relations_with_inv):
    """Reified fact graph + queries + the fact edges (for filters/lookups)."""
    fact_index, fact_type, R = strip_inverse_facts(data, num_relations_with_inv)
    graph = reify(fact_index, fact_type, data.num_nodes, R)
    queries = build_queries(data.target_edge_index, data.target_edge_type)
    return graph, queries, (fact_index, fact_type)


def build_fact_lookup(fact_index, fact_type):
    lookup = defaultdict(list)
    for i, (h, t, r) in enumerate(zip(fact_index[0].tolist(),
                                      fact_index[1].tolist(),
                                      fact_type.tolist())):
        lookup[(h, r, t)].append(i)
    return lookup


def facts_for_queries(target_edge_index, target_edge_type, fact_lookup):
    """fact node id (or -1) for each query produced by build_queries."""
    ids = []
    for h, t, r in zip(target_edge_index[0].tolist(),
                       target_edge_index[1].tolist(),
                       target_edge_type.tolist()):
        match = fact_lookup.get((h, r, t))
        ids.append(match[0] if match else -1)
    ids = torch.tensor(ids)
    return torch.cat([ids, ids])  # tail queries then head queries


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", nargs="+", required=True,
                   help="training graphs, e.g. FB15k237Inductive:v1 or FB15k237 WN18RR CoDExMedium")
    p.add_argument("--eval", nargs="+", default=[], help="test splits evaluated with best.pt")
    p.add_argument("--select_eval", nargs="+", default=[],
                   help="validation split(s) used to select best.pt (never a test split)")
    p.add_argument("--backbone", default="gat",
                   choices=["gat", "gine", "ginemax", "sage", "rgcn", "ginemaxlm"])
    p.add_argument("--agg", default="sum", choices=["sum", "mean"],
                   help="aggregation of the PyG gine / sage convolutions")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--edge_dropout", type=float, default=0.0,
                   help="training-time fact dropout (dense path / ginemaxlm only)")
    p.add_argument("--qc_readout", action=argparse.BooleanOptionalAction, default=False,
                   help="query-conditioned readout (generic 3-KG model; ginemaxlm only)")
    p.add_argument("--qc_complex", action=argparse.BooleanOptionalAction, default=False,
                   help="ComplEx-style term of the query-conditioned readout")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--max_train_secs", type=float, default=None,
                   help="wall-clock training budget (evaluation time excluded); "
                        "stops after the epoch that exceeds it")
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--eval_bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--cosine", action=argparse.BooleanOptionalAction, default=False,
                   help="cosine learning-rate schedule over --epochs")
    p.add_argument("--pos_mask", action=argparse.BooleanOptionalAction, default=True,
                   help="mask the other known true answers of a query out of the softmax")
    p.add_argument("--grad_ckpt", action=argparse.BooleanOptionalAction, default=False,
                   help="gradient checkpointing (memory)")
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--batches_per_epoch", type=int, nargs="+", default=None,
                   help="cap on batches per epoch, one value or one per --train graph")
    p.add_argument("--valid_every", type=int, default=1)
    p.add_argument("--valid_queries", type=int, default=None,
                   help="subsample of validation queries used for selection")
    p.add_argument("--no_train_valid", action=argparse.BooleanOptionalAction, default=False,
                   help="skip the validation eval on the training graphs")
    p.add_argument("--sampled", action=argparse.BooleanOptionalAction, default=False,
                   help="neighbour-sampled ego-subgraph regime (Appendix C.3)")
    p.add_argument("--num_hops", type=int, default=4)
    p.add_argument("--fanout", type=int, default=15)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--resume", default=None, help="<out>/state.pt to resume from")
    p.add_argument("--save_epochs", action=argparse.BooleanOptionalAction, default=False,
                   help="also save a snapshot per epoch (epoch_<k>.pt)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if (args.qc_readout or args.edge_dropout > 0) and args.backbone != "ginemaxlm":
        raise SystemExit("[reifm] --qc_readout / --edge_dropout exist on the dense "
                         "full-graph path only (backbone ginemaxlm)")
    device = torch.device(args.device)
    run_name = args.out or time.strftime("run_%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[reifm] device={device} out={out_dir}")
    print(f"[reifm] args: {vars(args)}")

    # ---- training graphs
    train_sets = []
    for spec in args.train:
        name, version = parse_spec(spec)
        tr, va, te, R2 = load_dataset(name, version)
        graph, queries, (fi, ft) = prepare_split(tr, R2)
        lookup = build_fact_lookup(fi, ft)
        qfacts = facts_for_queries(tr.target_edge_index, tr.target_edge_type, lookup)
        pos_filter = build_filter_index(fi, ft)
        graph = graph.to(device)
        # valid eval on the train graph
        va_queries = build_queries(va.target_edge_index, va.target_edge_type)
        if args.valid_queries is not None and va_queries[0].size(0) > args.valid_queries:
            keep = torch.randperm(va_queries[0].size(0))[: args.valid_queries]
            va_queries = tuple(t[keep] for t in va_queries)
        filt_index, filt_type = dedup_edges(
            torch.cat([fi, va.target_edge_index], dim=1),
            torch.cat([ft, va.target_edge_type]))
        va_filter = build_filter_index(filt_index, filt_type)
        train_sets.append(dict(spec=spec, graph=graph, queries=queries,
                               qfacts=qfacts, pos_filter=pos_filter,
                               va_queries=va_queries, va_filter=va_filter))
        print(f"[reifm] train {spec}: {graph.num_entities} entities, "
              f"{graph.num_facts} facts, {graph.num_rel_types} rel types, "
              f"{6*graph.num_facts} reified edges, "
              f"{queries[0].size(0)} train queries")

    # ---- eval graphs (test split, zero-shot when the graph is not a training graph)
    eval_sets = []
    for spec in args.eval:
        name, version = parse_spec(spec)
        tr, va, te, R2 = load_dataset(name, version)
        graph, queries, (fi, ft) = prepare_split(te, R2)
        filt_index, filt_type = test_filter_edges(name, R2, va, te, fi, ft)
        filt = build_filter_index(filt_index, filt_type)
        eval_sets.append(dict(spec=spec, graph=graph.to(device),
                              queries=queries, filter=filt))
        print(f"[reifm] eval {spec}: {graph.num_entities} entities, "
              f"{graph.num_facts} facts, {queries[0].size(0)} test queries")

    # ---- selection graphs: best.pt is chosen on these VALIDATION splits (never test)
    select_sets = []
    for spec in args.select_eval:
        name, version = parse_spec(spec)
        tr, va, te, R2 = load_dataset(name, version)
        graph, queries, (fi, ft) = prepare_split(va, R2)
        if args.valid_queries is not None and queries[0].size(0) > args.valid_queries:
            keep = torch.randperm(queries[0].size(0))[: args.valid_queries]
            queries = tuple(t[keep] for t in queries)
        filt_index, filt_type = dedup_edges(
            torch.cat([fi, va.target_edge_index], dim=1),
            torch.cat([ft, va.target_edge_type]))
        filt = build_filter_index(filt_index, filt_type)
        select_sets.append(dict(spec=spec, graph=graph.to(device),
                                queries=queries, filter=filt))
        print(f"[reifm] select-eval {spec}: {graph.num_entities} entities, "
              f"{queries[0].size(0)} valid queries")

    model = ReiFM(backbone=args.backbone, dim=args.dim, num_layers=args.layers,
                  dropout=args.dropout, agg=args.agg, qc_readout=args.qc_readout,
                  qc_complex=args.qc_complex, edge_dropout=args.edge_dropout).to(device)
    model.grad_checkpoint = args.grad_ckpt
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[reifm] model: backbone={args.backbone} dim={args.dim} "
          f"layers={args.layers} params={n_params}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 10) if args.cosine else None)

    # per-dataset batch budget: one value (broadcast) or one per --train spec
    bpe = args.batches_per_epoch
    if bpe is not None:
        if len(bpe) == 1:
            bpe = bpe * len(train_sets)
        elif len(bpe) != len(train_sets):
            raise SystemExit(f"--batches_per_epoch needs 1 or {len(train_sets)} "
                             f"values (got {len(bpe)})")
        print(f"[reifm] batch budget per dataset: "
              f"{dict(zip((ts['spec'] for ts in train_sets), bpe))}")

    history = []
    best_va = -1.0
    train_secs_used = 0.0  # cumulative TRAINING wall-clock (excl. eval/startup)
    start_epoch = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        best_va = state.get("best_va", -1.0)
        history = state.get("history", [])
        print(f"[reifm] RESUMED from {args.resume}: epoch {start_epoch}, "
              f"best={best_va:.4f} (note: RNG/data order not restored)")
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        if not args.pos_mask:
            for ts in train_sets:
                ts["pos_filter"] = None
        reach = None
        train_t0 = time.time()
        if args.sampled:
            losses, reach = train_epoch_sampled(
                model, optimizer, train_sets, args.bs, device,
                args.num_hops, args.fanout, batches_per_graph=bpe,
                num_workers=args.num_workers)
        else:
            losses = train_epoch_mixed(model, optimizer, train_sets, args.bs,
                                       device, batches_per_graph=bpe,
                                       accum_steps=args.accum_steps,
                                       grad_clip=args.grad_clip)
        train_secs_used += time.time() - train_t0  # eval below is NOT counted
        losses = {k: round(v, 4) for k, v in losses.items()}
        if scheduler is not None:
            scheduler.step()
        va_metrics = {}
        sel_metrics = {}
        if (epoch + 1) % args.valid_every == 0 or epoch == args.epochs - 1:
            for ts in ([] if args.no_train_valid else train_sets):
                if args.sampled:
                    m = evaluate_sampled(model, ts["graph"], ts["va_queries"],
                                         ts["va_filter"], args.eval_bs, device,
                                         args.num_hops, args.fanout)
                else:
                    m = evaluate(model, ts["graph"], ts["va_queries"],
                                 ts["va_filter"], args.eval_bs, device)
                va_metrics[ts["spec"]] = {k: round(v, 4) for k, v in m.items()}
                if device.type == "mps":
                    torch.mps.empty_cache()
            for ss in select_sets:
                m = evaluate(model, ss["graph"], ss["queries"], ss["filter"],
                             args.eval_bs, device)
                sel_metrics[ss["spec"]] = {k: round(v, 4) for k, v in m.items()}
                if device.type == "mps":
                    torch.mps.empty_cache()
        dt = time.time() - t0
        reach_str = f" reach={reach:.2f}" if reach is not None else ""
        sel_str = f" select={sel_metrics}" if sel_metrics else ""
        print(f"[reifm] epoch {epoch}: loss={losses}{reach_str} valid={va_metrics}"
              f"{sel_str} ({dt:.1f}s)", flush=True)
        history.append(dict(epoch=epoch, loss=losses, valid=va_metrics,
                            select=sel_metrics, secs=dt, reach=reach))
        mean_va_mrr = (sum(m["mrr"] for m in va_metrics.values()) / len(va_metrics)
                       if va_metrics else None)
        # best.pt: mean MRR on the --select_eval validation splits, else the
        # training graphs' validation mean
        sel_mrr = (sum(m["mrr"] for m in sel_metrics.values()) / len(sel_metrics)
                   if sel_metrics else mean_va_mrr)
        if sel_mrr is not None and sel_mrr > best_va:
            best_va = sel_mrr
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
        # full resumable state, refreshed every epoch (--resume <out>/state.pt)
        torch.save(dict(epoch=epoch, model=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict() if scheduler else None,
                        best_va=best_va, history=history, args=vars(args)),
                   os.path.join(out_dir, "state.pt"))
        if args.save_epochs:
            torch.save(model.state_dict(),
                       os.path.join(out_dir, f"epoch_{epoch}.pt"))
        if args.max_train_secs is not None and train_secs_used >= args.max_train_secs:
            print(f"[reifm] training budget reached: {train_secs_used:.0f}s "
                  f">= {args.max_train_secs:.0f}s — stopping after epoch "
                  f"{epoch} (eval excluded from budget)", flush=True)
            break

    # ---- final test evals with best checkpoint
    best_path = os.path.join(out_dir, "best.pt")
    if not os.path.exists(best_path):
        torch.save(model.state_dict(), best_path)
        print("[reifm] WARNING: no selection eval ran — saved final-epoch "
              "weights as best.pt", flush=True)
    model.load_state_dict(torch.load(best_path, weights_only=True))

    results = {}
    for es in eval_sets:
        t0 = time.time()
        if args.sampled:
            m = evaluate_sampled(model, es["graph"], es["queries"], es["filter"],
                                 args.eval_bs, device, args.num_hops, args.fanout)
        else:
            m = evaluate(model, es["graph"], es["queries"], es["filter"],
                         args.eval_bs, device)
        results[es["spec"]] = {k: round(v, 4) for k, v in m.items()}
        print(f"[reifm] TEST {es['spec']}: {results[es['spec']]} "
              f"({time.time()-t0:.1f}s)", flush=True)

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(dict(args=vars(args), history=history, test=results,
                       params=n_params), f, indent=2)
    print(f"[reifm] saved {out_dir}/results.json")


if __name__ == "__main__":
    main()
