"""Evaluate a saved ReiFM checkpoint on test splits, without retraining.

This is the evaluation harness behind every number of the paper.

Usage:
  uv run python scripts/eval_ckpt.py checkpoints/gat_seed0/best.pt \
      --eval FB15k237Inductive:v1 WN18RRInductive:v1 \
      --backbone gat --dim 64 --layers 12 [--device cuda] [--out name] [--dump_ranks]

Loading is strict: the architecture flags (--backbone, --agg, --dim, --layers,
--qc_readout, --qc_complex) must match the checkpoint's results.json.
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reifm.datasets import load_dataset  # noqa: E402
from reifm.engine import build_filter_index, evaluate, evaluate_sampled  # noqa: E402
from reifm.models import ReiFM  # noqa: E402

from train import dedup_edges, parse_spec, prepare_split, test_filter_edges  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--eval", nargs="+", required=True)
    p.add_argument("--backbone", default="gat",
                   choices=["gat", "gine", "ginemax", "sage", "rgcn", "ginemaxlm"])
    p.add_argument("--agg", default="sum", choices=["sum", "mean"])
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--qc_readout", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--qc_complex", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--split", default="test", choices=["test", "valid"],
                   help="valid = the same validation eval as train.py --select_eval "
                        "(the model-selection signal; never touches test)")
    p.add_argument("--eval_bs", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None,
                   help="write <ckpt dir>/<out>.json (and <out>_ranks.json with --dump_ranks)")
    p.add_argument("--sampled", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--num_hops", type=int, default=4)
    p.add_argument("--fanout", type=int, default=15)
    p.add_argument("--dump_ranks", action=argparse.BooleanOptionalAction, default=False,
                   help="also save the per-query filtered ranks")
    args = p.parse_args()

    device = torch.device(args.device)
    model = ReiFM(backbone=args.backbone, dim=args.dim, num_layers=args.layers,
                  agg=args.agg, qc_readout=args.qc_readout,
                  qc_complex=args.qc_complex).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    print(f"[eval_ckpt] loaded {args.ckpt}")

    results = {}
    ranks_dump = {}
    for spec in args.eval:
        name, version = parse_spec(spec)
        tr, va, te, R2 = load_dataset(name, version)
        if args.split == "valid":
            graph, queries, (fi, ft) = prepare_split(va, R2)
            filt_index, filt_type = dedup_edges(
                torch.cat([fi, va.target_edge_index], dim=1),
                torch.cat([ft, va.target_edge_type]))
        else:
            graph, queries, (fi, ft) = prepare_split(te, R2)
            # ULTRA-parity filter: includes the valid split on InGram/ILPC
            filt_index, filt_type = test_filter_edges(name, R2, va, te, fi, ft)
        filt = build_filter_index(filt_index, filt_type)
        t0 = time.time()
        if args.sampled:
            m = evaluate_sampled(model, graph.to(device), queries, filt,
                                 args.eval_bs, device, args.num_hops, args.fanout,
                                 return_ranks=args.dump_ranks)
        else:
            m = evaluate(model, graph.to(device), queries, filt, args.eval_bs, device,
                         return_ranks=args.dump_ranks)
        if args.dump_ranks:
            ranks_dump[spec] = m.pop("ranks")
        results[spec] = {k: round(v, 4) for k, v in m.items()}
        print(f"[eval_ckpt] {args.split.upper()} {spec}: {results[spec]} "
              f"({time.time()-t0:.1f}s)", flush=True)

    if args.out:
        out_path = os.path.join(os.path.dirname(args.ckpt), args.out + ".json")
        with open(out_path, "w") as f:
            json.dump(dict(ckpt=args.ckpt, args=vars(args), test=results), f, indent=2)
        print(f"[eval_ckpt] saved {out_path}")
        if args.dump_ranks:
            ranks_path = os.path.join(os.path.dirname(args.ckpt),
                                      args.out + "_ranks.json")
            with open(ranks_path, "w") as f:
                json.dump(dict(ckpt=args.ckpt, ranks=ranks_dump), f)
            print(f"[eval_ckpt] saved {ranks_path}")


if __name__ == "__main__":
    main()
