#!/usr/bin/env bash
# Integrity check of the released checkpoints against the frozen results of the
# paper: every KG checkpoint is evaluated on 5 splits and compared with
# results/broad_eval_matrix.csv. Tolerance 0.005 MRR (the pre-registered threshold
# of the paper's code non-drift control): on the two smallest splits GPU evaluation
# is not deterministic (NELL-995-v1: 30-40 of 402 query ranks change between two
# runs of the same checkpoint, up to +/-0.006 MRR). ~1 h on an A100.
#   bash scripts/integrity_check.sh            # default 5 splits
#   SPLITS="FB15k237Inductive:v1" bash scripts/integrity_check.sh
set -uo pipefail
cd "$(dirname "$0")/.."
DEVICE=${DEVICE:-cuda}
SPLITS=${SPLITS:-"FB15k237Inductive:v1 WN18RRInductive:v1 NELLInductive:v1 FBIngram:25 NLIngram:0"}
TOL=${TOL:-0.005}
fail=0
for bb in gat ginemax gine_sum sage rgcn; do
  BBF="--backbone $bb"; [ "$bb" = gine_sum ] && BBF="--backbone gine --agg sum"
  for s in 0 1 2; do
    d=checkpoints/${bb}_seed$s
    [ -f "$d/best.pt" ] || { echo "MISSING $d/best.pt"; fail=1; continue; }
    uv run python scripts/eval_ckpt.py "$d/best.pt" --eval $SPLITS $BBF --dim 64 --layers 12 \
      --device "$DEVICE" --eval_bs 16 --out integrity > "$d/integrity.log" 2>&1 || { echo "EVAL FAILED $d (see $d/integrity.log)"; fail=1; continue; }
    uv run python - "$bb" "$s" "$d/integrity.json" "$TOL" <<'EOF' || fail=1
import csv, json, sys
bb, seed, path, tol = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
got = json.load(open(path))["test"]
ref = {r["split"]: float(r["mrr"]) for r in csv.DictReader(open("results/broad_eval_matrix.csv"))
       if r["backbone"] == bb and r["seed"] == seed}
bad = 0
for split, m in got.items():
    d = m["mrr"] - ref[split]
    flag = "ok" if abs(d) <= tol else "DIFF"
    bad += flag == "DIFF"
    print(f"{bb}_seed{seed} {split}: paper {ref[split]:.4f} reproduced {m['mrr']:.4f} Δ {d:+.4f} {flag}")
sys.exit(1 if bad else 0)
EOF
  done
done
[ $fail -eq 0 ] && echo "INTEGRITY CHECK PASSED (tolerance $TOL)" || { echo "INTEGRITY CHECK FAILED"; exit 1; }
