#!/usr/bin/env bash
# The KG->RDB probe of Section 6 / Appendix D, both databases, paired queries
# (60 per foreign key, seeded generator qseed 0): ULTRA-3g/4g/50g on the raw
# foreign-key graph and the reified models re-ranked on the same queries.
# Requires: checkpoints/ (download_checkpoints.sh), vendor/ULTRA (setup_vendor.sh),
# a GPU with 80 GB for rel-stack (rel-f1 also runs on CPU, slowly).
# Usage: nohup bash scripts/probe_all.sh > results/probe/probe_all.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/probe
DEVICE=${DEVICE:-cuda}
GEN=checkpoints/generic3kg_seedA/best.pt

for DB in rel-f1 rel-stack; do
  BS=4; TB=""; [ "$DB" = rel-stack ] && { BS=1; TB="--timebox_s 10800"; }
  for CK in ultra_3g ultra_4g ultra_50g; do
    OUT=results/probe/probe_${DB}_${CK/_/}.csv
    [ -f "$OUT" ] && { echo "[skip] $OUT"; continue; }
    uv run python scripts/probe_rdb_ultra.py --dataset "$DB" --device "$DEVICE" --bs $BS \
      --max_queries 60 --qseed 0 $TB --ultra_ckpt vendor/ULTRA/ckpts/${CK}.pth --ultra_tag "$CK" \
      $( [ "$CK" = ultra_3g ] && echo "--paired_ckpt $GEN --paired_tag run48 --paired_recipe run48" ) \
      --dump_ranks results/probe/ranks_${CK} --out "$OUT"
  done
  # the three seeds of the generic model, paired (seed A is the run48_paired row above)
  for S in B C; do
    OUT=results/probe/probe_${DB}_generic3kg_seed${S}_paired.csv
    [ -f "$OUT" ] && continue
    uv run python scripts/probe_rdb_ultra.py --dataset "$DB" --device "$DEVICE" --bs $BS --max_queries 60 --qseed 0 $TB \
      --no_ultra --paired_recipe run48 --paired_ckpt checkpoints/generic3kg_seed${S}/best.pt --paired_tag generic3kg_seed${S} --out "$OUT"
  done
  # one-KG arms: the Section 5 GINE (mean+max) and GAT checkpoints (paper recipe), 3 seeds
  for BB in ginemax gat; do
    [ "$DB" = rel-stack ] && [ "$BB" = gat ] && continue   # GAT does not fit rel-stack in 80 GB
    OUT=results/probe/probe_${DB}_${BB}_1kg_paired.csv
    [ -f "$OUT" ] && continue
    R=ginemax_s1; [ "$BB" = gat ] && R=gat_s1; [ "$DB" = rel-stack ] && [ "$BB" = ginemax ] && R=ginemaxlm_s1
    uv run python scripts/probe_rdb_ultra.py --dataset "$DB" --device "$DEVICE" --bs $BS --max_queries 60 --qseed 0 $TB \
      --no_ultra --paired_recipe $R \
      --paired_ckpt checkpoints/${BB}_seed0/best.pt checkpoints/${BB}_seed1/best.pt checkpoints/${BB}_seed2/best.pt \
      --paired_tag ${BB}_seed0 ${BB}_seed1 ${BB}_seed2 --out "$OUT"
  done
done
echo "=== probe_all done ($(date -u)) ==="
