#!/usr/bin/env bash
# Download the released checkpoints from Hugging Face into checkpoints/.
#   15 KG models: <backbone>_seed<k>/best.pt (+ results.json), backbone in
#   {gat, ginemax, gine_sum, sage, rgcn}, k in {0,1,2}
#   generic 3-KG model of the probe: generic3kg_seed{A,B,C}/best.pt
#   one-graph models with the generic recipe (Appendix D): corpus_recipe_1kg_seed{0,1,2}/best.pt
set -euo pipefail
cd "$(dirname "$0")/.."
HF_REPO=${HF_REPO:-"itmatr/REIFM"}
BASE="https://huggingface.co/${HF_REPO}/resolve/main"
mkdir -p checkpoints
for bb in gat ginemax gine_sum sage rgcn; do
  for s in 0 1 2; do
    d="checkpoints/${bb}_seed${s}"; mkdir -p "$d"
    for f in best.pt results.json; do
      [ -f "$d/$f" ] || curl -fsSL "$BASE/${bb}_seed${s}/$f" -o "$d/$f"
    done
    echo "[download] $d"
  done
done
for tag in generic3kg_seedA generic3kg_seedB generic3kg_seedC corpus_recipe_1kg_seed0 corpus_recipe_1kg_seed1 corpus_recipe_1kg_seed2; do
  d="checkpoints/$tag"; mkdir -p "$d"
  for f in best.pt results.json; do
    [ -f "$d/$f" ] || curl -fsSL "$BASE/$tag/$f" -o "$d/$f"
  done
  echo "[download] $d"
done
