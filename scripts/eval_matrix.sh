#!/usr/bin/env bash
# Evaluate the 15 released checkpoints (checkpoints/<backbone>_seed<k>/best.pt) on the
# 40 inductive benchmarks, one invocation per (model, split); then run
# scripts/build_matrix.py to assemble results/broad_eval_matrix.csv and
# results/backbone_matrix.csv (paper Tables 2-4, C1, C2).
#
# ÉVAL ONLY — aucun train, aucune sélection (les ckpts sont gelés, sélection
# Tier-1 déjà faite sur la valid FB-v1 au moment du train).
#
# Flags : les 5 backbones du papier court sont tous en readout PLAIN, donc les
# défauts d'eval_ckpt.py (qc_* off, use_cofact off, rel_pe 0, rel_graph_layers
# -1, class_node off, ent_features vide) sont EXACTEMENT le protocole ; seuls
# --backbone/--agg/--dim/--layers sont passés. `load_state_dict` est strict :
# un mismatch de flags crashe, il ne dégrade pas silencieusement.
#
# GRANULARITÉ : UNE INVOCATION PAR (modèle, split), pour les 41.
# Raison — `eval_ckpt.py` n'a PAS de try/except par spec et n'écrit son JSON
# qu'APRÈS la boucle : un seul split qui casse dans une invocation groupée
# perdrait tous les splits de cette invocation. Et cette VM a déjà tué une
# chaîne multi-heures (2026-07-27). Le surcoût est le démarrage répété
# (imports + init CUDA + chargement du ckpt) ; on l'accepte pour que ni un
# crash ni un arrêt de VM ne coûte plus d'UN split. Chaque invocation garde
# son log complet (les tracebacks ne sont pas filtrés).
#
# Usage: nohup bash scripts/eval_matrix.sh > results/eval_matrix.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
DEVICE=${DEVICE:-cuda}
ROOT=checkpoints
LOGDIR=results/eval_logs; mkdir -p "$LOGDIR"

# PID de ce runner, pour que le watchdog teste sa vivacité SANS scanner les
# cmdlines (un `pgrep -f papershort_s2.sh` matche n'importe quel shell dont la
# ligne de commande mentionne le script — y compris celui d'un agent — et le
# watchdog ne relancerait alors jamais rien : vérifié, ce piège est réel).

PANEL_A="FB15k237Inductive:v1 FB15k237Inductive:v2 FB15k237Inductive:v3 FB15k237Inductive:v4 \
WN18RRInductive:v1 WN18RRInductive:v2 WN18RRInductive:v3 WN18RRInductive:v4 \
NELLInductive:v1 NELLInductive:v2 NELLInductive:v3 NELLInductive:v4"
PANEL_B="FBIngram:25 FBIngram:50 FBIngram:75 FBIngram:100 \
WKIngram:25 WKIngram:50 WKIngram:75 WKIngram:100 \
NLIngram:0 NLIngram:25 NLIngram:50 NLIngram:75 NLIngram:100"
EXT16="ILPC2022:small ILPC2022:large HM:1k HM:3k HM:5k HM:indigo \
FBNELL:FBNELL_v1 Metafam:Metafam \
WikiTopicsMT1:tax WikiTopicsMT1:health WikiTopicsMT2:org WikiTopicsMT2:sci \
WikiTopicsMT3:art WikiTopicsMT3:infra WikiTopicsMT4:sci WikiTopicsMT4:health"

bb_flags() {  # clé de config -> flags backbone
  case "$1" in
    gine_sum) echo "--backbone gine --agg sum" ;;
    gat)      echo "--backbone gat --agg sum" ;;
    sage)     echo "--backbone sage --agg sum" ;;
    rgcn)     echo "--backbone rgcn --agg sum" ;;
    ginemax)  echo "--backbone ginemax --agg sum" ;;
    *)        echo "UNKNOWN" ;;
  esac
}

MODELS=${MODELS:-"gat ginemax gine_sum sage rgcn"}
SEEDS=${SEEDS:-"0 1 2"}
FAILLOG=results/eval_failures.txt

eval_one() {  # $1=dir  $2=key  $3=seed  $4=spec  $5=outprefix
  local D="$1" key="$2" s="$3" spec="$4" pref="$5"
  local safe; safe=$(echo "$spec" | tr ':' '_')
  local out="${pref}${safe}"
  if [ -f "$D/${out}.json" ]; then
    echo "[s2] $key seed$s $spec — déjà fait"; return 0
  fi
  local lg="$LOGDIR/${key}_seed${s}_${safe}.log"
  echo "=== $(date -Is) [s2] $key seed$s : $spec ==="
  if uv run python scripts/eval_ckpt.py "$D/best.pt" --eval "$spec" \
       $(bb_flags "$key") --dim 64 --layers 12 --device "$DEVICE" \
       --eval_bs 16 --dump_ranks --out "$out" > "$lg" 2>&1; then
    grep -E '\[eval_ckpt\] (TEST|saved)' "$lg" | head -2
  else
    echo "[s2] ÉCHEC $key seed$s $spec — log $lg"
    echo "$key seed$s $spec  (log $lg)" >> "$FAILLOG"
    tail -12 "$lg" | sed 's/^/    | /'
  fi
}

for key in $MODELS; do
  for s in $SEEDS; do
    D="$ROOT/${key}_seed${s}"
    if [ ! -f "$D/best.pt" ]; then
      echo "[s2] !! ABSENT $D/best.pt — (backbone,seed) manquant, on saute"
      echo "$key seed$s — best.pt ABSENT" >> "$FAILLOG"; continue
    fi

    # 25-suite : réutilise l'éval gelée P0-3ter quand elle existe (GAT 0/1/2,
    # même harnais, même machine, avec ranks ; stabilité du chemin d'éval
    # établie par le gate S-1, max Δ 0.0009). Sinon, split par split.
    if [ -f "$D/eval25_frozen_p03ter.json" ]; then
      echo "[s2] $key seed$s : 25-suite = eval25_frozen_p03ter (gelée, réutilisée)"
    else
      for spec in $PANEL_A $PANEL_B; do
        eval_one "$D" "$key" "$s" "$spec" "s2_25_"
      done
    fi

    # 16 étendus : toujours split par split, pour les 15 modèles
    for spec in $EXT16; do
      eval_one "$D" "$key" "$s" "$spec" "s2_ext_"
    done

    echo "=== $(date -Is) [s2] $key seed$s TERMINÉ (sync GCS) ==="
  done
done
echo "=== S-2 ÉVALS COMPLÈTES ($(date -Is)) ==="
[ -f "$FAILLOG" ] && { echo "--- ÉCHECS consignés ---"; cat "$FAILLOG"; }
exit 0
