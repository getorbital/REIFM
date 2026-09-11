---
license: mit
library_name: pytorch
tags:
  - knowledge-graph
  - link-prediction
  - zero-shot
  - graph-neural-network
  - reification
  - arxiv:2609.11347
---

# REIFM checkpoints — Reification as a Transferable Vocabulary

Checkpoints of the paper **Reification as a Transferable Vocabulary: Zero-Shot
Link Prediction with Vanilla GNNs** (Camille Pradel, Matr, 2026; arXiv:2609.11347,
https://arxiv.org/abs/2609.11347). Code and evaluation pipeline:
https://github.com/getorbital/REIFM.

All models read *reified* graphs (every fact is a node linked to its subject,
object and relation-type node through six meta-relations; relation types are
anonymous shared nodes) and are applied zero-shot to unseen graphs: no
fine-tuning, no target-side adaptation, no entity or relation embeddings.

## Files

| directory | model | training data | selection | notes |
|---|---|---|---|---|
| `gat_seed{0,1,2}/` | GAT, PyG `GATConv` (4 heads, edge features), d=64, L=12 | FB15k-237-Inductive v1 train (4,245 triples) | validation MRR on the training graph's own validation split (Tier-1) | headline model of the paper (Tables 2–4) |
| `ginemax_seed{0,1,2}/` | GINE variant with mean+max aggregation (`reifm.models.ReifiedMeanMaxConv`), d=64, L=12 | same | same | |
| `gine_sum_seed{0,1,2}/` | GINE, PyG `GINEConv`, sum aggregation, d=64, L=12 | same | same | |
| `sage_seed{0,1,2}/` | GraphSAGE, PyG `SAGEConv` (no edge features), d=64, L=12 | same | same | |
| `rgcn_seed{0,1,2}/` | R-GCN, PyG `RGCNConv` over the 6 meta-relations, d=64, L=12 | same | same | |
| `corpus_recipe_1kg_seed{0,1,2}/` | same architecture and recipe as the generic model, trained on FB15k-237-Inductive v1 alone | FB15k-237-Inductive v1 train | Tier-1 (FB15k-237-Ind v1 validation) | the "ours, 1 KG, corpus recipe" column of Appendix D |
| `generic3kg_seed{A,B,C}/` | GINE mean+max (memory-lean implementation `ginemaxlm`), d=64, L=12, **query-conditioned readout** | FB15k-237 + WN18RR + CoDEx-Medium (transductive) | mean validation MRR of FB15k-237-Ind v1, WN18RR-Ind v1, NELL-995-Ind v1 (Tier-2, validation graphs distinct from the training graphs, never a test split) | the generic model of the KG→RDB probe (Section 6 / Appendix D); seed A is the original run, B and C re-pretrainings; a fourth draw failed the pretraining sanity gate and is not released as a probe model |

Every directory holds `best.pt` (state dict) and `results.json` (the full
argument block, per-epoch training history, validation-selection trace).

## Training recipe (15 KG models)

Dimension 64, 12 layers, dropout 0.2, cosine schedule from 5·10⁻⁴, weight
decay 0.01, batch 16 queries, full-graph propagation, plain readout (an MLP
over the candidate state), cross-entropy over all entities with known answers
masked; 20 epochs capped at 1,800 s on one NVIDIA A100 80 GB (the cap binds
for the GAT only: 14 epochs). Seeds {0, 1, 2} fixed a priori; no seed was
rejected or replaced. Only `backbone` and `seed` differ across the 15 runs
(`results.json` → `args`).

## Expected numbers

Filtered MRR, zero-shot, mean over the 3 seeds (paper Table 2):

| model | inductive-(e) (12) | inductive-(e,r) (13) | extended (15) | all 40 |
|---|---|---|---|---|
| GAT | 0.5457 | 0.3300 | 0.2281 | 0.3565 |
| GINE (mean+max) | 0.5276 | 0.3148 | 0.1807 | 0.3284 |
| GINE (sum) | 0.5069 | 0.2918 | 0.1987 | 0.3214 |
| GraphSAGE | 0.4676 | 0.2801 | 0.1625 | 0.2923 |
| R-GCN | 0.4344 | 0.2005 | 0.1055 | 0.2350 |
| ULTRA-3g (local re-run, reference) | 0.5224 | 0.3446 | 0.2786 | 0.3732 |

Per-seed and per-split values: `results/broad_eval_matrix.csv` in the code
repository. To reproduce one cell:

```bash
uv run python scripts/eval_ckpt.py checkpoints/gat_seed0/best.pt \
    --eval WN18RRInductive:v1 --backbone gat --dim 64 --layers 12
```

Loading is strict (`load_state_dict(strict=True)`): the architecture flags
must match the ones in `results.json`.

## Known limitation

On FBNELL (the one benchmark of the 40 whose inference graph contains isolated
entities), all 15 KG models collapse to MRR 0.08–0.11 with Hits@1 = 0: the five
isolated entities, unreachable by propagation and identical under the plain
readout, preempt ranks 1–5 of every query. The paper analyses this in
Section 5.3.

## License

MIT. Training data: FB15k-237-Inductive (GraIL), FB15k-237, WN18RR,
CoDEx-Medium, downloaded from their original sources through ULTRA's dataset
loaders; not redistributed here.
