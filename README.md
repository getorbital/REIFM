# REIFM — Reification as a Transferable Vocabulary

Code, checkpoints and evaluation pipeline for the paper

> **Reification as a Transferable Vocabulary: Zero-Shot Link Prediction with Vanilla GNNs**
> Camille Pradel (Matr), 2026. arXiv: [2609.11347](https://arxiv.org/abs/2609.11347).

The paper shows that five unmodified textbook GNNs (GAT, GINE with sum and with
mean+max aggregation, GraphSAGE, R-GCN), trained for 30 minutes on one small
knowledge graph, transfer zero-shot to 40 inductive link-prediction benchmarks
once the input graph is *reified*: every fact becomes a node, connected to its
subject, object and relation type through a fixed vocabulary of six
meta-relations. The best of them matches ULTRA on ULTRA's own evaluation suite.
The same vocabulary reads a relational database (a preliminary probe).

This repository contains exactly what is needed to reproduce the numbers of the
paper, and nothing else: the `reifm` package (reification, the five backbones
and the memory-lean variant, the query-conditioned wrapper with its two
readouts, training/evaluation engine), the scripts that produced every table,
the frozen result CSVs, and pointers to the released checkpoints. The code was
pruned from the research repository to these paths only; the pruned code
reproduces the research code's per-query ranks bit for bit on every released
checkpoint (`EQUIVALENCE.md`).

## Contents

| path | what |
|---|---|
| `reifm/` | the package: `reify.py` (the representation), `models.py` (backbones and the query-conditioned wrapper), `engine.py` (filtered ranking, training/eval loops), `datasets.py` (KG loading through ULTRA's dataset classes), `relbench_reify.py` (database → reified graph) |
| `scripts/train.py` | training (Table 2 models: `--max_train_secs 1800`, see `configs/`) |
| `scripts/eval_ckpt.py` | **the evaluation harness**: filtered MRR / Hits@k of a checkpoint on any split, with per-query rank dumps |
| `scripts/eval_matrix.sh`, `scripts/build_matrix.py` | evaluate the 15 checkpoints on the 40 benchmarks and assemble `results/broad_eval_matrix.csv` / `backbone_matrix.csv` (Tables 2–4, C1, C2) |
| `scripts/ultra_baseline.py`, `scripts/ultra_dump_ranks.py` | the locally re-run ULTRA-3g baseline (same splits, same filter, same machine) |
| `scripts/probe_rdb.py`, `scripts/probe_rdb_ultra.py`, `scripts/probe_all.sh` | the KG→RDB probe of Section 6 / Appendix D (reified models; ULTRA on the raw foreign-key graph) |
| `scripts/convert_to_lowmem.py` | converts a `ginemax` checkpoint to the memory-lean `ginemaxlm` implementation (same function; needed for the 10M-node rel-stack graph) |
| `scripts/integrity_check.sh` | re-evaluates every released KG checkpoint on 5 splits and compares with the frozen CSV |
| `scripts/figures.py` | Figures 2 and 3 from the CSVs and rank dumps |
| `configs/` | the frozen argument block of the 15 KG models and the recipe of the generic 3-KG model |
| `results/` | every CSV cited in the paper (frozen) |
| `checkpoints/` | download target for the released checkpoints (see below) |
| `vendor/` | third-party code, installed by `scripts/setup_vendor.sh` (not vendored in this repository) |

## Installation

Python ≥ 3.13, [uv](https://docs.astral.sh/uv/). CUDA is only needed to
reproduce the timings; every evaluation also runs on CPU (slowly) and on
Apple silicon.

```bash
git clone https://github.com/getorbital/REIFM.git && cd REIFM
uv sync                          # torch, torch-geometric, relbench, …
bash scripts/setup_vendor.sh     # clones ULTRA at the pinned commit into vendor/ULTRA
```

`vendor/ULTRA` provides the dataset loaders for all 40 benchmarks (they
download on first use into `data/kg-datasets/`) and the three released ULTRA
checkpoints (`vendor/ULTRA/ckpts/`). ULTRA is MIT-licensed; we pin the upstream
commit we evaluated against.

The optional Rust sampler (`reifm_sampler/`, needed only for the
sampled-subgraph regime of Appendix C.3) builds with `uv run maturin develop`
inside that directory.

## Checkpoints

The 15 KG models of the paper (5 backbones × seeds {0, 1, 2}), the generic
3-KG model of the probe (3 pretraining seeds) and the three one-graph models
trained with the generic model's recipe (Appendix D) are released on Hugging Face:
**`https://huggingface.co/itmatr/REIFM`**, one repository, one
model card. Download them into `checkpoints/`:

```bash
bash scripts/download_checkpoints.sh          # -> checkpoints/<backbone>_seed<k>/best.pt (+ results.json)
```

Each checkpoint ships with the `results.json` written at training time (full
argument block, training history, validation-selection trace).

## Reproducing the paper

One table, one command. Every command reads `checkpoints/` and writes under
`results/`; the frozen CSVs of the paper are in `results/` already, so each
command can be checked against them.

| paper item | command |
|---|---|
| one model on one split | `uv run python scripts/eval_ckpt.py checkpoints/gat_seed0/best.pt --eval WN18RRInductive:v1 --backbone gat --dim 64 --layers 12 --dump_ranks` |
| Tables 2, 3, 4, C1, C2 (15 models × 40 splits) | `bash scripts/eval_matrix.sh && uv run python scripts/build_matrix.py` |
| ULTRA-3g column | `uv run python scripts/ultra_baseline.py FB15k237Inductive v1 ckpts/ultra_3g.pth` (one split; loop over the 40) |
| Table C3 (alternative training graphs) | `results/monokg_transfer.csv` (frozen); training commands in `configs/monokg.md` |
| Section 6 / Appendix D (KG→RDB probe) | `bash scripts/probe_all.sh` (reified models and ULTRA, both databases, paired queries) |
| Figures 2–3 | `uv run --with matplotlib python scripts/figures.py` |
| retrain one of the 15 models | `uv run python scripts/train.py --train FB15k237Inductive:v1 --backbone gat --seed 0 --dim 64 --layers 12 --dropout 0.2 --cosine --lr 5e-4 --bs 16 --epochs 20 --max_train_secs 1800 --valid_queries 1000 --select_eval FB15k237Inductive:v1 --eval FB15k237Inductive:v1 --out gat_seed0` |

Timings in the paper refer to one NVIDIA A100 80 GB. On that device the full
15 × 40 evaluation takes 15–25 hours; a single small split takes seconds to
minutes.

`scripts/integrity_check.sh` re-evaluates the 15 KG checkpoints on 5 splits and
compares them with the frozen CSV at the pre-registered tolerance of 0.005 MRR:
GPU evaluation is not deterministic on the two smallest splits (NELL-995-v1,
NL-InGram-0), where re-running the same checkpoint moves the MRR by up to
±0.006; on an A100 the check reproduces 44 of the 75 cells exactly and all
but one within 0.005.

### Protocol in one paragraph

Filtered ranking, pessimistic ties, absolute MRR; 40 inductive benchmarks (12
GraIL, 13 InGram, 15 extended; HM:indigo excluded on measured compute
grounds, see the paper); models selected on the training graph's own
validation split only; seeds fixed a priori; ULTRA re-run locally from its
released checkpoint with the same code path. `docs/PROTOCOL.md` spells out
the regime labels used in the tables.

## Citing

```bibtex
@misc{pradel2026reification,
  title         = {Reification as a Transferable Vocabulary: Zero-Shot Link Prediction with Vanilla GNNs},
  author        = {Camille Pradel},
  year          = {2026},
  eprint        = {2609.11347},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2609.11347}
}
```

## License

MIT (code and released weights). Third-party components keep their own
licenses: ULTRA (MIT), RelBench (MIT), PyTorch Geometric (MIT). The benchmark
datasets are downloaded from their original sources and are not redistributed
here.
