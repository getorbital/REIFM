# Alternative single training graphs (paper Section 5.4, Table C3)

Frozen results: `results/monokg_transfer.csv`. Training commands (GAT, same
recipe as the 15 models unless stated):

* **CoDEx-Small, gradient-step-matched** (3 seeds): the reference GAT trains
  14 epochs × 531 batches = 7,434 steps; CoDEx-Small costs 13× per step, so the
  1,800 s cap is replaced by the same number of steps and selection points:

      uv run python scripts/train.py --train CoDExSmall --backbone gat --seed 0 \
          --dim 64 --layers 12 --dropout 0.2 --cosine --lr 5e-4 --bs 16 \
          --epochs 14 --batches_per_epoch 531 --grad_ckpt \
          --valid_queries 1000 --select_eval CoDExSmall --eval CoDExSmall --out gat_codexs_seed0

* **ConceptNet100k and AristoV4, sampled regime** (1 seed each) and the
  reference retrained in that regime on FB15k-237-Inductive v1: add
  `--sampled --fanout 15 --num_hops 4` (requires the Rust sampler,
  `reifm_sampler/`). These columns are comparable to their own reference only.
