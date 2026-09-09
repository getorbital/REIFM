# The generic 3-KG model of the probe (Section 6 / Appendix D)

Backbone `ginemaxlm` (memory-lean GINE with mean+max aggregation), d=64, L=12,
dropout 0.2, query-conditioned readout (`--qc_readout --qc_complex`); corpus
FB15k-237 + WN18RR + CoDEx-Medium with 200/100/100 batches per epoch; constant
learning rate 5e-4, weight decay 0.01, edge dropout 0.15, gradient clipping 1.0,
batch 8 × 4 accumulation steps, up to 40 epochs within 21,600 s (22 completed);
selection on the mean validation MRR of WN18RRInductive:v1, FB15k237Inductive:v1,
NELLInductive:v1 (Tier-2).

    uv run python scripts/train.py --train FB15k237 WN18RR CoDExMedium \
        --batches_per_epoch 200 100 100 --backbone ginemaxlm --dim 64 --layers 12 \
        --dropout 0.2 --edge_dropout 0.15 --lr 5e-4 --wd 0.01 --grad_clip 1.0 \
        --bs 8 --accum_steps 4 --epochs 40 --max_train_secs 21600 \
        --qc_readout --qc_complex --valid_queries 1000 --valid_every 1 \
        --select_eval WN18RRInductive:v1 FB15k237Inductive:v1 NELLInductive:v1 \
        --seed 0 --out generic3kg_seedA

Released as `checkpoints/generic3kg_seed{A,B,C}` (A = original run; B, C =
re-pretrainings with seeds 0 and 2 of the same recipe; the seed-1 draw failed the
95 % sanity gate on the selection score and is not used as a probe model).
