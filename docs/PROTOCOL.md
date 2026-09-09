# Evaluation protocol

* **Task.** Filtered link prediction, tail and head queries, on the test split of
  each benchmark's inference graph; the model sees the inference graph's fact
  edges only (the queried fact and the test/validation edges are never in the
  message-passing graph).
* **Metric.** MRR and Hits@k of the filtered rank; ties resolved pessimistically
  (a candidate scoring equal to the true answer counts against it). Aggregates are
  unweighted means over benchmarks; standard deviations are over the 3 seeds.
* **Benchmarks (40).** 12 GraIL inductive-(e) splits (FB15k-237, WN18RR,
  NELL-995 v1–v4); 13 InGram inductive-(e,r) splits (FB/WK 25–100, NL 0–100);
  15 extended splits (ILPC 2022 small/large, HM 1k/3k/5k, FBNELL, Metafam,
  WikiTopics MT1–4). HM:indigo (458 relations) is excluded on measured compute
  grounds and reported as out of scope. Loaders: ULTRA's `ultra/datasets.py`.
* **Regime labels** (`broad_eval_matrix.csv`, columns `regime_reifm`,
  `regime_ultra3g`): `train+select` = the training graph's own split;
  `in-family` = a benchmark whose parent KG is in the model's pretraining corpus;
  `new-KG` = family unseen; `in-family-partial` = FBNELL (Freebase and NELL).
  Our models saw FB15k-237-Inductive v1 only; ULTRA-3g saw FB15k-237, WN18RR and
  CoDEx-Medium.
* **Selection.** Tier-1: best validation MRR on the training graph's own
  validation split (1,000 queries), evaluated every epoch; never a test split.
  The generic 3-KG model of the probe uses Tier-2 selection (validation splits
  of three inductive benchmarks distinct from its training graphs).
* **Baseline.** ULTRA's released `ultra_3g.pth`, run locally through
  `scripts/ultra_baseline.py` on the same splits with the same filter; no number
  is quoted from the ULTRA paper.
* **Machine.** Every number of the paper was produced on one NVIDIA A100 80 GB,
  one job at a time. Evaluations on other devices reproduce the numbers up to
  CUDA nondeterminism (observed ≤ 0.001 MRR on re-evaluation).
