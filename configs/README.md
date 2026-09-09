# configs/

* `kg_models_frozen_recipe.json` — the recipe shared by the 15 KG models, restricted to the
  arguments of the released `scripts/train.py` (`frozen`), and the two that vary (`varying`:
  `backbone`, `seed`).
* `kg_models_args.json` — the complete argument block stored with each of the 15 runs
  (`results.json` → `args`, 80 fields). It comes from the research code, whose training
  script had many more options; every option absent from the released `train.py` was at its
  default (off) for these runs — the released code implements exactly that configuration.
* `generic_3kg_model.md` — the recipe of the generic 3-KG model of the probe.
* `monokg.md` — the alternative-training-graph runs of Appendix C.3.
