"""Thin Weights & Biases wrapper so every run (train + eval) is monitored.

Design goals:
- ON by default (the point is that NO run goes untracked), opt out with --no-wandb.
- Fail fast & loud at startup if W&B is enabled but not authenticated, so a
  long GPU job doesn't run for hours only to drop its metrics on the floor.
- A single place that flattens our nested metric dicts into W&B-friendly keys.

Add the standard flags to any argparse parser with `add_args(parser)`, then call
`init(args, ...)` once, `log_metrics(run, ...)` per step, and `finish(run)`.
"""

import argparse
import os


def add_args(parser):
    """Register the W&B CLI flags on an argparse parser. ON by default."""
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction,
                        default=True, help="log this run to Weights & Biases")
    parser.add_argument("--wandb_project", default=os.environ.get(
        "WANDB_PROJECT", "reifm-l4"))
    parser.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb_name", default=None,
                        help="W&B run name (defaults to --out / run id)")
    parser.add_argument("--wandb_tags", nargs="*", default=None)
    parser.add_argument("--wandb_group", default=None,
                        help="group runs (e.g. tierA / tierB) in the W&B UI")


def init(args, *, run_name, job_type, config_extra=None):
    """Start a W&B run, or return None if disabled.

    Raises a clear error if W&B is enabled but the environment isn't
    authenticated, rather than silently degrading to no tracking.
    """
    if not getattr(args, "wandb", False):
        print("[wandb] disabled (--no-wandb)")
        return None
    try:
        import wandb
    except ImportError as e:
        raise SystemExit(
            "[wandb] enabled but the `wandb` package is missing. Run `uv sync`, "
            "or pass --no-wandb to skip tracking.") from e

    if not (os.environ.get("WANDB_API_KEY") or _netrc_has_wandb()
            or os.environ.get("WANDB_MODE") in {"offline", "disabled"}):
        raise SystemExit(
            "[wandb] enabled but not authenticated. Do ONE of:\n"
            "  - `uv run wandb login` (interactive, stores the key in ~/.netrc), or\n"
            "  - export WANDB_API_KEY=<key from https://wandb.ai/authorize>, or\n"
            "  - export WANDB_MODE=offline  (log locally, sync later with `wandb sync`), or\n"
            "  - pass --no-wandb to run without tracking.")

    config = dict(vars(args))
    if config_extra:
        config.update(config_extra)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        job_type=job_type,
        tags=args.wandb_tags,
        group=args.wandb_group,
        config=config,
    )
    print(f"[wandb] tracking run {run.name} → {run.url}")
    return run


def _netrc_has_wandb():
    try:
        import netrc
        hosts = netrc.netrc().hosts
    except Exception:
        return False
    return any("wandb" in h for h in hosts)


def _flatten(metrics, prefix):
    """{'FB:v1': {'mrr': .3, 'hits@1': .2}} -> {'prefix/FB:v1/mrr': .3, ...}"""
    out = {}
    for spec, m in metrics.items():
        for k, v in m.items():
            out[f"{prefix}/{spec}/{k}"] = v
    return out


def log_metrics(run, *, step, train_loss=None, valid=None, test=None,
                reach=None, lr=None, secs=None, mean_valid_mrr=None,
                select=None, select_mrr=None):
    """Log one step's worth of metrics. No-op if run is None."""
    if run is None:
        return
    data = {}
    if train_loss is not None:
        for spec, v in train_loss.items():
            data[f"train/loss/{spec}"] = v
    if valid:
        data.update(_flatten(valid, "valid"))
    if test:
        data.update(_flatten(test, "test"))
    if reach is not None:
        data["train/reach"] = reach
    if lr is not None:
        data["train/lr"] = lr
    if secs is not None:
        data["train/epoch_secs"] = secs
    if mean_valid_mrr is not None:
        data["valid/mean_mrr"] = mean_valid_mrr
    if select:
        data.update(_flatten(select, "select"))
    if select_mrr is not None:
        data["select/mean_mrr"] = select_mrr
    # commit=True forces W&B to finalize THIS step now. With an explicit step= and
    # the default commit behaviour, W&B holds step N until the next log call with a
    # larger step, so the live UI always lags one epoch behind run.log. commit=True
    # uploads each step immediately (final data is identical either way).
    run.log(data, step=step, commit=True)


def log_summary(run, summary):
    """Set run-level summary values (e.g. mean test MRR). No-op if None."""
    if run is None:
        return
    for k, v in summary.items():
        run.summary[k] = v


def finish(run):
    if run is not None:
        run.finish()
