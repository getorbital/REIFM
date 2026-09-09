"""Run the pretrained ULTRA checkpoint on GraIL splits locally (CPU) to
validate our evaluation-protocol parity against published numbers.

Wraps ULTRA's script/run.py after installing our torch.load/torch_scatter
shims (reifm.datasets import has that side effect).

Usage: uv run python scripts/ultra_baseline.py FB15k237Inductive v1 [ckpt]
"""

import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import reifm.datasets  # noqa: F401  (torch.load + torch_scatter shims)

dataset, version = sys.argv[1], sys.argv[2]
ckpt = sys.argv[3] if len(sys.argv) > 3 else "ckpts/ultra_3g.pth"

ultra_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "ULTRA")
if not os.path.isabs(ckpt):
    ckpt = os.path.join(ultra_dir, ckpt)
os.chdir(ultra_dir)
sys.argv = [
    "run.py",
    "-c", "config/inductive/inference.yaml",
    "--dataset", dataset,
    "--version", version,
    "--epochs", "0",
    "--bpe", "null",
    "--gpus", os.environ.get("ULTRA_GPUS", "null"),  # null = CPU (parité Mac) ; "[0]" = GPU
    "--ckpt", ckpt,
]
runpy.run_path(os.path.join(ultra_dir, "script", "run.py"), run_name="__main__")
