#!/usr/bin/env bash
# Install the third-party code the pipeline imports, at the exact commits the
# paper was produced with. Nothing here is redistributed in this repository.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p vendor

# ULTRA (MIT) — dataset loaders for the 40 benchmarks and the released
# checkpoints ultra_3g/4g/50g (ckpts/). Upstream repository, pinned at the
# commit the paper was produced with.
ULTRA_URL=https://github.com/DeepGraphLearning/ULTRA.git
ULTRA_PIN=427966ad8ed60420eef034063d44f3153addff90
if [ ! -d vendor/ULTRA ]; then
  git clone "$ULTRA_URL" vendor/ULTRA
fi
git -C vendor/ULTRA fetch --quiet origin
git -C vendor/ULTRA checkout --quiet "$ULTRA_PIN"
echo "[setup_vendor] ULTRA at $(git -C vendor/ULTRA rev-parse --short HEAD)"
ls vendor/ULTRA/ckpts/ultra_3g.pth >/dev/null && echo "[setup_vendor] ULTRA checkpoints present (ckpts/)"
