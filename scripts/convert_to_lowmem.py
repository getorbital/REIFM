"""Convert a `ginemax` checkpoint (PyG mean+max path) to the `ginemaxlm`
memory-lean path, for graphs too large for the disjoint-union path (rel-stack).

The two implementations compute the same function; the only structural
difference is where the 6 meta-relation embeddings live: one table shared by
all layers at the model level (`edge_emb`, PyG path) versus one copy per layer
(`convs.k.edge_emb`, dense path). The conversion copies the shared table into
every layer; everything else is identical (the filtered ranks are bit-identical).

Usage: uv run python scripts/convert_to_lowmem.py checkpoints/ginemax_seed0/best.pt \
           checkpoints/ginemax_seed0/best_lowmem.pt --layers 12
"""
import argparse

import torch

p = argparse.ArgumentParser()
p.add_argument("src")
p.add_argument("dst")
p.add_argument("--layers", type=int, default=12)
a = p.parse_args()
sd = torch.load(a.src, map_location="cpu", weights_only=True)
for k in range(a.layers):
    sd[f"convs.{k}.edge_emb.weight"] = sd["edge_emb.weight"].clone()
torch.save(sd, a.dst)
print(f"[convert] {a.src} -> {a.dst} ({len(sd)} tensors)")
