# checkpoints/

Download target of `scripts/download_checkpoints.sh` (Hugging Face). Layout:

    checkpoints/<backbone>_seed<k>/best.pt        state dict
    checkpoints/<backbone>_seed<k>/results.json   argument block + training history
    checkpoints/generic3kg_seed{A,B,C}/...        the generic 3-KG model of the probe

`scripts/eval_matrix.sh` writes its per-(model, split) evaluation JSON and rank
dumps next to each checkpoint.
