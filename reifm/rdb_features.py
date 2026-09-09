"""Schema-agnostic entity features for reified relational DBs (cross-DB / RT-style).

To pretrain ONE model across many RelBench DBs (the reified vocabulary is already
dataset-agnostic), the *features* must also be encoded in a schema-agnostic way:
a "points" column in rel-f1 and a "score" column in rel-stack should land in the
same place. We do this RT-style, using the **column NAME** (not its position) as
the semantic key, via a frozen MiniLM text encoder:

    raw[entity] =  Σ_numeric  z(value) · colname_emb[col]          (value-weighted
                 + Σ_text     ( colname_emb[col] + MiniLM(value) )   column-name dir)

`colname_emb` and `MiniLM(value)` are FROZEN (precomputed, cached), so `raw` is a
fixed [num_entities, 384] matrix per DB — computed once, no gradients. The only
learned, SHARED part downstream is a single Linear(384 → dim) projector (in the
model), so a brand-new DB's entities are encoded by the exact same recipe → the
representation transfers zero-shot. Aligned to build_rdb_kg's entity ids.
"""
import numpy as np
import pandas as pd
import torch

_ENCODER = None
_STR_CACHE = {}  # text string -> 384 vector (frozen)


def _encoder(device="cpu"):
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer
        _ENCODER = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    return _ENCODER


def embed_strings(strings, device="cpu", batch_size=512):
    """Frozen MiniLM embeddings [len, 384] with a process-wide cache."""
    need = [s for s in dict.fromkeys(strings) if s not in _STR_CACHE]
    if need:
        enc = _encoder(device)
        vecs = enc.encode(need, convert_to_numpy=True, batch_size=batch_size,
                          show_progress_bar=False, normalize_embeddings=True)
        for s, v in zip(need, vecs):
            _STR_CACHE[s] = v.astype(np.float32)
    return np.stack([_STR_CACHE[s] for s in strings])


def column_types(tbl):
    """Per-column semantic type for a relbench Table: 'num' | 'text' | 'skip'
    (FK/pk/time columns are skipped — structure/time handled elsewhere)."""
    df = tbl.df
    fks = set(tbl.fkey_col_to_pkey_table)
    out = {}
    for c in df.columns:
        if c in fks or c == tbl.pkey_col or c == tbl.time_col:
            continue
        s = df[c].dropna()
        if len(s) == 0:
            continue
        out[c] = "num" if np.issubdtype(s.dtype, np.number) else "text"
    return out


def build_entity_cells(db, device="cpu", num_stats=None, max_text_card=2000,
                       kmax=12):
    """Per-entity UNPOOLED cells for cell-level attention (RT-style), instead of
    summing them. Returns (cells [num_entities, kmax, 384] float32, mask
    [num_entities, kmax] bool, num_stats). Each cell = colname_emb scaled by the
    standardised numeric value, or colname_emb + MiniLM(text value). This keeps
    the per-cell signal the pooled build collapses (the key RT ingredient)."""
    tables = list(db.table_dict.keys())
    n = sum(len(db.table_dict[t].df) for t in tables)
    cells = np.zeros((n, kmax, 384), dtype=np.float32)
    nfill = np.zeros(n, dtype=np.int64)

    colnames, coltypes = {}, {}
    for t in tables:
        coltypes[t] = column_types(db.table_dict[t])
        for c in coltypes[t]:
            colnames[(t, c)] = f"{t} {c}".replace("_", " ")
    name_emb = dict(zip(colnames.keys(),
                        embed_strings(list(colnames.values()), device)
                        if colnames else np.zeros((0, 384), np.float32)))
    if num_stats is None:
        num_stats = {}
    off = 0
    for t in tables:
        df = db.table_dict[t].df
        ln = len(df)
        for c, typ in coltypes[t].items():
            cn = name_emb[(t, c)]
            if typ == "num":
                v = df[c].to_numpy(dtype=np.float64, na_value=np.nan)
                key = (t, c)
                if key not in num_stats:
                    m = np.nanmean(v) if np.isfinite(v).any() else 0.0
                    sd = np.nanstd(v) if np.isfinite(v).any() else 1.0
                    num_stats[key] = (m, sd if sd > 1e-6 else 1.0)
                m, sd = num_stats[key]
                z = np.nan_to_num((v - m) / sd).astype(np.float32)
                vec = z[:, None] * cn[None, :]                     # [ln,384]
            else:
                vals = df[c].astype(str).fillna("").to_numpy()
                if pd.Series(vals).nunique() > max_text_card:
                    vec = np.tile(cn, (ln, 1))
                else:
                    vec = cn[None, :] + embed_strings(list(vals), device)
            # place this column's cells into the next free slot per row (< kmax)
            for r in range(ln):
                k = nfill[off + r]
                if k < kmax:
                    cells[off + r, k] = vec[r]
                    nfill[off + r] = k + 1
        off += ln
    mask = (np.arange(kmax)[None, :] < nfill[:, None])
    return (torch.tensor(cells), torch.tensor(mask), num_stats)


def build_entity_features(db, device="cpu", num_stats=None, max_text_card=2000):
    """[num_entities, 384] frozen schema-agnostic features aligned to
    build_rdb_kg entity ids (row offset per table). `num_stats` (per (table,col)
    mean/std) is fit on the given db if None and returned for reuse (no leakage
    when the same stats are passed for val/test)."""
    tables = list(db.table_dict.keys())
    n = sum(len(db.table_dict[t].df) for t in tables)
    raw = np.zeros((n, 384), dtype=np.float32)

    # 1) column-name embeddings (semantic key, shared across DBs)
    colnames, coltypes = {}, {}
    for t in tables:
        coltypes[t] = column_types(db.table_dict[t])
        for c in coltypes[t]:
            colnames[(t, c)] = f"{t} {c}".replace("_", " ")
    name_list = list(colnames.values())
    name_emb = dict(zip(colnames.keys(),
                        embed_strings(name_list, device) if name_list
                        else np.zeros((0, 384), np.float32)))

    if num_stats is None:
        num_stats = {}
    off = 0
    for t in tables:
        df = db.table_dict[t].df
        ln = len(df)
        for c, typ in coltypes[t].items():
            cn = name_emb[(t, c)]
            if typ == "num":
                v = df[c].to_numpy(dtype=np.float64, na_value=np.nan)
                key = (t, c)
                if key not in num_stats:
                    m = np.nanmean(v) if np.isfinite(v).any() else 0.0
                    sd = np.nanstd(v) if np.isfinite(v).any() else 1.0
                    num_stats[key] = (m, sd if sd > 1e-6 else 1.0)
                m, sd = num_stats[key]
                z = np.nan_to_num((v - m) / sd).astype(np.float32)  # [ln]
                # accumulate in-place per column (no [ln,384] float64 temporary)
                raw[off:off + ln] += z[:, None] * cn[None, :]
            else:  # text / categorical
                vals = df[c].astype(str).fillna("").to_numpy()
                if pd.Series(vals).nunique() > max_text_card:
                    # very high-cardinality identifiers (e.g. refs): colname only
                    raw[off:off + ln] += cn
                else:
                    ve = embed_strings(list(vals), device)           # [ln,384]
                    raw[off:off + ln] += cn + ve
        off += ln
    return torch.tensor(raw), num_stats
