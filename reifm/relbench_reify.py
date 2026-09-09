"""Reify a RelBench relational database into the SAME fixed vocabulary used for
KGs — to test whether the reified representation unifies knowledge graphs and
relational databases (the "maximal contribution" of the approach).

Mapping (multi-table RDB → reified KG, featureless / structural only):
  - each ROW of each table  → an entity node (its table = its class)
  - each FK column          → a relation type (anonymous node), e.g.
                              "results.driverId" links a results-row to a driver-row
  - each non-null FK cell    → a fact (subject=row, type=fk_col, object=ref row)

So a relational DB becomes a heterogeneous KG with the exact same
(has_subject/has_object/has_type) vocabulary. We deliberately ignore column
*features* here: the goal is to test the STRUCTURAL transfer of the vocabulary,
not to compete on RelBench's feature-rich tasks (which would need encoders —
extra machinery that would muddy the "generic GNN + reification" claim).

`entity_class` (table id per entity) is returned for the optional CLASS+is_a
extension (test 2); test 1 (zero-shot from a KG checkpoint) ignores it.
"""

import numpy as np
import pandas as pd
import torch

# entities/facts from time-less reference tables (drivers, circuits…) are always
# present → a very negative "timestamp" so the lazy temporal filter never drops them.
TIMELESS = -1e18


def build_rdb_kg(db):
    """relbench Database → (edge_index, edge_type, num_entities, num_relations,
    entity_class, rel_names, class_names). Edges are FK facts (subject→object).

    Also returns per-entity and per-fact epoch-second timestamps (entity_time,
    fact_time): a fact's time = its subject row's time_col (when the relation was
    created), TIMELESS for rows from reference tables — for the lazy temporal
    filter and relative-Δt features."""
    tables = list(db.table_dict.keys())
    class_names = tables
    # global entity id = row offset per table
    offset, ent_offset = {}, 0
    pkey_to_global = {}  # (table, pkey_value) -> global entity id
    entity_class, entity_time = [], []
    for t in tables:
        df = db.table_dict[t].df
        offset[t] = ent_offset
        tbl = db.table_dict[t]
        pk = tbl.pkey_col
        if pk is not None:
            for gid, pv in enumerate(df[pk].tolist()):
                pkey_to_global[(t, pv)] = ent_offset + gid
        entity_class.extend([tables.index(t)] * len(df))
        if tbl.time_col is not None:
            et = pd.to_datetime(df[tbl.time_col]).astype("int64").to_numpy() / 1e9
            entity_time.extend(et.tolist())
        else:
            entity_time.extend([TIMELESS] * len(df))
        ent_offset += len(df)
    num_entities = ent_offset

    # relation types = FK columns across tables
    rel_names, rel_id = [], {}
    src, dst, etype = [], [], []
    for t in tables:
        tbl = db.table_dict[t]
        df = tbl.df
        base = offset[t]
        for fk_col, ref_table in tbl.fkey_col_to_pkey_table.items():
            rname = f"{t}.{fk_col}"
            if rname not in rel_id:
                rel_id[rname] = len(rel_names)
                rel_names.append(rname)
            r = rel_id[rname]
            vals = df[fk_col].tolist()
            for row_i, fv in enumerate(vals):
                key = (ref_table, fv)
                g = pkey_to_global.get(key)
                if g is None:  # null / dangling FK
                    continue
                src.append(base + row_i)
                dst.append(g)
                etype.append(r)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(etype, dtype=torch.long)
    entity_time = torch.tensor(entity_time, dtype=torch.double)
    # a fact's time = its subject row's time (src is the subject entity global id)
    fact_time = entity_time[torch.tensor(src, dtype=torch.long)] if src \
        else torch.empty(0, dtype=torch.double)
    return dict(
        edge_index=edge_index, edge_type=edge_type, num_entities=num_entities,
        num_relations=len(rel_names), entity_class=torch.tensor(entity_class),
        rel_names=rel_names, class_names=class_names,
        pkey_to_global=pkey_to_global,  # (table, pkey value) -> entity node id
        table_offset=offset,            # table name -> first entity id
        entity_time=entity_time,        # [num_entities] epoch seconds (TIMELESS if none)
        fact_time=fact_time,            # [num_facts] epoch seconds of subject row
    )


def split_target(kg, target_rid, frac_support=0.5, seed=0):
    """Held-out split of one foreign-key column (the probe of Section 6): a
    support fraction of its cells stays in the fact graph, the rest become the
    queries (subject -> ?). Deterministic in `seed`."""
    et = kg["edge_type"]
    is_t = et == target_rid
    tgt_idx = is_t.nonzero(as_tuple=True)[0]
    g = torch.Generator().manual_seed(seed)
    perm = tgt_idx[torch.randperm(tgt_idx.numel(), generator=g)]
    n_sup = int(frac_support * perm.numel())
    sup, test = perm[:n_sup], perm[n_sup:]
    keep = ~is_t
    keep[sup] = True
    fact_index = kg["edge_index"][:, keep]
    fact_type = kg["edge_type"][keep]
    q_sub = kg["edge_index"][0, test]
    q_obj = kg["edge_index"][1, test]
    return fact_index, fact_type, q_sub, q_obj
