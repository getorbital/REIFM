#!/usr/bin/env python3
"""S-2 (papier court) — assemblage des matrices d'éval, périmètre 40 graphes.

Lit les JSON produits par scripts/eval_matrix.sh (+ les évals gelées
réutilisées) et écrit :

  results/broad_eval_matrix.csv  — 1 ligne / (backbone, seed, split)
  results/backbone_matrix.csv    — 1 ligne / (backbone, seed)

MRR ABSOLU uniquement (jamais de « % d'ULTRA » : composite interne, cf.
CLAUDE.md). Les colonnes ULTRA-3g local / record 018 sont des repères
per-split, reprises telles quelles des livrables Track B (déjà complets sur
les 41 : `results/reference/table1_main.csv` + `broad_eval_compare.csv`) — rien
n'est relancé côté dénominateur.

PÉRIMÈTRE : 40 graphes. HM:indigo est exclu (décision Camille 2026-07-30,
motif = coût mesuré ~4 h 40+/modèle). Voir le bloc EXCLUDED plus bas : le
split reste dans broad_eval_matrix.csv avec `in_scope=0`, il ne compte dans
aucun agrégat, et l'exclusion se déclare dans le papier.

Étiquetage des régimes (règle docs/ULTRA_PROTOCOL.md §9, « parent vu ⇒ plus
nouveau KG ») — les deux côtés sont étiquetés séparément parce que les
corpus d'entraînement diffèrent :
  - `regime_reifm`  : nos modèles S-1 s'entraînent sur FB15k237Inductive:v1
    SEUL ⇒ seules les 8 splits dérivées de Freebase sont « in-family » ;
    WN, NELL, Wikidata, HM, Metafam sont de vrais nouveaux KGs.
  - `regime_ultra3g`: ULTRA-3g pré-entraîne sur FB15k237+WN18RR+CoDExMedium
    ⇒ FB, WN **et** Wikidata sont in-family de son côté.
Cette asymétrie est à porter dans le papier : elle joue EN NOTRE DÉFAVEUR
sur Wikidata/WN (nous y sommes zéro-shot, ULTRA non).

Usage: uv run python scripts/build_matrix.py
"""

import csv
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P03 = os.path.join(ROOT, "checkpoints")
OUT = os.path.join(ROOT, "results")

BACKBONES = ["gat", "ginemax", "gine_sum", "sage", "rgcn"]
SEEDS = [0, 1, 2]

PANEL_A = [f"FB15k237Inductive:v{i}" for i in (1, 2, 3, 4)] + \
          [f"WN18RRInductive:v{i}" for i in (1, 2, 3, 4)] + \
          [f"NELLInductive:v{i}" for i in (1, 2, 3, 4)]
PANEL_B = [f"FBIngram:{k}" for k in (25, 50, 75, 100)] + \
          [f"WKIngram:{k}" for k in (25, 50, 75, 100)] + \
          [f"NLIngram:{k}" for k in (0, 25, 50, 75, 100)]
# HM:indigo est EXCLU du périmètre (décision Camille 2026-07-30, motif = coût :
# ~4 h 40+ par modèle mesurées → 30-70 h sur 15 modèles, la moitié du budget
# S-2). Le périmètre passe de 41 à 40 graphes inductifs.
#
# ⚠️ INTÉGRITÉ — à lire avant de toucher à cette liste. HM:indigo est le split
# où notre recette perd LE PLUS contre ULTRA (018 0.3091 vs 0.4361,
# Δ −0.127) : l'exclure améliore mécaniquement nos agrégats. Le motif est le
# coût, décidé sur des mesures de temps et non sur nos scores (nous n'avons
# jamais obtenu un seul chiffre ReiFM sur ce split), mais l'effet est réel.
# Conséquence non négociable : le split reste PRÉSENT dans
# broad_eval_matrix.csv avec `in_scope=0` et ses repères 018/ULTRA, et
# l'exclusion se déclare dans le papier avec sa raison. Rien n'est effacé.
EXCLUDED = ["HM:indigo"]

EXT_ALL = ["ILPC2022:small", "ILPC2022:large", "HM:1k", "HM:3k", "HM:5k",
           "HM:indigo", "FBNELL:FBNELL_v1", "Metafam:Metafam",
           "WikiTopicsMT1:tax", "WikiTopicsMT1:health",
           "WikiTopicsMT2:org", "WikiTopicsMT2:sci",
           "WikiTopicsMT3:art", "WikiTopicsMT3:infra",
           "WikiTopicsMT4:sci", "WikiTopicsMT4:health"]
EXT15 = [s for s in EXT_ALL if s not in EXCLUDED]      # périmètre : 15 étendus
ALL40 = PANEL_A + PANEL_B + EXT15                      # périmètre : 40 graphes
ALL_WITH_EXCLUDED = PANEL_A + PANEL_B + EXT_ALL        # pour les lignes du CSV

# split -> famille de KG
def family(spec):
    p = spec.split(":")[0]
    return {"FB15k237Inductive": "Freebase", "FBIngram": "Freebase",
            "WN18RRInductive": "WordNet",
            "NELLInductive": "NELL", "NLIngram": "NELL",
            "WKIngram": "Wikidata", "ILPC2022": "Wikidata",
            "WikiTopicsMT1": "Wikidata", "WikiTopicsMT2": "Wikidata",
            "WikiTopicsMT3": "Wikidata", "WikiTopicsMT4": "Wikidata",
            "HM": "INDIGO-HM", "Metafam": "Metafam",
            "FBNELL": "Freebase+NELL"}[p]

def panel(spec):
    if spec in PANEL_A:
        return "A_GraIL"
    if spec in PANEL_B:
        return "B_InGram"
    return "C_extended"

# régime, relatif au corpus d'entraînement de chaque côté (§9)
def regime_reifm(spec):
    """Nos modèles : FB15k237Inductive:v1 seul."""
    if spec == "FB15k237Inductive:v1":
        return "train+select"          # graphe d'entraînement et de sélection
    if family(spec) == "Freebase":
        return "in-family"             # parent Freebase vu
    if family(spec) == "Freebase+NELL":
        return "in-family-partial"     # FBNELL touche FB (vu) et NELL (non vu)
    return "new-KG"

def regime_ultra3g(spec):
    """ULTRA-3g : FB15k237 + WN18RR + CoDExMedium (Wikidata)."""
    if family(spec) in ("Freebase", "WordNet", "Wikidata"):
        return "in-family"
    if family(spec) == "Freebase+NELL":
        return "in-family-partial"
    return "new-KG"

TRUE_0SHOT_17 = [s for s in PANEL_A + PANEL_B
                 if regime_reifm(s) == "new-KG"]           # WN4+NELL4+WK4+NL5
NEW_KG_32 = [s for s in ALL40 if regime_reifm(s) == "new-KG"]


def load_refs():
    """Repères per-split ULTRA-3g local + record 018 (les 41, exclu compris)."""
    ultra, r018 = {}, {}
    with open(os.path.join(ROOT, "results", "reference", "table1_main.csv")) as f:
        for row in csv.DictReader(f):
            ultra[row["dataset"]] = float(row["ultra3g_mrr"])
            r018[row["dataset"]] = float(row["reifm018_mrr"])
    with open(os.path.join(ROOT, "results", "reference",
                           "broad_eval_compare.csv")) as f:
        for row in csv.DictReader(f):
            ultra[row["dataset"]] = float(row["mrr_ultra3g_local"])
            r018[row["dataset"]] = float(row["mrr_018"])
    return ultra, r018


def load_model(key, seed):
    """Renvoie (per_split, sources) pour un (backbone, seed).

    per_split[spec] = dict de métriques. `sources` trace de quel fichier
    chaque bloc vient (traçabilité : évals gelées réutilisées vs S-2).
    """
    d = os.path.join(P03, f"{key}_seed{seed}")
    per, src = {}, {}
    if not os.path.isdir(d):
        return per, src

    # 25-suite : éval gelée P0-3ter si présente (GAT 0/1/2), sinon les
    # fichiers par split produits par S-2 (s2_25_<split>.json).
    frozen = os.path.join(d, "eval25_frozen_p03ter.json")
    if os.path.exists(frozen):
        for spec, m in json.load(open(frozen))["test"].items():
            per[spec] = m
            src[spec] = "frozen_p03ter"
    else:
        for spec in PANEL_A + PANEL_B:
            p = os.path.join(d, f"s2_25_{spec.replace(':', '_')}.json")
            if os.path.exists(p):
                t = json.load(open(p))["test"]
                if spec in t:
                    per[spec] = t[spec]
                    src[spec] = "s2"

    # étendus : un fichier par split (on CHARGE aussi l'exclu s'il existe,
    # pour qu'il apparaisse dans le CSV avec in_scope=0)
    for spec in EXT_ALL:
        safe = spec.replace(":", "_")
        for fname, tag in ((f"s2_ext_{safe}.json", "s2"),
                           (f"gat_eval_{safe}.json", "p1_8_gat")):
            p = os.path.join(d, fname)
            if os.path.exists(p):
                t = json.load(open(p))["test"]
                if spec in t:
                    per[spec] = t[spec]
                    src[spec] = tag
                break
    return per, src


def macro(per, specs, require_all=True):
    """Moyenne non pondérée des MRR (convention 'flat' du papier).

    `require_all` (défaut) : renvoie None si un seul split manque. Un
    agrégat partiel qui s'affiche comme un agrégat complet est un piège —
    p.ex. un « flat_40 » calculé sur 25 splits vaudrait le flat_25 et se
    lirait comme un résultat sur les 41. Le compte (2e valeur) reste
    toujours renvoyé pour le rapport de complétude.
    """
    vals = [per[s]["mrr"] for s in specs if s in per]
    if not vals or (require_all and len(vals) != len(specs)):
        return None, len(vals)
    return sum(vals) / len(vals), len(vals)


def fmt(x, n=4):
    return "" if x is None else f"{x:.{n}f}"


def main():
    os.makedirs(OUT, exist_ok=True)
    ultra, r018 = load_refs()

    # --- broad_eval_matrix.csv : 1 ligne / (backbone, seed, split) --------
    broad = os.path.join(OUT, "broad_eval_matrix.csv")
    n_rows = 0
    with open(broad, "w", newline="") as f:
        w = csv.writer(f)
        # `in_scope` : 0 pour les splits exclus du périmètre (HM:indigo). La
        # ligne reste écrite si le chiffre existe — on n'efface rien, on
        # marque. Les agrégats de backbone_matrix.csv ignorent in_scope=0.
        w.writerow(["backbone", "seed", "split", "in_scope", "panel", "family",
                    "regime_reifm", "regime_ultra3g", "num_queries",
                    "mrr", "hits@1", "hits@3", "hits@10",
                    "mrr_ultra3g_local", "mrr_018", "source"])
        for key in BACKBONES:
            for seed in SEEDS:
                per, src = load_model(key, seed)
                for spec in ALL_WITH_EXCLUDED:
                    if spec not in per:
                        continue
                    m = per[spec]
                    w.writerow([key, seed, spec,
                                0 if spec in EXCLUDED else 1,
                                panel(spec), family(spec),
                                regime_reifm(spec), regime_ultra3g(spec),
                                m.get("num_queries", ""),
                                fmt(m["mrr"]), fmt(m.get("hits@1")),
                                fmt(m.get("hits@3")), fmt(m.get("hits@10")),
                                fmt(ultra.get(spec)), fmt(r018.get(spec)),
                                src.get(spec, "")])
                    n_rows += 1

    # --- backbone_matrix.csv : 1 ligne / (backbone, seed) ----------------
    mat = os.path.join(OUT, "backbone_matrix.csv")
    summary = []
    with open(mat, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backbone", "seed", "params", "best_valid_sel_fbv1",
                    "panelA_12", "panelB_13", "ext_15", "flat_25", "flat_40",
                    "panelmean_25", "true0shot_17", "newKG_32",
                    "n_splits_ok", "n_splits_expected", "sources"])
        for key in BACKBONES:
            for seed in SEEDS:
                per, src = load_model(key, seed)
                d = os.path.join(P03, f"{key}_seed{seed}")
                params, bvs = "", None
                rj = os.path.join(d, "results.json")
                if os.path.exists(rj):
                    j = json.load(open(rj))
                    params = j.get("params", "")
                    sel = [e["select"]["FB15k237Inductive:v1"]["mrr"]
                           for e in j.get("history", [])
                           if "FB15k237Inductive:v1" in e.get("select", {})]
                    bvs = max(sel) if sel else None
                a, na = macro(per, PANEL_A)
                b, nb = macro(per, PANEL_B)
                e, ne = macro(per, EXT15)
                f25, n25 = macro(per, PANEL_A + PANEL_B)
                f40, n40 = macro(per, ALL40)
                pm = (a + b) / 2 if a is not None and b is not None else None
                t17, _ = macro(per, TRUE_0SHOT_17)
                n32, _ = macro(per, NEW_KG_32)
                w.writerow([key, seed, params, fmt(bvs), fmt(a), fmt(b),
                            fmt(e), fmt(f25), fmt(f40), fmt(pm), fmt(t17),
                            fmt(n32), n40, len(ALL40),
                            "|".join(sorted(set(src.values())))])
                summary.append((key, seed, na, nb, ne, n40, f40))

    # --- rapport de complétude (stdout, pas un livrable) -----------------
    print(f"écrit {mat}")
    print(f"écrit {broad}  ({n_rows} lignes)")
    print(f"\n{'config':<16}{'A/12':<7}{'B/13':<7}{'ext/15':<8}{'tot/40':<8}{'flat_40'}")
    print("-" * 54)
    inc = 0
    for key, seed, na, nb, ne, n40, f40 in summary:
        flag = "" if n40 == 40 else "  ← INCOMPLET"
        if n40 != 40:
            inc += 1
        print(f"{key+'_seed'+str(seed):<16}{na:<7}{nb:<7}{ne:<8}{n40:<8}"
              f"{fmt(f40):<9}{flag}")
    # repères, pour la lecture de la fiche (MRR absolu)
    print(f"\nrepères (MRR absolu, mêmes 40 splits du périmètre) :")
    for name, ref in (("ULTRA-3g local", ultra), ("record 018", r018)):
        va = [ref[s] for s in PANEL_A if s in ref]
        vb = [ref[s] for s in PANEL_B if s in ref]
        vx = [ref[s] for s in EXT15 if s in ref]
        vall = [ref[s] for s in ALL40 if s in ref]
        print(f"  {name:<16} A {sum(va)/len(va):.4f}  B {sum(vb)/len(vb):.4f}"
              f"  ext {sum(vx)/len(vx):.4f}  flat-40 {sum(vall)/len(vall):.4f}")
    if inc:
        print(f"\n⚠️  {inc} config(s) incomplète(s) — relancer "
              f"scripts/eval_matrix.sh (resumable) avant de lire les tables.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
