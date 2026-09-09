"""M1-bis — dump des rangs PAR REQUÊTE d'ULTRA-3g local sur les splits InGram.

But : comparer le profil head/tail (et par covariable) d'ULTRA au nôtre —
distinguer « pathologie de notre readout » vs « difficulté intrinsèque du
problème » sur l'asymétrie head découverte par le diagnostic M1.

Reprend exactement le protocole de vendor/ULTRA/script/run.py (__main__) :
filtre = inference + valid + test pour les classes InGram / ILPC, inference +
test pour les autres splits inductifs (MTDEA : FBNELL, Metafam, WikiTopics…),
mais conserve les rangs par triplet.
Sortie : results/diag_m1_ultra/ultra_<spec>.npz (rank_tail, rank_head, rel,
filter_kind) ; si le graphe d'inférence contient des entités isolées (degré 0,
FBNELL) : n_iso, n_above_iso_{tail,head} (nb de candidats filtrés qu'ULTRA
score strictement au-dessus du bloc isolé), gold_gt_iso_{tail,head} (la vraie
réponse est-elle scorée au-dessus du bloc), iso_spread_{tail,head} (max-min
des scores du bloc, doit être ~0 : entités structurellement identiques).

Historique : jusqu'au 2026-09-09 le filtre InGram était appliqué à tout
split (S-6, FBNELL : MRR 0,4121 au lieu de la cellule run.py 0,4734) ; les
dumps InGram antérieurs ne changent pas.

Usage :
  uv run python scripts/ultra_dump_ranks.py FBIngram:25 FBIngram:100 \
      WKIngram:100 NLIngram:100
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import reifm.datasets  # noqa: F401  (shims torch.load / torch_scatter)

ULTRA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "vendor", "ULTRA")
sys.path.insert(0, ULTRA_DIR)

from torch_geometric.data import Data          # noqa: E402
from ultra import tasks, util                  # noqa: E402
from ultra.models import Ultra                 # noqa: E402


def main():
    specs = sys.argv[1:] or ["FBIngram:25"]
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "diag_m1_ultra")
    os.makedirs(out_dir, exist_ok=True)

    os.chdir(ULTRA_DIR)  # les chemins de config/données d'ULTRA sont relatifs
    device = torch.device("cpu")
    torch.set_num_threads(8)

    model = None
    for spec in specs:
        name, version = spec.split(":")
        cfg = util.load_config(
            "config/inductive/inference.yaml",
            context=dict(dataset=name, version=version, epochs=0,
                         bpe="null", gpus="null",
                         ckpt=os.path.join(ULTRA_DIR, "ckpts/ultra_3g.pth")))
        dataset = util.build_dataset(cfg)
        train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
        if model is None:
            model = Ultra(rel_model_cfg=cfg.model.relation_model,
                          entity_model_cfg=cfg.model.entity_model)
            state = torch.load(cfg.checkpoint, map_location="cpu")
            model.load_state_dict(state["model"])
            model = model.to(device).eval()

        # filtre de test : même branche que run.py (__main__)
        if "Ingram" in name or "ILPC" in name:
            filter_kind = "ingram:inference+valid+test"
            fe = torch.cat([valid_data.edge_index, valid_data.target_edge_index,
                            test_data.target_edge_index], dim=1)
            ft = torch.cat([valid_data.edge_type, valid_data.target_edge_type,
                            test_data.target_edge_type])
        else:
            filter_kind = "inductive:inference+test"
            fe = torch.cat([test_data.edge_index, test_data.target_edge_index], dim=1)
            ft = torch.cat([test_data.edge_type, test_data.target_edge_type])
        filtered = Data(edge_index=fe, edge_type=ft,
                        num_nodes=test_data.num_nodes)
        print(f"[ultra-dump] {spec}: filter={filter_kind}", flush=True)

        # entités isolées du graphe d'inférence (degré 0) — diagnostic R-50
        deg = torch.zeros(test_data.num_nodes, dtype=torch.long)
        deg.scatter_add_(0, test_data.edge_index.reshape(-1),
                         torch.ones(test_data.edge_index.numel(), dtype=torch.long))
        iso = (deg == 0).nonzero().flatten()
        diag = {k: [] for k in ("n_above_iso_tail", "n_above_iso_head",
                                "gold_gt_iso_tail", "gold_gt_iso_head",
                                "iso_spread_tail", "iso_spread_head")}

        def iso_diag(pred, pos, mask, side):
            if iso.numel() == 0:
                return
            iso_scores = pred[:, iso]
            iso_ref = iso_scores[:, 0]
            diag[f"iso_spread_{side}"].append(
                (iso_scores.max(1).values - iso_scores.min(1).values))
            above = (pred > iso_ref.unsqueeze(1)) & mask
            diag[f"n_above_iso_{side}"].append(above.sum(1))
            gold = pred.gather(1, pos.unsqueeze(1)).squeeze(1)
            diag[f"gold_gt_iso_{side}"].append(gold > iso_ref)

        triplets = torch.cat([test_data.target_edge_index,
                              test_data.target_edge_type.unsqueeze(0)]).t()
        rt, rh = [], []
        with torch.no_grad():
            for start in range(0, triplets.size(0), 32):
                batch = triplets[start:start + 32]
                t_batch, h_batch = tasks.all_negative(test_data, batch)
                t_pred = model(test_data, t_batch)
                h_pred = model(test_data, h_batch)
                t_mask, h_mask = tasks.strict_negative_mask(filtered, batch)
                pos_h, pos_t, _ = batch.t()
                rt.append(tasks.compute_ranking(t_pred, pos_t, t_mask))
                rh.append(tasks.compute_ranking(h_pred, pos_h, h_mask))
                iso_diag(t_pred, pos_t, t_mask, "tail")
                iso_diag(h_pred, pos_h, h_mask, "head")
        rt = torch.cat(rt).numpy()
        rh = torch.cat(rh).numpy()
        rel = test_data.target_edge_type.numpy()
        both = np.concatenate([rt, rh])
        mrr = (1.0 / both).mean()
        print(f"[ultra-dump] {spec}: MRR={mrr:.4f} "
              f"tail={np.mean(1.0/rt):.4f} head={np.mean(1.0/rh):.4f} "
              f"n={len(both)}", flush=True)
        extra = {k: torch.cat(v).numpy() for k, v in diag.items() if v}
        if iso.numel():
            print(f"[ultra-dump] {spec}: {iso.numel()} isolated entities; "
                  f"gold>iso tail={extra['gold_gt_iso_tail'].mean():.3f} "
                  f"head={extra['gold_gt_iso_head'].mean():.3f}; "
                  f"median #cands above iso block tail="
                  f"{np.median(extra['n_above_iso_tail']):.0f} "
                  f"head={np.median(extra['n_above_iso_head']):.0f}; "
                  f"max iso spread={max(extra['iso_spread_tail'].max(), extra['iso_spread_head'].max()):.2e}",
                  flush=True)
        np.savez(os.path.join(out_dir, f"ultra_{spec.replace(':', '_')}.npz"),
                 rank_tail=rt, rank_head=rh, rel=rel, filter_kind=filter_kind,
                 n_iso=iso.numel(), iso_nodes=iso.numpy(), **extra)


if __name__ == "__main__":
    main()
