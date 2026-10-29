"""Thesis evaluation protocol on L2-normalised embeddings (cosine = dot product). numpy only.

Retrieval, per split: every photo is a query; the gallery is the other photos of the **same
species** (the app filters by species), leave-one-out. A query whose animal has no other photo is
skipped (it cannot be found), and so is a species with a single animal (nothing to confuse it
with: Rank-1 would be 100 % by construction).
- Rank-k (by animal, like MatchPets): each animal scores its best photo; hit if the query's animal
  is among the top k animals.
- mAP (by photo): average precision of the query's own photos in the ranked gallery.
Verification: the match threshold is fixed at the EER on `val` pairs, then FAR/FRR are measured on
`test` pairs. Every result carries its gallery size: Rank-k means little without it.
"""

import numpy as np

KS = (1, 5, 10)


def retrieval(E: np.ndarray, animals: list[str], species: list[str], ks=KS) -> dict:
    """Rank-k and mAP, global and per species, plus one row per query."""
    S = E @ E.T
    per_query = []
    for sp in sorted(set(species)):
        idx = [i for i, s in enumerate(species) if s == sp]
        n_animals = len({animals[i] for i in idx})
        if n_animals < 2:
            continue
        for q in idx:
            gallery = [g for g in idx if g != q]
            own = [g for g in gallery if animals[g] == animals[q]]
            if not own:
                continue
            best: dict[str, float] = {}
            for g in gallery:
                best[animals[g]] = max(best.get(animals[g], -np.inf), float(S[q, g]))
            ranked = sorted(best, key=best.get, reverse=True)
            rank = ranked.index(animals[q]) + 1
            order = sorted(gallery, key=lambda g: S[q, g], reverse=True)
            hits, precisions = 0, []
            for pos, g in enumerate(order, 1):
                if animals[g] == animals[q]:
                    hits += 1
                    precisions.append(hits / pos)
            rival = next(a for a in ranked if a != animals[q]) if len(ranked) > 1 else ""
            per_query.append({
                "query": q, "animal": animals[q], "species": sp, "rank": rank,
                "ap": float(np.mean(precisions)), "gallery_animals": n_animals,
                "own_best": best[animals[q]], "rival": rival,
                "rival_best": best[rival] if rival else float("nan"),
            })
    return {"global": _summary(per_query, ks), **{
        sp: _summary([r for r in per_query if r["species"] == sp], ks) for sp in sorted({r["species"] for r in per_query})
    }, "per_query": per_query}


def _summary(rows: list[dict], ks) -> dict:
    if not rows:
        return {"queries": 0}
    ranks = np.array([r["rank"] for r in rows])
    out = {"queries": len(rows), "gallery_animals": max(r["gallery_animals"] for r in rows)}
    for k in ks:
        out[f"rank{k}"] = float(np.mean(ranks <= k))
    out["mAP"] = float(np.mean([r["ap"] for r in rows]))
    return out


def pair_scores(E: np.ndarray, animals: list[str], species: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Cosine scores of same-animal and different-animal pairs (same species only). A species with
    a single animal is skipped, as in `retrieval`: it has no different-animal pairs to compare with."""
    S = E @ E.T
    rivals = {sp: len({a for a, s in zip(animals, species) if s == sp}) > 1 for sp in set(species)}
    same, diff = [], []
    for i in range(len(animals)):
        for j in range(i + 1, len(animals)):
            if species[i] != species[j] or not rivals[species[i]]:
                continue
            (same if animals[i] == animals[j] else diff).append(float(S[i, j]))
    return np.array(same), np.array(diff)


def eer(same: np.ndarray, diff: np.ndarray) -> tuple[float, float]:
    """Equal error rate and the threshold where false accepts = false rejects."""
    if len(same) == 0 or len(diff) == 0:
        return float("nan"), float("nan")
    thresholds = np.unique(np.concatenate([same, diff]))
    far = np.array([np.mean(diff >= t) for t in thresholds])
    frr = np.array([np.mean(same < t) for t in thresholds])
    k = int(np.argmin(np.abs(far - frr)))
    return float((far[k] + frr[k]) / 2), float(thresholds[k])


def at_threshold(same: np.ndarray, diff: np.ndarray, t: float) -> dict:
    """False accept / reject rates of a fixed threshold."""
    if np.isnan(t) or len(same) == 0 or len(diff) == 0:
        return {"far": float("nan"), "frr": float("nan")}
    return {"far": float(np.mean(diff >= t)), "frr": float(np.mean(same < t))}

