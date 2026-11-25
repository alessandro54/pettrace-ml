"""Conservative text-assisted reranking for an already ranked visual shortlist."""

import numpy as np

from pettrace.evaluation import metrics

_FIELD_WEIGHTS = {
    "primary_color": 3.0,
    "secondary_color": 1.0,
    "coat_pattern": 3.0,
    "coat_length": 1.0,
    "ear_shape": 1.0,
    "white_markings": 2.0,
    "distinctive_features": 2.0,
}
_MISSING = {"unknown", "not_visible"}


def attribute_similarity(left, right):
    """Weighted, interpretable compatibility in [-1, 1], ignoring unavailable evidence."""
    score = weight = 0.0
    for field, importance in _FIELD_WEIGHTS.items():
        a, b = left[field], right[field]
        if field in {"white_markings", "distinctive_features"}:
            a, b = set(a) - {"none"}, set(b) - {"none"}
            if not a and not b:
                continue
            elif not a or not b:
                value = -1.0
            else:
                value = (2 * len(a & b) / len(a | b)) - 1
        else:
            if a in _MISSING or b in _MISSING:
                continue
            value = 1.0 if a == b else -1.0
        score += importance * value
        weight += importance
    return score / weight if weight else 0.0


def gated_topk_rerank(
    visual_scores,
    text_scores,
    *,
    visual_margin_threshold,
    text_advantage_threshold,
    text_weight,
    top_k=5,
):
    """Rerank only an ambiguous visual Top-k when text clearly favors another candidate."""
    visual = np.asarray(visual_scores, dtype=float)
    text = np.asarray(text_scores, dtype=float)
    if visual.ndim != 1 or text.shape != visual.shape or len(visual) < 2:
        raise ValueError("visual_scores and text_scores must be equal one-dimensional arrays")
    if not np.isfinite(visual).all() or not np.isfinite(text).all():
        raise ValueError("scores must be finite")
    if min(visual_margin_threshold, text_advantage_threshold, text_weight) < 0:
        raise ValueError("thresholds and text_weight must be non-negative")
    if not 2 <= top_k <= len(visual):
        raise ValueError("top_k must be between 2 and the number of candidates")

    visual_order = np.argsort(-visual, kind="stable")
    shortlist = visual_order[:top_k]
    visual_top1 = int(shortlist[0])
    visual_margin = float(visual[shortlist[0]] - visual[shortlist[1]])
    fused = visual[shortlist] + text_weight * text[shortlist]
    proposed = int(shortlist[np.argsort(-fused, kind="stable")[0]])
    text_advantage = float(text[proposed] - text[visual_top1])
    changed = bool(
        visual_margin < visual_margin_threshold
        and proposed != visual_top1
        and text_advantage >= text_advantage_threshold
    )

    order = visual_order.copy()
    if changed:
        order[:top_k] = shortlist[np.argsort(-fused, kind="stable")]
    return order, {
        "gate_open": visual_margin < visual_margin_threshold,
        "changed": changed,
        "visual_margin": visual_margin,
        "text_advantage": text_advantage,
        "visual_top1": visual_top1,
        "reranked_top1": int(order[0]),
        "top_k": top_k,
    }


def rerank_evaluation(evaluation, attributes, *, text_weight, gated,
                      visual_margin_threshold=0.0, text_advantage_threshold=0.0, top_k=5):
    """Apply identity-level E5 to one frozen E4 Evaluation without changing its embeddings."""
    if len(attributes) != len(evaluation.photos):
        raise ValueError("One validated attribute record is required per evaluation photo")
    embeddings = np.asarray(evaluation.embeddings, dtype=float)
    scores = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    animals = [photo.animal for photo in evaluation.photos]
    species = [photo.species for photo in evaluation.photos]
    rows = []
    for sp in sorted(set(species)):
        indices = [i for i, value in enumerate(species) if value == sp]
        if len({animals[i] for i in indices}) < 2:
            continue
        for query in indices:
            gallery = [index for index in indices if index != query]
            own = [index for index in gallery if animals[index] == animals[query]]
            if not own:
                continue
            identities = list(dict.fromkeys(animals[index] for index in gallery))
            visual = np.array([max(scores[query, index] for index in gallery if animals[index] == animal)
                               for animal in identities])
            text = np.array([max(attribute_similarity(attributes[query], attributes[index])
                                 for index in gallery if animals[index] == animal)
                             for animal in identities])
            baseline = np.argsort(-visual, kind="stable")
            shortlist = baseline[:min(top_k, len(baseline))]
            if gated:
                order, audit = gated_topk_rerank(
                    visual, text, visual_margin_threshold=visual_margin_threshold,
                    text_advantage_threshold=text_advantage_threshold,
                    text_weight=text_weight, top_k=len(shortlist))
                apply_text = audit["changed"]
            else:
                fused = visual[shortlist] + text_weight * text[shortlist]
                order = baseline.copy()
                order[:len(shortlist)] = shortlist[np.argsort(-fused, kind="stable")]
                audit = {
                    "gate_open": True,
                    "changed": int(order[0]) != int(baseline[0]),
                    "visual_margin": float(visual[baseline[0]] - visual[baseline[1]]),
                    "text_advantage": float(text[order[0]] - text[baseline[0]]),
                }
                apply_text = True
            ranked_animals = [identities[index] for index in order]
            rank = ranked_animals.index(animals[query]) + 1
            photo_scores = scores[query].copy()
            if apply_text:
                for identity_index in shortlist:
                    animal = identities[identity_index]
                    for index in gallery:
                        if animals[index] == animal:
                            photo_scores[index] += text_weight * text[identity_index]
            photo_order = sorted(gallery, key=lambda index: photo_scores[index], reverse=True)
            hits, precisions = 0, []
            for position, index in enumerate(photo_order, 1):
                if animals[index] == animals[query]:
                    hits += 1
                    precisions.append(hits / position)
            baseline_correct = identities[baseline[0]] == animals[query]
            reranked_correct = ranked_animals[0] == animals[query]
            own_index = identities.index(animals[query])
            rival_index = next(index for index in order if identities[index] != animals[query])
            rows.append({
                "query": query, "animal": animals[query], "species": sp, "rank": rank,
                "ap": float(np.mean(precisions)), "gallery_animals": len(identities),
                "own_best": float(visual[own_index]), "rival": identities[rival_index],
                "rival_best": float(visual[rival_index]),
                "baseline_top1": identities[baseline[0]], "reranked_top1": ranked_animals[0],
                "gate_open": bool(audit["gate_open"]), "changed": bool(audit["changed"]),
                "rescued": bool(not baseline_correct and reranked_correct),
                "damaged": bool(baseline_correct and not reranked_correct),
                "visual_margin": audit["visual_margin"], "text_advantage": audit["text_advantage"],
            })
    result = {
        "global": metrics._summary(rows, metrics.KS),
        **{sp: metrics._summary([row for row in rows if row["species"] == sp], metrics.KS)
           for sp in sorted({row["species"] for row in rows})},
        "per_query": rows,
    }
    result["audit"] = {
        "queries": len(rows),
        "gate_activation_rate": float(np.mean([row["gate_open"] for row in rows])) if rows else 0.0,
        "rescued_queries": sum(row["rescued"] for row in rows),
        "damaged_queries": sum(row["damaged"] for row in rows),
        "changed_queries": sum(row["changed"] for row in rows),
    }
    return result


def calibrate(validation, attributes, grid):
    """Choose zero-damage double-gate settings on validation, then Rank-1 and mAP."""
    candidates = []
    for tau_v, text_weight, tau_t in grid:
        result = rerank_evaluation(
            validation, attributes, text_weight=text_weight, gated=True,
            visual_margin_threshold=tau_v, text_advantage_threshold=tau_t)
        candidates.append({
            "tau_v": tau_v, "lambda": text_weight, "tau_t": tau_t,
            "rank1": result["global"]["rank1"], "mAP": result["global"]["mAP"],
            **result["audit"],
        })
    if not candidates:
        raise ValueError("Calibration grid cannot be empty")
    minimum_damage = min(candidate["damaged_queries"] for candidate in candidates)
    pool = [candidate for candidate in candidates if candidate["damaged_queries"] == minimum_damage]
    selected = max(pool, key=lambda row: (
        row["rank1"], row["mAP"], -row["damaged_queries"],
        -row["gate_activation_rate"], -row["lambda"], -row["tau_v"], -row["tau_t"]))
    return selected, candidates
