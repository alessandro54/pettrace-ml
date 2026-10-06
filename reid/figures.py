"""Thesis figures and tables from one Evaluation (needs the `figures` dependency group).

    figs = all_figures(ev, label)          # name → matplotlib Figure (shown inline in notebooks)
    s = summary(ev, model)                 # the numbers the report tables and the thesis use
    save(ev, model, "reports/e1_v3")       # summary.json + scores.json + PNGs
"""

import json
import os

import numpy as np

COLORS = {"cat": "#0F766E", "dog": "#B45309"}
SPECIES_ES = {"cat": "gatos", "dog": "perros"}


def _rows(ev) -> list[dict]:
    return ev.result["test"]["per_query"]


def _animals(ev) -> tuple[list[str], dict[str, str]]:
    species_of = {r["animal"]: r["species"] for r in _rows(ev)}
    return sorted(species_of, key=lambda a: (species_of[a], a)), species_of


def _mean_similarity(ev) -> dict:
    """Per species: mean cosine of photo pairs of the same animal vs of different animals."""
    out = {}
    sim = ev.embeddings @ ev.embeddings.T
    animal = np.array([p.animal for p in ev.photos])
    species = np.array([p.species for p in ev.photos])
    off = ~np.eye(len(animal), dtype=bool)
    for sp in ("cat", "dog"):
        m = species == sp
        if m.sum() < 2:
            continue
        pair = np.outer(m, m) & off
        same = pair & (animal[:, None] == animal[None, :])
        out[sp] = {"same": float(sim[same].mean()), "diff": float(sim[pair & ~same].mean())}
    return out


def summary(ev, model=None) -> dict:
    rows, (animals, _) = _rows(ev), _animals(ev)
    test = ev.result["test"]
    return {
        "global": test["global"],
        "per_species": {sp: test[sp] for sp in ("cat", "dog") if sp in test},
        "per_animal": {a: {"rank1": float(np.mean([r["rank"] == 1 for r in rows if r["animal"] == a])),
                           "queries": sum(r["animal"] == a for r in rows),
                           "median_margin": float(np.median([r["own_best"] - r["rival_best"] for r in rows if r["animal"] == a])),
                           "rivals": {x: sum(1 for r in rows if r["animal"] == a and r["rank"] > 1 and r["rival"] == x)
                                      for x in animals if x != a}}
                       for a in animals},
        "eer": ev.result["val"]["eer"], "threshold": ev.result["val"]["threshold"],
        "pairs_same": len(ev.same), "pairs_diff": len(ev.diff),
        "same_range": [float(ev.same.min()), float(ev.same.max())],
        "diff_range": [float(ev.diff.min()), float(ev.diff.max())],
        "mean_similarity": _mean_similarity(ev),
        "dataset": ev.dataset.name, "commit": ev.dataset.commit, "protocol": ev.protocol,
        "model_version": model.model_version if model else "",
        "model_sha256": model.sha256 if model else {},
    }


def rank1_per_animal(ev, label: str):
    import matplotlib.pyplot as plt

    rows, (animals, species_of) = _rows(ev), _animals(ev)
    fig, ax = plt.subplots(figsize=(max(6.5, 0.5 * len(animals) + 1.5), 3.4))
    hits = [np.mean([r["rank"] == 1 for r in rows if r["animal"] == a]) * 100 for a in animals]
    bars = ax.bar(animals, hits, color=[COLORS[species_of[a]] for a in animals])
    ax.bar_label(bars, fmt="%.0f%%", fontsize=8)
    ax.axhline(100 / max(r["gallery_animals"] for r in rows), ls="--", color="#6B7280", label="azar")
    for sp in sorted(set(species_of.values())):
        ax.bar(0, 0, color=COLORS[sp], label=SPECIES_ES[sp])
    ax.set_ylim(0, 115)
    ax.set_ylabel("Rank-1 (%)")
    ax.set_xticks(range(len(animals)), animals, rotation=45, ha="right")
    ax.set_title(f"Rank-1 por animal — {label}", fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout()
    return fig


def score_distribution(ev, label: str):
    import matplotlib.pyplot as plt

    same, diff, thr = ev.same, ev.diff, ev.result["val"]["threshold"]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    bins = np.linspace(min(same.min(), diff.min()), max(same.max(), diff.max()), 30)
    ax.hist(diff, bins=bins, alpha=0.6, density=True, color="#E4572E", label=f"distinto animal (n={len(diff)})")
    ax.hist(same, bins=bins, alpha=0.6, density=True, color="#0F766E", label=f"mismo animal (n={len(same)})")
    ax.axvline(thr, color="black", ls="--", label=f"umbral EER = {thr:.3f}")
    ax.set_xlabel("similitud coseno")
    ax.set_ylabel("densidad")
    ax.set_title(f"Distribución de similitudes — {label}", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def cmc(ev, label: str):
    import matplotlib.pyplot as plt

    rows = _rows(ev)
    max_rank = max(r["gallery_animals"] for r in rows)
    ranks = np.array([r["rank"] for r in rows])
    curve = [np.mean(ranks <= k) * 100 for k in range(1, max_rank + 1)]
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.plot(range(1, max_rank + 1), curve, marker="o", color="#0F766E")
    for k, v in enumerate(curve, 1):
        ax.annotate(f"{v:.0f}%", (k, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9)
    ax.set_xticks(range(1, max_rank + 1))
    ax.set_ylim(0, 110)
    ax.set_xlabel("rango k")
    ax.set_ylabel("aciertos ≤ k (%)")
    ax.set_title(f"Curva CMC — {label}", fontsize=11)
    fig.tight_layout()
    return fig


def similarity_matrix(ev, label: str = ""):
    """Mean similarity between animals, one panel per species (species are never compared)."""
    import matplotlib.pyplot as plt

    animals, species_of = _animals(ev)
    sim = ev.embeddings @ ev.embeddings.T
    groups = [sp for sp in ("cat", "dog") if sp in species_of.values()]
    fig, axes = plt.subplots(1, len(groups), figsize=(4.4 * len(groups), 4.2), squeeze=False)
    im = None
    for ax, sp in zip(axes[0], groups):
        names = [a for a in animals if species_of[a] == sp]
        idx = {a: [i for i, p in enumerate(ev.photos) if p.animal == a] for a in names}
        M = np.array([[sim[np.ix_(idx[a], idx[b])].mean() for b in names] for a in names])
        im = ax.imshow(M, cmap="viridis", vmin=0.6, vmax=0.95)
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(names)), names, fontsize=9)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] < 0.8 else "black")
        ax.set_title(SPECIES_ES[sp].capitalize(), fontsize=11)
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025)
    fig.suptitle(f"Similitud media entre animales (diagonal: mismo animal){' — ' + label if label else ''}", fontsize=11)
    return fig


def confusion(ev, label: str = ""):
    """Top-1 confusion per species: row = queried animal, column = animal ranked first (the
    diagonal is a hit). Cells show the number of queries, colour the share of the row."""
    import matplotlib.pyplot as plt

    rows, (animals, species_of) = _rows(ev), _animals(ev)
    groups = [sp for sp in ("cat", "dog") if sp in species_of.values()]
    fig, axes = plt.subplots(1, len(groups), figsize=(4.9 * len(groups), 4.4), squeeze=False,
                             gridspec_kw={"wspace": 0.35})
    im = None
    for k, (ax, sp) in enumerate(zip(axes[0], groups)):
        names = [a for a in animals if species_of[a] == sp]
        pos = {a: i for i, a in enumerate(names)}
        C = np.zeros((len(names), len(names)), dtype=int)
        for r in rows:
            if r["species"] == sp:
                C[pos[r["animal"]], pos[r["animal"] if r["rank"] == 1 else r["rival"]]] += 1
        share = C / np.maximum(C.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(names)), names, fontsize=9)
        for i in range(len(names)):
            for j in range(len(names)):
                if C[i, j]:
                    ax.text(j, i, str(C[i, j]), ha="center", va="center", fontsize=9,
                            color="white" if share[i, j] > 0.5 else "black", fontweight="bold" if i == j else None)
        ax.set_xlabel("primer lugar del ranking")
        if k == 0:
            ax.set_ylabel("animal consultado")
        ax.set_title(f"{SPECIES_ES[sp].capitalize()} (aciertos {np.trace(C)}/{C.sum()})", fontsize=11)
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, label="proporción de la fila")
    fig.suptitle(f"Confusión top-1 por especie{' — ' + label if label else ''}", fontsize=11)
    return fig


def all_figures(ev, label: str) -> dict:
    return {"confusion": confusion(ev, label), "rank1_por_animal": rank1_per_animal(ev, label),
            "distribucion_scores": score_distribution(ev, label), "cmc": cmc(ev, label),
            "matriz_similitud": similarity_matrix(ev)}


def save(ev, model, outdir: str, label: str | None = None) -> dict:
    """summary.json, scores.json (per-query rows + pair scores, no images) and the PNGs."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    os.makedirs(outdir, exist_ok=True)
    s = summary(ev, model)
    json.dump(s, open(os.path.join(outdir, "summary.json"), "w"), indent=2, default=float)
    json.dump({"result": ev.result, "same": ev.same.round(5).tolist(), "diff": ev.diff.round(5).tolist(),
               "photos": [{"animal": p.animal, "species": p.species, "path": os.path.relpath(p.path, ev.dataset.root)}
                          for p in ev.photos]},
              open(os.path.join(outdir, "scores.json"), "w"), default=float)
    label = label or f"{s['model_version']} · {ev.dataset.name}"
    for name, fig in all_figures(ev, label).items():
        fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=200, bbox_inches="tight")
    return s


# ---- Markdown tables (notebooks render them with IPython.display.Markdown)

def _pct(x: float) -> str:
    return f"{x * 100:.1f} %"


def metrics_table(summaries: dict[str, dict]) -> str:
    """One column per experiment: {"E1": summary, "E2": summary}."""
    names = list(summaries)
    sp = {n: summaries[n]["per_species"] for n in names}
    chance = 1 / next(iter(summaries.values()))["global"]["gallery_animals"]
    rows = [
        ("Consultas / galería por especie", lambda s, n: f"{s['global']['queries']} / {s['global']['gallery_animals']} animales"),
        (f"Rank-1 global (azar {_pct(chance)})", lambda s, n: f"**{_pct(s['global']['rank1'])}**"),
        ("Rank-1 gatos / perros", lambda s, n: f"{_pct(sp[n]['cat']['rank1'])} / {_pct(sp[n]['dog']['rank1'])}"),
        (f"Rank-5 global (azar {_pct(5 * chance)})", lambda s, n: _pct(s["global"]["rank5"])),
        ("mAP global", lambda s, n: f"{s['global']['mAP']:.3f}"),
        ("mAP gatos / perros", lambda s, n: f"{sp[n]['cat']['mAP']:.3f} / {sp[n]['dog']['mAP']:.3f}"),
        ("EER / umbral", lambda s, n: f"{_pct(s['eer'])} / {s['threshold']:.3f}"),
        ("Similitud mínima, mismo animal", lambda s, n: f"{s['same_range'][0]:.3f}"),
        ("Similitud máxima, distinto animal", lambda s, n: f"{s['diff_range'][1]:.3f}"),
    ]
    out = ["| Métrica | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for title, f in rows:
        out.append(f"| {title} | " + " | ".join(f(summaries[n], n) for n in names) + " |")
    return "\n".join(out)


def per_animal_table(summaries: dict[str, dict]) -> str:
    """Rank-1 hits per animal for each experiment + who the last one confuses it with."""
    names = list(summaries)
    last = summaries[names[-1]]["per_animal"]
    out = ["| Animal | " + " | ".join(names) + f" | {names[-1]}: confundido con |", "|---" * (len(names) + 2) + "|"]
    for a in sorted(last):
        hits = [f"{round(summaries[n]['per_animal'][a]['rank1'] * summaries[n]['per_animal'][a]['queries'])}"
                f"/{summaries[n]['per_animal'][a]['queries']}" for n in names]
        riv = ", ".join(f"{k} ({v})" for k, v in sorted(last[a]["rivals"].items(), key=lambda kv: -kv[1]) if v) or "—"
        out.append(f"| {a} | " + " | ".join(hits) + f" | {riv} |")
    return "\n".join(out)


# ---- Captions (Spanish, for the notebooks): how to read each figure and what this run shows

def _list(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " y " + items[-1] if items else ""


def explain(name: str, s: dict) -> str:
    pa, g, sp = s["per_animal"], s["global"], s["per_species"]
    n = g["gallery_animals"]
    chance = 1 / n
    hits = lambda a: round(pa[a]["rank1"] * pa[a]["queries"])  # noqa: E731
    if name == "confusion":
        pairs = [(c, a, b) for a in pa for b, c in pa[a]["rivals"].items() if c]
        pairs.sort(reverse=True)
        top = _list([f"`{a}` → `{b}` ({c} de {pa[a]['queries']})" for c, a, b in pairs[:3]])
        per = " y ".join(f"{round(sp[x]['rank1'] * sp[x]['queries'])}/{sp[x]['queries']} en "
                         f"{SPECIES_ES[x]}" for x in sp)
        return ("**Cómo leerlo.** Cada fila es el animal consultado y cada columna el animal que el modelo puso en "
                "primer lugar; el número es la cantidad de consultas. La diagonal son los aciertos y todo lo que "
                "queda fuera de ella son confusiones; solo se compara dentro de cada especie, como en la app. "
                f"**En esta ejecución** acierta {per}. Las confusiones más frecuentes: {top}. Si se concentran "
                "dentro de un grupo de parecido, el error es de pelaje similar y no del método.")
    if name == "rank1_por_animal":
        best = [a for a in sorted(pa) if pa[a]["rank1"] == 1.0]
        worst = sorted(pa, key=lambda a: pa[a]["rank1"])[:3]
        return ("**Cómo leerlo.** Cada barra es el porcentaje de consultas de ese animal en que el animal correcto "
                f"quedó primero; la línea punteada es el azar ({chance:.1%} con {n} animales por especie). "
                "Muestra si el promedio esconde animales mucho más difíciles que otros. "
                f"**En esta ejecución** {_list([f'`{a}`' for a in best]) or 'ningún animal'} aciertan todas sus "
                "consultas y los más difíciles son "
                + _list([f"`{a}` ({hits(a)}/{pa[a]['queries']})" for a in worst]) + ".")
    if name == "distribucion_scores":
        return ("**Cómo leerlo.** Histogramas de la similitud coseno entre pares de fotos del mismo animal (verde) y "
                "de animales distintos de la misma especie (rojo); la línea punteada es el umbral donde ambos "
                "errores se igualan (EER). Cuanto menos se solapan, mejor separa el modelo identidades con un "
                f"umbral fijo. **En esta ejecución** la EER es {s['eer']:.1%} con umbral {s['threshold']:.3f}: "
                f"pares del mismo animal van de {s['same_range'][0]:.3f} a {s['same_range'][1]:.3f} y de animales "
                f"distintos de {s['diff_range'][0]:.3f} a {s['diff_range'][1]:.3f}, así que un umbral fijo no "
                "basta y el producto muestra un ranking que el dueño confirma.")
    if name == "cmc":
        return ("**Cómo leerlo.** Curva CMC: para cada k, el porcentaje de consultas en que el animal correcto "
                "aparece entre los k primeros del ranking. Es lo que ve el dueño: cuántos candidatos debe revisar. "
                f"Con {n} animales por especie, en k = {n} siempre es 100 %. **En esta ejecución** el correcto "
                f"está primero en {g['rank1']:.1%} de las consultas (azar {chance:.1%}) y entre los 5 primeros en "
                f"{g['rank5']:.1%} (azar {5 * chance:.1%}).")
    if name == "matriz_similitud":
        ms = s.get("mean_similarity", {})
        detail = "; ".join(f"en {SPECIES_ES[x]} {v['same']:.3f} (mismo) frente a {v['diff']:.3f} (distinto)"
                           for x, v in ms.items())
        return ("**Cómo leerlo.** Similitud coseno media entre las fotos de cada par de animales; la diagonal es el "
                "mismo animal. Una diagonal clara sobre un fondo oscuro indica que el embedding agrupa por "
                "identidad; celdas claras fuera de la diagonal señalan animales que el modelo ve parecidos. "
                f"**En esta ejecución** la similitud media es {detail}: la diferencia es pequeña, por eso el "
                "ranking funciona pero un umbral fijo no.")
    return ""
