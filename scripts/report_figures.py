"""Figures and tables for the thesis report from one model on one dataset version.

    make report DATASET=v2 MODEL=models:/pettrace-embedder/2 NAME=e2_v2 RUN=<evaluate run id>

runs the three steps in the project image:
    dump     embed with the registered pipeline, write scores to reports/<NAME>.json
    plot     figures + summary.json into reports/<NAME>/
    publish  attach them to the evaluate.py run in MLflow (artifacts/report/)

Same zero-shot protocol as evaluate.py (every photo is a query, same-species gallery,
leave-one-out); species with a single animal are skipped.
"""

import argparse
import json
import os
import sys


def dump(args) -> None:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import evaluate as ev  # noqa: E402
    from reid import data, metrics  # noqa: E402
    from reid.pipeline import Pipeline  # noqa: E402

    ds = data.from_dir(args.dataset) if os.path.isdir(args.dataset) else data.load(args.dataset)
    pipeline = Pipeline(ev.model_dir(args.model))
    photos = ds.photos
    E = ev.embed(pipeline, photos)
    animals, species = [p.animal for p in photos], [p.species for p in photos]
    result = ev.evaluate_zero_shot(ds, pipeline, {"dataset": ds.name, "splits": {}, "protocol": "zero-shot"})
    same, diff = metrics.pair_scores(E, animals, species)
    out = {
        "dataset": ds.name,
        "commit": ds.commit,
        "model": args.model,
        "model_version": pipeline.spec.get("model_version", ""),
        "photos": [{"animal": a, "species": s, "path": os.path.relpath(p.path, ds.root)} for a, s, p in zip(animals, species, photos)],
        "similarity": (E @ E.T).round(5).tolist(),
        "same": same.round(5).tolist(),
        "diff": diff.round(5).tolist(),
        "result": result,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, default=float)
    print(f"wrote {args.out}: {len(photos)} photos, {len(same)} same / {len(diff)} diff pairs")


def plot(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    d = json.load(open(args.json))
    outdir = args.outdir or os.path.splitext(args.json)[0]
    os.makedirs(outdir, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "figure.dpi": 200, "axes.spines.top": False, "axes.spines.right": False})
    rows = d["result"]["test"]["per_query"]
    species_of = {r["animal"]: r["species"] for r in rows}
    animals = sorted(species_of, key=lambda a: (species_of[a], a))  # grouped by species
    label = f"{d['model_version']} · {d['dataset']}"
    colors = {"cat": "#0F766E", "dog": "#B45309"}
    names_es = {"cat": "gatos", "dog": "perros"}

    # Rank-1 per animal, colored by species
    fig, ax = plt.subplots(figsize=(max(6.5, 0.5 * len(animals) + 1.5), 3.4))
    hits = [np.mean([r["rank"] == 1 for r in rows if r["animal"] == a]) * 100 for a in animals]
    bars = ax.bar(animals, hits, color=[colors[species_of[a]] for a in animals])
    ax.bar_label(bars, fmt="%.0f%%", fontsize=8)
    ax.axhline(100 / max(r["gallery_animals"] for r in rows), ls="--", color="#6B7280", label="azar")
    for sp in sorted(set(species_of.values())):
        ax.bar(0, 0, color=colors[sp], label=names_es[sp])
    ax.set_ylim(0, 115)
    ax.set_ylabel("Rank-1 (%)")
    ax.set_xticks(range(len(animals)), [a.capitalize() for a in animals], rotation=45, ha="right")
    ax.set_title(f"Rank-1 por animal — {label}", fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout()
    fig.savefig(f"{outdir}/rank1_por_animal.png")

    # Same vs different score distributions + EER threshold
    same, diff = np.array(d["same"]), np.array(d["diff"])
    thr = d["result"]["val"]["threshold"]
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
    fig.savefig(f"{outdir}/distribucion_scores.png")

    # CMC curve
    max_rank = max(r["gallery_animals"] for r in rows)
    ranks = np.array([r["rank"] for r in rows])
    cmc = [np.mean(ranks <= k) * 100 for k in range(1, max_rank + 1)]
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.plot(range(1, max_rank + 1), cmc, marker="o", color="#0F766E")
    for k, v in enumerate(cmc, 1):
        ax.annotate(f"{v:.0f}%", (k, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9)
    ax.set_xticks(range(1, max_rank + 1))
    ax.set_ylim(0, 110)
    ax.set_xlabel("rango k")
    ax.set_ylabel("aciertos ≤ k (%)")
    ax.set_title(f"Curva CMC — {label}", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{outdir}/cmc.png")

    # Mean similarity between animals: one panel per species (species are never compared)
    sim = np.array(d["similarity"])
    groups = [sp for sp in ("cat", "dog") if sp in species_of.values()]
    fig, axes = plt.subplots(1, len(groups), figsize=(4.4 * len(groups), 4.2), squeeze=False)
    for ax, sp in zip(axes[0], groups):
        names = [a for a in animals if species_of[a] == sp]
        idx = {a: [i for i, p in enumerate(d["photos"]) if p["animal"] == a] for a in names}
        M = np.array([[sim[np.ix_(idx[a], idx[b])].mean() for b in names] for a in names])
        im = ax.imshow(M, cmap="viridis", vmin=0.6, vmax=0.95)
        caps = [n.capitalize() for n in names]
        ax.set_xticks(range(len(names)), caps, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(names)), caps, fontsize=9)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] < 0.8 else "black")
        ax.set_title(names_es[sp].capitalize(), fontsize=11)
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025)
    fig.suptitle("Similitud media entre animales (diagonal: mismo animal)", fontsize=11)
    fig.savefig(f"{outdir}/matriz_similitud.png", bbox_inches="tight")

    # Summary for the report tables
    summary = {
        "global": {k: v for k, v in d["result"]["test"]["global"].items()},
        "per_species": {sp: d["result"]["test"][sp] for sp in ("cat", "dog") if sp in d["result"]["test"]},
        "per_animal": {a: {"rank1": float(np.mean([r["rank"] == 1 for r in rows if r["animal"] == a])),
                           "queries": sum(r["animal"] == a for r in rows),
                           "median_margin": float(np.median([r["own_best"] - r["rival_best"] for r in rows if r["animal"] == a])),
                           "rivals": {x: sum(1 for r in rows if r["animal"] == a and r["rank"] > 1 and r["rival"] == x) for x in animals if x != a}}
                       for a in animals},
        "eer": d["result"]["val"]["eer"], "threshold": thr,
        "pairs_same": len(same), "pairs_diff": len(diff),
        "same_range": [float(same.min()), float(same.max())], "diff_range": [float(diff.min()), float(diff.max())],
        "dataset": d["dataset"], "commit": d["commit"], "model_version": d["model_version"],
    }
    json.dump(summary, open(f"{outdir}/summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


def publish(args) -> None:
    import mlflow

    outdir = args.outdir or os.path.splitext(args.json)[0]
    with mlflow.start_run(run_id=args.run):
        mlflow.log_artifact(args.json, artifact_path="report")
        mlflow.log_artifacts(outdir, artifact_path="report")
    print(f"published {outdir} + {os.path.basename(args.json)} to run {args.run} (artifacts/report)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("dump")
    a.add_argument("--dataset", required=True)
    a.add_argument("--model", default="models:/pettrace-embedder@production")
    a.add_argument("--out", required=True)
    b = sub.add_parser("plot")
    b.add_argument("json")
    b.add_argument("--outdir")
    c = sub.add_parser("publish")
    c.add_argument("json")
    c.add_argument("--run", required=True, help="evaluate.py run id")
    c.add_argument("--outdir")
    args = ap.parse_args()
    {"dump": dump, "plot": plot, "publish": publish}[args.cmd](args)


if __name__ == "__main__":
    main()
