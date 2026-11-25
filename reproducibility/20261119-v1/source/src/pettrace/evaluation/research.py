"""Research-specific views of immutable prepared data; no preparation or training here.

PetTrace's closed-set, same-species protocol is an explicit study decision. CMC/mAP
background: Zheng et al., ICCV 2015, Scalable Person Re-Identification: A Benchmark.
"""

import hashlib
import json
import re
from pathlib import Path

from pettrace.data.benchmark import Dataset, Photo

PUBLIC_REPO = "alessandro54/pettrace-training"
PUBLIC_VERSION = "v1"
PUBLIC_COMMIT = "dd7a00cb8b3638759ffa0216251e62d334b5bf10"
PUBLIC_FINGERPRINT = "38ef098d26afa792a1712f9e37487d143f60a9a3bc647a5fdde5505f790f8c7d"
REFINED_PUBLIC_REPO = "composite/lcw+dow-refined"
REFINED_PUBLIC_VERSION = "v2"


def refined_public_receipt(prepared):
    """Describe the two pinned refined sources as one reproducible benchmark."""
    provenance = prepared.provenance
    sources = provenance.get("sources", [])
    if prepared.version != REFINED_PUBLIC_VERSION or len(sources) != 2:
        raise ValueError("Public refined evaluation requires both audited v2 sources")
    identity = {
        "version": prepared.version,
        "prepared_fingerprint": prepared.fingerprint,
        "sources": sources,
    }
    return {
        "repo": REFINED_PUBLIC_REPO,
        "version": REFINED_PUBLIC_VERSION,
        "commit": hashlib.sha1(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
        "fingerprint": prepared.fingerprint,
        "sources": sources,
    }


def public_split_dataset(prepared, receipt, split):
    """Expose one published identity-disjoint split without resampling it.

    The caller must use load_training/load_local_training to verify the snapshot first.
    Only ``val`` may select E5 parameters; ``test`` remains sealed until selection is frozen.
    """
    if split not in {"val", "test"}:
        raise ValueError("Public evaluation split must be val or test")
    if (not receipt or not re.fullmatch(r"[0-9a-f]{40}", receipt.get("commit", ""))
            or receipt.get("fingerprint") != prepared.fingerprint):
        raise ValueError("Public evaluation requires a pinned, matching publication receipt")
    rows = [row for row in prepared.rows if row["split"] == split]
    other = {row["animal"] for row in prepared.rows if row["split"] != split}
    if not rows or other.intersection(row["animal"] for row in rows):
        raise ValueError(f"Public {split} must be nonempty and identity-disjoint from other splits")
    if {row["species"] for row in rows} != {"cat", "dog"}:
        raise ValueError("Public test requires both cats and dogs")
    animals = {}
    for row in rows:
        if row["animal"] in animals and animals[row["animal"]]["species"] != row["species"]:
            raise ValueError("An identity has conflicting species")
        animals[row["animal"]] = {"species": row["species"], "split": split, "source": row["source"]}
    identity = {"prepared_fingerprint": prepared.fingerprint, "split": split,
                "rows": sorted((r["animal"], r["species"], r["sha256"]) for r in rows)}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return Dataset(receipt["repo"], receipt["version"], receipt["commit"], digest,
                   str(getattr(prepared, "root", "")),
                   animals, [Photo(str(r["path"]), r["animal"], r["species"], split, 0.0) for r in rows])


def public_test_dataset(prepared, receipt):
    """Expose ONLY the published test identities; never resplit or include train/val."""
    return public_split_dataset(prepared, receipt, "test")


def save_benchmark_comparison(public_ev, private_ev, output, label):
    """Compare domains without combining their galleries or inferring a domain causal effect."""
    from pettrace.evaluation.figures import comparison_figure

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table = ["| Dataset | Group | Queries | Gallery (identities) | Rank-1 | Rank-2 | mAP |",
             "|---|---|---:|---:|---:|---:|---:|"]
    records = []
    evaluations = {f"Public {public_ev.dataset.version} · test": public_ev,
                   f"PetTrace {private_ev.dataset.version}": private_ev}
    for name, ev in evaluations.items():
        for scope in ("global", "cat", "dog"):
            values = ev.result["test"].get(scope)
            if not values or not values.get("queries"):
                continue
            row = {"domain": name, "scope": scope, "dataset": ev.dataset.name,
                   "dataset_commit": ev.dataset.commit, "dataset_fingerprint": ev.dataset.fingerprint,
                   **{key: values[key] for key in ("queries", "gallery_animals", "rank1", "rank2", "mAP")}}
            records.append(row)
            table.append(f"| {name} | {scope} | {row['queries']} | {row['gallery_animals']} | "
                         f"{row['rank1']:.2%} | {row['rank2']:.2%} | {row['mAP']:.4f} |")
    (output / "benchmark-comparison.json").write_text(json.dumps({
        "model": label, "gallery_policy": "separate_same_species_leave_one_out",
        "limitation": "Different identities, gallery sizes and capture conditions; no causal domain attribution.",
        "records": records}, indent=2))
    figure = comparison_figure(evaluations, label)
    figure.savefig(output / "benchmark-comparison.png", dpi=180, bbox_inches="tight")
    return "\n".join(table), figure
