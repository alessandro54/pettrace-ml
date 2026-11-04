"""Paired E3 curation check on an already verified E3 notebook kernel.

Call run(globals(), output) after loading E2 and both immutable benchmarks in E3.
Re-running with the same output retries publication and reuses completed fits.
It never acquires raw data, changes splits, or selects a model using test results.
"""
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time
from types import SimpleNamespace

import numpy as np

SEEDS = (42, 43, 44)
ORIGINS = {
    "v2.50": "ab63e3732cb64f2e9ba6fbfa3584f4e0",
    "v3.50": "50770cc30ff84aadad4840e502339fe2",
}
MODEL_KEY = "34b5868b55efae9fabf7b79c3e49c6d3c91edf5895a89ba784872c1c80975a26"
PREPARATION_EVIDENCE = (
    "manifest.csv", "review.csv", "prepared.json", "audit.json", "summary.json",
    "excluded.csv", "sources.json", "animal-screening.json", "lookup.json",
    "requirements-e3-runtime.txt", "dataset-coverage.json",
)


def write_json(path, value):
    partial = path.with_suffix(path.suffix + ".tmp")
    partial.write_text(json.dumps(value, indent=2) + "\n")
    partial.replace(path)


def copy_preparation_evidence(source, report):
    destination = Path(report) / "preparation"
    destination.mkdir(exist_ok=True)
    for filename in PREPARATION_EVIDENCE:
        path = Path(source) / filename
        if path.is_file():
            shutil.copy2(path, destination / filename)


def cached_vectors(manifest, config, database):
    """Use only the immutable manifest and checksum-verified cached E2 vectors."""
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != config["manifest_sha256"]:
        raise ValueError("Manifest differs from the published experiment")
    with manifest.open() as source:
        all_rows = list(csv.DictReader(source))
    rows = [r for r in all_rows if r["split"] in {"train", "val"}]
    identity = [{k: r[k] for k in ("animal", "species", "split", "sha256")} for r in rows]
    digest = hashlib.sha256(json.dumps({"model": MODEL_KEY, "rows": identity}, sort_keys=True).encode()).hexdigest()
    if digest != config["cache_sha256"]:
        raise ValueError("Feature identity/order differs from the published experiment")
    values = []
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as cache:
        for row in rows:
            saved = cache.execute("SELECT vector, checksum FROM embeddings WHERE image=?", (row["sha256"],)).fetchone()
            if saved is None:
                raise ValueError("Missing frozen embedding; do not start an unplanned extraction")
            blob, checksum = saved
            if len(blob) != 2048 or hashlib.sha256(blob).hexdigest() != checksum:
                raise ValueError("Corrupt frozen embedding")
            values.append(np.frombuffer(blob, dtype=np.float32))
    values = np.stack(values)
    if not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-3):
        raise ValueError("Invalid frozen embeddings")
    return all_rows, rows, values


def image_key(image):
    return image.mode, image.size, hashlib.sha256(image.tobytes()).hexdigest()


def run(context, output, *, prepared_inputs=None):
    import torch
    from PIL import Image, ImageOps
    from mlflow.entities import Metric, Param
    from pettrace.data.curation import sha256
    from pettrace.data.checkpoints import DriveBackup
    from pettrace.evaluation.core import embed, evaluate, load_model
    from pettrace.evaluation.figures import save
    from pettrace.experiments.e3 import frozen_controls
    from pettrace.training.features import feature_cache
    from pettrace.training.triplet import Config, Head, fit, project, export_encoder
    from pettrace.tracking import production_context, source_identity, save_provenance, publish_evaluation

    client, _ = production_context()
    base = context["base_model"]
    datasets = {"public": context["public_ds"], "private": context["ds"]}
    if not torch.cuda.is_available() or not base.golden_ok:
        raise ValueError("A verified E2 model and GPU are required")
    if not context["REGISTER_E3"] or not context["LOG_TO_MLFLOW"]:
        raise ValueError("This closure preserves model registration and mandatory publication")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(context["FEATURE_CACHE_ROOT"])
    database = cache_dir / f"frozen-e2-{MODEL_KEY}.sqlite3"
    source = source_identity()
    driver_sha = sha256(__file__)
    origins = ORIGINS if prepared_inputs is None else {
        name: "prepared:" + json.loads((Path(folder) / "training-config.json").read_text())["prepared_dataset_fingerprint"]
        for name, folder in prepared_inputs.items()}
    variants = {}
    reference = None
    for name, origin in origins.items():
        folder = output / "inputs" / name if prepared_inputs is None else Path(prepared_inputs[name])
        folder.mkdir(parents=True, exist_ok=True)
        if prepared_inputs is None:
            for filename in ("manifest.csv", "training-config.json", "dataset-coverage.json"):
                if not (folder / filename).exists():
                    client.download_artifacts(origin, f"report/{filename}", str(folder))
        config = json.loads((folder / "training-config.json").read_text())
        if config["base_model_sha256"] != base.sha256 or config["training"] != asdict(Config()):
            raise ValueError("Base model or historical hyperparameters differ")
        if (config["dataset_commit"] != datasets["private"].commit
                or config["dataset_fingerprint"] != datasets["private"].fingerprint):
            raise ValueError("Private benchmark differs from the historical experiments")
        all_rows, rows, features = cached_vectors(folder / "manifest.csv", config, database)
        variants[name] = (config, rows, features)
        # Require the original immutable validation/test, not only matching counts.
        held_out = sorted((r["animal"], r["species"], r["split"], r["sha256"])
                          for r in all_rows if r["split"] in {"val", "test"})
        if reference is None:
            reference = held_out
        elif held_out != reference:
            raise ValueError("Validation/test differs between curation versions")
        train = [r for r in rows if r["split"] == "train"]
        if ({r["animal"] for r in train} & {r[0] for r in held_out}
                or {r["sha256"] for r in train} & {r[3] for r in held_out}):
            raise ValueError("Training overlaps held-out data")
        if {r["sha256"] for r in all_rows} & {sha256(p.path) for p in datasets["private"].photos}:
            raise ValueError("Private images overlap training data")
    plan = {"seeds": SEEDS, "origins": origins, "training": asdict(Config()),
            "source": source, "driver_sha256": driver_sha, "selection": "public validation macro-species mAP",
            "comparison": ("curation at 50 percent; not a volume effect" if prepared_inputs is None
                           else "source-photo volume with a common automatic curation policy"),
            "register_model": True, "launcher": context.get("launcher", {}),
            "benchmarks": {k: {"commit": d.commit, "fingerprint": d.fingerprint} for k, d in datasets.items()},
            "manifest_sha256": {k: v[0]["manifest_sha256"] for k, v in variants.items()}}
    plan = json.loads(json.dumps(plan))
    if (output / "plan.json").exists() and json.loads((output / "plan.json").read_text()) != plan:
        raise ValueError("Closure plan changed; preserve the existing output")
    write_json(output / "plan.json", plan)
    for domain, key in (("public", "public_ev"), ("private", "ev")):
        if key not in context:
            continue  # Standalone series checks identical baseline queries at evaluation below.
        controls = output / "inputs" / domain
        controls.mkdir(exist_ok=True)
        frozen_controls(context[key], domain, controls)
    backup = DriveBackup({"closure": output}, Path("/content/drive/MyDrive/PetTrace/checkpoints/e3-closure") / output.name)
    backup()

    # Fit all predeclared seeds before reading their test metrics.
    for seed in SEEDS:
        for name, (original, rows, features) in variants.items():
            folder = output / f"{name}-seed{seed}"
            folder.mkdir(exist_ok=True)
            config = Config(seed=seed)
            if (folder / "training-summary.json").exists():
                print(f"Reuse completed fit: {name}, seed {seed}", flush=True)
                continue
            if (folder / "head.pt").exists():
                raise RuntimeError("Interrupted fit: keep its evidence and explicitly choose a new output")
            current = {**original, "seed": seed, "training": asdict(config), "source": source,
                       "origin_run_id": origins[name], "driver_sha256": driver_sha}
            write_json(folder / "training-config.json", current)
            print(f"Fit {name}, seed {seed}, {len(rows)} train/val photos", flush=True)
            with backup:
                _, training = fit(features, rows, folder, config, device="cuda", callback=lambda _: backup())
                write_json(folder / "training-summary.json", training)

    # Extract each benchmark's frozen E2 representation once, then reuse it for every head.
    frozen, pixels = {}, {}
    for domain, dataset in datasets.items():
        rows = [{"animal": p.animal, "species": p.species, "split": "test", "path": p.path,
                 "sha256": sha256(p.path)} for p in dataset.photos]
        frozen[domain], _ = feature_cache(base.pipeline, rows, cache_dir, base.dir)
        pixels[domain] = {}
        for p, vector in zip(dataset.photos, frozen[domain]):
            with Image.open(p.path) as im:
                pixels[domain][image_key(ImageOps.exif_transpose(im).convert("RGB"))] = vector

    for seed in SEEDS:
        for name, (original, _, _) in variants.items():
            folder = output / f"{name}-seed{seed}"
            os.environ["PETTRACE_EXECUTION_ID"] = f"{output.name}-{name}-seed{seed}"
            training = json.loads((folder / "training-summary.json").read_text())
            head = Head(training["config"]["hidden"])
            head.load_state_dict(torch.load(folder / "head.pt", map_location="cpu", weights_only=True)["state_dict"])
            head.eval()
            if not (folder / "model" / "golden.npy").exists():
                export_encoder(base.dir, head, folder / "model", f"e3-{name}-seed{seed}@{output.name}")
            model = load_model(str(folder / "model"))
            if not model.golden_ok:
                raise RuntimeError("Exported model failed golden verification")
            # Verify real-image ONNX/PyTorch parity in both domains before using cached projection.
            for domain, dataset in datasets.items():
                actual = embed(model.pipeline, dataset.photos[:8], batch=8)
                if not np.allclose(actual, project(head, frozen[domain][:8]), atol=1e-3):
                    raise RuntimeError("Cached projection differs from exported model")
            runs = {}
            for domain, dataset in datasets.items():
                report = folder / domain
                report.mkdir(exist_ok=True)
                pipeline = SimpleNamespace(embed=lambda images, domain=domain: project(
                    head, np.stack([pixels[domain][image_key(im)] for im in images])))
                evaluation = evaluate(dataset, pipeline, zero_shot=True)
                save(evaluation, model, str(report))
                frozen_controls(evaluation, domain, report)
                write_json(report / "evaluation.json", evaluation.result)
                for filename in ("training-config.json", "training-summary.json", "history.json", "head.pt"):
                    shutil.copy2(folder / filename, report / filename)
                for filename in ("plan.json",):
                    shutil.copy2(output / filename, report / filename)
                shutil.copy2(__file__, report / "repeat_e3.py")
                if prepared_inputs is not None:
                    copy_preparation_evidence(prepared_inputs[name], report)
                    for filename in ("run_e3_volume.py", "volume-plan.json", "volume-series.json"):
                        shutil.copy2(output / filename, report / filename)
                runtime = save_provenance(report, model)
                runtime["driver_sha256"] = driver_sha
                runtime["inference"] = "cached frozen E2 + PyTorch head; real-image ONNX parity verified"
                write_json(report / "runtime.json", runtime)
                with backup:
                    run_id = publish_evaluation(evaluation, model, report, source["source_sha256"],
                        experiment="E3", expected_dataset_version=dataset.version, expected_protocol="zero-shot")
                    timestamp = int(time.time() * 1000)
                    metrics = [Metric(k, float(v), timestamp, int(r["epoch"])) for r in training["history"]
                               for k, v in r.items() if k != "epoch"]
                    metrics += [Metric(k, float(v), timestamp, 0) for k, v in original["dataset_coverage"].items()]
                    metrics += [Metric("training.best_epoch", training["best_epoch"], timestamp, 0),
                                Metric("training.best_val_mAP", training["best_val_mAP"], timestamp, 0)]
                    params = {f"training.{k}": v for k, v in training["config"].items()}
                    params.update({"training.dataset_fingerprint": original["prepared_dataset_fingerprint"],
                                   "training.origin_run": origins[name], "driver.sha256": driver_sha})
                    params.update(original.get("coverage_params", {}))
                    client.log_batch(run_id, metrics=metrics, params=[Param(k, str(v)) for k, v in params.items()])
                    for k, v in {"source_sample_pct": str(original.get("source_sample_pct", 50)),
                                 "curation": name, "closure": output.name,
                                 "training_seed": str(seed)}.items():
                        client.set_tag(run_id, k, v)
                    if client.get_run(run_id).info.status != "FINISHED":
                        raise RuntimeError("Publication did not finish")
                    write_json(report / "publication.json", {"published": True, "run_id": run_id})
                    client.log_artifact(run_id, str(report / "publication.json"), artifact_path="report")
                    runs[domain] = run_id
                    print(f"Published {name} seed {seed} {domain}: {run_id}", flush=True)
            # Register candidates only, after both evaluations are published. Never promote an alias.
            run_id = runs["private"]
            model_source = client.get_run(run_id).info.artifact_uri.rstrip("/") + "/model"
            versions = [v for v in client.search_model_versions("name='pettrace-embedder'")
                        if v.run_id == run_id and v.source == model_source]
            if len(versions) > 1:
                raise RuntimeError("Duplicate candidate registrations")
            if not versions:
                client.log_artifacts(run_id, str(folder / "model"), artifact_path="model")
            version = versions[0] if versions else client.create_model_version("pettrace-embedder",
                source=model_source, run_id=run_id, tags={"experiment": "E3", "status": "candidate"})
            if client.get_model_version("pettrace-embedder", version.version).status != "READY":
                raise RuntimeError("Candidate registration is not ready")
            write_json(folder / "publication.json", {"published": True, "runs": runs,
                       "registry_version": str(version.version), "alias_promoted": False})
            backup()
    print(f"{len(SEEDS) * len(variants)} paired fits and {2 * len(SEEDS) * len(variants)} domain evaluations published.", flush=True)
