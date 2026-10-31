"""Read frozen controls for E3 without running or publishing E1/E2 again."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from pettrace.tracking import production_context

FROZEN_RUNS = {
    "public": {"E1": "670af363cda5492899ce6466cc0962e9", "E2": "59011833455c4190bc4d92e156bf5b64"},
    "private": {"E1": "4e6391cc267940ba952aa75aee43dad7", "E2": "4b09272fa13e48d7af6e5b694d416cfe"},
}


def frozen_controls(evaluation, domain, output):
    """Validate finished runs, immutable data/model and identical queries before comparing."""
    client, _ = production_context()
    evaluations, summaries, receipts = {}, {}, {}
    for label, run_id in FROZEN_RUNS[domain].items():
        run = client.get_run(run_id)
        expected = {"encoder.onnx": "4773e649274ca8774c4a63d45f0438fe598ee7ba67e65def97537aa0eca9a576"}
        if label == "E2":
            expected["detector.onnx"] = "da8f9b9ebea5157cdcefa7bea9fd782249ebdf119f917c07d64e5b41942b9c9d"
        if (run.info.status != "FINISHED" or run.data.tags.get("experiment") != label
                or run.data.params.get("dataset.commit") != evaluation.dataset.commit
                or run.data.params.get("dataset.fingerprint") != evaluation.dataset.fingerprint
                or not run.data.params.get("protocol", "").startswith("zero-shot:")
                or any(run.data.params.get(f"model.sha256.{name}") != digest for name, digest in expected.items())):
            raise ValueError(f"Frozen {label} does not match this benchmark/protocol/model")
        folder = Path(output) / f"baseline_{label.lower()}"
        folder.mkdir(parents=True, exist_ok=True)
        paths = {name: Path(client.download_artifacts(run_id, artifact, str(folder))) for name, artifact in
                 (("evaluation", "evaluation.json"), ("summary", "report/summary.json"))}
        result = json.loads(paths["evaluation"].read_text())
        def signature(rows):
            return [(r["query"], r["photo"], r["animal"], r["species"], r["gallery_animals"]) for r in rows]
        if signature(result["test"]["per_query"]) != signature(evaluation.result["test"]["per_query"]):
            raise ValueError("Frozen baseline queries/gallery differ")
        summaries[label] = json.loads(paths["summary"].read_text())
        evaluations[label] = SimpleNamespace(result=result, dataset=evaluation.dataset)
        receipts[label] = {"run_id": run_id, "sha256": {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in paths.items()}}
    (Path(output) / "frozen-controls.json").write_text(json.dumps(receipts, indent=2))
    return evaluations, summaries
