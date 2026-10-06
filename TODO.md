# TODO

## Now (OE2/OE3 evidence)

- [x] E1 and E2 on `pettrace-reid@v3`: one notebook per experiment, executed and attached to runs
      `af8a88b1` (E1) and `e0d3e49a` (E2).
- [ ] E3–E5 notebooks (`notebooks/README.md`): training inside the notebook, Colab GPU setup cell
      (install this repo at the pinned commit, MLflow/HF secrets from Colab).
- [ ] **Confidence intervals**: bootstrap over animals (and queries) for Rank-1, mAP and EER; report
      `73.4 % [a–b]` everywhere. With 128 queries the E1/E2 gap is within noise.
- [ ] **Model cards** per registered version (purpose, data, metrics with CIs, limits, licences:
      YOLOv8 AGPL-3.0, CLIP MIT), stored with the version in MLflow and linked here.
- [ ] **Datasheet** for `pettrace-reid`: collection, consent, pseudonymization, biases (7 animals
      per species, few households, lighting), intended use.

## OE3 (experiments E3–E5)

- [ ] **Training data**: E3–E4 train on public pet re-ID datasets (check licences); `pettrace-reid`
      stays an independent test set. Keep growing it (test first, then `val`).
- [ ] **Sighting-style photos** (outside the home, far, from behind) and an owner-photo vs
      street-photo protocol: removes the background advantage E2 exposed in E1.
- [ ] **Rank-k vs gallery size** (7, 50, 200 candidates, public distractors): the city-scale view.
- [ ] E3: frozen CLIP + MLP head, triplet loss, P×K sampler; training notebook (Colab GPU) per
      [notebooks/README.md](notebooks/README.md); seeds fixed and logged.
- [ ] E4: LoRA (r = 8, α = 16, q/v), merged into the ONNX encoder.
- [ ] E5: re-ranking with the sighting's description (multilingual text encoder).
- [ ] Threshold calibrated at the EER on `val`, FAR/FRR reported on `test` (no more optimistic EER).

## OE4 (MLOps)

- [ ] **Promotion gate in CI**: evaluate `@candidate` vs `@production` on the same `@vN` `test`
      split; block on a per-species regression, golden-set mismatch, p95 latency ≥ 3 s or ONNX
      parity failure; promotion = alias move (`@staging` → `@production`).
- [ ] **Blue-green re-index**: new Qdrant collection with the promoted model, alias swap, rollback
      = swap back (app repo).
- [ ] **Monitoring**: latency, similarity-score drift per `model_version`, owner confirmation rate
      of top-K matches (OpenTelemetry + Grafana).
- [ ] **Feedback loop**: owner-confirmed matches (with consent) become Studio candidates for the
      next dataset version.
- [ ] **Trained-model promotion across registries**: copy validated artifacts keeping source run,
      `git_sha`, dataset hash and sha256 (GPU training is not bit-exact).
- [ ] MLflow from the interim VPS to the AWS target (ECS + RDS): `pg_dump` of schema `mlflow`,
      artifacts stay in R2, point the domain.
- [ ] Repo hygiene: pre-commit (ruff, nbstripout), Dependabot, pinned image digest.
