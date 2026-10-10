# pettrace-ml

Re-identification experiments for **PetTrace**, a mobile app that helps find lost pets in Lima
(UPC thesis). Given a photo of a sighted animal, the system ranks the lost-pet reports of the same
species by visual similarity; the owner confirms. This repository holds everything needed to
measure that ranking: the evaluation harness, model registration, notebooks and the published
reports. The app itself (iOS, Android, Go core, vision service) lives in a separate repository.

> **Resumen (ES).** Experimentos de re-identificación de mascotas de la tesis PetTrace (UPC):
> arnés de evaluación, registro de modelos, notebooks y resultados. Cada resultado es
> reproducible con los comandos de abajo y está trazado en MLflow al commit, al modelo (sha256)
> y a la versión del dataset.

## Experiments

| ID | Pipeline | Training | Status |
|---|---|---|---|
| E1 | full image → CLIP ViT-B/32 → 512-d, L2 | none (zero-shot) | registered, in production |
| E2 | YOLOv8n detects the animal → crop (10 % margin) → same CLIP | none (zero-shot) | registered, candidate |
| E3 | E2 + frozen CLIP + MLP head, triplet loss | head on `train` | design |
| E4 | E3 + LoRA fine-tuning (r = 8, α = 16, q/v) | LoRA on `train` | design |
| E5 | E4 + re-ranking with the sighting's text description | weight on `val` | design |

### Results on `pettrace-reid@v3` (zero-shot)

14 animals (7 cats, 7 dogs), 128 photos. Every photo is a query against the other photos of the
same species (leave-one-out); each animal scores with its best photo. Chance Rank-1 = 1/7 = 14.3 %.
The EER threshold is chosen on the same pairs (optimistic).

| Metric | E1 | E2 |
|---|---|---|
| Rank-1 (all / cats / dogs) | 73.4 % / 75.4 % / 71.6 % | 70.3 % / 65.6 % / 74.6 % |
| Rank-5 | 98.4 % | 99.2 % |
| mAP | 0.509 | 0.527 |
| EER (threshold) | 39.2 % (0.794) | 35.0 % (0.812) |

Cropping separates same-animal from different-animal scores better (lower EER, higher mAP) and
helps dogs, but hurts look-alike cats (`cat-01` is taken for `cat-02`, same tabby pattern, in 6 of
10 queries): part of E1's hits came from the photo's background. With 128 queries the Rank-1 gap is
within noise (confidence intervals: [TODO.md](TODO.md)). Per-animal tables and figures:
[`reports/e1_v3`](reports/e1_v3), [`reports/e2_v3`](reports/e2_v3).

| | Notebook (canonical record) | MLflow run (`pettrace-reid-eval`) | Code commit |
|---|---|---|---|
| E1 | [`e1_clip_zero_shot.ipynb`](notebooks/e1_clip_zero_shot.ipynb) | [`af8a88b1`](https://ml.pettrace.app/#/experiments/3/runs/af8a88b1e2a6434c9eeb31b62c9010e4) | `f80d1d6` |
| E2 | [`e2_yolo_crop_zero_shot.ipynb`](notebooks/e2_yolo_crop_zero_shot.ipynb) | [`e0d3e49a`](https://ml.pettrace.app/#/experiments/3/runs/e0d3e49aec554a4588efa3bce99955e1) | `c9dd2b9` |

Each notebook shows the data, how the model was built, the whole evaluation computed step by step
(preprocessing, embeddings, similarities, ranking, metric formulas, checked against the library),
tables, figures with how to read them, and conclusions,
and is attached (executed) to its run together with the figures.

## What identifies a result

Registry version numbers and run IDs are local to one MLflow server; these are not:

| | Identity |
|---|---|
| Dataset | Hugging Face `alessandro54/pettrace-reid`, tag `v3`, commit `354763693e9105eb953c559620037dd98995326b` |
| E1 model | `clip-vit-b32@v1`: `encoder.onnx` sha256 `4773e649274ca8774c4a63d45f0438fe598ee7ba67e65def97537aa0eca9a576`, golden fingerprint `5e40c7774bd7ad51` |
| E2 model | `clip-vit-b32-crop@v1`: same encoder + `detector.onnx` sha256 `da8f9b9ebea5157cdcefa7bea9fd782249ebdf119f917c07d64e5b41942b9c9d` |
| Code | the commit of this repository recorded on each run (`git_sha` tag) |

Every evaluation run in MLflow (`https://ml.pettrace.app`, experiment `pettrace-reid-eval`)
records the dataset commit and fingerprint, the resolved model version, the sha256 of every ONNX
file it evaluated, the code commit, the ONNX Runtime version and the CPU.

## Reproduce

Needs Docker, read access to the private dataset (Hugging Face token) and an MLflow user
(reviewers get a read-only one; ask the author).

```bash
git clone https://github.com/alessandro54/pettrace-ml && cd pettrace-ml
git checkout <commit from the run's git_sha tag>   # f80d1d6 (E1), c9dd2b9 (E2)
cp .env.example .env            # MLflow user/password + HF_TOKEN
make image                      # Linux, Python 3.11, versions from uv.lock

make notebook NB=notebooks/e1_clip_zero_shot.ipynb        # E1: executes, logs a run, attaches itself
make notebook NB=notebooks/e2_yolo_crop_zero_shot.ipynb    # E2 (reads E1's report for the comparison)
make evaluate DATASET=v3 MODEL=models:/pettrace-embedder/2 PROTOCOL=zero-shot REPORT=reports/e2_v3  # CLI
```

Evaluating writes a run, so a read-only user can reproduce the numbers without logging:
`docker run --rm --env-file .env -v $PWD:/repo pettrace-ml scripts/evaluate.py --dataset v3 --model models:/pettrace-embedder/2 --protocol zero-shot --no-mlflow`.

The models are downloaded from MLflow and checked by hash. To rebuild them from scratch instead:
`make image-export && make register-v1` re-exports E1 from the pinned Hugging Face revision and
must give the same sha256; E2's detector is the COCO-pretrained `yolov8n.pt` exported with
Ultralytics 8.3.253 (command in [`scripts/register_e2.py`](scripts/register_e2.py)); a re-export
is not guaranteed to be byte-identical, so the validated file is the one inside version 2.

## Layout

```
reid/        harness: data (pinned dataset versions), metrics (Rank-k, mAP, EER), evaluation
             (protocol, model identity, MLflow run), figures, pipeline (ONNX runner), golden
             (fingerprint), clip_reference (PyTorch parity reference)
scripts/     evaluate, register_v1, register_e2, verify_model, attach_notebook, mlflow_smoke
notebooks/   one notebook per experiment (E1–E5): data, model, training, evaluation, results, MLflow run
reports/     published scores, figures and summaries per model × dataset version
```

`reid/pipeline.py`, `reid/golden.py` and `reid/clip_reference.py` mirror the vision service's
modules in the app repository; the golden fingerprint on every registered version (checked by the
service at startup and by `make verify-model`) fails if they diverge.

## Data and licences

- **Photos are not in this repository.** `pettrace-reid` is a private dataset: owners gave consent
  for a private research dataset; EXIF/GPS is stripped. Reports contain scores and file names only.
- **Animals are pseudonymous.** The dataset and everything here use codes (`cat-01`, `dog-03`);
  pets' names never leave the curation tool, because names plus photos identify the owners.
  `v3` is the only dataset version (earlier ones used names and were removed).
- Code: MIT ([LICENSE](LICENSE)).
- YOLOv8 weights (E2 detector) are AGPL-3.0 (Ultralytics); they are not distributed here.
- CLIP ViT-B/32: OpenAI, MIT.
