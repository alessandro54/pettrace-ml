.PHONY: image image-export e1 evaluate notebook register-v1 register-e2 verify-model test test-e3 lint manifest

IMAGE ?= pettrace-ml
# Every MLflow run records the commit it came from; GIT_DIRTY=true marks uncommitted code.
GIT_ENV = -e GIT_SHA=$$(git rev-parse HEAD) -e GIT_DIRTY=$$(test -z "$$(git status --porcelain)" && echo false || echo true)
DOCKER = docker run --rm --env-file .env $(GIT_ENV) -e PETTRACE_OUTPUT_DIR=/repo/reports -v $(CURDIR):/repo -v pettrace-ml-hf:/root/.cache/huggingface

image:
	docker build -t $(IMAGE) .

image-export:
	docker build --build-arg GROUPS="--group figures --group notebooks --group export" -t $(IMAGE):export .

# make evaluate DATASET=v3 MODEL=models:/pettrace-embedder/1 PROTOCOL=zero-shot [REPORT=reports/e1_v3]
evaluate:
	$(DOCKER) $(IMAGE) -m pettrace.cli evaluate --dataset $(DATASET) --model $(or $(MODEL),models:/pettrace-embedder@production) $(if $(PROTOCOL),--protocol $(PROTOCOL)) $(if $(REPORT),--report $(REPORT))

# Headless E1; set PRIVATE_VERSION/PRIVATE_COMMIT after publishing a new private benchmark.
e1:
	uv run --extra tracking pettrace run e1 $(if $(PRIVATE_VERSION),--private-version $(PRIVATE_VERSION)) $(if $(PRIVATE_COMMIT),--private-commit $(PRIVATE_COMMIT))

# Execute an experiment notebook in the pinned image and save it with its outputs (the canonical
# record of the experiment): make notebook NB=notebooks/e1_clip_zero_shot.ipynb
notebook:
	$(DOCKER) --entrypoint jupyter $(IMAGE) nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 $(NB)
	$(DOCKER) $(IMAGE) tools/attach_notebook.py $(NB)

# E1: pettrace-embedder v1 @production (static: verifies an existing version, never re-creates it)
register-v1:
	$(DOCKER) $(IMAGE):export tools/register_v1.py

# E2: YOLOv8n crop + v1 encoder → @candidate; needs artifacts/yolov8n.onnx (see the script)
register-e2:
	$(DOCKER) $(IMAGE):export tools/register_e2.py

MODEL ?= models:/pettrace-embedder@production
verify-model:
	$(DOCKER) $(IMAGE) tools/verify_model.py $(MODEL)

test:
	uv run --locked --group dev --group notebooks --extra notebooks python -m pytest -q

test-e3:
	uv run --locked --group dev --extra training python -m pytest -q tests/integration/pettrace/experiments/test_e3_training.py

lint:
	uv run --group dev ruff check .

# Refresh the Git-tracked run index; MLflow reads only, local .env supplied by Docker.
manifest:
	$(DOCKER) $(IMAGE) tools/sync_experiments.py
