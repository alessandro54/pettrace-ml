.PHONY: image image-export evaluate report register-v1 register-e2 verify-model mlflow-smoke test lint

IMAGE ?= pettrace-ml
# Every MLflow run records the commit it came from; GIT_DIRTY=true marks uncommitted code.
GIT_ENV = -e GIT_SHA=$$(git rev-parse HEAD) -e GIT_DIRTY=$$(test -z "$$(git status --porcelain)" && echo false || echo true)
DOCKER = docker run --rm --env-file .env $(GIT_ENV) -v $(CURDIR):/repo -v pettrace-ml-hf:/root/.cache/huggingface

image:
	docker build -t $(IMAGE) .

image-export:
	docker build --build-arg GROUPS="--group figures --group export" -t $(IMAGE):export .

# make evaluate DATASET=v2 MODEL=models:/pettrace-embedder/1 PROTOCOL=zero-shot
evaluate:
	$(DOCKER) $(IMAGE) scripts/evaluate.py --dataset $(DATASET) --model $(or $(MODEL),models:/pettrace-embedder@production) $(if $(PROTOCOL),--protocol $(PROTOCOL))

# make report DATASET=v2 MODEL=models:/pettrace-embedder/1 NAME=e1_v2 RUN=<evaluate run id>
report:
	$(DOCKER) $(IMAGE) scripts/report_figures.py dump --dataset $(DATASET) --model $(MODEL) --out reports/$(NAME).json
	$(DOCKER) $(IMAGE) scripts/report_figures.py plot reports/$(NAME).json > /dev/null
	$(DOCKER) $(IMAGE) scripts/report_figures.py publish reports/$(NAME).json --run $(RUN)

# E1: pettrace-embedder v1 @production (static: verifies an existing version, never re-creates it)
register-v1:
	$(DOCKER) $(IMAGE):export scripts/register_v1.py

# E2: YOLOv8n crop + v1 encoder → @candidate; needs artifacts/yolov8n.onnx (see the script)
register-e2:
	$(DOCKER) $(IMAGE):export scripts/register_e2.py

MODEL ?= models:/pettrace-embedder@production
verify-model:
	$(DOCKER) $(IMAGE) scripts/verify_model.py $(MODEL)

mlflow-smoke:
	$(DOCKER) $(IMAGE) scripts/mlflow_smoke.py

test:
	uv run --group dev pytest -q

lint:
	uv run --group dev ruff check .
