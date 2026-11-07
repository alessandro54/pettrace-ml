# E4 audit and E5 inference optimization plan

Status: **E4 winner frozen; E5 text evidence and validation protocol pending implementation**.

This document records the implementation audit and the proposed experiments after freezing the
E3-100 and E4 checkpoints. It distinguishes current evidence from prospective changes. Any new
condition must save complete, separate public and PetTrace reports locally. An official thesis run
must also publish those reports to `https://ml.pettrace.app`; publication can be retried from the
saved evidence without retraining. Independent users may exercise the underlying methods locally,
but that does not constitute a completed PetTrace experiment.

## Current evidence

| Condition | Domain | Rank-1 | Rank-2 | Rank-5 | mAP |
|---|---|---:|---:|---:|---:|
| E3-100, seed 46 | Public v1 | 80.1% | — | — | 0.7640 |
| E3-100, seed 46 | PetTrace v4 | 78.4% | — | — | 0.6880 |
| E4, seed 46 | Public v1 | 85.56% | 94.95% | 99.3% | 0.8126 |
| E4, seed 46 | PetTrace v4 | 83.61% | 92.79% | 99.7% | 0.7218 |
| E4, exploratory seed | Public v1 | 88.81% | — | — | 0.8374 |
| E4, exploratory seed | PetTrace v4 | 87.54% | — | — | 0.7489 |

The exploratory seed is descriptive. It must not replace seed 46 merely because its test results
were higher; candidate selection must use validation evidence only.

The main observed error pattern is the gap between Rank-1 and Rank-2/Rank-5. The correct identity
is usually present among the first candidates, but a visually similar rival occupies Rank-1. The
maximum observed similarity between different animals is approximately 0.972–0.979.

## Audit of the E3 to E4 transition

### Initialization

E4 does not initialize a new random MLP. It downloads the fixed E3-100 seed-46 head from MLflow,
checks registry version 18 and verifies the artifact SHA-256 before loading it. The MLP remains
frozen throughout E4.

The CLIP base weights and visual projection also remain frozen. LoRA updates are inserted only in
the `q_proj` and `v_proj` attention projections. The LoRA `B` matrices start at zero, so the initial
E4 model is functionally equivalent to E3; `control_equivalence` verifies this before training.

Therefore, E4 should be described as **LoRA adaptation of the visual encoder in the fixed E3 metric
space**, not joint LoRA and MLP training.

Relevant implementation:

- [`src/pettrace/experiments/e4.py`](../src/pettrace/experiments/e4.py)
- [`src/pettrace/training/lora.py`](../src/pettrace/training/lora.py)

### Triplet loss and mining

The fixed E4 configuration is:

- batch-hard triplet loss on L2-normalized embeddings;
- Euclidean distance and margin `0.2`;
- eight identities and four photos per identity (`P=8`, `K=4`);
- one species per batch, alternating cats and dogs;
- AdamW with learning rate `1e-4` and weight decay `1e-4`;
- 100 steps per epoch, at most 30 epochs and patience 7.

The hardest positive and negative are selected only from the current 32-image batch. This prevents
trivial cross-species negatives but does not reliably expose the globally most similar rival. With
many identities, the visual twin responsible for a Rank-1 error may never share a batch with the
anchor. Identities with fewer than four photos also require repeated photos to fill the batch.

This is the highest-leverage training-side limitation. A later retraining study could evaluate
offline hard-negative mining or a cross-batch memory, but neither is part of the proposed
zero-retraining ablation.

### Checkpoint selection

The E4 checkpoint selector is correct:

1. `fit` accepts only train and validation rows and rejects test rows;
2. training and validation identities must be disjoint;
3. every epoch is scored using validation macro mAP across species;
4. `adapter.pt` is replaced only when that score improves;
5. early stopping uses patience 7;
6. the selected `adapter.pt` is explicitly reloaded after training.

E4 therefore evaluates the best validation checkpoint, not the last epoch.

## Evaluation bottlenecks

The frozen evaluator extracts one deterministic embedding per photo and calculates the cosine
matrix as `E @ E.T`. A separate validation-only entry point now implements flip TTA and robust
identity aggregation without changing that baseline; query expansion remains prospective.

Rank-k and mAP use different aggregation units:

- Rank-k assigns each identity its maximum score across gallery photos;
- mAP ranks every gallery photo independently.

Maximum aggregation is sensitive to one accidental high-scoring rival photo and may explain part
of the observed impostor maximum. Nevertheless, the completed validation-only diagnostic found
that both an identity prototype and a top-two mean reduced macro-species Rank-1. The existing
`single + max` rule therefore remains fixed for E4 and the E5 visual candidate list.

The implemented benchmark is **closed-set, same-species, photo-level leave-one-out retrieval**. It
does not yet establish open-set identification because all queries are evaluated against galleries
that contain their identity and there is no unknown/reject decision.

Relevant implementation:

- [`src/pettrace/evaluation/core.py`](../src/pettrace/evaluation/core.py)
- [`src/pettrace/evaluation/metrics.py`](../src/pettrace/evaluation/metrics.py)
- [`src/pettrace/models/pipeline.py`](../src/pettrace/models/pipeline.py)

## Locked architecture

- Deployment is online: one incoming query against a static registered gallery.
- The first ablation is horizontal-flip TTA followed by an identity-level aggregation diagnostic.
- `max` remains the reference; validation compares `top2_mean` and `prototype`.
- k-reciprocal re-ranking is excluded from the current study because it is transductive and does
  not match the intended online service.
- E5 may use only identity-agnostic observable attributes and may alter a result only when both
  the visual ambiguity gate and the text-margin gate are open.

## Zero-retraining experiments

All hyperparameters below must be selected using validation only. Public v1 test and PetTrace v4
must remain separate and must be evaluated only after freezing each condition.

### 1. Horizontal-flip test-time augmentation

Use the same YOLO crop for both views:

1. crop the animal once;
2. embed the original crop;
3. horizontally flip that crop and embed it;
4. add the two unit embeddings and L2-normalize the result.

The augmentation must not run YOLO independently on the flipped raw image because a different box
would confound augmentation with detection. Apply the same procedure to queries and gallery photos.

This deterministic, non-transductive experiment was completed on validation. With maximum
aggregation it changed macro-species Rank-1 by `-0.007613` and mAP by `+0.001724`; every TTA
combination reduced Rank-1. It was therefore rejected before either test was loaded.

### 2. Gated alpha query expansion

For an ambiguous query, construct an expanded embedding from the query and its highest-scoring
same-species neighbors, weight the neighbors by similarity raised to `alpha`, and normalize the
result. Alpha query expansion follows the retrieval method discussed by
[Radenović et al.](https://arxiv.org/abs/1711.02512).

Use one contribution per candidate identity, or identity representatives, so identities with more
gallery photos do not dominate. Use one expansion iteration. Select the neighbor count, `alpha` and
ambiguity threshold on validation.

The principal risk is confirmation of an incorrect first match. For that reason, query expansion
should initially be restricted to queries whose visual Top-1/Top-2 margin is below a frozen
threshold.

### Recorded decision and remaining ablation

1. Retain frozen E4 seed-46 with `single + max` as the visual baseline.
2. Do not evaluate flip TTA or alternative identity aggregation on either test.
3. Treat gated alpha query expansion as a separate prospective study, not as a consequence of the
   rejected flip condition.

Each condition must record its configuration, validation selection evidence, public and PetTrace
metrics, per-query ranks and runtime cost. Its MLflow publication receipt is required before the
official experiment is considered complete.

## E5 natural-language re-ranking

E5 should be a conservative identity-level reranker over the visual Top-5. It should not retrieve
new identities outside that candidate set.

The fixed visual control is MLflow model `pettrace-embedder/19`, E4 seed 46 at epoch 9. Its public
and private run identifiers, ONNX hashes and `single + max` inference rule are verified by
`pettrace.experiments.e5.load_frozen_e4(model_uri, output)`. E5 must reuse this control without
retraining it; `model_uri` may be a local directory downloaded from an immutable public release.

For visual candidates `c`, define:

- `s1` and `s2`: the best two visual identity scores;
- `delta_v = s1 - s2`: the visual confidence margin;
- `z_v(q, c)`: calibrated visual compatibility;
- `z_t(q, c)`: calibrated text compatibility.

Use a hard ambiguity gate:

```text
g(q) = 1 when delta_v < tau_v, otherwise 0
S_total(q, c) = z_v(q, c) + g(q) * lambda * z_t(q, c)
```

Safety constraints:

- preserve visual Top-1 exactly when the gate is closed;
- consider only the visual Top-5 when the gate is open;
- allow a swap only when the text advantage exceeds a second frozen threshold;
- choose `tau_v`, `lambda` and the text threshold using validation only;
- report gate coverage, rescued errors, damaged correct matches and net Rank-1 change.

Text evidence must be identity-agnostic: coat color and pattern, distinctive markings, ear and tail
shape, size and other observable attributes. Names, source paths, publisher labels or descriptions
derived from identity identifiers would constitute leakage.

## Recorded validation protocol

The completed diagnostic accepted validation rows only. It embedded each fixed YOLO crop in its
original and mirrored orientations, then evaluated the full 2 × 3 factorial: single/TTA
representation against max/top-2/prototype identity scoring. Its one-off execution API was removed
from the maintained package after publication; the immutable run and this protocol preserve the
evidence without expanding the production evaluator.

Identity aggregation cannot change the current photo-level mAP by definition; it changes Rank-k.
Horizontal-flip TTA can change both. The diagnostic selected `single + max`; no alternative
qualified for test evaluation. MLflow run:
[`8d3549cfa0114529972ea1b42b90b1e5`](https://ml.pettrace.app/#/experiments/3/runs/8d3549cfa0114529972ea1b42b90b1e5).

## Risks and limits

- The exploratory E4 seed cannot become the confirmatory winner based on observed test metrics.
- The public and PetTrace benchmarks have already been observed; new results remain exploratory.
- Test-time expansion and re-ranking change the inference protocol and must be reported explicitly.
- Text generation can leak identity or dataset artifacts unless its inputs and vocabulary are fixed.
- None of these experiments alone establishes open-set recognition or population-level uncertainty.
