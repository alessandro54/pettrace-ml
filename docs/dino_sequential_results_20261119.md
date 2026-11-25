# DINO sequential E1–E4: verified results

Study: `20261119-v1`. Six training runs and 16 evaluation runs completed. Selected MLP seed 46 and LoRA seed 47 by validation; no test-based checkpoint selection. Research registry: `pettrace-dinov2-research/1`; no serving alias promoted.

| Model | Public Rank-1 | Public mAP | PetTrace Rank-1 | PetTrace mAP |
|---|---:|---:|---:|---:|
| raw | 90.61% | 84.69% | 95.02% | 86.03% |
| frozen | 92.06% | 83.83% | 94.82% | 86.50% |
| mlp-46 | 94.58% | 89.44% | 93.82% | 85.37% |
| mlp_lora-47 | 96.39% | 92.17% | 94.42% | 87.29% |

Public: 277 photos / 60 identities, immutable v1 test split (the reporting label remains `lcw+dow-refined@v2` because prepared crops are v2). PetTrace v6: 502 photos / 50 identities. Separate same-species closed-set galleries, photo-level leave-one-out. The logged `zero-shot` protocol label describes evaluation infrastructure; trained MLP/LoRA are not zero-shot models.

## Uncertainty and limits

- public: E4 minus frozen mAP +8.33 pp; descriptive 95% identity-cluster interval [5.04, 12.02] pp. Rank-1 +4.33 pp, interval [0.37, 8.60] pp; 17 rescues, 5 damages.
- private: E4 minus frozen mAP +0.78 pp; descriptive 95% identity-cluster interval [-1.18, 2.82] pp. Rank-1 -0.40 pp, interval [-2.19, 1.23] pp; 16 rescues, 18 damages.

These intervals condition on a fixed gallery/checkpoint; they do not establish cross-session generalization or capture training variance. All E4 seeds share the selected E3 parent. The research is retrospective: earlier DINO test results motivated the study. PetTrace improvement over frozen is inconclusive; MLP alone regressed there. The earlier linear-projection alternative remains a relevant comparator, not superseded on every metric.

## Registration repair

MLflow 3 `register_model(runs:/.../research-model)` expects an MLmodel flavor or logged-model entity; this bundle intentionally contains research weights and code. Recovered by creating the version from the existing artifact URI, verifying all 16 evaluation statuses and selected head/adapter SHA-256 against run parameters. A repeated registration reused version 1. Receipts, summaries, per-query evaluation inputs, paired comparisons and selected weights were recovered locally without retraining.

Evidence: [registration](../reports/dino-sequential/20261119-v1/registration.json), [publications](../reports/dino-sequential/20261119-v1/publication.json), [paired comparisons](../reports/dino-sequential/20261119-v1/comparison.json).

Selected runs: [public](https://ml.pettrace.app/#/experiments/3/runs/e3857cc943cb40ab8b32269aa125235c), [PetTrace](https://ml.pettrace.app/#/experiments/3/runs/59643220608b4add946ae14d0276932d).
