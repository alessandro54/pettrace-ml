# Short paper evidence

Updated 21 November 2026. This page links the recorded evidence for *Adapting CLIP and DINOv2 for Pet Re-Identification through Cropping, Metric Projection, and LoRA* and distinguishes the sequential research model from the serving model.

Project code is licensed under MIT. See [LICENSE](../LICENSE). The [refined cat dataset](https://huggingface.co/datasets/alessandro54/labeled-cats-in-the-wild-refined) and [refined dog dataset](https://huggingface.co/datasets/alessandro54/dogs-of-the-world-refined) have their own provenance and upstream notices.

## Sequential DINOv2 study

Study `20261119-v1` compares full-image frozen DINOv2 (E1, `raw`), cropped frozen DINOv2 (E2, `frozen`), an MLP projection (E3, `mlp`), and LoRA over the selected frozen MLP (E4, `mlp_lora`). Six training runs and 16 evaluation runs completed.

Validation selected MLP seed 46 and sequential LoRA seed 47. All three E4 runs share the selected E3 parent; they are conditional E4 repetitions, not independent repetitions of the entire E3-to-E4 sequence. Checkpoints were selected using public validation macro-species mAP, not test scores.

The public test contains 277 photographs of 60 identities, pinned to composite commit `a724c1099a5432609c853958e88b68c4cf82d871`. PetTrace v6 contains 502 photographs of 50 identities, pinned to `52bfe26384c4230008fe4707aa0cb145988766a8`. Evaluation uses separate same-species galleries and photo-level leave-one-out retrieval. The logged `zero-shot` label identifies this evaluator; trained MLP/LoRA checkpoints are not zero-shot models.

### Six training runs

| Seed | E3 MLP | E4 LoRA with frozen MLP |
|---|---|---|
| 45 | [`1826964031ee46469b32771a91e8f397`](https://ml.pettrace.app/#/experiments/3/runs/1826964031ee46469b32771a91e8f397) | [`59c75a8f5b88421985b08bbd296be09b`](https://ml.pettrace.app/#/experiments/3/runs/59c75a8f5b88421985b08bbd296be09b) |
| 46 | [`0b46d1e179a1491897b2bbbe9130b85d`](https://ml.pettrace.app/#/experiments/3/runs/0b46d1e179a1491897b2bbbe9130b85d) | [`c5a1d8eef6a140f0bcf0b072077cd08a`](https://ml.pettrace.app/#/experiments/3/runs/c5a1d8eef6a140f0bcf0b072077cd08a) |
| 47 | [`3dccefa2a7564d7e9dcf42693e2cd1c1`](https://ml.pettrace.app/#/experiments/3/runs/3dccefa2a7564d7e9dcf42693e2cd1c1) | [`e75c53869b0e4b3fbca2f548dbb379d8`](https://ml.pettrace.app/#/experiments/3/runs/e75c53869b0e4b3fbca2f548dbb379d8) |

### Sixteen evaluation runs

| Configuration | Public test | PetTrace v6 |
|---|---|---|
| E1 frozen full image | [`5f0383b06f5641859574eaa031a692e3`](https://ml.pettrace.app/#/experiments/3/runs/5f0383b06f5641859574eaa031a692e3) | [`b9d2b4c8459342b29948cd1496e683fc`](https://ml.pettrace.app/#/experiments/3/runs/b9d2b4c8459342b29948cd1496e683fc) |
| E2 frozen crop | [`6d0c1096eb774341877d9e735adc14b1`](https://ml.pettrace.app/#/experiments/3/runs/6d0c1096eb774341877d9e735adc14b1) | [`544c56ddf3584c8f8518e28c71384a24`](https://ml.pettrace.app/#/experiments/3/runs/544c56ddf3584c8f8518e28c71384a24) |
| E3 seed 45 | [`a0d3d3a38f18462ab7dc106f3a9c94b4`](https://ml.pettrace.app/#/experiments/3/runs/a0d3d3a38f18462ab7dc106f3a9c94b4) | [`c6a81d13db364288a4c743f98a0bd3af`](https://ml.pettrace.app/#/experiments/3/runs/c6a81d13db364288a4c743f98a0bd3af) |
| E3 seed 46 (selected) | [`57cb0d512a034953a6e8ba25a383e16d`](https://ml.pettrace.app/#/experiments/3/runs/57cb0d512a034953a6e8ba25a383e16d) | [`d16a9618c16b455aa187728c50a54c30`](https://ml.pettrace.app/#/experiments/3/runs/d16a9618c16b455aa187728c50a54c30) |
| E3 seed 47 | [`9f340cb376b84fab8bb89d79f46374c5`](https://ml.pettrace.app/#/experiments/3/runs/9f340cb376b84fab8bb89d79f46374c5) | [`affd850286b24dfcae439cc1eb82fab3`](https://ml.pettrace.app/#/experiments/3/runs/affd850286b24dfcae439cc1eb82fab3) |
| E4 seed 45 | [`42d874d833164d4db4e0e13b13d2d928`](https://ml.pettrace.app/#/experiments/3/runs/42d874d833164d4db4e0e13b13d2d928) | [`cda7a09b23ee426b98d5c58a8dc02f8a`](https://ml.pettrace.app/#/experiments/3/runs/cda7a09b23ee426b98d5c58a8dc02f8a) |
| E4 seed 46 | [`1a3c34251e4348a4b1571bb16fb06bf7`](https://ml.pettrace.app/#/experiments/3/runs/1a3c34251e4348a4b1571bb16fb06bf7) | [`898919ebb1f54c1aa0223d800bd4c8d1`](https://ml.pettrace.app/#/experiments/3/runs/898919ebb1f54c1aa0223d800bd4c8d1) |
| E4 seed 47 (selected) | [`e3857cc943cb40ab8b32269aa125235c`](https://ml.pettrace.app/#/experiments/3/runs/e3857cc943cb40ab8b32269aa125235c) | [`59643220608b4add946ae14d0276932d`](https://ml.pettrace.app/#/experiments/3/runs/59643220608b4add946ae14d0276932d) |

The selected E4 evaluations report public Rank-1 96.39% and mAP 92.17%, and PetTrace Rank-1 94.42% and mAP 87.29%. The research bundle is registered as `pettrace-dinov2-research/1`; that registration did not promote a serving alias.

## Separate LoRA-only serving model

The [public production bundle](https://huggingface.co/alessandro54/pettrace-production/tree/bbf288479e67455cdac894953f28320ca7ce19a1) contains DINOv2-L/14 with YOLOv8n cropping and LoRA, without the MLP or E5. It corresponds to the separate LoRA-only control, selected at seed 45, rather than sequential E4. Its training run is [`209b96e9182c45d3a69e5671a5201e3c`](https://ml.pettrace.app/#/experiments/3/runs/209b96e9182c45d3a69e5671a5201e3c). The selected evaluations are [`ab0b0aaf98e24ae18737c7f41da42d03`](https://ml.pettrace.app/#/experiments/3/runs/ab0b0aaf98e24ae18737c7f41da42d03) (public) and [`34150624a2d4450b9b61e2cf8dac314b`](https://ml.pettrace.app/#/experiments/3/runs/34150624a2d4450b9b61e2cf8dac314b) (PetTrace v6).

The bundle is released under AGPL-3.0 with upstream notices, exports 1024-dimensional embeddings, and is pinned by the application to Hugging Face revision `bbf288479e67455cdac894953f28320ca7ce19a1`. Its model card, preprocessing configuration, checksums and export-verification records describe the serving artifact. Deployment demonstrates integration; it does not establish field effectiveness.

## Access and reproducibility limits

MLflow links require authorized read access. An anonymous request returns HTTP 401; published runs are not anonymously accessible. Private PetTrace photographs are not redistributed, and reproducing private-image inference requires separately approved data access.

This page publishes the evidence index and selection decisions. The current DINOv2 source and research notebooks are not yet included in the public repository snapshot accompanying this update; this documentation update is not a complete source release of those experiments. The recorded runtime, source bundles and artifacts must be inspected with authorized access before claiming a complete reproduction from the public repository alone.

The study is retrospective: earlier DINOv2 test results informed its design. Fixed-gallery identity-cluster intervals do not measure new-session generalization or training variance. Sequential E4 improves public mAP, but its PetTrace mAP difference from cropped frozen DINOv2 is inconclusive. E5 remains exploratory and is outside the paper's E1-to-E4 comparison. No claim of open-set rejection or capture-session-separated performance follows from these runs.
