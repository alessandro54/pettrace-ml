# DINO E3 hidden-width ablation

Status: prospective, not executed. Defined 2026-11-20 after inspecting existing public and
PetTrace results. No result or improvement is assumed.

Question: does increasing the hidden width from 256 to 512 improve DINOv2 metric adaptation
under the same training protocol? The narrower relative bottleneck is a hypothesis, not
an established explanation for transfer degradation. DINO's existing head has more trainable
parameters than CLIP's; hidden dimension is not a count of retained backbone parameters.

Intervention: 1024→256→1024 versus 1024→512→1024, ReLU and L2. Parameter counts 525,568
versus 1,050,112. Only the MLP is trained. DINOv2-L/14 and YOLO stay frozen; no E4/LoRA.

Controls: exact public prepared fingerprint d0ca5c0b1bed8844f943b9fd355ce912524797d0dc71b50c7a1624e39f49fe58;
193,276 training and 273 validation photos with disjoint identities; same frozen embeddings,
AdamW lr 0.001, weight decay 0.0001, triplet margin 0.2, P8×K4, gradient clipping 5,
maximum 30×100 steps, patience 7. Seeds 45,46,47 in each width, six fits. Same seeds fix
sampler order but do not produce identical initial weights across different architectures.
Early stopping may yield different realized exposure; report it rather than call it equal
training compute. Existing E3 results remain historical controls, not overwritten.

Selection: per-run checkpoint by validation macro-species mAP. Choose width by mean of the
three validation mAP scores, ties to smaller width; choose seed within that width by validation
mAP, ties to smaller seed. Keep frozen E2 eligible if its validation mAP is no lower than the
winning family mean. Freeze selection before test. Test scores must not select a width.

Evaluation: immutable public v1 test (277 photos / 60 identities), PetTrace v6 commit
52bfe26384c4230008fe4707aa0cb145988766a8 (502 / 50), separately. Same-species closed-set
photo-level leave-one-out; identity Rank-1/5, photo mAP, descriptive EER. Publish all six
candidates and frozen E2 in each domain, not just a winner. Report mean/sample SD, matched-seed
512−256 differences and identity-cluster intervals, with fixed-gallery limits. Six training
publications and fourteen evaluation publications, plus the analysis, are mandatory.

Interpretation: improvement would support sensitivity to width under this protocol; it
would not prove that lost information was the cause. Doubling width also increases capacity,
so bottleneck and parameter count are not causally separated. A negative result must remain
in the report. This is exploratory evidence on previously inspected benchmarks, not an
independent confirmation of cross-session transfer.

Execution: reuse verified frozen features if retained; otherwise the full training corpus
must be loaded and DINO features extracted again. Do not substitute the small local pilot
or test embeddings for the training corpus. Operational setup stays in tools/setup_dino_colab.py.
Registration is outside this ablation; previous notebooks' registration settings and aliases
are not modified. Publication retry never requires fitting again.
