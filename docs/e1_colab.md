# E1 in VS Code with a Colab kernel

1. Select the Colab kernel and run the dependency cell. It keeps compatible Colab NumPy/Hub
   libraries (`numpy>=2.1,<2.3`, `huggingface-hub>=1.32,<2`), preserves Requests, and fixes
   ONNX Runtime/MLflow versions. These bounds address the Numba and datasets conflicts seen
   in the previous runtime output.
2. Run the package-install cell. It uses an installed `pettrace-ml` package; if absent, it installs
   from `PETTRACE_PACKAGE_SOURCE` (currently the public Git source). Publish the current code
   before using it remotely. Set this variable to a release wheel URL once a wheel is published.
3. Provide HF and production MLflow credentials through environment variables, Secrets or hidden
   prompts. A local `.env` on your Mac is not automatically visible to the Colab kernel.
4. Run the experiment cells in order. The published HF `v4` tag and the fixed registered E1
   artifact must be available. E1 uses CPU inference and does not train a model.
5. Review the generated results, analysis and conclusions. The final section always publishes to MLflow. Save the executed notebook from VS Code.

No package source files, local root path or report from another experiment are required beside
the notebook. The implementation lives in the installed `pettrace` package. Package version,
source hash, model hashes, dataset commit and runtime versions are recorded for reproducibility.

If an earlier installation downgraded Colab's base NumPy/Hub/Requests, use a fresh runtime; a
kernel restart alone does not undo installed packages. Restart the kernel after changing the
installed package or replacing already imported dependencies.

## Evidence and interpretation

The notebook reports retrieval metrics, gallery sizes and chance, descriptive EER, five figures,
per-animal metrics and margins, the strongest confusions and exploratory lighting strata.
`analysis.json` preserves the extra tables and dataset commit; `findings.md` preserves
the generated conclusions. Publication is the last notebook step, after reviewing these findings. The actual v4 findings require a real execution;
synthetic verification is not experimental evidence.

EER is fitted and reported on the same pairs. Lighting strata can be confounded by animal and
scene; absent light values are loaded as zero. Compare models on the same dataset/protocol.

## Production publication

The shared `pettrace.tracking` module requires the existing `pettrace-reid-eval` experiment on
`https://ml.pettrace.app`. E1 verifies its fixed encoder hash/golden set, and repeats its expected
model/dataset checks at publication. Matching runs are reused; incomplete uploads are resumed.
The evaluation does not promote models or change serving aliases.

Outputs are under `reports/e1_v4` relative to the kernel directory (`PETTRACE_OUTPUT_DIR` can
change the parent). Preserve unpublished reports before disconnecting. Published reports are
saved to MLflow, while `make notebook` can additionally attach a local executed notebook.
