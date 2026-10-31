# E1 in VS Code with a Colab kernel

1. Use a fresh Python 3.11 or 3.12 runtime and run the installation cell. It installs the
   versioned v0.2.1 wheel and all dependencies with hashes exported from `uv.lock`.
2. Restart the kernel and continue from the package-verification cell. It displays the source
   commit and lockfile hash embedded in the wheel. See [release installation](releases.md).
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
