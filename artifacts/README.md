# Saved lab results

Each notebook creates `artifacts/<notebook-name>/<timestamp-id>/`. Open that run's `README.md` to find its results.

- `tables/`: metrics, predictions, examples, and diagnostic tables (CSV).
- `figures/`: saved plots (PNG).
- `experiment.json`: problem, coverage, model choices, selection, and baseline-relative validation evidence.
- `manifest.json`: configuration, environment, and source/data fingerprints.
- `model/`: selected models and required references. An ensemble stores its members and averaging metadata here. If an S4 forecast reference wins, its definition is in `experiment.json`. Load only your own trusted models.
- `training/`: S5 neural progress, best-weight recovery checkpoints, and completed family winners. These files do not mark a run complete or provide automatic resume; later notebooks use the final `model/` bundles.

Rerunning an analysis cell updates its files within the current run. Rerunning setup creates a separate run. Existing runs are not removed. Notebooks can show one compact table and one figure per cell; complete results remain saved here.

Keep the starting run as reference evidence when trying a change. Open saved learning curves after neural training. S5 uses labels `s5_00` through `s5_05`; its later notebooks require the exact preceding run folder. Follow the [S5 guide](../docs/s5_classification.md#4-follow-the-investigation) for these handoffs. No run is selected automatically.

Start with **Inspect first** in each new run overview: summary, reference comparisons, and relevant TODO evidence. Other generated files remain linked below.

Use reference comparisons and error diagnostics to work through the TODOs.
