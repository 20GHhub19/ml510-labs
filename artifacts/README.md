# Saved lab results

Each notebook creates `artifacts/<notebook-name>/<timestamp-id>/`. Open that run's `README.md` to find its results.

- `tables/`: metrics, predictions, examples, and diagnostic tables (CSV).
- `figures/`: saved plots (PNG).
- `experiment.json`: problem, coverage, model choices, selection, and baseline-relative validation evidence.
- `manifest.json`: configuration, environment, and source/data fingerprints.
- `model/`: the selected saved model when applicable. A selected ensemble stores its members and equal-averaging metadata here. If a forecast reference wins, its definition is in `experiment.json`. Load only your own trusted models.

Rerunning an analysis cell updates its files within the current run. Rerunning setup creates a separate run. Existing runs are not removed. Notebooks can show one compact table and one figure per cell; complete results remain saved here.

Keep the starting run as reference evidence when trying a change. After notebook 04 training, open that run?s learning curves. An instructor may show a prepared local run; no run is selected automatically.

Start with **Inspect first** in each new run overview: summary, reference comparisons, and relevant TODO evidence. Other generated files remain linked below.

Use reference comparisons and error diagnostics to work through the TODOs.
