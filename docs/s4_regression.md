# S4: Regression and forecasting

[Back to all labs](../README.md)

Start here to prepare and run the first lab in the repository. Basic Python and pandas are assumed. Complete environment setup and data verification before beginning the notebook sequence.

An operations analyst wants to assess whether aggregate rental predictions could support workload planning. Estimate an observed hour first, then predict the next hour from available history. Use the evidence to choose a next step: keep a reference, investigate further, or revise the proposed use. Lower model error alone does not establish business value.

Consider one action a supervisor could take before the predicted hour and one operational measure needed to judge usefulness.

The records describe completed rentals across the system. They do not measure unmet demand, station availability, or the number of bicycles needed. Do not convert prediction improvements into claimed financial savings.

## 1. Prepare the environment

Complete installation and data verification before class.

Open a terminal at the repository root (the folder containing `pyproject.toml`). Use standard **64-bit Python 3.12**. The shared environment includes TensorFlow for notebooks 04–05; CPU execution is the course default.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python -m ipykernel install --user --name ml510 --display-name "Python (ML510)"
python scripts/check_environment.py --deep
```

If activation is blocked, use `.\.venv\Scripts\python.exe` in place of `python` in these commands; no execution-policy change is needed.

### Linux / macOS Apple Silicon

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python -m ipykernel install --user --name ml510 --display-name "Python (ML510)"
python scripts/check_environment.py --deep
```

The installation provides the dependencies for all six notebooks and installs the local library in editable mode. Obtain the notebooks separately from Moodle. Keep using this environment when you return to the course; recreate it on another computer instead of copying `.venv`.

The selected TensorFlow version targets Windows x86-64, Linux x86-64, and macOS Apple Silicon with native ARM64 Python. Intel macOS requires a compatible alternative environment. No GPU setup is required. See the [TensorFlow installation guide](https://www.tensorflow.org/install/pip) for platform requirements.

## 2. Obtain and verify the data

Download the [official UCI Bike Sharing ZIP](https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip). Extract the original `hour.csv` and `Readme.txt` into `data/raw/bike_sharing/`. `day.csv` is unused. Use the original UCI files rather than the Kaggle competition files.

Then, from the repository root:

```bash
python scripts/verify_data.py
```

This checks your local data and writes `data/processed/bike_sharing_manifest.json` without changing the CSV. The hash identifies your local snapshot. Notebooks do not download data or install packages automatically.

Dataset: Fanaee-T, H. (2013), *Bike Sharing*, UCI Machine Learning Repository, [DOI 10.24432/C5W894](https://doi.org/10.24432/C5W894), CC BY 4.0. Keep the original description and attribution with redistributed data.

## 3. Obtain and open the notebooks

Download the S4 notebook assignments from the course Moodle. Extract the archive if necessary, then copy the `s4_regression` folder into this repository's `notebooks/` folder. The ZIP alone cannot be opened by the execution script. Check that `notebooks/s4_regression/00_start_here.ipynb` exists, without an extra nested `s4_regression` folder. Keep all six notebooks together. See the [placement instructions](../notebooks/README.md).

With the environment active, launch Jupyter from the repository root:

```bash
python -m jupyterlab
```

In Jupyter, open `notebooks/s4_regression/00_start_here.ipynb`. Select **Python (ML510)**. In VS Code, select the same `.venv` interpreter for the notebook kernel.

Run cells from top to bottom. Each notebook starts independently from the raw CSV and shared configuration; use a fresh kernel for each notebook.

Rerun from setup after changing settings so saved results remain consistent. For command-line execution, run these commands from the repository root:

```bash
python scripts/execute_notebooks.py --scope core
# Include the at-home sequence and capacity experiments
python scripts/execute_notebooks.py --scope all
```

`core` runs notebooks 00–03; `all` runs all six. The command creates local output copies in `notebooks/s4_corrections/` and saves detailed results in `artifacts/`. The assignments and generated output copies remain local; they are not included on GitHub. Execution stops before overwriting an annotated instructor copy; preserve or rename it first. The runner allows 7200 seconds per cell for CPU training; `--timeout` changes this limit without changing the epoch budget.

## 4. Follow the learning phases

Start with a reference, inspect its errors, then test one change. Keep extra complexity only when the evidence supports it.

| Phase | Notebook | Main task |
|---|---|---|
| Understand | `00_start_here.ipynb` | Discover target leakage and missing hours; compare working-day and non-working-day activity. |
| Represent | `01_linear_regression.ipynb` | Compare context with constants, inspect squared temperature, then change Ridge `alpha`. |
| Compare | `02_trees_and_ensembles.ipynb` | Change tree depth; compare overall error, peak-period error, and severe misses. |
| Use history | `03_lagged_forecasting.ipynb` | Trace a historical prediction and change one history choice. |
| Learn history at home | `04_sequence_forecasting.ipynb` | Train GRU 32, interpret history inputs and learning curves; keep test reserved. |
| Improve and combine at home | `05_model_capacity.ipynb` | Compare recurrent models, a Transformer, learning-rate tuning, and bagged Transformers on the same next-hour task. |

Pause at each question before running the next check. Use the embedded TODOs to investigate a change or verify a choice, then decide what the evidence supports. Notebook 02 runs shallow tree, deep tree, forest, and boosting in order. Notebook 03 bridges familiar models to ordered history; notebooks 04-05 continue that practice at home.

### References and errors

Notebook 01 keeps mean and median references. Notebook 02 refits these and the earlier contextual linear and engineered Ridge models. Forecasting notebooks use timestamped historical references. Choose the stronger reference on validation and keep it fixed for test evaluation.

Use **MAE** as the primary comparison and **RMSE** to inspect larger errors. Notebook 02 also checks working-day hours 07-09 and 16-19, chosen before fitting. Its saved diagnostics include slice counts, slice MAE, the 90th-percentile absolute error, and mean underprediction magnitude. The latter averages underpredictions only and is zero when none occur.

`reference-comparisons.csv` shows the reference score, candidate score, and improvement. Positive improvement favors the candidate; relative error reduction is a fraction. A validation-score winner is a starting point for a recommendation: consider remaining errors, data requirements, explanation, and maintenance effort.

### TensorFlow preview

Open `04_sequence_forecasting.ipynb` locally and find **Classroom preview: stop before training**. Inspect its timestamped example and the [TensorFlow adapter](../src/ml_lib/models/deep_learning.py): GRU, output layer, Adam, MSE, and early stopping. Stop before training; continue the experiment at home.

The preview needs only the example inputs and model code. After training, inspect the learning curves saved by your own run. An instructor may show privately prepared results; no downloaded model or pre-existing run is required.

### Modeling choices

Edit experiment settings in notebook cells. TOML holds data paths, split dates, seed, and execution settings.

| Choice | Controls to understand | Evidence |
|---|---|---|
| Inputs | Columns, scaling, missing-input policy | Errors, coefficients, retained rows |
| Ridge | `alpha` | Validation error and shrinkage |
| Tree | Depth and minimum leaf size | Training/validation gap |
| Forest | Tree count, depth, leaf size, feature fraction | Error versus ensemble complexity |
| Boosting | Learning rate, iteration count, leaf limits | Validation error |
| History | Lags and lookback | Input example and retained rows |
| Recurrent models | Hidden units, learning rate, batch size, epochs, patience | Learning curves and validation error |
| Transformer | Representation width, heads, blocks, feed-forward width, dropout | Validation error, parameter count, learning curves |
| Bagging | Resampled training days and member seeds | Single-model versus averaged predictions, total effort |

Tree counts, boosting iterations, and epochs are budgets. Depth, leaf size, and hidden units affect capacity. Sequence models predict one next-hour count from a 24-hour history.

### Rules to keep

- Fit preprocessing and models on training data. Use validation to choose; reserve test for final evaluation. Split examples by their target timestamp; history may include earlier training observations.
- Missing hours are unknown, not zero. Compare forecasts on rows with complete inputs, labels, and every reference; inspect retained counts.
- `casual + registered = cnt` reveals the same-hour answer. Historical counts become usable after observation. Future calendar information is assumed known; measured future weather is unavailable.
- Observations are assumed to arrive after each completed hour. Later observed outcomes may enter later histories while the fitted model stays fixed.
- Timestamps and peak-period slices use the dataset's local hour labels. Weather fields are source-normalized; see the downloaded `Readme.txt`. UCI weekday codes start on Sunday; diagnostic labels start on Monday.
- Engineered inputs in 01-02 add only `temp_squared`, keeping calendar encoding fixed. Coefficients describe associations, not causal effects.
- Starting tabular models use training-fitted imputation, numeric scaling, and one-hot encoding. Scaling matters for Ridge and is held constant for tree comparisons. Unknown categories are ignored; missing forecast labels are never imputed.
- The GRU trains with standardized-target MSE and Adam. Early stopping monitors validation loss and restores the best weights. MAE evaluates predictions in rental units.

## 5. Inspect results and try extensions

Open `artifacts/<notebook-name>/<timestamp-id>/README.md`. **Inspect first** links the summary and TODO evidence; detailed tables, predictions, figures, and model files follow. `experiment.json` records your choices; `manifest.json` identifies the data, source, and environment.

Useful tables and figures appear beside the relevant code, with at most one compact table and one figure per cell. Rerunning a result cell updates its files; rerunning setup creates a new run. Keep the starting run when trying a change. See the [artifact instructions](../artifacts/README.md).

## 6. Continue at home

In notebook 04, train the starting GRU and inspect its input example, historical references, and learning curves. Use validation evidence; leave test evaluation to notebook 05. Comparisons with notebook 03 do not isolate architecture because inputs differ.

Notebook 05 keeps the next-hour task fixed and runs GRU 32, GRU 128, BiLSTM 128 per direction, then a Transformer. The two GRUs isolate width at their starting settings; BiLSTM changes cell type and directionality together. All models see only completed history and known next-hour calendar inputs.

The [Transformer encoder](../src/ml_lib/models/transformer_layers.py) uses a 128-dimensional representation, four attention heads, two encoder blocks, and a 256-unit feed-forward layer. Positions preserve order; attention relates observations within the history window. These are explicit starting choices, not a claim that the architecture is optimal.

Each architecture tries learning rates 0.001 and 0.0003, with the same epoch budget, batch size, and stopping rule. This is hyperparameter tuning, not pretrained-model fine-tuning. Inspect starting architectures separately from the validation-selected settings. Learning curves mark the best validation-loss epoch; saved tables separate selected-model fitting time from total tuning effort.

Finally, three new Transformers use the selected Transformer settings and bootstrap training days with replacement. Each sampled day contributes its complete, already-built windows. Each member learns its own preprocessing from that sample. Average all three predictions equally. This day-block bagging keeps windows intact but does not remove all temporal dependence; it supplies neither out-of-bag evaluation nor calibrated uncertainty.

Compare MAE, larger misses, and effort against the tuned single Transformer. Greater total capacity is an attempt to improve predictions, not a guarantee. Freeze all choices on validation before inspecting test. This historical test period has already been examined during lab development, so it is not fresh confirmation of the expanded experiment.

## Troubleshooting

| Problem | What to check |
|---|---|
| Notebook files are missing | Obtain `s4_regression` from Moodle and follow the [placement instructions](../notebooks/README.md). |
| Library import fails | Install `requirements.txt` from the repository root; select the `.venv` kernel. |
| `hour.csv` is missing | Check its location against the [download instructions](#2-obtain-and-verify-the-data). |
| Wrong Python or packages | Check `python --version` and `python -c "import sys; print(sys.executable)"`. |
| TensorFlow import fails | Check the shared installation, supported platform, and [TensorFlow installation guide](https://www.tensorflow.org/install/pip). |
| Too few complete windows | Inspect missing hours and configured splits/lookback; keep missing targets unknown. |
| Imports or device settings seem stale | Restart the notebook kernel after changing imported code or TensorFlow settings. |

On Windows, a deeply nested repository can make TensorFlow installation fail with a long-path error. A short repository location avoids this. If you already created `.venv` here, retry the same pinned installation using Windows extended-path syntax, then continue with the usual `python` commands:

```powershell
$labPython = '\\?\' + (Resolve-Path .venv\Scripts\python.exe).Path
& $labPython -m pip install -r requirements.txt
```

## Validation status

The six-notebook sequence has been executed on Windows CPU, including TensorFlow tuning and bagging. The latest review executed notebooks 00–03 from fresh kernels and passed all 91 shared-library tests, including TensorFlow checks. The unchanged 04–05 training uses the earlier validated runs. Browser workflow and other operating systems have not been validated. Detailed validation evidence is retained locally, outside the student release.

## References

- [UCI Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
- [scikit-learn leakage guidance](https://scikit-learn.org/1.7/common_pitfalls.html) and [lagged forecasting example](https://scikit-learn.org/1.7/auto_examples/applications/plot_time_series_lagged_features.html)
- [TensorFlow time-series tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Keras time-series Transformer example](https://keras.io/examples/timeseries/timeseries_classification_transformer/) and [scikit-learn bagging guidance](https://scikit-learn.org/1.7/modules/ensemble.html#bagging-meta-estimator)
- [Connecting model metrics to business success](https://developers.google.com/machine-learning/managing-ml-projects/success)

The progression follows ML510 Session 4. Split dates, model settings, and availability assumptions are lab choices; no published benchmark is claimed as a notebook result.
