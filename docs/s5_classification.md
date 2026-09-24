# S5: From fraud scores to review decisions

[All labs](../README.md)

A fraud team cannot investigate every account application. Can a model help it find more fraud while limiting unnecessary reviews of legitimate customers? Build a scoring function, examine its probabilities, then choose a review policy. Recorded fraud caught by a policy is not proof that the review would prevent a loss.

## 1. Reuse the course environment

Use the Python 3.12 environment prepared for S4. No additional package is needed. For a first installation, follow the [shared environment instructions](s4_regression.md#1-prepare-the-environment), including Windows and Linux/macOS commands. The existing [troubleshooting guidance](s4_regression.md#troubleshooting) also applies.

From the repository root, with that environment active:

```bash
python -m pip install -r requirements.txt
python -m pip check
python scripts/check_environment.py --deep --strict-pins
```

CPU execution is the starting setting. Keep the selected kernel at **Python (ML510)**. Model size, epoch budgets and measured execution effort are different concepts; a larger dataset does not guarantee that a neural model will beat boosting.

The neural notebook prints one progress line per epoch and saves learning history and best validation-loss checkpoints under `training/<trial-name>/` in its run folder. Attention can take much longer than classical models on CPU. Completed family winners are saved before the next family starts.

Recovery folders contain configuration, training-fitted preprocessing, `best.keras`, `history.csv` and `training.json`. The history CSV numbers epochs from zero; progress messages count from one. These are recovery files, not completed experiment handoffs. An interrupted run remains incomplete; a hard process termination may leave its status as `training`. There is no automatic resume or promise of identical continuation. Later notebooks load the final `model/` bundles from a completed run.

## 2. Obtain and verify BAF Base

Download **Base.csv**, version 2, from the authors' [Bank Account Fraud dataset](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022). Extract it if compressed and place the CSV at `data/raw/bank_account_fraud/Base.csv`. Use Base only, not Variant I–V. The CSV is about 213 MB and contains one million rows.

```bash
python scripts/verify_data.py --lab s5
```

Verification checks the source fingerprint and writes `data/processed/bank_fraud_manifest.json`. It does not edit the raw file. Notebooks read local data; they do not install packages or download data during an experiment.

The helper assigns `case_id` from the original row position in this unchanged file. It is a comparison key, not a real applicant identifier.

BAF is **synthetic**, generated from anonymized bank-application data. Its numeric and categorical fields represent applications, sessions, devices and previous activity. `fraud_bool=1` denotes recorded fraud. The [authors' datasheet](https://github.com/feedzai/bank-account-fraud/blob/main/documents/datasheet.pdf) documents the fields and limitations. Use the supplied field meanings rather than treating every negative value as missing.

The dataset has its own **CC BY-NC-SA 4.0** license; the repository's MIT license does not replace it. Attribution: Jesus et al. (2022), *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation*, NeurIPS Datasets and Benchmarks. [Paper and source](https://github.com/feedzai/bank-account-fraud).

## 3. Place and open the notebooks

Obtain `s5_classification` from Moodle. Extract it if zipped, then copy the folder into `notebooks/`. The first file should be `notebooks/s5_classification/00_start_here.ipynb`, without an extra nested folder. Keep the six files together.

```bash
python -m jupyterlab
```

Run each notebook top to bottom with a fresh kernel. Shut down the preceding notebook's kernel to release its data and models from memory. Notebooks 00–02 rebuild their inputs independently. For notebooks 03–05, paste the preceding stage's printed artifact-folder path into `source_run`, for example:

```python
source_run = "artifacts/s5_02/<your-run-id>"
```

The handoffs are **02 → 03 → 04 → 05**. They reload saved models rather than relying on another kernel's variables. The loader checks the data fingerprint, selected features, preprocessing choices, month partitions and completed stage. It never chooses a folder called “latest.” Load only your own trusted model artifacts.

For the complete sequence from a terminal:

```bash
python scripts/execute_notebooks.py --lab s5
```

The script passes the exact artifact paths from that execution between stages and creates local output notebooks in `notebooks/s5_corrections/`. It stops before overwriting annotated instructor corrections. Preserve or rename those copies first. `--timeout` changes the per-cell execution limit, not the training budget.

## 4. Follow the investigation

Ask, predict, run a check, then interpret. Complete the embedded TODOs; nothing is submitted. Continue the neural experiments at home if needed, then return to the probability and policy stages.

| Notebook | Main question | TODO |
|---|---|---|
| `00_start_here.ipynb` | What is measured and available at the decision moment? | Investigate the intended transfer amount's missing rule and timing using training records. |
| `01_logistic_regression.ipynb` | Can a simple score improve on a constant reference? | Change the cutoff without refitting. |
| `02_trees_and_ensembles.ipynb` | Do more flexible boundaries help? | Change shallow-tree depth from 5 to 8. |
| `03_neural_models.ipynb` | Do learned representations and tuning improve detection under the same error limit? | Change dropout in the deeper MLP. |
| `04_probabilities.ipynb` | Do score values agree with observed frequencies? | Compare the same applications before and after calibration within populated raw-score bins. |
| `05_review_policy.ipynb` | Which cases should receive review attention? | Tighten the validation false-positive limit from 5% to 1%. |

The starting inputs exclude the target, month, age, elapsed time since the request, device fraud counts, and the supplied credit-risk score. The latter fields need timing or upstream-model assumptions beyond this starting investigation. Age remains available for inspecting errors; omitting it from inputs does not establish fairness.

Six explicit missingness flags preserve whether address history, bank-account history, session length, device email count or intended transfer amount was absent. All models receive the same flags and selected raw fields. Medians still come only from training data; an imputed value and an observed median need not carry the same information.

### Read the benchmark carefully

The authors evaluate **recall at 5% false-positive rate**. Their synthetic-data generation study reports roughly 55% recall, while their main model comparison uses 100 LightGBM configurations, training months 0-5 and test months 6-7. Our Base v2 file, selected fields, reserved calibration month, model budgets and evaluation partitions do not reproduce that experiment. Treat the result as context, not a promised score or an AP target. [Authors' appendix](https://arxiv.org/html/2211.13358v1#A3), [benchmark code](https://github.com/feedzai/bank-account-fraud/blob/main/notebooks/empirical_results.ipynb).

Our main question is whether a change detects more fraud while flagging at most 5% of legitimate validation applications. A 5% false-positive rate means about 50 flags per 1,000 legitimate applications, not 5% of reviews being wrong. More recall alone does not establish acceptable workload or business value.

### Reserve separate evidence

| Role | Months | Use |
|---|---|---|
| Training | 0–3 | Fit preparation and model parameters |
| Calibration | 4 | Fit the probability adjustment |
| Validation | 5 | Select settings, models, adjustments and operating rules |
| Test | 6–7 | Evaluate frozen choices |

Keep natural prevalence and identical evaluation cases. These are retrospective month partitions, not a complete reconstruction of deployment with delayed labels. BAF has no exact daily timestamps or label-availability dates.

### Choose the change deliberately

| Control | What it changes |
|---|---|
| Logistic `C` | Inverse regularization strength; larger values mean less constraint |
| Tree depth and leaf size | Flexibility and the evidence required for a leaf |
| Forest tree count and feature sampling | Averaging effort and diversity |
| Boosting iterations and learning rate | The budget and size of successive corrections |
| Boosting class weight | The relative emphasis on fraud in the training loss; compare `None` with `"balanced"` |
| MLP width and depth | Learned representation capacity |
| Transformer dimensions, heads and blocks | Attention between fields; tokens are not timestamps |
| Dropout | Training regularization |
| Learning rate, batch size, epoch budget and patience | Optimization and training effort |
| Cutoff and false-positive limit | Decisions made with scores; they do not refit the scorer |

The neural comparison uses MLP [32], MLP [128], MLP [128, 64], a feature-token Transformer. Each architecture tries the same two learning rates. The sequence ends with the Transformer comparison, then moves to calibration and review policies.

## 5. Read the evidence

Each stage creates `artifacts/s5_XX/<run-id>/`. Open its README and start with **Inspect first**. `manifest.json` records source/data fingerprints and environment details; `experiment.json` records the actual choices. Detailed predictions, comparisons and histories are CSV files, and figures are saved as PNGs. Repeating a result cell updates its stable filename; rerunning setup starts another run.

- **Recall at a maximum 5% FPR** selects the validation-score winner. Compare models on identical applications and under the same limit; thresholds move tied scores together. Candidate ties keep the earlier candidate, and cutoff ties retain the higher cutoff.
- **Average precision and ROC AUC** supply supporting ranking evidence. A gain in AP need not improve detection at the chosen operating point.
- **Log loss, Brier score and reliability bins** examine probabilities. Lower loss is better; log loss penalizes confident mistakes strongly. Brier score also reflects discrimination, so it is not a pure calibration measure.
- **Counts, precision, recall and false-positive rate** describe the review rule's errors and workload. Read slice scores with their sample counts.

Recall is the fraction of recorded fraud reviewed. Precision is the fraction of reviews concerning recorded fraud. False-positive rate is the fraction of legitimate applications reviewed: with rare fraud, even a small rate can create many unnecessary reviews.

Tied scores move together at a cutoff. A shallow tree can therefore use less than the full 5% allowance. Compare achieved FPR as well as recall. Age slices apply the model's globally selected cutoff to both groups; they do not tune separate group thresholds.

The neural summary reports additional fraud detected relative to the classical reference, fitting effort and neural parameter counts. Inspect whether the gain justifies the extra effort. Parameter counts are not a comparable capacity measure for trees. The worked dropout change illustrates that supporting metrics can improve while constrained recall stays unchanged.

Neural models train with binary cross-entropy. Their best validation-loss weights are restored; validation recall at 5% FPR selects between configurations. These are different choices. Each stage retains unsuccessful comparisons as well as successful ones; extra capacity must justify its effort.

Calibration fits a logistic mapping of clipped log-odds using the separate calibration month. Keep it only when validation log loss improves. Weighted boosting supplies a concrete probability-quality investigation: changing training emphasis can distort raw probabilities. A strictly increasing recalibration preserves ranking, so better probability meaning does not inherently improve recall at a fixed FPR; clipping may create endpoint ties.

The overall winner, strongest classical candidate, logistic reference and weighted candidate retain their names through the handoffs, with duplicates saved once. The final policy compares a 0.5 cutoff with each model's validation-selected cutoff at 5% FPR. The TODO tightens this limit to 1% using separate validation-only choices. No mandatory capacity cap obscures the error-limit comparison; inspect the resulting review counts.

Freeze the starting thresholds before test. Report actual test recall and FPR without choosing new test cutoffs: a 5% validation limit is not a guarantee for later months. Each split is one evaluation batch, not a reconstruction of daily staffing.

Higher test recall with a higher actual FPR is a trade-off, not a clean ranking at an identical error rate. Repeating a policy evaluation cell replaces its method/split/policy results rather than adding duplicate rows.

The practical recommendation may differ from the validation-score winner. Review quality, delays, legitimate-customer friction and prevented losses require operational evidence. The existing test period has already been examined during lab development; refreshed experiments are not fresh confirmation. Further model or policy changes need new confirmatory evidence.

## Instructor material and validation

Private corrections contain executed starting cells plus marked **Instructor solution** cells with worked TODO code and measured answers. Extra experiments use separate variables and runs. Students' starting notebooks remain unexecuted.

The full-data experiments and worked TODO fits were validated on Windows CPU. This teaching refinement reuses those fits and refreshes the analysis; it does not repeat full neural training. Shared-library tests cover input checks, probabilities, alignment, calibration, policy rules, saving/loading, and interruption checkpoints on tiny neural fixtures. Other operating systems and the browser workflow remain unchecked. Detailed evidence stays local.

## References

- [BAF dataset and datasheet](https://github.com/feedzai/bank-account-fraud)
- [Scikit-learn: classification thresholds](https://scikit-learn.org/1.7/modules/classification_threshold.html)
- [Scikit-learn: probability calibration](https://scikit-learn.org/1.7/modules/calibration.html)
- [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959)
- [Keras: early stopping](https://keras.io/api/callbacks/early_stopping/)
