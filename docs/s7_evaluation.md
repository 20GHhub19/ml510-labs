# S7: Five practical questions about model evaluation

[All labs](../README.md)

A better average score is only part of the evidence. In S7, trust the comparison, diagnose learning, inspect important errors, challenge model behavior, then decide what the evidence supports. State an expectation before inspecting the result. Integrity checks protect an evaluation; model tests examine predictions; an intervention experiment studies the effect of using them.

## 1. Prepare

Reuse the course environment and select **Python (ML510)**. For a first installation, follow the [S4 setup](s4_regression.md#1-prepare-the-environment). No new dependencies are needed.

Copy the five notebooks and evidence folder from Moodle into:

```text
notebooks/s7_evaluation/00_validation.ipynb
data/processed/s7_evidence/manifest.json
```

Keep the folders intact. You also need the [Bike Sharing CSV](s4_regression.md#2-obtain-and-verify-the-data). The active notebooks use rental data and saved S4 predictions; no BAF data or neural training is required. The evidence retains the source attribution and fingerprints; dataset licenses remain separate from the code license.

```bash
python scripts/verify_data.py --lab s4
python -m jupyterlab
```

Open `00_validation.ipynb`. Run each notebook in a fresh kernel; no notebook depends on another notebook's variables.

## 2. Investigate five questions

| Notebook | Main experiment | Your TODO |
|---|---|---|
| `00_validation.ipynb` | Inner folds choose Ridge settings; outer folds assess the procedure. Detect preprocessing fitted outside its training boundary. | Repair the boundary and explain the roles of the scores. |
| `01_learning.ipynb` | Compare shallow and unrestricted trees using whole-day samples of the same training pool. | Test whether depth 8 improves assessment error, not just the training gap. |
| `02_groups.ipynb` | Compare saved rental predictions overall and during working-day peaks, with paired intervals and severe shortfalls. | Separate a predictive preference from adequate planning performance. |
| `03_behavior.ipynb` | Check rental output validity, local temperature stability and the response to warmer, less windy conditions. | Change minimum leaf size from 10 to 20 and compare accuracy and behavior on identical cases. |
| `04_decision.ipynb` | Apply a primary requirement and a subgroup guardrail to a constructed intervention. | Recommend proceed, hold or redesign and identify missing evidence. |

One principal TODO per notebook; nothing is submitted. Questions precede evidence. Deliberately damaged inputs or reports are labelled integrity demonstrations. The behavioral counterexamples are actual responses from fitted models, not altered predictions.

### Read the evidence carefully

- **Validation:** preprocessing is fitted inside each training fold. Inner scores select settings; outer scores assess the selection procedure. The three outer/two inner folds use only the original S4 training period. Choosing the same alpha does not make different fitted models or assessment periods interchangeable.
- **Learning curves:** fractions of 25%, 50% and 100% draw complete observed days from a fixed pool. Three seeds show sensitivity to which days are available; full-data fits are reused. An unrestricted tree can grow with additional data despite unchanged settings. A smaller training gap alone does not justify replacement.
- **Group comparisons:** peak periods are working-day hours 07-09 and 16-19. Check that unique case identities cover the expected population before summarizing errors. Negative candidate-minus-reference MAE favors the candidate, but inspect the remaining shortfall too. Day-block intervals condition on the saved fitted models and do not include retraining uncertainty.
- **Behavior:** negative rental estimates violate the output domain. A 0.5°C temperature correction is checked against an illustrative 20-rental tolerance. The directional example raises temperature by 3°C and lowers wind by 5 km/h, expecting an increase greater than 5 rentals. It starts at 10–20°C in category-1 weather during hours 06–20, with other inputs fixed. These are teaching requirements, not proven causal laws. Training-derived weather ranges screen out obvious extrapolation, but counterexamples still need inspection.
- **Decisions:** the final trial is constructed course data. Check outcome coverage against the expected roster before calculating effects. Its improvement target and subgroup guardrail are separate requirements. Shadow operation observes without acting; a canary limits exposure; a randomized A/B experiment can assess an intervention against an actual comparator.

Earlier evaluation periods have already been examined. This lab develops evaluation skills; it does not supply fresh confirmation or deployment approval.

## 3. Inspect and rerun

Each setup creates an `artifacts/s7_XX/<run-id>/` folder. Follow the notebook's overview link for complete tables, failing cases and settings. Repeating a result cell replaces its file within that run.

For command-line execution:

```bash
python scripts/execute_notebooks.py --lab s7
```

Output copies go to `notebooks/s7_corrections/`. The script refuses to overwrite annotated corrections. Privately supplied corrections contain separate runnable TODO solutions and measured answers; students do not need them to run the lab.

If an evidence fingerprint differs, restore the supplied Moodle folder rather than mixing results from different experiments.

## References

- Session 7 course slides: validation, learning curves, group evidence, behavioral tests and intervention experiments.
- [Scikit-learn cross-validation](https://scikit-learn.org/1.7/modules/cross_validation.html).
- [Training-only preprocessing](https://scikit-learn.org/1.7/common_pitfalls.html).
- [VerifIA domain rules](https://docs.verifia.ca/concepts/domain/): define feasible inputs and expected output relationships. This lab implements its checks directly in Python.
