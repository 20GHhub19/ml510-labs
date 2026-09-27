# S6: Understand operating behaviour before raising an alarm

[All labs](../README.md)

Can train-compressor measurements identify periods that deserve maintenance inspection? Define similarity, interpret operating profiles, then investigate what drives unusualness scores and whether added complexity contributes useful evidence. A cluster is not a diagnosis, and an alert does not prove that maintenance would prevent downtime.

## 1. Prepare the environment

Reuse the Python 3.12 course environment. No new dependency is required. For a first installation, follow the [environment instructions](s4_regression.md#1-prepare-the-environment).

From the repository root, activate that environment and run:

```bash
python -m pip install -r requirements.txt
python -m pip check
python scripts/check_environment.py --deep --strict-pins
```

CPU execution is the starting policy. Select **Python (ML510)** in Jupyter.

## 2. Obtain and verify MetroPT-3

Download the [official MetroPT-3 archive](https://archive.ics.uci.edu/static/public/791/metropt%2B3%2Bdataset.zip). Extract `MetroPT3(AirCompressor).csv` into `data/raw/metropt3/`. Use this 2020 release, not the different MetroPT dataset collected in 2022.

```bash
python scripts/verify_data.py --lab s6
```

The verifier checks the official CSV fingerprint and records gaps, complete-window counts, and partitions in `data/processed/metropt3_manifest.json`. It does not modify the raw file. Raw data and results stay local.

Attribution: Davari, Veloso, Ribeiro, and Gama (2021), *MetroPT-3 Dataset*, DOI [10.24432/C5VW3R](https://doi.org/10.24432/C5VW3R). The data has a separate **CC BY 4.0** license. See the [field descriptions and failure reports](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset).

The file contains 1,516,948 observations. Most intervals are about ten seconds, but gaps exist; its final observations extend into September 1. The source descriptions disagree about sampling frequency, repeat an incident number, and contain a contradictory maintenance-date note. We use observed timestamps and preserve those reporting uncertainties.

## 3. Open the notebooks

Obtain `s6_unsupervised` from Moodle, extract it if zipped, and copy it into `notebooks/`. The first file must be `notebooks/s6_unsupervised/00_start_here.ipynb`, without a duplicated folder level.

```bash
python -m jupyterlab
```

Run each notebook top to bottom in a fresh kernel. Close preceding kernels to release memory. Notebooks 00–03 prepare their own data. In 04 and 05, paste the `source_run = ...` assignment printed by the preceding notebook into the labelled input cell. The printed next-notebook link opens the next stage. Scripted execution supplies these paths automatically. Models, features, windows, partitions, and fingerprints are checked; no arbitrary latest run is loaded. Load only your own trusted models.

For command-line execution:

```bash
python scripts/execute_notebooks.py --lab s6
```

The script passes its exact run paths between stages and writes local output copies in `notebooks/s6_corrections/`. It refuses to overwrite annotated instructor corrections. Preserve or rename them before regenerating. These files remain private.

## 4. Follow the investigation

| Notebook | Question | TODO |
|---|---|---|
| 00 — Understand the compressor | What do the measurements and gaps establish? | Explain one sensor change and one observation gap. |
| 01 — Define similarity | Do observed examples and digital signals support the group descriptions? | Remove temperature summaries and inspect what changes. |
| 02 — Challenge groups | Does another group add a meaningful distinction? | Inspect changed assignments under two DBSCAN radii. |
| 03 — Detect departures | Which sensors drive alerts, and do detectors agree? | Reduce PCA dimensions and compare alert evidence. |
| 04 — Learn nonlinear relationships | Does an autoencoder improve on linear reconstruction? | Reduce its bottleneck and inspect the consequences. |
| 05 — Learn ordered behaviour | Does a GRU resolve a limitation shared by simpler detectors? | Investigate a detected incident, an unverified alert, and the missed July incident; recommend the next evidence to collect. |

Questions precede the checks that answer them. Complete the TODOs using separate experiment variables; nothing is submitted. Continue neural practice at home if needed. Inspect the actual [autoencoder code](../src/ml_lib/models/autoencoder.py) alongside the notebook.

### Define the observation

One example is a completed ten-minute window; its score becomes available at the end. Seven analog sensors define the starting inputs. Digital operating signals remain interpretation evidence, not predictors or fault labels.

The helper computes minute means, requiring at least five observations per minute. It rejects windows touched by raw gaps over twenty seconds, missing minutes, or nonfinite inputs. Windows do not overlap or cross partition boundaries. Averaging may hide short transients; the chosen representation is an assumption to examine.

Clustering, Isolation Forest, and LOF use window means and standard deviations. PCA and both autoencoders reconstruct the same ordered minute measurements: PCA and the dense model flatten them; the GRU processes their order. Thus score differences between summary detectors and reconstructors also reflect representation choices.

### Reserve evidence

| Role | Period | Purpose |
|---|---|---|
| Fit | February 1–21 | Fit preparation and models |
| Monitor | February 22–29 | Stop neural training and restore weights |
| Threshold reference | March | Set cutoffs without failure labels |
| Validation | April–May | Compare incident evidence and alert burden |
| Test | June onward | Evaluate frozen choices |

The early reference is not certified healthy. Failure reports are never model inputs. All comparisons use matching windows; missing observations reduce what can be evaluated.

### Choose controls deliberately

| Control | What to investigate |
|---|---|
| Sensor choice and scaling | What makes two windows neighbours? |
| K and initialization | How much profile detail is stable and interpretable? |
| DBSCAN radius and minimum samples | Which density and noise assumptions change assignments? |
| PCA components | What linear information survives compression? |
| Isolation Forest tree/sample counts | How much randomized isolation evidence is collected? |
| LOF neighbours | What local context defines unusualness? |
| Autoencoder bottleneck and hidden units | Which nonlinear representation can be learned? |
| GRU architecture | Does sequential processing help on the same observed windows? |
| Learning rate, epochs, patience | How the model is optimized; training loss is not maintenance utility |
| March score percentile | How a score becomes an inspection alert |

Both neural models use Adam and standardized-input MSE, with a maximum of 100 epochs and patience 8, restoring the best monitoring-loss weights. The training summary shows the stopping reason, retained epoch, monitoring loss, and fitting effort. The starting bottleneck has eight dimensions. PCA is the linear reference. Lower reconstruction loss need not produce better anomaly evidence. Check whether the lowest monitoring loss occurs at the epoch budget: such a run does not establish convergence, and additional training is not guaranteed to help.

## 5. Interpret and inspect artifacts

Open `artifacts/s6_XX/<run-id>/README.md` and start with **Inspect first**. Complete profiles, assignments, sensor contributions, scores, incident coverage, alert episodes, and learning histories are saved. Experiment choices and thresholds go to `experiment.json`; the manifest records data/source fingerprints and environment.

Reported durations measure flagged observation time, not time spent reviewing or repairing equipment. Cutoffs use the March 99th percentile (higher interpolation), with strict greater-than alerts. This is a reference alert-rate policy, not a guaranteed false-positive rate. Consecutive alerts form episodes; missing windows break episodes. Scores become available only at window end.

An incident is adequately covered when retained windows cover at least half its reported duration and at least one score is available within the interval. Report all coverage fractions and missed incidents. Original reported endpoints have minute precision; the final stated minute is included when making half-open intervals.

The validation evidence leader detects the most adequately covered reported incidents, then has the fewest outside-report alert episodes per observed day. Ties keep candidate order and are listed explicitly. Also inspect alerted hours inside and outside reports and episode duration. A partially overlapping episode contributes only its overlapping duration to inside-report hours; overlapping reports are not counted twice. Few long episodes can still require substantial attention. There are very few incidents: this rule does not establish a robust ranking or replace an engineer's recommendation.

Unreported operation is not confirmed healthy. Outside-report alerts are **unverified**, not proven false alarms. Do not infer definitive precision, specificity, early-warning performance, prevented downtime, or financial benefit. Reconstruction contributions locate mismatch, not its cause.

### Questions to carry through the lab

Inspect group sizes, representative observations, and digital loaded-state fractions before naming operating profiles. Compare local neighbours separately from overall partition agreement. Check whether related pressure channels contribute repeated information to the distance definition.

For each detector, ask which sensors drive its alerts and how much its decisions differ from the reference. Agreement may reflect a shared signal. Compare incident evidence and unverified duration alongside reconstruction loss; check whether neural training exhausted its budget.

The final investigation includes a missed reported incident. Inspect its sensor departures before concluding that no change occurred. This already-examined period can reveal limitations, but must not be used to tune a cutoff until the event is detected. Identify what a maintenance engineer should verify before recommending an operational pilot.

Neural recovery files live under `training/<model>/`: configuration, preparation, epoch history, best model, and status. Incomplete recovery files are not completed handoffs; there is no automatic resume. Rerunning setup creates a new run; repeated result cells overwrite their stable outputs.

Private corrections retain the starting experiments and add marked instructor-only worked solutions. Public notebooks stay unexecuted; detailed validation evidence remains local.

## References

- [MetroPT-3 source and documentation](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset)
- [Authors' sparse-autoencoder study](https://doi.org/10.1109/DSAA53316.2021.9564181): related evidence, not a claimed reproduction of its protocol or results.
- [Scikit-learn clustering](https://scikit-learn.org/1.7/modules/clustering.html)
- [Novelty and outlier detection](https://scikit-learn.org/1.7/modules/outlier_detection.html)
- [PCA](https://scikit-learn.org/1.7/modules/generated/sklearn.decomposition.PCA.html)

## Validation status

The six starting notebooks and worked exercises have been executed on Windows CPU, including dense and GRU training. The shared-library suite and saved-model restoration checks pass. Historical 25-epoch runs remain available locally for comparison. The 100-epoch dense and GRU runs and the worked bottleneck exercise have been executed from scratch; saved best-model predictions and matching evaluation windows were verified. Source notebooks remain unexecuted for delivery. Browser workflow and other operating systems have not been validated; detailed results stay local.
