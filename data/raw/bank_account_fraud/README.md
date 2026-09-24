# BAF Base data

Place the official version-2 `Base.csv` here. Use the one-million-row Base dataset, not the biased variants. The CSV and generated results remain local and ignored by Git.

Follow the [S5 data instructions](../../../docs/s5_classification.md#2-obtain-and-verify-baf-base), then run `python scripts/verify_data.py --lab s5` from the repository root. Verification records the source fingerprint without editing the file.

BAF contains synthetic applications derived from anonymized banking data. Dataset attribution: Jesus et al. (2022), *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation*. [Authors' repository](https://github.com/feedzai/bank-account-fraud). Dataset license: CC BY-NC-SA 4.0, separate from the course code's MIT license.
