# Generated data records

Data verification writes a local CSV fingerprint and audit here:

- `python scripts/verify_data.py`: `bike_sharing_manifest.json` for [S4](../../docs/s4_regression.md#2-obtain-and-verify-the-data).
- `python scripts/verify_data.py --lab s5`: `bank_fraud_manifest.json` for [S5](../../docs/s5_classification.md#2-obtain-and-verify-baf-base).

This folder can also hold explicitly saved derived data. The current notebooks construct features in memory. Keep raw CSV files unchanged and record the source/configuration when saving a derived dataset.
