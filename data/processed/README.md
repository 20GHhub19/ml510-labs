# Generated data records

`python scripts/verify_data.py` writes `bike_sharing_manifest.json` here with the local CSV fingerprint and validation audit. See the [dataset notes](../../docs/s4_regression.md#2-obtain-and-verify-the-data).

This folder can also hold explicitly saved derived data. The current notebooks construct features in memory. Keep raw CSV files unchanged and record the source/configuration when saving a derived dataset.
