"""Verify a downloaded local course CSV and write a separate manifest."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
from lab_helpers.s4_regression.bike_sharing import load_bike_hourly, data_audit
from ml_lib.experiment.configuration import project_root, load_config
from ml_lib.experiment.artifacts import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab", choices=["s4", "s5"], default="s4")
    parser.add_argument("--path", type=Path, help="Optional explicit local dataset path.")
    args = parser.parse_args()
    root = project_root()
    if args.lab == "s5":
        from lab_helpers.s5_classification.bank_fraud import verify_baf
        path = args.path or root / load_config(root / "configs/bank_fraud.toml")["data"]["path"]
        report = verify_baf(path)
        report["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["source_url"] = "https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022"
        target = root / "data/processed/bank_fraud_manifest.json"
        write_json(target, report)
        print(json.dumps(report, indent=2))
        print("Manifest:", target)
        print("The raw dataset was not changed.")
        return
    path = args.path or root / load_config(root / "configs/bike_sharing.toml")["data"]["path"]
    raw = load_bike_hourly(path)
    report = data_audit(raw, path)
    report["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["source_url"] = "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset"
    print(json.dumps(report, indent=2))
    target = root / "data/processed/bike_sharing_manifest.json"
    write_json(target, report)
    print("Manifest:", target)
    print("The raw dataset was not changed. This SHA-256 is a local snapshot fingerprint.")


if __name__ == "__main__":
    main()
