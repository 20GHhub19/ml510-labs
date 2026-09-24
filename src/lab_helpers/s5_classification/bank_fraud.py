"""Read BAF applications and keep dataset-specific meanings out of ml_lib."""
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
from ml_lib.data.loading import sha256_file
from ml_lib.problems.classification import ClassificationProblem, binary_labels
from ml_lib.problems.contracts import PredictionContract


DATASET_SHA256 = "7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809"
CATEGORICAL = ("payment_type", "employment_status", "housing_status", "source", "device_os")
EXCLUDED = {"fraud_bool", "month", "customer_age", "days_since_request", "device_fraud_count", "credit_risk_score"}
SENTINEL_MINUS_ONE = ("prev_address_months_count", "current_address_months_count", "bank_months_count",
                      "session_length_in_minutes", "device_distinct_emails_8w")
FIELDS = ("fraud_bool", "income", "name_email_similarity", "prev_address_months_count",
          "current_address_months_count", "customer_age", "days_since_request", "intended_balcon_amount",
          "payment_type", "zip_count_4w", "velocity_6h", "velocity_24h", "velocity_4w", "bank_branch_count_8w",
          "date_of_birth_distinct_emails_4w", "employment_status", "credit_risk_score", "email_is_free",
          "housing_status", "phone_home_valid", "phone_mobile_valid", "bank_months_count", "has_other_cards",
          "proposed_credit_limit", "foreign_request", "source", "session_length_in_minutes", "device_os",
          "keep_alive_session", "device_distinct_emails_8w", "device_fraud_count", "month")


def load_baf(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Download official BAF Base.csv; see docs/s5_classification.md.")
    if sha256_file(path) != DATASET_SHA256:
        raise ValueError("Expected official BAF Base.csv version 2. See docs/s5_classification.md for its source.")
    frame = pd.read_csv(path)
    if set(frame.columns) != set(FIELDS):
        raise ValueError("Expected the official BAF Base.csv fields. Do not use a biased variant or an exported index column.")
    binary_labels(frame.fraud_bool, both_classes=True)
    if frame.isna().any().any() or not frame.month.isin(range(8)).all():
        raise ValueError("Unexpected missing raw fields or month values in BAF Base.csv.")
    numeric = frame.drop(columns=list(CATEGORICAL)).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("BAF raw numeric fields must be finite.")
    frame.index = pd.RangeIndex(len(frame), name="case_id")
    return frame


def prepare_baf(raw):
    frame = raw.copy()
    for column in SENTINEL_MINUS_ONE:
        frame[column] = frame[column].replace(-1, np.nan)
    # The datasheet defines all negative values in this field as missing.
    frame["intended_balcon_amount"] = frame.intended_balcon_amount.mask(frame.intended_balcon_amount < 0)
    # Keep absence visible: an imputed median is not an observed value.
    for column in (*SENTINEL_MINUS_ONE, "intended_balcon_amount"):
        frame[column + "_missing"] = frame[column].isna().astype(int)
    return frame


def baf_contract():
    return PredictionContract(name="BAF completed-application review", task="classification",
        unit="one completed account application", target="fraud_bool", target_unit="0 legitimate; 1 fraudulent",
        prediction_moment="after application completion, before approval",
        information_set="selected application, session and historical activity fields; no later outcome",
        action="prioritize human review under a hypothetical false-positive constraint")


def make_baf_problem(frame, schema):
    schema.validate(frame)
    forbidden = EXCLUDED.intersection(schema.columns)
    if forbidden:
        raise ValueError(f"Fields excluded from the S5 starting contract: {sorted(forbidden)}")
    if not set(schema.categorical).issubset(CATEGORICAL) or set(schema.numeric).intersection(CATEGORICAL):
        raise ValueError("Keep BAF categorical fields in the categorical schema.")
    metadata = frame[["month", "customer_age", "source"]].copy()
    metadata["age_group"] = np.where(frame.customer_age < 50, "below_50", "50_or_above")
    return ClassificationProblem(frame[schema.columns], frame.fraud_bool, metadata, baf_contract())


def month_masks(frame, split):
    if set(split) != {"train", "calibration", "validation", "test"}:
        raise ValueError("Expected training, calibration, validation and test month lists.")
    flattened = [month for values in split.values() for month in values]
    if sorted(flattened) != list(range(8)):
        raise ValueError("Assign every month exactly once.")
    if not (max(split["train"]) < min(split["calibration"]) <= max(split["calibration"])
            < min(split["validation"]) <= max(split["validation"]) < min(split["test"])):
        raise ValueError("BAF split roles must follow the declared chronological order.")
    masks = {name: frame.month.isin(values).to_numpy() for name, values in split.items()}
    for name, mask in masks.items():
        try:
            binary_labels(frame.loc[mask, "fraud_bool"], both_classes=True)
        except ValueError as exc:
            raise ValueError(f"Split {name} must contain both classes.") from exc
    return masks


def data_audit(raw):
    rows = []
    clean = prepare_baf(raw)
    for column in raw:
        rows.append({"field": column, "dtype": str(raw[column].dtype), "unique_values": raw[column].nunique(),
                     "documented_missing": int(clean[column].isna().sum()), "model_eligible": column not in EXCLUDED})
    return pd.DataFrame(rows)


def handoff_signature(schema, preprocessing, config):
    return {"schema": asdict(schema), "preprocessing": asdict(preprocessing), "split": config["split"],
            "contract": baf_contract().to_dict()}


def validate_handoff(source_run, *, root, dataset_fingerprint, signature, expected_phase):
    if source_run is None or not str(source_run).strip():
        raise ValueError("Set source_run to the explicit artifact folder printed by the preceding modeling notebook.")
    path = Path(source_run)
    if not path.is_absolute():
        path = Path(root) / path
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    experiment = json.loads((path / "experiment.json").read_text(encoding="utf-8"))
    if manifest["dataset"]["sha256"] != dataset_fingerprint:
        raise ValueError("Source run uses a different dataset fingerprint.")
    # JSON normalizes tuple-valued feature schemas into lists.
    normalized = json.loads(json.dumps(signature))
    if experiment.get("handoff") != normalized or experiment.get("phase") != expected_phase:
        raise ValueError("Source run has incompatible feature definitions, partitions, contract or phase.")
    if not experiment.get("completed"):
        raise ValueError("The source run did not complete successfully.")
    return path, experiment


def verify_baf(path):
    raw = load_baf(path)
    fingerprint = sha256_file(path)
    if fingerprint != DATASET_SHA256:
        raise ValueError("This file differs from official BAF Base.csv version 2; inspect its source before proceeding.")
    return {"dataset": "BAF Base", "version": 2, "sha256": fingerprint, "rows": len(raw),
            "columns": len(raw.columns), "positive_class": 1,
            "months": raw.groupby("month").fraud_bool.agg(["size", "sum", "mean"]).reset_index().to_dict("records")}
