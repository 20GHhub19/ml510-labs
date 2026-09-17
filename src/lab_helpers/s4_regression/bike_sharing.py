"""Prepare Bike Sharing observations for regression and sequence examples."""
from pathlib import Path
import numpy as np
import pandas as pd
from ml_lib.data.loading import read_local_csv, sha256_file
from ml_lib.data.validation import require_columns, require_time_index, hourly_grid
from ml_lib.features.transformations import add_cyclic_features
from ml_lib.features.temporal import lag_features, make_sequences
from ml_lib.problems.contracts import PredictionContract
from ml_lib.problems.regression import RegressionProblem

REQUIRED = (
    "instant", "dteday", "season", "yr", "mnth", "hr", "holiday", "weekday",
    "workingday", "weathersit", "temp", "atemp", "hum", "windspeed", "casual", "registered", "cnt",
)
FORBIDDEN_TABULAR = {"cnt", "casual", "registered", "instant", "dteday"}
CALENDAR = ("hr_sin", "hr_cos", "weekday_sin", "weekday_cos", "mnth_sin", "mnth_cos", "holiday", "workingday")
HISTORY = ("cnt", "temp", "hum", "windspeed", *CALENDAR)


def load_bike_hourly(path: str | Path) -> pd.DataFrame:
    """Read the UCI hourly records, check the fields, and sort by timestamp."""
    raw = read_local_csv(path)
    require_columns(raw, REQUIRED)
    for name in REQUIRED:
        if name != "dteday":
            raw[name] = pd.to_numeric(raw[name], errors="raise")
            if not np.isfinite(raw[name].to_numpy(dtype=float)).all():
                raise ValueError(f"Non-finite source values in {name}. Investigate before modeling.")
    for name in ("instant", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "season", "weathersit", "casual", "registered", "cnt"):
        if not np.equal(raw[name], np.floor(raw[name])).all():
            raise ValueError(f"Expected integer values in {name}.")
    for name, low, high in [("hr", 0, 23), ("mnth", 1, 12), ("weekday", 0, 6), ("season", 1, 4),
                            ("weathersit", 1, 4), ("holiday", 0, 1), ("workingday", 0, 1), ("yr", 0, 1)]:
        if not raw[name].between(low, high).all():
            raise ValueError(f"Values outside [{low}, {high}] in {name}.")
    if (raw[["cnt", "casual", "registered"]] < 0).any().any():
        raise ValueError("Rental counts must be nonnegative.")
    if not (raw.casual + raw.registered == raw.cnt).all():
        raise ValueError("Source contract violated: casual + registered must equal cnt.")
    if raw.instant.duplicated().any():
        raise ValueError("Duplicate source record IDs.")
    dates = pd.to_datetime(raw.dteday, format="%Y-%m-%d", errors="raise")
    if not raw.mnth.eq(dates.dt.month).all() or not raw.yr.eq(dates.dt.year - 2011).all():
        raise ValueError("Date, month and year fields disagree; use the original 2011-2012 UCI data.")
    # UCI uses Sunday=0; pandas uses Monday=0. Do not silently mix these codes.
    if not raw.weekday.eq((dates.dt.dayofweek + 1) % 7).all():
        raise ValueError("UCI weekday must use Sunday=0 and agree with dteday.")
    expected_working = ((dates.dt.dayofweek < 5) & raw.holiday.eq(0)).astype(int)
    if not raw.workingday.eq(expected_working).all():
        raise ValueError("workingday disagrees with weekday/holiday.")
    if raw.groupby("dteday").holiday.nunique().gt(1).any():
        raise ValueError("Holiday flag changes within one calendar day.")
    raw.index = pd.DatetimeIndex(dates + pd.to_timedelta(raw.hr, unit="h"), name="timestamp")
    raw = raw.sort_index()
    require_time_index(raw)
    return raw


def data_audit(raw: pd.DataFrame, path: str | Path) -> dict:
    grid = hourly_grid(raw)
    missing = grid.index.difference(raw.index)
    return {
        "source": "UCI Bike Sharing, DOI 10.24432/C5W894", "file_sha256": sha256_file(path),
        "rows": len(raw), "first_bucket": str(raw.index.min()), "last_bucket": str(raw.index.max()),
        "expected_hourly_buckets": len(grid), "missing_hourly_buckets": len(missing),
        "first_missing_bucket_labels": [str(x) for x in missing[:12]],
        "target_identity_passed": bool((raw.casual + raw.registered == raw.cnt).all()),
        "time_convention": "naive source-local hourly bucket labels; not UTC or Toronto conversion",
        "gap_policy": "keep missing targets unknown; never replace with zero",
    }


def add_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    """Deterministic future calendar; holiday schedule is assumed known in advance.

    Source holiday flags are a calendar lookup, not an outcome-derived feature.
    No weather or rental outcomes are filled.
    """
    out = frame.copy()
    out["hr"] = out.index.hour
    out["mnth"] = out.index.month
    out["weekday"] = (out.index.dayofweek + 1) % 7
    by_date = out["holiday"].groupby(out.index.normalize()).first()
    out["holiday"] = pd.Series(out.index.normalize().map(by_date), index=out.index, dtype=float)
    out["workingday"] = np.where(out.holiday.isna(), np.nan,
                                  ((out.index.dayofweek < 5) & out.holiday.eq(0)).astype(float))
    return add_cyclic_features(out, {"hr": 24, "weekday": 7, "mnth": 12})


def regression_contract() -> PredictionContract:
    return PredictionContract(
        name="bike-context-regression-v1", task="regression", unit="system-hour", target="cnt",
        target_unit="completed rentals per hour", prediction_moment="conditional on that hour's measured context",
        information_set="same-hour observed weather and calendar; no rental components or history",
        action="study completed rental activity for planning; not a deployable start-of-hour forecast",
    )


def forecasting_contract(*, representation: str = "lagged", lookback: int = 24) -> PredictionContract:
    if representation not in {"lagged", "sequence"}:
        raise ValueError("Unknown forecasting representation.")
    information = ("past count lags and averages, last observed weather, next-hour calendar"
                   if representation == "lagged" else
                   f"last {lookback} observed hours of counts/weather/calendar plus next-hour calendar")
    return PredictionContract(
        name=f"bike-{representation}-next-hour-v2", task="forecasting", unit="system-hour",
        target="cnt", target_unit="completed rentals per hour",
        prediction_moment="hourly, after the preceding hour's observations arrive; zero reporting delay assumed",
        information_set=information, action="support aggregate next-hour planning",
    )


def make_tabular_problem(raw: pd.DataFrame) -> RegressionProblem:
    frame = add_calendar(raw)
    frame["temp_squared"] = frame.temp ** 2
    frame["temp_workingday"] = frame.temp * frame.workingday
    columns = ["temp", "hum", "windspeed", "hr", "weekday", "season", "holiday", "workingday", "weathersit", "mnth",
               "temp_squared", "temp_workingday", "hr_sin", "hr_cos", "weekday_sin", "weekday_cos"]
    if set(columns) & FORBIDDEN_TABULAR:
        raise ValueError("Tabular feature selector contains a leaking or identifier column.")
    metadata = frame[["workingday", "weathersit", "season"]].copy()
    return RegressionProblem(frame[columns], frame.cnt.astype(float), metadata, regression_contract(),
                             {"input_rows": len(raw), "retained_rows": len(frame)})


def make_forecasting_problem(raw: pd.DataFrame, *, lags=(1, 2, 24, 168), rolling=(3, 24)) -> RegressionProblem:
    """Build one row per target hour s using observations available through s-1."""
    frame = add_calendar(hourly_grid(raw))
    X = lag_features(frame, "cnt", list(lags), list(rolling))
    for name in ("temp", "hum", "windspeed"):
        X[f"{name}_lag_1"] = frame[name].shift(1)
    for name in CALENDAR:
        X[name] = frame[name]
    valid = X.notna().all(axis=1) & frame.cnt.notna()
    metadata = frame.loc[valid, ["workingday", "weathersit", "season"]].copy()
    metadata["origin"] = metadata.index - pd.Timedelta(hours=1)
    contract = forecasting_contract(representation="lagged")
    return RegressionProblem(X.loc[valid], frame.loc[valid, "cnt"].astype(float), metadata, contract,
        {"hourly_grid_rows": len(frame), "retained_rows": int(valid.sum()), "dropped_incomplete_rows": int((~valid).sum()),
         "retained_fraction": float(valid.mean()), "feature_names": list(X.columns),
         "index_semantics": "target hour s; every observed input is from s-1 or earlier"})


def make_bike_sequences(raw: pd.DataFrame, *, history_columns, future_columns, lookback):
    if not history_columns or len(set(history_columns)) != len(history_columns) or not set(history_columns) <= set(HISTORY):
        raise ValueError(f"Choose unique history fields from {HISTORY}.")
    if len(set(future_columns)) != len(future_columns) or not set(future_columns) <= set(CALENDAR):
        raise ValueError(f"Known-future fields must be unique calendar fields from {CALENDAR}.")
    frame = add_calendar(hourly_grid(raw))
    dataset = make_sequences(frame, history_columns=history_columns, future_columns=future_columns,
                             target="cnt", lookback=lookback)
    return dataset, frame
