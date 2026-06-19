"""
Robust Event Congestion Duration Model Training

Run from the src directory:

    python train_impact_model.py

Outputs:
    ../models/duration_q10.joblib
    ../models/duration_q50.joblib
    ../models/duration_q90.joblib
    ../models/encoders.joblib
    ../models/duration_metadata.joblib
    ../models/feature_importance.csv
    ../models/duration_validation.csv
    ../models/priority_classifier.joblib
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    mean_pinball_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


# =========================================================
# PATHS
# =========================================================

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent

DATA_PATH = PROJECT_DIR / "data" / "events_clean.csv"
MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# FEATURES
# =========================================================

CAT_FEATURES = [
    "event_type",
    "event_cause",
    "corridor",
    "zone",
    "time_bucket",
    "veh_type",
]

NUM_FEATURES = [
    "latitude",
    "longitude",
    "hour",
    "dow",
    "month",
    "is_weekend",
    "requires_road_closure",
]

FEATURES = CAT_FEATURES + NUM_FEATURES


# =========================================================
# CONFIGURATION
# =========================================================

QUANTILES = [0.1, 0.5, 0.9]

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Cap training duration at dataset P95 or 12 hours,
# whichever is lower.
TARGET_CAP_PERCENTILE = 0.95
MAX_OPERATIONAL_DURATION_MIN = 720.0

# The dashboard P90 will also be constrained relative to P50.
OPERATIONAL_P90_MULTIPLIER = 3.0
OPERATIONAL_P90_MIN_EXTRA_MIN = 30.0

USE_LOG_TARGET = True


# =========================================================
# HELPERS
# =========================================================

def create_time_bucket(hour_series: pd.Series) -> pd.Series:
    hours = pd.to_numeric(
        hour_series,
        errors="coerce",
    ).fillna(0).astype(int)

    return pd.Series(
        np.select(
            [
                hours.between(7, 10),
                hours.between(17, 20),
                (hours >= 23) | (hours <= 5),
            ],
            [
                "morning_peak",
                "evening_peak",
                "night",
            ],
            default="off_peak",
        ),
        index=hours.index,
    )


def convert_binary(series: pd.Series) -> pd.Series:
    mapping = {
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0,
        "high": 1,
        "low": 0,
    }

    text_result = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    numeric_result = pd.to_numeric(
        series,
        errors="coerce",
    )

    result = text_result.fillna(numeric_result).fillna(0)

    return result.clip(0, 1).astype(int)


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "hour" not in df.columns:
        df["hour"] = 0

    if "dow" not in df.columns:
        df["dow"] = 0

    if "month" not in df.columns:
        df["month"] = 1

    if "time_bucket" not in df.columns:
        df["time_bucket"] = create_time_bucket(df["hour"])

    if "is_weekend" not in df.columns:
        dow = pd.to_numeric(
            df["dow"],
            errors="coerce",
        ).fillna(0)

        df["is_weekend"] = dow.isin([5, 6]).astype(int)

    if "requires_road_closure" not in df.columns:
        df["requires_road_closure"] = 0

    for column in CAT_FEATURES:
        if column not in df.columns:
            df[column] = "unknown"

        df[column] = (
            df[column]
            .fillna("unknown")
            .astype(str)
            .str.strip()
        )

        invalid_values = df[column].isin(
            ["", "nan", "None", "NaN"]
        )

        df.loc[invalid_values, column] = "unknown"

    for column in [
        "latitude",
        "longitude",
        "hour",
        "dow",
        "month",
    ]:
        if column not in df.columns:
            df[column] = np.nan

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["is_weekend"] = convert_binary(
        df["is_weekend"]
    )

    df["requires_road_closure"] = convert_binary(
        df["requires_road_closure"]
    )

    return df


def calculate_numeric_medians(
    train_df: pd.DataFrame,
) -> dict:
    medians = {}

    for column in NUM_FEATURES:
        values = pd.to_numeric(
            train_df[column],
            errors="coerce",
        )

        median = values.median()

        if pd.isna(median):
            median = 0.0

        medians[column] = float(median)

    return medians


def apply_numeric_medians(
    df: pd.DataFrame,
    medians: dict,
) -> pd.DataFrame:
    df = df.copy()

    for column in NUM_FEATURES:
        fallback = medians.get(column, 0.0)

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        ).fillna(fallback)

    return df


def encode_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=np.float64,
    )

    x_train = train_df[FEATURES].copy()
    x_test = test_df[FEATURES].copy()

    x_train[CAT_FEATURES] = encoder.fit_transform(
        x_train[CAT_FEATURES].astype(str)
    )

    x_test[CAT_FEATURES] = encoder.transform(
        x_test[CAT_FEATURES].astype(str)
    )

    return x_train, x_test, encoder


def transform_target(values):
    values = np.asarray(values, dtype=float)

    if USE_LOG_TARGET:
        return np.log1p(values)

    return values


def inverse_target(values):
    values = np.asarray(values, dtype=float)

    if USE_LOG_TARGET:
        return np.expm1(values)

    return values


def create_duration_model(
    quantile: float,
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        max_iter=400,
        max_depth=4,
        learning_rate=0.035,
        min_samples_leaf=40,
        l2_regularization=3.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=RANDOM_STATE,
    )


def create_duration_strata(
    duration: pd.Series,
):
    try:
        strata = pd.qcut(
            duration,
            q=5,
            labels=False,
            duplicates="drop",
        )

        counts = strata.value_counts()

        if (
            strata.nunique() >= 2
            and counts.min() >= 2
        ):
            return strata

    except (ValueError, TypeError):
        pass

    return None


# =========================================================
# TRAINING
# =========================================================

def train():
    print("=" * 72)
    print("EVENT CONGESTION IMPACT MODEL")
    print("=" * 72)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        low_memory=False,
    )

    df = prepare_dataframe(df)

    if "duration_min" not in df.columns:
        raise ValueError(
            "duration_min column is missing from events_clean.csv"
        )

    df["duration_min"] = pd.to_numeric(
        df["duration_min"],
        errors="coerce",
    )

    duration_df = df[
        df["duration_min"].notna()
        & np.isfinite(df["duration_min"])
        & (df["duration_min"] >= 1)
    ].copy()

    if len(duration_df) < 100:
        raise ValueError(
            "Not enough valid duration records for training."
        )

    print(f"\nTotal records: {len(df):,}")
    print(
        f"Records with valid duration: "
        f"{len(duration_df):,}"
    )

    print("\nOriginal duration distribution:")

    print(
        duration_df["duration_min"]
        .describe(
            percentiles=[
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .round(2)
        .to_string()
    )

    # -----------------------------------------------------
    # SPLIT BEFORE CALCULATING CAP
    # -----------------------------------------------------

    strata = create_duration_strata(
        duration_df["duration_min"]
    )

    train_df, test_df = train_test_split(
        duration_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=strata,
    )

    training_percentile_cap = float(
        train_df["duration_min"].quantile(
            TARGET_CAP_PERCENTILE
        )
    )

    duration_cap = min(
        training_percentile_cap,
        MAX_OPERATIONAL_DURATION_MIN,
    )

    if not np.isfinite(duration_cap):
        duration_cap = MAX_OPERATIONAL_DURATION_MIN

    duration_cap = max(
        60.0,
        duration_cap,
    )

    print(
        f"\nTraining P95 duration: "
        f"{training_percentile_cap:.1f} minutes"
    )

    print(
        f"Operational training cap: "
        f"{duration_cap:.1f} minutes"
    )

    train_df["duration_original"] = (
        train_df["duration_min"]
    )

    test_df["duration_original"] = (
        test_df["duration_min"]
    )

    train_df["duration_min"] = (
        train_df["duration_min"]
        .clip(
            lower=1,
            upper=duration_cap,
        )
    )

    test_df["duration_min"] = (
        test_df["duration_min"]
        .clip(
            lower=1,
            upper=duration_cap,
        )
    )

    capped_training_rows = int(
        (
            train_df["duration_original"]
            > duration_cap
        ).sum()
    )

    capped_testing_rows = int(
        (
            test_df["duration_original"]
            > duration_cap
        ).sum()
    )

    print(
        f"Training rows capped: "
        f"{capped_training_rows:,}"
    )

    print(
        f"Testing rows capped: "
        f"{capped_testing_rows:,}"
    )

    # -----------------------------------------------------
    # FEATURE PREPARATION
    # -----------------------------------------------------

    numeric_medians = calculate_numeric_medians(
        train_df
    )

    train_df = apply_numeric_medians(
        train_df,
        numeric_medians,
    )

    test_df = apply_numeric_medians(
        test_df,
        numeric_medians,
    )

    x_train, x_test, encoder = encode_train_test(
        train_df,
        test_df,
    )

    y_train_original = (
        train_df["duration_min"]
        .to_numpy(dtype=float)
    )

    y_test_original = (
        test_df["duration_min"]
        .to_numpy(dtype=float)
    )

    y_train_transformed = transform_target(
        y_train_original
    )

    y_test_transformed = transform_target(
        y_test_original
    )

    print(
        f"\nTraining rows: {len(x_train):,}"
    )

    print(
        f"Testing rows: {len(x_test):,}"
    )

    # -----------------------------------------------------
    # QUANTILE MODELS
    # -----------------------------------------------------

    quantile_models = {}
    predictions_by_quantile = {}

    print("\n=== Quantile duration models ===")

    for quantile in QUANTILES:
        model = create_duration_model(
            quantile
        )

        model.fit(
            x_train,
            y_train_transformed,
        )

        prediction_transformed = model.predict(
            x_test
        )

        prediction = inverse_target(
            prediction_transformed
        )

        prediction = np.clip(
            prediction,
            1,
            duration_cap,
        )

        quantile_models[quantile] = model
        predictions_by_quantile[quantile] = prediction

        output_path = (
            MODEL_DIR
            / f"duration_q{int(quantile * 100)}.joblib"
        )

        joblib.dump(
            model,
            output_path,
        )

        raw_loss = mean_pinball_loss(
            y_test_original,
            prediction,
            alpha=quantile,
        )

        print(
            f"P{int(quantile * 100)} "
            f"pinball loss: {raw_loss:.2f}"
        )

    # -----------------------------------------------------
    # FIX QUANTILE CROSSING
    # -----------------------------------------------------

    raw_prediction_matrix = np.column_stack([
        predictions_by_quantile[0.1],
        predictions_by_quantile[0.5],
        predictions_by_quantile[0.9],
    ])

    crossing_mask = (
        (
            raw_prediction_matrix[:, 0]
            > raw_prediction_matrix[:, 1]
        )
        |
        (
            raw_prediction_matrix[:, 1]
            > raw_prediction_matrix[:, 2]
        )
    )

    crossing_rate = float(
        crossing_mask.mean()
    )

    ordered_predictions = np.sort(
        raw_prediction_matrix,
        axis=1,
    )

    p10_prediction = ordered_predictions[:, 0]
    p50_prediction = ordered_predictions[:, 1]
    p90_prediction = ordered_predictions[:, 2]

    # -----------------------------------------------------
    # EVALUATION
    # -----------------------------------------------------

    p10_loss = mean_pinball_loss(
        y_test_original,
        p10_prediction,
        alpha=0.1,
    )

    p50_loss = mean_pinball_loss(
        y_test_original,
        p50_prediction,
        alpha=0.5,
    )

    p90_loss = mean_pinball_loss(
        y_test_original,
        p90_prediction,
        alpha=0.9,
    )

    median_mae = mean_absolute_error(
        y_test_original,
        p50_prediction,
    )

    interval_coverage = float(
        np.mean(
            (
                y_test_original
                >= p10_prediction
            )
            &
            (
                y_test_original
                <= p90_prediction
            )
        )
    )

    average_interval_width = float(
        np.mean(
            p90_prediction
            - p10_prediction
        )
    )

    print("\n=== Validation results ===")

    print(
        f"P10 pinball loss: "
        f"{p10_loss:.2f}"
    )

    print(
        f"P50 pinball loss: "
        f"{p50_loss:.2f}"
    )

    print(
        f"P90 pinball loss: "
        f"{p90_loss:.2f}"
    )

    print(
        f"Median MAE: "
        f"{median_mae:.2f} minutes"
    )

    print(
        f"P10-P90 coverage: "
        f"{interval_coverage * 100:.1f}%"
    )

    print(
        f"Average interval width: "
        f"{average_interval_width:.1f} minutes"
    )

    print(
        f"Quantile crossing rate: "
        f"{crossing_rate * 100:.1f}%"
    )

    validation_df = pd.DataFrame({
        "actual_duration_min":
            y_test_original,

        "original_duration_min":
            test_df[
                "duration_original"
            ].to_numpy(dtype=float),

        "p10":
            p10_prediction,

        "p50":
            p50_prediction,

        "p90":
            p90_prediction,

        "absolute_error":
            np.abs(
                y_test_original
                - p50_prediction
            ),
    })

    validation_df.to_csv(
        MODEL_DIR / "duration_validation.csv",
        index=False,
    )

    print("\nSample validation predictions:")

    sample_size = min(
        10,
        len(validation_df),
    )

    print(
        validation_df.head(sample_size)
        .round(1)
        .to_string(index=False)
    )

    # -----------------------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------------------

    print("\nCalculating permutation importance...")

    importance = permutation_importance(
        quantile_models[0.5],
        x_test,
        y_test_transformed,
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame({
        "feature": FEATURES,
        "importance":
            importance.importances_mean,
        "importance_std":
            importance.importances_std,
    }).sort_values(
        "importance",
        ascending=False,
    )

    importance_df.to_csv(
        MODEL_DIR / "feature_importance.csv",
        index=False,
    )

    print("\nTop features:")

    print(
        importance_df.head(10)
        .round(5)
        .to_string(index=False)
    )

    # -----------------------------------------------------
    # SAVE ENCODER AND METADATA
    # -----------------------------------------------------

    joblib.dump(
        encoder,
        MODEL_DIR / "encoders.joblib",
    )

    metadata = {
        "model_version":
            "duration-quantile-v2",

        "features":
            FEATURES,

        "categorical_features":
            CAT_FEATURES,

        "numerical_features":
            NUM_FEATURES,

        "quantiles":
            QUANTILES,

        "use_log_target":
            USE_LOG_TARGET,

        "target_cap_percentile":
            TARGET_CAP_PERCENTILE,

        "duration_cap_min":
            float(duration_cap),

        "max_operational_duration_min":
            MAX_OPERATIONAL_DURATION_MIN,

        "operational_p90_multiplier":
            OPERATIONAL_P90_MULTIPLIER,

        "operational_p90_min_extra_min":
            OPERATIONAL_P90_MIN_EXTRA_MIN,

        "numeric_medians":
            numeric_medians,

        "validation": {
            "p10_pinball_loss":
                float(p10_loss),

            "p50_pinball_loss":
                float(p50_loss),

            "p90_pinball_loss":
                float(p90_loss),

            "median_mae_min":
                float(median_mae),

            "interval_coverage":
                interval_coverage,

            "average_interval_width_min":
                average_interval_width,

            "quantile_crossing_rate":
                crossing_rate,
        },
    }

    joblib.dump(
        metadata,
        MODEL_DIR / "duration_metadata.joblib",
    )

    # -----------------------------------------------------
    # PRIORITY CLASSIFIER
    # -----------------------------------------------------

    if "is_high_priority" in df.columns:
        print(
            "\n=== Priority classifier "
            "(reference only) ==="
        )

        priority_df = df[
            df["is_high_priority"].notna()
        ].copy()

        priority_df = apply_numeric_medians(
            priority_df,
            numeric_medians,
        )

        x_priority = (
            priority_df[FEATURES]
            .copy()
        )

        x_priority[CAT_FEATURES] = (
            encoder.transform(
                x_priority[
                    CAT_FEATURES
                ].astype(str)
            )
        )

        y_priority = convert_binary(
            priority_df[
                "is_high_priority"
            ]
        )

        if y_priority.nunique() >= 2:
            (
                x_priority_train,
                x_priority_test,
                y_priority_train,
                y_priority_test,
            ) = train_test_split(
                x_priority,
                y_priority,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=y_priority,
            )

            priority_model = (
                HistGradientBoostingClassifier(
                    max_iter=250,
                    max_depth=5,
                    learning_rate=0.05,
                    min_samples_leaf=30,
                    l2_regularization=2.0,
                    random_state=RANDOM_STATE,
                )
            )

            priority_model.fit(
                x_priority_train,
                y_priority_train,
            )

            priority_prediction = (
                priority_model.predict(
                    x_priority_test
                )
            )

            print(
                classification_report(
                    y_priority_test,
                    priority_prediction,
                    labels=[0, 1],
                    target_names=[
                        "Low",
                        "High",
                    ],
                    zero_division=0,
                )
            )

            joblib.dump(
                priority_model,
                MODEL_DIR
                / "priority_classifier.joblib",
            )

        else:
            print(
                "Priority classifier skipped: "
                "only one target class exists."
            )

    print("\n" + "=" * 72)
    print("TRAINING COMPLETED")
    print(f"Models saved in: {MODEL_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    train()