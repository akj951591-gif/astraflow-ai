"""
Duration prediction wrapper.

Returns:

    p10:
        Optimistic congestion duration.

    p50:
        Expected congestion duration.

    p90:
        Operational severe-case duration.

    raw_p90:
        Original upper quantile before operational constraints.
"""

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
DEFAULT_MODEL_DIR = PROJECT_DIR / "models"


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


def time_bucket(hour: int) -> str:
    hour = int(hour)

    if 7 <= hour <= 10:
        return "morning_peak"

    if 17 <= hour <= 20:
        return "evening_peak"

    if hour >= 23 or hour <= 5:
        return "night"

    return "off_peak"


def safe_text(
    value,
    default="unknown",
) -> str:
    if value is None:
        return default

    value = str(value).strip()

    if value in [
        "",
        "nan",
        "None",
        "NaN",
    ]:
        return default

    return value


class DurationPredictor:
    def __init__(
        self,
        model_dir=None,
    ):
        if model_dir is None:
            model_dir = DEFAULT_MODEL_DIR

        self.model_dir = Path(model_dir)

        required_files = [
            "encoders.joblib",
            "duration_q10.joblib",
            "duration_q50.joblib",
            "duration_q90.joblib",
        ]

        missing_files = [
            filename
            for filename in required_files
            if not (
                self.model_dir / filename
            ).exists()
        ]

        if missing_files:
            raise FileNotFoundError(
                "Missing model files: "
                + ", ".join(missing_files)
                + ". Run train_impact_model.py first."
            )

        self.encoder = joblib.load(
            self.model_dir
            / "encoders.joblib"
        )

        self.q10 = joblib.load(
            self.model_dir
            / "duration_q10.joblib"
        )

        self.q50 = joblib.load(
            self.model_dir
            / "duration_q50.joblib"
        )

        self.q90 = joblib.load(
            self.model_dir
            / "duration_q90.joblib"
        )

        metadata_path = (
            self.model_dir
            / "duration_metadata.joblib"
        )

        if metadata_path.exists():
            self.metadata = joblib.load(
                metadata_path
            )
        else:
            self.metadata = {
                "model_version":
                    "legacy",

                "use_log_target":
                    False,

                "duration_cap_min":
                    720.0,

                "operational_p90_multiplier":
                    3.0,

                "operational_p90_min_extra_min":
                    30.0,

                "numeric_medians":
                    {},
            }

        self.use_log_target = bool(
            self.metadata.get(
                "use_log_target",
                False,
            )
        )

        self.duration_cap = float(
            self.metadata.get(
                "duration_cap_min",
                720.0,
            )
        )

        self.p90_multiplier = float(
            self.metadata.get(
                "operational_p90_multiplier",
                3.0,
            )
        )

        self.p90_min_extra = float(
            self.metadata.get(
                "operational_p90_min_extra_min",
                30.0,
            )
        )

        self.numeric_medians = (
            self.metadata.get(
                "numeric_medians",
                {},
            )
        )

    def inverse_target(
        self,
        values,
    ):
        values = np.asarray(
            values,
            dtype=float,
        )

        if self.use_log_target:
            maximum_safe_log = np.log1p(
                max(
                    self.duration_cap * 10,
                    100,
                )
            )

            values = np.clip(
                values,
                -10,
                maximum_safe_log,
            )

            return np.expm1(values)

        return values

    def build_input(
        self,
        lat,
        lon,
        event_cause,
        hour,
        dow,
        month,
        event_type,
        corridor,
        zone,
        veh_type,
        requires_road_closure,
    ) -> pd.DataFrame:
        hour = int(hour)
        dow = int(dow)

        if month is None:
            month = datetime.now().month

        month = int(month)

        row = pd.DataFrame([{
            "event_type":
                safe_text(
                    event_type,
                    "unplanned",
                ),

            "event_cause":
                safe_text(
                    event_cause,
                    "others",
                ),

            "corridor":
                safe_text(
                    corridor,
                    "Non-corridor",
                ),

            "zone":
                safe_text(
                    zone,
                    "unknown",
                ),

            "time_bucket":
                time_bucket(hour),

            "veh_type":
                safe_text(
                    veh_type,
                    "unknown",
                ),

            "latitude":
                float(lat),

            "longitude":
                float(lon),

            "hour":
                hour,

            "dow":
                dow,

            "month":
                month,

            "is_weekend":
                int(
                    dow in [5, 6]
                ),

            "requires_road_closure":
                int(
                    bool(
                        requires_road_closure
                    )
                ),
        }])

        for column in NUM_FEATURES:
            fallback = float(
                self.numeric_medians.get(
                    column,
                    0.0,
                )
            )

            row[column] = pd.to_numeric(
                row[column],
                errors="coerce",
            ).fillna(fallback)

        row[CAT_FEATURES] = (
            self.encoder.transform(
                row[
                    CAT_FEATURES
                ].astype(str)
            )
        )

        return row[FEATURES]

    def predict(
        self,
        lat,
        lon,
        event_cause,
        hour,
        dow,
        month=None,
        event_type="unplanned",
        corridor="Non-corridor",
        zone="unknown",
        veh_type="unknown",
        requires_road_closure=False,
    ):
        x_input = self.build_input(
            lat=lat,
            lon=lon,
            event_cause=event_cause,
            hour=hour,
            dow=dow,
            month=month,
            event_type=event_type,
            corridor=corridor,
            zone=zone,
            veh_type=veh_type,
            requires_road_closure=
                requires_road_closure,
        )

        q10_model_output = float(
            self.inverse_target(
                self.q10.predict(
                    x_input
                )
            )[0]
        )

        q50_model_output = float(
            self.inverse_target(
                self.q50.predict(
                    x_input
                )
            )[0]
        )

        q90_model_output = float(
            self.inverse_target(
                self.q90.predict(
                    x_input
                )
            )[0]
        )

        raw_model_predictions = np.array([
            q10_model_output,
            q50_model_output,
            q90_model_output,
        ])

        raw_model_predictions = np.nan_to_num(
            raw_model_predictions,
            nan=1.0,
            posinf=self.duration_cap,
            neginf=1.0,
        )

        # Prevent quantile crossing.
        ordered_predictions = np.sort(
            raw_model_predictions
        )

        raw_p10 = float(
            ordered_predictions[0]
        )

        raw_p50 = float(
            ordered_predictions[1]
        )

        raw_p90 = float(
            ordered_predictions[2]
        )

        p10 = np.clip(
            raw_p10,
            1.0,
            self.duration_cap,
        )

        p50 = np.clip(
            raw_p50,
            p10,
            self.duration_cap,
        )

        # The raw model may produce an extremely high P90.
        # Convert it into an operational severe-case estimate.
        p90_relative_limit = max(
            p50 + self.p90_min_extra,
            p50 * self.p90_multiplier,
        )

        p90 = min(
            raw_p90,
            self.duration_cap,
            p90_relative_limit,
        )

        p90 = max(
            p50,
            p90,
        )

        interval_width = float(
            p90 - p10
        )

        uncertainty_ratio = (
            interval_width
            / max(p50, 1.0)
        )

        if uncertainty_ratio >= 2.0:
            confidence = "VERY LOW"
            confidence_score = 25

        elif uncertainty_ratio >= 1.25:
            confidence = "LOW"
            confidence_score = 45

        elif uncertainty_ratio >= 0.65:
            confidence = "MODERATE"
            confidence_score = 70

        else:
            confidence = "HIGH"
            confidence_score = 90

        operationally_capped = bool(
            p90 < raw_p90
        )

        warnings = []

        if confidence in ["LOW", "VERY LOW"]:
            warnings.append(
                "Similar historical incidents show high duration variation. "
                "Manual review is recommended."
            )

        if operationally_capped:
            warnings.append(
                "The raw upper-quantile prediction was constrained to produce "
                "a practical operational severe-case estimate."
            )

        if p50 >= self.duration_cap * 0.85:
            warnings.append(
                "The expected duration is close to the maximum operational "
                "forecast horizon."
            )

        warning = (
            " ".join(warnings)
            if warnings else None
        )
        
        return {
            "p10":
                round(float(p10), 1),

            "p50":
                round(float(p50), 1),

            "p90":
                round(float(p90), 1),

            "raw_p10":
                round(float(raw_p10), 1),

            "raw_p50":
                round(float(raw_p50), 1),

            "raw_p90":
                round(float(raw_p90), 1),

            "interval_width":
                round(interval_width, 1),

            "confidence":
                confidence,

            "confidence_score":
                confidence_score,

            "warning":
                warning,

            "operationally_capped":
                operationally_capped,

            "duration_cap_min":
                round(
                    self.duration_cap,
                    1,
                ),

            "model_version":
                self.metadata.get(
                    "model_version",
                    "unknown",
                ),
        }