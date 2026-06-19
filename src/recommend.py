from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# =========================================================
# PROJECT PATHS
# =========================================================
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
DEFAULT_DATA_PATH = PROJECT_DIR / "data" / "events_clean.csv"


# =========================================================
# TRANSPARENT RESOURCE RULES
# =========================================================
# Format:
# (
#     minimum_duration,
#     road_closure_required,
#     named_corridor,
#     recommended_constables,
# )
MANPOWER_RULES = [
    (180, True, True, 8),
    (60, True, True, 5),
    (60, False, True, 3),
    (0, False, False, 1),
]


class IncidentRecommender:
    """
    Historical-similarity and rule-based resource recommender.

    It finds incidents with similar:
    - latitude
    - longitude
    - hour
    - day of week

    It then uses their historical outcomes to recommend:
    - road closure
    - diversion
    - barricades
    - traffic constables
    """

    def __init__(
        self,
        data_path: str | Path | None = None,
        k: int = 15,
    ):
        # Use an absolute project-relative path so this works
        # locally and on Streamlit Community Cloud.
        self.data_path = Path(
            data_path or DEFAULT_DATA_PATH
        ).resolve()

        if not self.data_path.exists():
            raise FileNotFoundError(
                "Historical incident dataset was not found.\n"
                f"Expected path: {self.data_path}\n"
                "Ensure data/events_clean.csv is committed to GitHub."
            )

        self.df = pd.read_csv(self.data_path)

        if self.df.empty:
            raise ValueError(
                f"The dataset is empty: {self.data_path}"
            )

        self.feat_cols = [
            "latitude",
            "longitude",
            "hour",
            "dow",
        ]

        required_columns = [
            "latitude",
            "longitude",
            "hour",
            "dow",
            "event_cause",
            "duration_min",
            "requires_road_closure",
            "is_high_priority",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in self.df.columns
        ]

        if missing_columns:
            raise ValueError(
                "events_clean.csv is missing required columns: "
                + ", ".join(missing_columns)
            )

        # Convert model features to numeric values.
        for column in self.feat_cols:
            self.df[column] = pd.to_numeric(
                self.df[column],
                errors="coerce",
            )

        self.df["duration_min"] = pd.to_numeric(
            self.df["duration_min"],
            errors="coerce",
        )

        self.df["requires_road_closure"] = (
            self.df["requires_road_closure"]
            .apply(self._to_boolean_number)
        )

        self.df["is_high_priority"] = (
            self.df["is_high_priority"]
            .apply(self._to_boolean_number)
        )

        self.df["event_cause"] = (
            self.df["event_cause"]
            .fillna("others")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        # Rows without geographic/time information cannot be
        # used by nearest-neighbour search.
        self.df = (
            self.df
            .dropna(subset=self.feat_cols)
            .reset_index(drop=True)
        )

        if self.df.empty:
            raise ValueError(
                "No valid rows remain after cleaning the "
                "nearest-neighbour feature columns."
            )

        self.k = max(
            1,
            min(int(k), len(self.df)),
        )

        # Standard deviation scaling prevents latitude and
        # longitude from dominating hour and day-of-week.
        self.scale = (
            self.df[self.feat_cols]
            .std()
            .replace(0, 1)
            .fillna(1)
        )

        feature_matrix = (
            self.df[self.feat_cols]
            .div(self.scale)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .to_numpy(dtype=float)
        )

        self.nn = NearestNeighbors(
            n_neighbors=self.k,
            metric="euclidean",
        )

        self.nn.fit(feature_matrix)

    @staticmethod
    def _to_boolean_number(value) -> float:
        """
        Convert common true/false representations to 1.0 or 0.0.
        """

        if pd.isna(value):
            return 0.0

        if isinstance(value, bool):
            return float(value)

        if isinstance(value, (int, float, np.number)):
            return 1.0 if float(value) > 0 else 0.0

        normalized = str(value).strip().lower()

        true_values = {
            "true",
            "yes",
            "y",
            "1",
            "high",
            "required",
        }

        return 1.0 if normalized in true_values else 0.0

    def _similar_incidents(
        self,
        lat: float,
        lon: float,
        hour: int,
        dow: int,
        cause: str | None = None,
    ) -> pd.DataFrame:
        """
        Return geographically and temporally similar incidents.
        """

        query = pd.DataFrame(
            [
                {
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "hour": int(hour),
                    "dow": int(dow),
                }
            ]
        )

        query_matrix = (
            query[self.feat_cols]
            .div(self.scale)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .to_numpy(dtype=float)
        )

        _, neighbour_indexes = self.nn.kneighbors(
            query_matrix
        )

        similar_incidents = (
            self.df
            .iloc[neighbour_indexes[0]]
            .copy()
        )

        if cause:
            normalized_cause = (
                str(cause)
                .strip()
                .lower()
            )

            cause_matches = similar_incidents[
                similar_incidents["event_cause"]
                == normalized_cause
            ]

            # Use cause-specific results only when enough
            # examples are available.
            if len(cause_matches) >= 5:
                similar_incidents = cause_matches

        return similar_incidents

    @staticmethod
    def _recommend_constables(
        median_duration: float,
        needs_closure: bool,
        is_named_corridor: bool,
    ) -> int:
        """
        Apply transparent manpower rules.
        """

        # Strongest conditions first.
        if (
            median_duration >= 180
            and needs_closure
            and is_named_corridor
        ):
            return 8

        if (
            median_duration >= 60
            and needs_closure
            and is_named_corridor
        ):
            return 5

        if (
            median_duration >= 60
            and not needs_closure
            and is_named_corridor
        ):
            return 3

        if needs_closure and is_named_corridor:
            return 4

        if needs_closure:
            return 3

        if is_named_corridor:
            return 2

        return 1

    def recommend(
        self,
        lat: float,
        lon: float,
        event_cause: str,
        start_hour: int,
        day_of_week: int,
        is_named_corridor: bool = False,
    ) -> dict:
        """
        Generate an explainable operational recommendation.
        """

        similar_incidents = self._similar_incidents(
            lat=lat,
            lon=lon,
            hour=start_hour,
            dow=day_of_week,
            cause=event_cause,
        )

        number_of_incidents = len(similar_incidents)

        median_duration = (
            similar_incidents["duration_min"]
            .median()
        )

        closure_rate = (
            similar_incidents[
                "requires_road_closure"
            ]
            .mean()
        )

        high_priority_rate = (
            similar_incidents[
                "is_high_priority"
            ]
            .mean()
        )

        if pd.isna(median_duration):
            median_duration = 0.0

        if pd.isna(closure_rate):
            closure_rate = 0.0

        if pd.isna(high_priority_rate):
            high_priority_rate = 0.0

        median_duration = max(
            0.0,
            float(median_duration),
        )

        closure_rate = min(
            1.0,
            max(0.0, float(closure_rate)),
        )

        high_priority_rate = min(
            1.0,
            max(0.0, float(high_priority_rate)),
        )

        # Recommend closure when at least 40% of similar
        # incidents historically required closure.
        needs_closure = closure_rate >= 0.40

        # Recommend diversion for closure events or incidents
        # whose median duration is longer than one hour.
        needs_diversion = (
            needs_closure
            or median_duration > 60
        )

        constables = self._recommend_constables(
            median_duration=median_duration,
            needs_closure=needs_closure,
            is_named_corridor=bool(
                is_named_corridor
            ),
        )

        barricades = 0

        if needs_closure:
            barricades = (
                4
                if median_duration > 120
                else 2
            )

        elif needs_diversion:
            barricades = 2

        normalized_cause = (
            str(event_cause)
            .replace("_", " ")
            .title()
        )

        corridor_text = (
            "a named traffic corridor"
            if is_named_corridor
            else "a non-corridor location"
        )

        rationale = (
            f"Based on {number_of_incidents} similar historical "
            f"incidents for {normalized_cause} near this location "
            f"and time, the median disruption was "
            f"{median_duration:.0f} minutes. "
            f"{closure_rate * 100:.0f}% of similar incidents "
            f"required road closure and "
            f"{high_priority_rate * 100:.0f}% were high priority. "
            f"The event is located on {corridor_text}."
        )

        return {
            "n_similar_incidents_used": int(
                number_of_incidents
            ),
            "expected_duration_min": round(
                median_duration,
                1,
            ),
            "pct_similar_required_closure": round(
                closure_rate,
                2,
            ),
            "pct_similar_high_priority": round(
                high_priority_rate,
                2,
            ),
            "recommend_road_closure": bool(
                needs_closure
            ),
            "recommend_diversion": bool(
                needs_diversion
            ),
            "recommend_barricades": int(
                barricades
            ),
            "recommend_traffic_constables": int(
                constables
            ),
            "rationale": rationale,
        }


# =========================================================
# LOCAL TEST
# =========================================================
if __name__ == "__main__":
    recommender = IncidentRecommender()

    example_result = recommender.recommend(
        lat=12.95,
        lon=77.52,
        event_cause="water_logging",
        start_hour=18,
        day_of_week=4,
        is_named_corridor=True,
    )

    print(
        "Example recommendation for a water-logging event:"
    )

    for key, value in example_result.items():
        print(f"{key}: {value}")