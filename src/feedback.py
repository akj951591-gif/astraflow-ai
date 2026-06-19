from pathlib import Path
from datetime import datetime
import pandas as pd


class PostEventLogger:
    COLUMNS = [
        "event_id",
        "recorded_at",
        "event_cause",
        "event_type",
        "latitude",
        "longitude",
        "hour",
        "day_of_week",

        "predicted_p10_min",
        "predicted_p50_min",
        "predicted_p90_min",
        "actual_duration_min",

        "duration_error_min",
        "duration_error_percent",

        "recommended_constables",
        "actual_constables",

        "recommended_barricades",
        "actual_barricades",

        "recommended_road_closure",
        "actual_road_closure",

        "recommended_diversion",
        "actual_diversion",

        "average_speed_reduction_percent",
        "public_complaints",
        "response_success_score",
        "notes",
    ]

    def __init__(self, file_path="../data/post_event_feedback.csv"):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.file_path.exists():
            pd.DataFrame(columns=self.COLUMNS).to_csv(
                self.file_path,
                index=False
            )

    def save_review(
        self,
        latest_result,
        actual_duration,
        actual_constables,
        actual_barricades,
        actual_road_closure,
        actual_diversion,
        speed_reduction,
        complaints,
        notes="",
    ):
        duration = latest_result["duration"]
        recommendation = latest_result["result"]

        predicted_duration = float(duration["p50"])
        actual_duration = float(actual_duration)

        error_minutes = abs(actual_duration - predicted_duration)

        error_percent = (
            error_minutes / actual_duration * 100
            if actual_duration > 0
            else 0
        )

        within_prediction_range = (
            float(duration["p10"])
            <= actual_duration
            <= float(duration["p90"])
        )

        duration_score = max(0, 100 - error_percent)

        resource_difference = abs(
            int(actual_constables)
            - int(recommendation["recommend_traffic_constables"])
        )

        resource_score = max(0, 100 - resource_difference * 5)

        barricade_difference = abs(
            int(actual_barricades)
            - int(recommendation["recommend_barricades"])
        )

        barricade_score = max(0, 100 - barricade_difference * 4)

        complaint_score = max(0, 100 - int(complaints) * 5)

        prediction_range_bonus = 10 if within_prediction_range else 0

        success_score = (
            duration_score * 0.40
            + resource_score * 0.20
            + barricade_score * 0.15
            + complaint_score * 0.25
            + prediction_range_bonus
        )

        success_score = min(100, max(0, success_score))

        event_id = datetime.now().strftime("EVT-%Y%m%d-%H%M%S")

        record = {
            "event_id": event_id,
            "recorded_at": datetime.now().isoformat(),

            "event_cause": latest_result["cause"],
            "event_type": latest_result["event_type"],

            "latitude": latest_result["lat"],
            "longitude": latest_result["lon"],

            "hour": latest_result["hour"],
            "day_of_week": latest_result["dow"],

            "predicted_p10_min": round(float(duration["p10"]), 2),
            "predicted_p50_min": round(predicted_duration, 2),
            "predicted_p90_min": round(float(duration["p90"]), 2),

            "actual_duration_min": round(actual_duration, 2),

            "duration_error_min": round(error_minutes, 2),
            "duration_error_percent": round(error_percent, 2),

            "recommended_constables":
                recommendation["recommend_traffic_constables"],

            "actual_constables": int(actual_constables),

            "recommended_barricades":
                recommendation["recommend_barricades"],

            "actual_barricades": int(actual_barricades),

            "recommended_road_closure":
                recommendation["recommend_road_closure"],

            "actual_road_closure": bool(actual_road_closure),

            "recommended_diversion":
                recommendation["recommend_diversion"],

            "actual_diversion": bool(actual_diversion),

            "average_speed_reduction_percent":
                float(speed_reduction),

            "public_complaints": int(complaints),

            "response_success_score": round(success_score, 2),

            "notes": notes,
        }

        existing = self.load_reviews()

        updated = pd.concat(
            [existing, pd.DataFrame([record])],
            ignore_index=True
        )

        updated.to_csv(self.file_path, index=False)

        return record

    def load_reviews(self):
        try:
            return pd.read_csv(self.file_path)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return pd.DataFrame(columns=self.COLUMNS)

    def summary(self):
        reviews = self.load_reviews()

        if reviews.empty:
            return {
                "total_reviews": 0,
                "mean_error_percent": 0,
                "mean_success_score": 0,
                "mean_actual_duration": 0,
            }

        return {
            "total_reviews": len(reviews),

            "mean_error_percent":
                reviews["duration_error_percent"].mean(),

            "mean_success_score":
                reviews["response_success_score"].mean(),

            "mean_actual_duration":
                reviews["actual_duration_min"].mean(),
        }