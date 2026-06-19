from __future__ import annotations

import math
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, GEOSPHERE, MongoClient
from pymongo.errors import PyMongoError, OperationFailure
from pymongo.server_api import ServerApi

try:
    import streamlit as st
except ImportError:
    st = None


# Load local .env file when running on the computer.
load_dotenv()


# =========================================================
# CONFIGURATION HELPERS
# =========================================================
def get_config(
    name: str,
    default: str | None = None,
) -> str | None:
    """
    Read configuration in this order:

    1. Environment variable / .env
    2. Streamlit Cloud secrets
    3. Default value
    """

    environment_value = os.getenv(name)

    if environment_value:
        return environment_value

    if st is not None:
        try:
            if name in st.secrets:
                return str(st.secrets[name])
        except Exception:
            pass

    return default


def utc_now() -> datetime:
    """
    Return the current timezone-aware UTC datetime.
    """

    return datetime.now(timezone.utc)


def make_id(prefix: str) -> str:
    """
    Generate an application-friendly unique ID.
    """

    random_part = uuid.uuid4().hex[:12].upper()
    return f"{prefix}-{random_part}"


def to_native(value: Any) -> Any:
    """
    Convert NumPy, pandas and nested values into values
    that MongoDB can safely store.
    """

    if value is None:
        return None

    if isinstance(value, dict):
        return {
            str(key): to_native(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            to_native(item)
            for item in value
        ]

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()

    if isinstance(value, datetime):
        return value

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None

    return value


# =========================================================
# MONGODB STORE
class MongoStore:
    def __init__(
        self,
        uri: str | None = None,
        database_name: str | None = None,
    ):
        self.uri = uri or get_config("MONGODB_URI")

        self.database_name = (
            database_name
            or get_config(
                "MONGODB_DATABASE",
                "astraflow_ai",
            )
        )

        if not self.uri:
            raise ValueError(
                "MONGODB_URI is missing. "
                "Add it to .env locally or Streamlit Secrets."
            )

        self.client = MongoClient(
            self.uri,
            server_api=ServerApi("1"),
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000,
            socketTimeoutMS=20000,
            retryWrites=True,
        )

        self.client.admin.command("ping")

        self.db = self.client[
            str(self.database_name)
        ]

        self.forecasts = self.db["forecasts"]
        self.resource_plans = self.db["resource_plans"]
        self.scenario_results = self.db["scenario_results"]
        self.live_monitoring = self.db["live_monitoring"]
        self.post_event_feedback = self.db["post_event_feedback"]

        self.create_indexes()

    def create_indexes(self) -> None:
        """
        Create MongoDB indexes.

        Index names are not manually specified, so existing
        indexes such as forecast_id_1 are reused.
        """

        self.forecasts.create_index(
            [("forecast_id", ASCENDING)],
            unique=True,
        )

        self.forecasts.create_index(
            [("created_at", DESCENDING)]
        )

        self.forecasts.create_index(
            [("event.cause", ASCENDING)]
        )

        self.forecasts.create_index(
            [("event.event_type", ASCENDING)]
        )

        self.forecasts.create_index(
            [("risk.score", DESCENDING)]
        )

        self.forecasts.create_index(
            [("location", GEOSPHERE)]
        )

        self.resource_plans.create_index(
            [("resource_plan_id", ASCENDING)],
            unique=True,
        )

        self.resource_plans.create_index(
            [("forecast_id", ASCENDING)],
            unique=True,
        )

        self.scenario_results.create_index(
            [("scenario_id", ASCENDING)],
            unique=True,
        )

        self.scenario_results.create_index(
            [("forecast_id", ASCENDING)]
        )

        self.scenario_results.create_index(
            [("created_at", DESCENDING)]
        )

        self.live_monitoring.create_index(
            [("monitoring_id", ASCENDING)],
            unique=True,
        )

        self.live_monitoring.create_index(
            [("forecast_id", ASCENDING)]
        )

        self.live_monitoring.create_index(
            [("created_at", DESCENDING)]
        )

        self.post_event_feedback.create_index(
            [("feedback_id", ASCENDING)],
            unique=True,
        )

        self.post_event_feedback.create_index(
            [("forecast_id", ASCENDING)]
        )

        self.post_event_feedback.create_index(
            [("created_at", DESCENDING)]
        )

    def health_check(self) -> dict:
        try:
            self.client.admin.command("ping")

            return {
                "connected": True,
                "database": self.database_name,
                "message": "MongoDB connected successfully",
            }

        except PyMongoError as error:
            return {
                "connected": False,
                "database": self.database_name,
                "message": str(error),
            }
    # =====================================================
    # FORECASTS
    # =====================================================
    def save_forecast(
        self,
        latest: dict,
    ) -> str:
        """
        Save or update a congestion forecast.
        """

        forecast_id = (
            latest.get("forecast_id")
            or make_id("FRC")
        )

        duration = latest.get(
            "duration",
            {},
        )

        recommendation = latest.get(
            "result",
            {},
        )

        longitude = float(
            latest.get("lon", 0) or 0
        )

        latitude = float(
            latest.get("lat", 0) or 0
        )

        document = {
            "forecast_id": forecast_id,

            "event": {
                "cause": latest.get("cause"),
                "event_type": latest.get(
                    "event_type"
                ),
                "hour": latest.get("hour"),
                "day_of_week": latest.get(
                    "dow"
                ),
                "month": latest.get("month"),
                "corridor": latest.get(
                    "corridor"
                ),
                "zone": latest.get("zone"),
                "vehicle_type": latest.get(
                    "vehicle_type"
                ),
                "time_bucket": latest.get(
                    "time_bucket"
                ),
                "named_corridor": bool(
                    latest.get(
                        "named_corridor",
                        False,
                    )
                ),
                "requires_road_closure": bool(
                    latest.get(
                        "requires_road_closure",
                        False,
                    )
                ),
            },

            # GeoJSON uses [longitude, latitude].
            "location": {
                "type": "Point",
                "coordinates": [
                    longitude,
                    latitude,
                ],
            },

            "prediction": {
                "p10": duration.get("p10"),
                "p50": duration.get("p50"),
                "p90": duration.get("p90"),
                "raw_p10": duration.get(
                    "raw_p10"
                ),
                "raw_p50": duration.get(
                    "raw_p50"
                ),
                "raw_p90": duration.get(
                    "raw_p90"
                ),
                "interval_width": duration.get(
                    "interval_width"
                ),
                "confidence": duration.get(
                    "confidence"
                ),
                "confidence_score": duration.get(
                    "confidence_score"
                ),
                "operationally_capped": bool(
                    duration.get(
                        "operationally_capped",
                        False,
                    )
                ),
                "duration_cap_min": duration.get(
                    "duration_cap_min"
                ),
                "model_version": duration.get(
                    "model_version"
                ),
                "warning": duration.get(
                    "warning"
                ),
            },

            "risk": {
                "score": latest.get(
                    "risk_score"
                ),
                "level": latest.get(
                    "risk_text"
                ),
            },

            "historical_recommendation": (
                recommendation
            ),

            "updated_at": utc_now(),
        }

        document = to_native(document)

        self.forecasts.update_one(
            {
                "forecast_id": forecast_id,
            },
            {
                "$set": document,
                "$setOnInsert": {
                    "created_at": utc_now(),
                },
            },
            upsert=True,
        )

        return forecast_id

    def get_forecast(
        self,
        forecast_id: str,
    ) -> dict | None:
        """
        Return one forecast.
        """

        return self.forecasts.find_one(
            {
                "forecast_id": forecast_id,
            },
            {
                "_id": 0,
            },
        )

    def get_recent_forecasts(
        self,
        limit: int = 20,
    ) -> list[dict]:
        """
        Return recent forecasts.
        """

        safe_limit = max(
            1,
            min(int(limit), 100),
        )

        cursor = (
            self.forecasts
            .find(
                {},
                {"_id": 0},
            )
            .sort(
                "created_at",
                DESCENDING,
            )
            .limit(safe_limit)
        )

        return list(cursor)

    # =====================================================
    # RESOURCE PLANS
    # =====================================================
    def save_resource_plan(
        self,
        forecast_id: str,
        resource_plan: dict,
    ) -> str:
        """
        Save or update the resource deployment plan linked
        to a forecast.
        """

        if not forecast_id:
            raise ValueError(
                "forecast_id is required."
            )

        resource_plan_id = (
            f"RSC-{forecast_id}"
        )

        document = {
            "resource_plan_id": (
                resource_plan_id
            ),
            "forecast_id": forecast_id,
            "plan": to_native(
                resource_plan
            ),
            "updated_at": utc_now(),
        }

        self.resource_plans.update_one(
            {
                "forecast_id": forecast_id,
            },
            {
                "$set": document,
                "$setOnInsert": {
                    "created_at": utc_now(),
                },
            },
            upsert=True,
        )

        return resource_plan_id

    def get_resource_plan(
        self,
        forecast_id: str,
    ) -> dict | None:
        """
        Return the resource plan for a forecast.
        """

        return self.resource_plans.find_one(
            {
                "forecast_id": forecast_id,
            },
            {
                "_id": 0,
            },
        )

    # =====================================================
    # SCENARIO RESULTS
    # =====================================================
    def save_scenario(
        self,
        forecast_id: str,
        input_plan: dict,
        simulation_result: dict,
    ) -> str:
        """
        Save a scenario simulation.
        """

        if not forecast_id:
            raise ValueError(
                "forecast_id is required."
            )

        scenario_id = make_id("SCN")

        document = {
            "scenario_id": scenario_id,
            "forecast_id": forecast_id,
            "input_plan": to_native(
                input_plan
            ),
            "simulation_result": to_native(
                simulation_result
            ),
            "created_at": utc_now(),
        }

        self.scenario_results.insert_one(
            document
        )

        return scenario_id

    def get_scenarios(
        self,
        forecast_id: str,
        limit: int = 20,
    ) -> list[dict]:
        """
        Return scenario simulations linked to a forecast.
        """

        safe_limit = max(
            1,
            min(int(limit), 100),
        )

        cursor = (
            self.scenario_results
            .find(
                {
                    "forecast_id": forecast_id,
                },
                {
                    "_id": 0,
                },
            )
            .sort(
                "created_at",
                DESCENDING,
            )
            .limit(safe_limit)
        )

        return list(cursor)

    # =====================================================
    # LIVE MONITORING
    # =====================================================
    def save_live_monitoring(
        self,
        forecast_id: str | None,
        sensor_input: dict,
        monitoring_result: dict,
    ) -> str:
        """
        Save a live monitoring snapshot.
        """

        monitoring_id = make_id("LIVE")

        document = {
            "monitoring_id": monitoring_id,
            "forecast_id": forecast_id,
            "sensor_input": to_native(
                sensor_input
            ),
            "monitoring_result": to_native(
                monitoring_result
            ),
            "created_at": utc_now(),
        }

        self.live_monitoring.insert_one(
            document
        )

        return monitoring_id

    def get_live_monitoring(
        self,
        forecast_id: str,
        limit: int = 50,
    ) -> list[dict]:
        """
        Return live monitoring records for a forecast.
        """

        safe_limit = max(
            1,
            min(int(limit), 200),
        )

        cursor = (
            self.live_monitoring
            .find(
                {
                    "forecast_id": forecast_id,
                },
                {
                    "_id": 0,
                },
            )
            .sort(
                "created_at",
                DESCENDING,
            )
            .limit(safe_limit)
        )

        return list(cursor)

    # =====================================================
    # POST-EVENT FEEDBACK
    # =====================================================
    def save_post_event_feedback(
        self,
        forecast_id: str,
        feedback: dict,
    ) -> str:
        """
        Save actual post-event outcomes.
        """

        if not forecast_id:
            raise ValueError(
                "forecast_id is required."
            )

        feedback_id = make_id("FDB")

        document = {
            "feedback_id": feedback_id,
            "forecast_id": forecast_id,
            "feedback": to_native(
                feedback
            ),
            "created_at": utc_now(),
        }

        self.post_event_feedback.insert_one(
            document
        )

        return feedback_id

    def get_post_event_feedback(
        self,
        forecast_id: str,
    ) -> list[dict]:
        """
        Return all feedback records for one forecast.
        """

        cursor = (
            self.post_event_feedback
            .find(
                {
                    "forecast_id": forecast_id,
                },
                {
                    "_id": 0,
                },
            )
            .sort(
                "created_at",
                DESCENDING,
            )
        )

        return list(cursor)

    # =====================================================
    # STATISTICS
    # =====================================================
    def get_statistics(self) -> dict:
        """
        Return high-level database statistics.
        """

        total_forecasts = (
            self.forecasts.count_documents({})
        )

        total_resource_plans = (
            self.resource_plans.count_documents({})
        )

        total_scenarios = (
            self.scenario_results.count_documents({})
        )

        total_live_records = (
            self.live_monitoring.count_documents({})
        )

        total_feedback = (
            self.post_event_feedback.count_documents({})
        )

        average_pipeline = [
            {
                "$group": {
                    "_id": None,
                    "average_p10": {
                        "$avg": "$prediction.p10"
                    },
                    "average_p50": {
                        "$avg": "$prediction.p50"
                    },
                    "average_p90": {
                        "$avg": "$prediction.p90"
                    },
                    "average_risk": {
                        "$avg": "$risk.score"
                    },
                }
            }
        ]

        aggregate = list(
            self.forecasts.aggregate(
                average_pipeline
            )
        )

        averages = (
            aggregate[0]
            if aggregate
            else {}
        )

        return {
            "total_forecasts": (
                total_forecasts
            ),
            "total_resource_plans": (
                total_resource_plans
            ),
            "total_scenarios": (
                total_scenarios
            ),
            "total_live_records": (
                total_live_records
            ),
            "total_feedback": (
                total_feedback
            ),
            "average_p10": round(
                averages.get(
                    "average_p10",
                    0,
                ) or 0,
                1,
            ),
            "average_p50": round(
                averages.get(
                    "average_p50",
                    0,
                ) or 0,
                1,
            ),
            "average_p90": round(
                averages.get(
                    "average_p90",
                    0,
                ) or 0,
                1,
            ),
            "average_risk": round(
                averages.get(
                    "average_risk",
                    0,
                ) or 0,
                1,
            ),
        }

    # =====================================================
    # DELETE HELPERS
    # =====================================================
    def delete_forecast_pipeline(
        self,
        forecast_id: str,
    ) -> dict:
        """
        Delete a forecast and all linked operational data.
        """

        if not forecast_id:
            raise ValueError(
                "forecast_id is required."
            )

        deleted_forecast = (
            self.forecasts.delete_one(
                {
                    "forecast_id": forecast_id,
                }
            ).deleted_count
        )

        deleted_resource_plans = (
            self.resource_plans.delete_many(
                {
                    "forecast_id": forecast_id,
                }
            ).deleted_count
        )

        deleted_scenarios = (
            self.scenario_results.delete_many(
                {
                    "forecast_id": forecast_id,
                }
            ).deleted_count
        )

        deleted_live_records = (
            self.live_monitoring.delete_many(
                {
                    "forecast_id": forecast_id,
                }
            ).deleted_count
        )

        deleted_feedback = (
            self.post_event_feedback.delete_many(
                {
                    "forecast_id": forecast_id,
                }
            ).deleted_count
        )

        return {
            "forecast": deleted_forecast,
            "resource_plans": (
                deleted_resource_plans
            ),
            "scenarios": deleted_scenarios,
            "live_records": deleted_live_records,
            "feedback": deleted_feedback,
        }

    # =====================================================
    # CLOSE CONNECTION
    # =====================================================
    def close(self) -> None:
        """
        Close the MongoDB client connection.
        """

        self.client.close()