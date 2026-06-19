
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

IN_PATH = "../data/events_clean.csv"
OUT_PATH = "../data/hotspots.csv"

# ~120m clustering radius in degrees (rough, fine at Bengaluru's latitude)
EPS_DEGREES = 0.0011
MIN_SAMPLES = 4


def build_hotspots(df: pd.DataFrame) -> pd.DataFrame:
    coords = df[["latitude", "longitude"]].to_numpy()
    db = DBSCAN(eps=EPS_DEGREES, min_samples=MIN_SAMPLES).fit(coords)
    df = df.copy()
    df["hotspot_id"] = db.labels_  # -1 = noise / one-off location

    clustered = df[df["hotspot_id"] != -1]

    agg = clustered.groupby("hotspot_id").agg(
        n_incidents=("id", "count"),
        lat=("latitude", "mean"),
        lon=("longitude", "mean"),
        pct_high_priority=("is_high_priority", "mean"),
        pct_road_closure=("requires_road_closure", "mean"),
        median_duration_min=("duration_min", "median"),
        top_corridor=("corridor", lambda s: s.mode().iat[0] if not s.mode().empty else "unknown"),
        top_cause=("event_cause", lambda s: s.mode().iat[0] if not s.mode().empty else "unknown"),
        top_zone=("zone", lambda s: s.mode().iat[0] if not s.mode().empty else "unknown"),
    ).reset_index()

    # simple 0-100 risk score: frequency + severity + closure likelihood
    agg["freq_norm"] = agg["n_incidents"] / agg["n_incidents"].max()
    agg["risk_score"] = (
        0.5 * agg["freq_norm"]
        + 0.3 * agg["pct_high_priority"]
        + 0.2 * agg["pct_road_closure"]
    ) * 100

    return agg.sort_values("risk_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_csv(IN_PATH)
    hotspots = build_hotspots(df)
    hotspots.to_csv(OUT_PATH, index=False)
    print(f"Found {len(hotspots)} hotspots (clusters with >= {MIN_SAMPLES} incidents)")
    print(hotspots.head(10)[["hotspot_id", "n_incidents", "top_corridor", "top_cause", "risk_score"]])
