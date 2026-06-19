
import pandas as pd
import numpy as np

IN_PATH = "../data/astram_events.csv"
OUT_PATH = "../data/events_clean.csv"

KEEP_COLS = [
    "id", "event_type", "latitude", "longitude", "address",
    "event_cause", "requires_road_closure", "start_datetime",
    "closed_datetime", "status", "priority", "corridor", "zone",
    "junction", "veh_type", "description",
]

def load_raw(path=IN_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return df[[c for c in KEEP_COLS if c in df.columns]].copy()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- timestamps ---
    df["start_datetime"] = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True)
    df["closed_datetime"] = pd.to_datetime(df["closed_datetime"], errors="coerce", utc=True)

    # --- drop rows with no location or no start time (can't model these) ---
    df = df.dropna(subset=["latitude", "longitude", "start_datetime"])
    df = df[(df["latitude"] != 0) & (df["longitude"] != 0)]

    # --- duration in minutes, only when closed_datetime exists and is sane ---
    dur = (df["closed_datetime"] - df["start_datetime"]).dt.total_seconds() / 60.0
    # clip impossible values: negative, or longer than 3 days (3*1440 = 4320 min)
    # these are data-entry errors (closed weeks later), not real congestion duration
    dur = dur.where((dur >= 1) & (dur <= 4320))
    df["duration_min"] = dur

    # --- fill categorical NaNs explicitly so the model treats them as their own category ---
    for col in ["event_cause", "priority", "corridor", "zone", "veh_type"]:
        df[col] = df[col].fillna("unknown")

    df["requires_road_closure"] = df["requires_road_closure"].fillna(False).astype(bool)

    # --- time features ---
    df["hour"] = df["start_datetime"].dt.hour
    df["dow"] = df["start_datetime"].dt.dayofweek           # 0=Mon
    df["is_weekend"] = df["dow"].isin([5, 6])
    df["month"] = df["start_datetime"].dt.month

    def time_bucket(h):
        if 7 <= h <= 10:
            return "morning_peak"
        if 17 <= h <= 20:
            return "evening_peak"
        if 23 <= h or h <= 5:
            return "night"
        return "off_peak"
    df["time_bucket"] = df["hour"].apply(time_bucket)

    # --- binary severity label used for the classifier (High vs not-High) ---
    df["is_high_priority"] = (df["priority"] == "High").astype(int)

    return df.reset_index(drop=True)


if __name__ == "__main__":
    raw = load_raw()
    clean_df = clean(raw)
    clean_df.to_csv(OUT_PATH, index=False)
    print(f"Loaded {len(raw)} raw rows -> {len(clean_df)} clean rows")
    print(f"Rows with usable duration: {clean_df['duration_min'].notna().sum()}")
    print(clean_df[["event_cause", "priority", "time_bucket", "duration_min"]].head())
