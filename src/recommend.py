
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

DATA_PATH = "../data/events_clean.csv"

# tunable business-rule thresholds -- replace with learned values later
MANPOWER_RULES = [
    # (min_duration, requires_closure, corridor_is_named) -> constables
    (180, True, True, 8),
    (60,  True, True, 5),
    (60,  False, True, 3),
    (0,   False, False, 1),
]


class IncidentRecommender:
    def __init__(self, data_path=DATA_PATH, k=15):
        self.df = pd.read_csv(data_path)
        self.k = k
        self.feat_cols = ["latitude", "longitude", "hour", "dow"]
        # normalize scale so lat/long don't dominate hour/dow in distance calc
        self.scale = self.df[self.feat_cols].std()
        X = (self.df[self.feat_cols] / self.scale).to_numpy()
        self.nn = NearestNeighbors(n_neighbors=min(k, len(self.df))).fit(X)

    def _similar_incidents(self, lat, lon, hour, dow, cause=None):
        query = pd.DataFrame([{"latitude": lat, "longitude": lon, "hour": hour, "dow": dow}])
        Xq = (query[self.feat_cols] / self.scale).to_numpy()
        dist, idx = self.nn.kneighbors(Xq)
        sims = self.df.iloc[idx[0]]
        if cause:
            cause_match = sims[sims["event_cause"] == cause]
            if len(cause_match) >= 5:
                sims = cause_match
        return sims

    def recommend(self, lat, lon, event_cause, start_hour, day_of_week,
                   is_named_corridor=False):
        sims = self._similar_incidents(lat, lon, start_hour, day_of_week, event_cause)

        median_duration = sims["duration_min"].median()
        pct_closure = sims["requires_road_closure"].mean()
        pct_high = sims["is_high_priority"].mean()
        n = len(sims)

        median_duration = 0 if pd.isna(median_duration) else median_duration
        needs_closure = pct_closure >= 0.4
        needs_diversion = needs_closure or median_duration > 60

        # manpower from rule table (first matching rule wins)
        constables = 1
        for min_dur, req_close, named, count in MANPOWER_RULES:
            if (median_duration >= min_dur and needs_closure == req_close
                    and is_named_corridor == named):
                constables = count
                break

        barricades = 0
        if needs_closure:
            barricades = 4 if median_duration > 120 else 2

        return {
            "n_similar_incidents_used": int(n),
            "expected_duration_min": round(float(median_duration), 1),
            "pct_similar_required_closure": round(float(pct_closure), 2),
            "pct_similar_high_priority": round(float(pct_high), 2),
            "recommend_road_closure": bool(needs_closure),
            "recommend_diversion": bool(needs_diversion),
            "recommend_barricades": int(barricades),
            "recommend_traffic_constables": int(constables),
            "rationale": (
                f"Based on {n} similar past incidents ({event_cause}) near this "
                f"location/time: median disruption {median_duration:.0f} min, "
                f"{pct_closure*100:.0f}% needed road closure."
            ),
        }


if __name__ == "__main__":
    rec = IncidentRecommender()

    # Example: a water-logging event reported on Mysore Road area at 6pm on a Friday
    result = rec.recommend(
        lat=12.95, lon=77.52, event_cause="water_logging",
        start_hour=18, day_of_week=4, is_named_corridor=True
    )
    print("Example recommendation for a new water-logging report:")
    for k, v in result.items():
        print(f"  {k}: {v}")
