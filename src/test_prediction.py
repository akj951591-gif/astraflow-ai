from model_predict import DurationPredictor


predictor = DurationPredictor()

result = predictor.predict(
    lat=12.95500,
    lon=77.55200,
    event_cause="protest",
    hour=18,
    dow=0,
    month=6,
    event_type="unplanned",
    corridor="Mysore Road",
    zone="unknown",
    veh_type="unknown",
    requires_road_closure=False,
)

print("\nPredicted result:")

for key, value in result.items():
    print(f"{key}: {value}")