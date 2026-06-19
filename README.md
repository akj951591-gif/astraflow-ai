# Event-Driven Congestion Forecasting and Resource Planning System

AstraFlow AI is an uncertainty-aware traffic operations platform built using the ASTRAM Bengaluru traffic-incident dataset containing **8,173 incidents from November 2023 to April 2024**.

The system forecasts how long an event may disrupt traffic, estimates congestion risk, recommends operational resources, compares alternative response strategies, monitors live conditions, and stores outcomes for continuous learning.

---

## Problem Statement

Political rallies, festivals, sports events, construction activities, VIP movements, accidents, and sudden gatherings can create severe localized traffic congestion.

Traffic authorities currently face three major challenges:

1. Event impact is not quantified in advance.
2. Manpower, barricades, and diversions are often planned using experience.
3. Post-event outcomes are not systematically recorded for future improvement.

AstraFlow AI addresses these challenges through:

```text
Historical Intelligence
        ↓
Impact Forecasting
        ↓
Risk Assessment
        ↓
Resource Recommendation
        ↓
Scenario Simulation
        ↓
Live Monitoring
        ↓
Post-Event Learning
```

---

## What the Dataset Supports

| Requirement                       | Current Status                                                | Explanation                                                                                                                       |
| --------------------------------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Forecast unplanned incidents      | Strong                                                        | Approximately 7,700 historical incidents covering breakdowns, accidents, potholes, waterlogging, congestion, and road conditions  |
| Forecast planned events           | Limited but usable                                            | Only around 191 planned-event examples are available, so predictions should be supplemented with event-permit and attendance data |
| Predict disruption duration       | Implemented                                                   | Quantile models predict optimistic, expected, and severe-case durations                                                           |
| Recommend manpower and barricades | Implemented using an explainable operational engine           | Historical data does not contain actual manpower or barricade deployment labels                                                   |
| Recommend diversion and closure   | Implemented using historical similarity and transparent rules | Recommendations are designed for human approval                                                                                   |
| Scenario comparison               | Implemented                                                   | Multiple response strategies can be simulated before deployment                                                                   |
| Live congestion monitoring        | Implemented as a prototype                                    | Accepts sensor-like inputs such as speed, density, queue length, rainfall, and road blockage                                      |
| Post-event learning               | Implemented                                                   | Forecasts, resources, scenarios, monitoring records, and actual outcomes can be stored in MongoDB                                 |

---

## Important Data Finding

The original `priority` field should not be treated as a meaningful machine-learning target.

In the dataset, priority is almost entirely determined by whether the incident occurred on a named traffic corridor. A classifier can therefore achieve nearly perfect accuracy by reproducing this existing rule.

AstraFlow AI instead uses **incident duration** as the primary predictive target because it provides more useful operational information.

---

## Core Features

## 1. Congestion Hotspot Intelligence

Historical incidents are geographically clustered using DBSCAN.

Each hotspot receives a composite risk score based on:

* Incident frequency
* Historical duration
* High-priority frequency
* Road-closure frequency
* Event cause
* Corridor characteristics

The system identified approximately **373 recurring congestion hotspots**.

The dashboard displays:

* Interactive hotspot map
* Top-risk corridors
* Historical incident count
* Median duration
* Dominant incident cause
* Composite hotspot risk score

---

## 2. Uncertainty-Aware Duration Forecasting

AstraFlow AI uses three quantile regression models:

* **P10:** optimistic recovery duration
* **P50:** expected congestion duration
* **P90:** operational severe-case duration

Example:

```text
P10: 12.1 minutes
P50: 107.0 minutes
P90: 321.0 minutes
```

The application also displays:

* Prediction confidence
* Confidence score
* Uncertainty interval
* Raw quantile output
* Operational severe-case adjustment
* Model version
* Manual-review warning

Duration is log-transformed during training to reduce the influence of extreme values.

The severe-case prediction is constrained to remain operationally useful while preserving the raw model output for diagnostics.

---

## 3. Explainable Historical Recommendation Engine

For each new event, the system searches for similar historical incidents using:

* Event cause
* Geographic proximity
* Hour of day
* Day of week
* Corridor type
* Closure history

The system returns:

* Number of similar incidents used
* Historical median duration
* Percentage requiring road closure
* Percentage classified as high priority
* Plain-language recommendation rationale

This allows an operator to understand why a recommendation was generated.

---

## 4. Resource Deployment Planning

The resource engine converts forecast and risk information into a practical field plan.

It recommends:

* Total traffic officers
* Total barricades
* Officers at major junctions
* Officers at entry and exit points
* Mobile patrol officers
* Control-room and reserve personnel
* Entry barricades
* Exit barricades
* Closure and buffer barricades
* Road closure
* Diversion
* Response time
* Monitoring interval
* Field checklist

Because the source dataset does not contain historical manpower deployment labels, this module uses:

```text
Historical similarity
+ predicted duration
+ event type
+ congestion risk
+ peak-hour conditions
+ corridor status
+ closure/diversion requirement
+ transparent safety rules
```

The original historical recommendation is stored separately from the final operationally adjusted plan.

---

## 5. Scenario Simulation Laboratory

The Scenario Lab compares different traffic-control strategies before field deployment.

Available strategies include:

* Lean Deployment
* Balanced Response
* Recommended Plan
* Aggressive Response
* Emergency Control
* Custom Plan

Each strategy can modify:

* Number of officers
* Number of barricades
* Diversion status
* Road-closure status
* Manual signal control
* Public advisory
* Standby resource percentage

The simulator estimates:

* Duration after intervention
* Congestion minutes saved
* Residual risk
* Resource cost index
* Efficiency score
* Decision score
* Remaining operational gaps

This works as a lightweight traffic-response digital twin.

---

## 6. Live Traffic Monitoring

The live-monitoring module accepts real-time or simulated traffic inputs:

* Current traffic speed
* Normal traffic speed
* Vehicle density
* Queue length
* Rainfall
* Crowd intensity
* Road blockage percentage
* Emergency vehicle requirement

It generates:

* Live congestion threat index
* Speed-reduction percentage
* Estimated additional delay
* Automated alerts
* Review interval
* Immediate operational actions
* Emergency green-corridor recommendation

Future deployments can connect this module to CCTV analytics, GPS feeds, traffic sensors, and weather APIs.

---

## 7. Post-Event Learning

After an event ends, operators can record:

* Actual congestion duration
* Officers actually deployed
* Barricades actually used
* Road closure implemented
* Diversion implemented
* Speed reduction
* Citizen complaints
* Operational notes

The system calculates:

* Duration prediction error
* Error percentage
* Resource difference
* Response success score
* Long-term model performance trend

These records provide the missing labels needed to improve resource recommendations over time.

---

## MongoDB Integration

MongoDB Atlas is used for persistent operational storage.

Collections:

```text
forecasts
resource_plans
scenario_results
live_monitoring
post_event_feedback
```

Relationships are maintained through `forecast_id`.

Example:

```text
Forecast
└── forecast_id: FRC-XXXXXXXXXXXX

Resource Plan
├── resource_plan_id
└── forecast_id

Scenario Result
├── scenario_id
└── forecast_id

Post-Event Feedback
├── feedback_id
└── forecast_id
```

Historical training data remains in CSV format, while new application records are stored in MongoDB.

---

## Machine-Learning Pipeline

## Features

### Categorical

* Event type
* Event cause
* Corridor
* Zone
* Time bucket
* Vehicle type

### Numerical

* Latitude
* Longitude
* Hour
* Day of week
* Month
* Weekend indicator
* Road-closure indicator

## Models

Three `HistGradientBoostingRegressor` models are trained using quantile loss:

```text
Quantile 0.10 → P10
Quantile 0.50 → P50
Quantile 0.90 → P90
```

Additional safeguards include:

* Log-transformed duration target
* Training-duration clipping
* Unknown-category handling
* Numerical-value imputation
* Quantile-order correction
* Operational P90 constraint
* Permutation feature importance
* Prediction-confidence calculation
* Validation interval coverage

---

## Project Structure

```text
astram_project/
│
├── data/
│   ├── raw_events.csv
│   ├── events_clean.csv
│   ├── hotspots.csv
│   └── post_event_feedback.csv
│
├── models/
│   ├── duration_q10.joblib
│   ├── duration_q50.joblib
│   ├── duration_q90.joblib
│   ├── encoders.joblib
│   ├── duration_metadata.joblib
│   ├── duration_validation.csv
│   ├── feature_importance.csv
│   └── priority_classifier.joblib
│
├── src/
│   ├── app.py
│   ├── clean_and_features.py
│   ├── hotspot_clustering.py
│   ├── train_impact_model.py
│   ├── model_predict.py
│   ├── recommend.py
│   ├── scenario_simulator.py
│   ├── live_monitor.py
│   ├── feedback.py
│   ├── mongodb_store.py
│   ├── test_prediction.py
│   └── test_mongodb.py
│
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

## Installation

## 1. Create a virtual environment

```bash
python -m venv venv
```

Windows:

```powershell
venv\Scripts\activate
```

Linux or macOS:

```bash
source venv/bin/activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

Required packages include:

```text
pandas
numpy
scikit-learn
joblib
streamlit
plotly
folium
streamlit-folium
pymongo
python-dotenv
```

---

## MongoDB Configuration

Create a `.env` file in the project root:

```env
MONGODB_URI=mongodb+srv://USERNAME:PASSWORD@CLUSTER.mongodb.net/?retryWrites=true&w=majority
MONGODB_DATABASE=astraflow_ai
```

Add this to `.gitignore`:

```gitignore
.env
__pycache__/
*.pyc
```

Test the connection:

```bash
cd src
python test_mongodb.py
```

---

## Run the Complete Pipeline

From the `src` directory:

```bash
python clean_and_features.py
python hotspot_clustering.py
python train_impact_model.py
python recommend.py
python test_prediction.py
streamlit run app.py
```

The preprocessing and training scripts only need to be rerun when the dataset or model changes.

For normal use:

```bash
cd src
streamlit run app.py
```

---

## Dashboard Modules

The current dashboard includes:

```text
1. Operations Map
2. Incident Planner
3. Resource Deployment
4. Live Monitor
5. Scenario Lab
6. Intelligence Center
7. Post-Event Review
```

---

## Current Limitations

### Planned-event data

The dataset includes relatively few planned events. Reliable planned-event forecasting will require additional information such as:

* Event permits
* Expected attendance
* Venue capacity
* Procession route
* Event duration
* Road-closure plan
* Recurring-event attendance history

### Resource labels

The historical dataset does not include:

* Officers deployed
* Barricades used
* Diversion selected
* Signal plan
* Operational cost

Therefore, resource recommendations are currently explainable and rule-assisted rather than purely supervised machine-learning predictions.

### Live feeds

The live monitor currently accepts manual or simulated sensor values. Real deployment would integrate:

* CCTV vehicle counts
* GPS speed data
* Traffic-signal status
* Weather data
* Crowd-density estimation
* Emergency vehicle feeds

### Forecast uncertainty

Some historical incidents have highly variable durations. The platform displays confidence and uncertainty explicitly instead of hiding this limitation.

---

## Future Development

### Planned-event intelligence

Integrate:

* BBMP and police event permits
* Festival calendars
* Sports schedules
* Political-event notifications
* Historical attendance data

### Graph-based diversion planning

Use the live road network with:

* Dijkstra
* A*
* Dynamic traffic assignment
* Road-capacity constraints

## Automated learning

Create a scheduled retraining pipeline using newly collected post-event records.

## Smart-city integration

Support:

* CCTV analytics
* Traffic sensors
* Signal controllers
* Control-room alerts
* Public traffic advisories
* Emergency green corridors

---

## Responsible AI

AstraFlow AI is a decision-support system.

It does not automatically:

* Close roads
* Issue fines
* Deploy police personnel
* Override traffic signals

All major actions require approval from authorized traffic personnel.

The platform supports responsible use through:

* Explainable recommendations
* Confidence display
* Manual-review warnings
* Raw and adjusted predictions
* Audit records
* Post-event evaluation
* Human approval

---

## Final Value Proposition

Most traffic dashboards only show where congestion already exists.

AstraFlow AI supports the complete operational cycle:

```text
Predict before disruption
Plan resources scientifically
Simulate alternatives safely
Monitor changing conditions
Record actual outcomes
Improve future decisions
```

> AstraFlow AI transforms reactive traffic control into predictive, explainable, and continuously improving urban mobility management.
