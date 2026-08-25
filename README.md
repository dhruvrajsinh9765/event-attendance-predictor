# Event Attendance Predictor

Predicts whether a student who registered for a club event will actually attend, using their registration behaviour and event details.

## Problem

Clubs often see a gap between how many students register for an event and how many actually show up. This model estimates a per-student attendance probability so the club can plan capacity or follow up with likely no-shows.

## Dataset

- `data/train.csv` — 508 labeled registrations (target: `attended`, 0/1)
- `data/test.csv` — unlabeled registrations to score

Features: `event_type`, `registration_days_before`, `previous_events_registered`, `previous_events_attended`, `club_member`, `event_day`, `event_time`, `travel_distance_km`.

## Approach

### 1. Cleaning
- Dropped 5 rows with a missing target (`attended`) — can't train on unknown labels.
- Dropped exact duplicate rows.
- Standardized inconsistent categorical casing (`Yes`/`yes`/`YES` → `Yes`) in `club_member`.
- Left as 496 clean, usable training rows.

### 2. Feature engineering
- `event_hour` — extracted the hour from `event_time` (e.g. `"14:00"` → `14`).
- `past_attendance_rate` — `previous_events_attended / previous_events_registered`, guarding against division by zero for first-time registrants. Captures behavioral consistency better than either raw count alone.
- `is_weekend` — flags Saturday/Sunday events.

### 3. Preprocessing pipeline
- Numeric features: median imputation + standard scaling.
- Categorical features: most-frequent imputation + one-hot encoding.
- Built with `sklearn.Pipeline` + `ColumnTransformer` so preprocessing is learned only from the training data (no leakage) and applies identically at inference time.

### 4. Models compared
Two models trained and compared on a held-out 20% validation split (stratified by target):

| Model               | Precision | Recall | F1    | ROC-AUC |
|----------------------|-----------|--------|-------|---------|
| Logistic Regression  | 0.679     | 0.841  | 0.752 | 0.697   |
| **Random Forest**    | **0.667** | **0.889** | **0.762** | **0.718** |

**Random Forest** was selected based on F1-score and ROC-AUC, then refit on the full training set (496 rows) for the final model.

### 5. Inference
The trained pipeline outputs an attendance **probability** via `predict_proba`, matching the required output format:

S1379 → 81.9% Likely
S1273 → 33.7% Unlikely

Full predictions saved to `data/test_predictions.csv`.

## Insights

From Random Forest feature importances:

1. **Registration lead time is the strongest predictor** (17.3% importance) — how early or late a student registers matters more than any other factor. The club could use last-minute registrations as a signal to send a reminder nudge before the event.
2. **Travel distance is the second strongest predictor** (15.0%) — students traveling further are less reliable attendees. Hybrid/virtual attendance options, or more centrally located venues, could help.
3. **Behavioral consistency (`past_attendance_rate`) outranks raw attendance count** — a student's historical follow-through ratio is more predictive than simply how many events they've attended before. The club could flag low-ratio students for targeted engagement regardless of their raw attendance numbers.

## How to run

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python train_model.py                       # trains, evaluates, and saves the model
cd streamlit_app && streamlit run app.py     # launches the demo app
```

## Project structure

.
├── data/
│ ├── train.csv
│ ├── test.csv
│ └── test_predictions.csv
├── streamlit_app/
│ ├── app.py
│ └── model.joblib
├── train_model.py
├── requirements.txt
├── .gitignore
└── README.md