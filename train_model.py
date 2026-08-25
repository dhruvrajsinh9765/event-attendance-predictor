import pandas as pd

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

print(train_df.shape)
print(train_df.head())
print(train_df.info())
print(train_df.isnull().sum())

# Drop rows where the target itself is missing — can't train on unknown labels
train_df = train_df.dropna(subset=["attended"]).reset_index(drop=True)

# Drop exact duplicate rows
train_df = train_df.drop_duplicates().reset_index(drop=True)

# Standardize inconsistent text casing
train_df["club_member"] = train_df["club_member"].astype(str).str.strip().str.capitalize()
test_df["club_member"] = test_df["club_member"].astype(str).str.strip().str.capitalize()

print("After cleaning:", train_df.shape)
print(train_df["club_member"].unique())

import numpy as np

def time_to_hour(t):
    try:
        return int(str(t).split(":")[0])
    except (ValueError, AttributeError):
        return np.nan

for df in (train_df, test_df):
    df["event_hour"] = df["event_time"].apply(time_to_hour)
    df["past_attendance_rate"] = np.where(
        df["previous_events_registered"] > 0,
        df["previous_events_attended"] / df["previous_events_registered"],
        np.nan,
    )
    df["is_weekend"] = df["event_day"].isin(["Saturday", "Sunday"]).astype(int)

print(train_df[["event_time", "event_hour", "previous_events_registered", "previous_events_attended", "past_attendance_rate", "event_day", "is_weekend"]].head())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

numeric_features = [
    "registration_days_before", "previous_events_registered",
    "previous_events_attended", "travel_distance_km",
    "event_hour", "past_attendance_rate",
]
categorical_features = ["event_type", "club_member", "event_day", "is_weekend"]

X = train_df[numeric_features + categorical_features]
y = train_df["attended"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train:", X_train.shape, " Val:", X_val.shape)

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42),
}

results = {}
for name, clf in models.items():
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    proba = pipe.predict_proba(X_val)[:, 1]

    precision = precision_score(y_val, preds, zero_division=0)
    recall = recall_score(y_val, preds, zero_division=0)
    f1 = f1_score(y_val, preds, zero_division=0)
    auc = roc_auc_score(y_val, proba)

    results[name] = {"pipeline": pipe, "f1": f1}

    print(f"\n=== {name} ===")
    print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}  ROC-AUC: {auc:.3f}")
    print(classification_report(y_val, preds, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_val, preds))

best_name = max(results, key=lambda k: results[k]["f1"])
print(f"\nBest model: {best_name}")


import joblib

final_clf = models[best_name]
final_pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", final_clf)])
final_pipe.fit(X, y)  # refit on ALL training data, not just the 80% split

X_test_final = test_df[numeric_features + categorical_features]
test_proba = final_pipe.predict_proba(X_test_final)[:, 1]
test_pred = final_pipe.predict(X_test_final)

output = test_df[["student_id"]].copy()
output["attendance_probability"] = (test_proba * 100).round(1)
output["predicted_attendance"] = np.where(test_pred == 1, "Likely", "Unlikely")
output.to_csv("data/test_predictions.csv", index=False)

print(output.head(10))

joblib.dump(final_pipe, "streamlit_app/model.joblib")
ohe_cols = final_pipe.named_steps["preprocess"].named_transformers_["cat"] \
    .named_steps["onehot"].get_feature_names_out(categorical_features)
all_cols = numeric_features + list(ohe_cols)
importances = final_pipe.named_steps["model"].feature_importances_
imp_df = pd.DataFrame({"feature": all_cols, "importance": importances}) \
    .sort_values("importance", ascending=False)
print(imp_df.head(10))
print("Model saved to streamlit_app/model.joblib")