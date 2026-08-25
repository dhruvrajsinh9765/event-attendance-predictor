import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Event Attendance Predictor", page_icon="🎟️")
st.title("🎟️ Event Attendance Predictor")
st.write("Predicts how likely a student is to attend an event they registered for.")

model = joblib.load("model.joblib")

def time_to_hour(t):
    try:
        return int(str(t).split(":")[0])
    except (ValueError, AttributeError):
        return np.nan

with st.form("prediction_form"):
    event_type = st.selectbox("Event type", ["Workshop", "Hackathon", "Social", "Competition", "Talk"])
    registration_days_before = st.number_input("Days before event registered", 0, 60, 5)
    previous_events_registered = st.number_input("Previous events registered", 0, 50, 3)
    previous_events_attended = st.number_input("Previous events attended", 0, 50, 2)
    club_member = st.selectbox("Club member?", ["Yes", "No"])
    event_day = st.selectbox("Event day", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    event_time = st.selectbox("Event time", ["9:00","11:00","14:00","16:00","18:00"])
    travel_distance_km = st.number_input("Travel distance (km)", 0.0, 200.0, 5.0)
    submitted = st.form_submit_button("Predict")

if submitted:
    row = pd.DataFrame([{
        "event_type": event_type, "registration_days_before": registration_days_before,
        "previous_events_registered": previous_events_registered,
        "previous_events_attended": previous_events_attended, "club_member": club_member,
        "event_day": event_day, "event_time": event_time, "travel_distance_km": travel_distance_km,
    }])
    row["event_hour"] = row["event_time"].apply(time_to_hour)
    row["past_attendance_rate"] = np.where(
        row["previous_events_registered"] > 0,
        row["previous_events_attended"] / row["previous_events_registered"], np.nan)
    row["is_weekend"] = row["event_day"].isin(["Saturday", "Sunday"]).astype(int)

    cols = ["registration_days_before","previous_events_registered","previous_events_attended",
            "travel_distance_km","event_hour","past_attendance_rate","event_type","club_member",
            "event_day","is_weekend"]
    proba = model.predict_proba(row[cols])[0, 1]
    st.metric("Attendance probability", f"{proba*100:.1f}%")

    if proba >= 0.5:
        st.success("✅ Likely to attend")
    else:
        st.warning("⚠️ Unlikely to attend")