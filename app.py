import streamlit as st
import requests
import json

API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Sleep Disorder Predictor",
    page_icon="",
    layout="centered",
)

st.title("Sleep Disorder Predictor")
st.markdown(
    "Fill in your lifestyle and health metrics below to predict "
    "whether you have **Insomnia**, **Sleep Apnea**, or **no disorder**."
)

# Check API health
try:
    health = requests.get(f"{API_URL}/", timeout=3)
    if health.status_code == 200:
        st.success("API is running")
    else:
        st.warning("API returned an unexpected status.")
except requests.exceptions.ConnectionError:
    st.error(
        "Cannot reach the API at `http://localhost:8000`. "
        "Make sure FastAPI (or Docker) is running first."
    )

st.divider()

# Input form
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 80, 35)
    occupation = st.selectbox("Occupation", [
        "Software Engineer", "Doctor", "Nurse", "Teacher",
        "Engineer", "Accountant", "Scientist", "Lawyer",
        "Sales Representative", "Salesperson",
    ])
    bmi_category = st.selectbox(
        "BMI Category", ["Normal", "Normal Weight", "Overweight", "Obese"]
    )
    sleep_duration = st.slider("Sleep Duration (hours/night)", 4.0, 12.0, 7.0, 0.1)
    quality_of_sleep = st.slider("Quality of Sleep (1–10)", 1, 10, 7)

with col2:
    physical_activity = st.slider("Physical Activity (min/day)", 0, 120, 60)
    stress_level = st.slider("Stress Level (1–10)", 1, 10, 5)
    heart_rate = st.slider("Resting Heart Rate (bpm)", 40, 130, 72)
    daily_steps = st.number_input("Daily Steps", 0, 30000, 7000, step=500)
    bp_systolic = st.slider("Blood Pressure — Systolic (mmHg)", 80, 200, 120)
    bp_diastolic = st.slider("Blood Pressure — Diastolic (mmHg)", 50, 130, 80)

st.divider()

# Predict button
if st.button("Predict Sleep Disorder", use_container_width=True, type="primary"):
    payload = {
        "gender": gender,
        "age": age,
        "occupation": occupation,
        "sleep_duration": sleep_duration,
        "quality_of_sleep": quality_of_sleep,
        "physical_activity_level": physical_activity,
        "stress_level": stress_level,
        "bmi_category": bmi_category,
        "heart_rate": heart_rate,
        "daily_steps": int(daily_steps),
        "blood_pressure_systolic": bp_systolic,
        "blood_pressure_diastolic": bp_diastolic,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        result = response.json()

        prediction = result["prediction"]
        probabilities = result["probabilities"]

        # Result display
        color_map = {
            "None": "green",
            "Insomnia": "orange",
            "Sleep Apnea": "red",
        }
        emoji_map = {
            "None": "✅",
            "Insomnia": "😴",
            "Sleep Apnea": "😮‍💨",
        }

        color = color_map.get(prediction, "gray")
        emoji = emoji_map.get(prediction, "❓")

        st.markdown(
            f"### {emoji} Prediction: "
            f"<span style='color:{color}; font-weight:bold'>{prediction}</span>",
            unsafe_allow_html=True,
        )

        # Probability bars
        st.markdown("**Class Probabilities:**")
        for cls, prob in sorted(probabilities.items(), key=lambda x: -x[1]):
            st.progress(prob, text=f"{cls}: {prob:.1%}")

        # Raw JSON expander
        with st.expander("Raw API response"):
            st.json(result)

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API.")
    except Exception as e:
        st.error(f"Something went wrong: {e}")

# Footer
st.divider()
st.caption(
    "Model: Random Forest · "
    "Dataset: Sleep Health & Daily Performance (Kaggle) · "
    "Tracked with MLflow"
)