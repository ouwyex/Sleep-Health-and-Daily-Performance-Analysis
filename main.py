from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import numpy as np
import os

# App
app = FastAPI(
    title="Sleep Disorder Prediction API",
    description=(
        "Predicts sleep disorder (None / Insomnia / Sleep Apnea) "
        "based on lifestyle and health metrics."
    ),
    version="1.0.0",
)

# Load model once at startup
MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found. "
            "Please run train.py first."
        )
    return joblib.load(MODEL_PATH)

artefacts = load_model()
model         = artefacts["model"]
le_gender     = artefacts["le_gender"]
le_occupation = artefacts["le_occupation"]
le_bmi        = artefacts["le_bmi"]
le_target     = artefacts["le_target"]

# Request / Response schemas
class PredictRequest(BaseModel):
    gender: Literal["Male", "Female"] = Field(
        ..., example="Male"
    )
    age: int = Field(..., ge=18, le=80, example=35)
    occupation: str = Field(
        ...,
        example="Software Engineer",
        description=(
            "E.g. Software Engineer, Doctor, Nurse, Teacher, "
            "Sales Representative, Engineer, Accountant, "
            "Scientist, Lawyer, Salesperson"
        ),
    )
    sleep_duration: float = Field(
        ..., ge=1.0, le=12.0, example=7.2,
        description="Hours of sleep per night"
    )
    quality_of_sleep: int = Field(
        ..., ge=1, le=10, example=7,
        description="Subjective quality rating (1–10)"
    )
    physical_activity_level: int = Field(
        ..., ge=0, le=120, example=60,
        description="Minutes of physical activity per day"
    )
    stress_level: int = Field(
        ..., ge=1, le=10, example=5,
        description="Subjective stress rating (1–10)"
    )
    bmi_category: Literal["Normal", "Normal Weight", "Overweight", "Obese"] = Field(
        ..., example="Normal"
    )
    heart_rate: int = Field(
        ..., ge=40, le=130, example=72,
        description="Resting heart rate (bpm)"
    )
    daily_steps: int = Field(
        ..., ge=0, le=30000, example=7000
    )
    blood_pressure_systolic: int = Field(
        ..., ge=80, le=200, example=120,
        description="Systolic blood pressure (mmHg)"
    )
    blood_pressure_diastolic: int = Field(
        ..., ge=50, le=130, example=80,
        description="Diastolic blood pressure (mmHg)"
    )


class PredictResponse(BaseModel):
    prediction: str
    probabilities: dict[str, float]


# Endpoints
@app.get("/", tags=["Health"])
def root():
    """Health-check endpoint."""
    return {"message": "ML API is running"}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """
    Predict the sleep disorder for a given set of lifestyle metrics.

    Returns **prediction** (one of: *None*, *Insomnia*, *Sleep Apnea*)
    and per-class **probabilities**.
    """
    # Encode categorical features
    try:
        gender_enc = le_gender.transform([request.gender])[0]
    except ValueError:
        raise HTTPException(400, f"Unknown gender: {request.gender}")

    try:
        occ_enc = le_occupation.transform([request.occupation])[0]
    except ValueError:
        # Fall back to most-common class (index 0) for unseen occupations
        occ_enc = 0

    try:
        bmi_enc = le_bmi.transform([request.bmi_category])[0]
    except ValueError:
        raise HTTPException(400, f"Unknown BMI category: {request.bmi_category}")

    features = np.array([[
        gender_enc,
        request.age,
        occ_enc,
        request.sleep_duration,
        request.quality_of_sleep,
        request.physical_activity_level,
        request.stress_level,
        bmi_enc,
        request.heart_rate,
        request.daily_steps,
        request.blood_pressure_systolic,
        request.blood_pressure_diastolic,
    ]])

    pred_idx   = model.predict(features)[0]
    pred_proba = model.predict_proba(features)[0]

    prediction = le_target.inverse_transform([pred_idx])[0]
    probabilities = {
        cls: round(float(prob), 4)
        for cls, prob in zip(le_target.classes_, pred_proba)
    }

    return PredictResponse(prediction=prediction, probabilities=probabilities)
