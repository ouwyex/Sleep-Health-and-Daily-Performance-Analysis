"""
train.py - Train a Sleep Disorder classification model
Dataset: Sleep Health and Daily Performance Dataset (Kaggle)
Target: Predict Sleep Disorder (None / Insomnia / Sleep Apnea)

Includes MLflow experiment tracking and model registry.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, accuracy_score,
    f1_score, precision_score, recall_score,
)
import joblib
import mlflow
import mlflow.sklearn

# ── Synthetic dataset that mirrors the real Kaggle CSV ──────────────────────
# Columns: Gender, Age, Occupation, Sleep Duration, Quality of Sleep,
#          Physical Activity Level, Stress Level, BMI Category,
#          Heart Rate, Daily Steps, Blood Pressure, Sleep Disorder

np.random.seed(42)
N = 374

occupations = [
    "Software Engineer", "Doctor", "Sales Representative",
    "Teacher", "Nurse", "Engineer", "Accountant",
    "Scientist", "Lawyer", "Salesperson",
]
bmi_cats = ["Normal", "Normal Weight", "Overweight", "Obese"]
genders   = ["Male", "Female"]

data = {
    "Gender":                  np.random.choice(genders, N),
    "Age":                     np.random.randint(27, 59, N),
    "Occupation":              np.random.choice(occupations, N),
    "Sleep Duration":          np.round(np.random.uniform(5.8, 8.5, N), 1),
    "Quality of Sleep":        np.random.randint(4, 10, N),
    "Physical Activity Level": np.random.randint(30, 90, N),
    "Stress Level":            np.random.randint(3, 9, N),
    "BMI Category":            np.random.choice(bmi_cats, N,
                                   p=[0.25, 0.25, 0.35, 0.15]),
    "Heart Rate":              np.random.randint(62, 86, N),
    "Daily Steps":             np.random.randint(3000, 10000, N),
}

# Derive systolic / diastolic from stress & BMI (realistic pattern)
sys = (110 + data["Stress Level"] * 2 +
       (data["BMI Category"] == "Obese").astype(int) * 10 +
       np.random.randint(-5, 6, N))
dia = (sys * 0.62 + np.random.randint(-3, 4, N)).astype(int)
data["Blood Pressure"] = [f"{s}/{d}" for s, d in zip(sys, dia)]

# Rule-based target (mimics real-world patterns)
disorder = []
for i in range(N):
    if (data["Stress Level"][i] >= 7 and
            data["Sleep Duration"][i] < 6.5 and
            data["Quality of Sleep"][i] <= 5):
        disorder.append("Insomnia")
    elif (data["BMI Category"][i] in ["Overweight", "Obese"] and
          data["Heart Rate"][i] > 75 and
          data["Sleep Duration"][i] < 7.0):
        disorder.append("Sleep Apnea")
    else:
        disorder.append("None")
data["Sleep Disorder"] = disorder

df = pd.DataFrame(data)
print(f"Dataset shape: {df.shape}")
print(f"Sleep Disorder distribution:\n{df['Sleep Disorder'].value_counts()}\n")

# ── Feature engineering ──────────────────────────────────────────────────────
le_gender     = LabelEncoder().fit(df["Gender"])
le_occupation = LabelEncoder().fit(df["Occupation"])
le_bmi        = LabelEncoder().fit(df["BMI Category"])
le_target     = LabelEncoder().fit(df["Sleep Disorder"])

df["Gender_enc"]     = le_gender.transform(df["Gender"])
df["Occupation_enc"] = le_occupation.transform(df["Occupation"])
df["BMI_enc"]        = le_bmi.transform(df["BMI Category"])

# Parse blood pressure into two numeric features
df[["BP_Systolic", "BP_Diastolic"]] = (
    df["Blood Pressure"].str.split("/", expand=True).astype(int)
)

FEATURES = [
    "Gender_enc", "Age", "Occupation_enc",
    "Sleep Duration", "Quality of Sleep",
    "Physical Activity Level", "Stress Level",
    "BMI_enc", "Heart Rate", "Daily Steps",
    "BP_Systolic", "BP_Diastolic",
]

X = df[FEATURES]
y = le_target.transform(df["Sleep Disorder"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Hyperparameters ───────────────────────────────────────────────────────────
PARAMS = {
    "n_estimators": 100,
    "max_depth":    None,
    "random_state": 42,
    "min_samples_split": 2,
}

# ── MLflow: experiment setup ──────────────────────────────────────────────────
mlflow.set_experiment("sleep-disorder-classification")

with mlflow.start_run(run_name="random-forest-v1"):

    # ── Train ──────────────────────────────────────────────────────────────────
    clf = RandomForestClassifier(**PARAMS)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # ── Metrics ────────────────────────────────────────────────────────────────
    acc       = accuracy_score(y_test, y_pred)
    f1_macro  = f1_score(y_test, y_pred, average="macro",     zero_division=0)
    f1_weight = f1_score(y_test, y_pred, average="weighted",  zero_division=0)
    prec      = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec       = recall_score(y_test, y_pred, average="weighted",    zero_division=0)

    print(f"Accuracy : {acc:.4f}")
    print(f"F1 macro : {f1_macro:.4f}")
    print(f"F1 weight: {f1_weight:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=le_target.classes_,
                                 zero_division=0))

    # ── Log params & metrics to MLflow ────────────────────────────────────────
    mlflow.log_params(PARAMS)
    mlflow.log_metrics({
        "accuracy":           acc,
        "f1_macro":           f1_macro,
        "f1_weighted":        f1_weight,
        "precision_weighted": prec,
        "recall_weighted":    rec,
    })

    # ── Save joblib artefact (for FastAPI) ────────────────────────────────────
    artefacts = {
        "model":          clf,
        "le_gender":      le_gender,
        "le_occupation":  le_occupation,
        "le_bmi":         le_bmi,
        "le_target":      le_target,
        "feature_names":  FEATURES,
    }
    joblib.dump(artefacts, "model.joblib")
    mlflow.log_artifact("model.joblib")          # attach file to the run

    # ── Log the sklearn model + register it ───────────────────────────────────
    mlflow.sklearn.log_model(
        sk_model=clf,
        artifact_path="sklearn-model",
        registered_model_name="SleepDisorderClassifier",   # Model Registry
    )

    run_id = mlflow.active_run().info.run_id
    print(f"\nMLflow run id : {run_id}")
    print("Model registered → SleepDisorderClassifier (version 1)")
    print("Model saved     → model.joblib")