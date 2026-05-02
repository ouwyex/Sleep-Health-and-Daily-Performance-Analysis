"""
batch_predict.py - Batch Prediction Pipeline

Reads unpredicted rows from input_data, runs the ML model,
and writes results into the predictions table.

Can be run manually:
    python batch_predict.py

Or scheduled automatically via scheduler.py (APScheduler).
"""

import sqlite3
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

DB_PATH    = "sleep_pipeline.db"
MODEL_PATH = "model.joblib"


def load_model():
    artefacts    = joblib.load(MODEL_PATH)
    model        = artefacts["model"]
    le_gender    = artefacts["le_gender"]
    le_occ       = artefacts["le_occupation"]
    le_bmi       = artefacts["le_bmi"]
    le_target    = artefacts["le_target"]
    return model, le_gender, le_occ, le_bmi, le_target

def fetch_unpredicted(conn):
    query = """
        SELECT i.*
        FROM   input_data i
        LEFT JOIN predictions p ON p.input_id = i.id
        WHERE  p.id IS NULL
    """
    rows = conn.execute(query).fetchall()
    cols = [d[0] for d in conn.execute(query).description
            ] if rows else []

    # Re-fetch with column names
    cursor = conn.execute(query)
    cols   = [d[0] for d in cursor.description]
    rows   = cursor.fetchall()
    return cols, rows


def run_batch():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}] Starting batch prediction...")

    model, le_gender, le_occ, le_bmi, le_target = load_model()

    with sqlite3.connect(DB_PATH) as conn:
        cols, rows = fetch_unpredicted(conn)

        if not rows:
            print("  No new rows to predict. Skipping.")
            return

        print(f"  Found {len(rows)} unpredicted rows.")

        results = []
        for row in rows:
            record = dict(zip(cols, row))

            # Encode categorical features (fallback to 0 for unseen values)
            try:
                g_enc = int(le_gender.transform([record["gender"]])[0])
            except ValueError:
                g_enc = 0

            try:
                o_enc = int(le_occ.transform([record["occupation"]])[0])
            except ValueError:
                o_enc = 0

            try:
                b_enc = int(le_bmi.transform([record["bmi_category"]])[0])
            except ValueError:
                b_enc = 0

            FEATURE_COLS = ["Gender_enc","Age","Occupation_enc","Sleep Duration","Quality of Sleep","Physical Activity Level","Stress Level","BMI_enc","Heart Rate","Daily Steps","BP_Systolic","BP_Diastolic"]
            features = pd.DataFrame([[
                g_enc,
                record["age"],
                o_enc,
                record["sleep_duration"],
                record["quality_of_sleep"],
                record["physical_activity_level"],
                record["stress_level"],
                b_enc,
                record["heart_rate"],
                record["daily_steps"],
                record["blood_pressure_systolic"],
                record["blood_pressure_diastolic"],
            ]], columns=FEATURE_COLS)

            pred_idx   = model.predict(features)[0]
            pred_proba = model.predict_proba(features)[0]
            prediction = le_target.inverse_transform([pred_idx])[0]

            # Map probabilities to class names
            proba_dict = dict(zip(le_target.classes_, pred_proba))

            results.append((
                record["id"],
                prediction,
                round(float(proba_dict.get("None", 0.0)), 4),
                round(float(proba_dict.get("Insomnia", 0.0)), 4),
                round(float(proba_dict.get("Sleep Apnea", 0.0)), 4),
                timestamp,
            ))

        # Write all predictions in one transaction
        conn.executemany("""
            INSERT INTO predictions (
                input_id, prediction,
                probability_none, probability_insomnia, probability_apnea,
                prediction_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, results)
        conn.commit()

        print(f"  Saved {len(results)} predictions to database.")

        # Print summary
        from collections import Counter
        counts = Counter(r[1] for r in results)
        for label, cnt in counts.items():
            print(f"    {label}: {cnt}")

if __name__ == "__main__":
    run_batch()
