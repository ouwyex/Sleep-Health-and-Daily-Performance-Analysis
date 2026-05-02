"""
setup_db.py - Creates the SQLite database and populates input_data table
              with sample records for batch prediction.

Run once before starting the pipeline:
    python setup_db.py
"""

import sqlite3
import numpy as np
import pandas as pd

DB_PATH = "sleep_pipeline.db"

np.random.seed(99)
N = 50  # sample records to populate input_data

def create_tables(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS input_data (
            id                        INTEGER PRIMARY KEY AUTOINCREMENT,
            gender                    TEXT    NOT NULL,
            age                       INTEGER NOT NULL,
            occupation                TEXT    NOT NULL,
            sleep_duration            REAL    NOT NULL,
            quality_of_sleep          INTEGER NOT NULL,
            physical_activity_level   INTEGER NOT NULL,
            stress_level              INTEGER NOT NULL,
            bmi_category              TEXT    NOT NULL,
            heart_rate                INTEGER NOT NULL,
            daily_steps               INTEGER NOT NULL,
            blood_pressure_systolic   INTEGER NOT NULL,
            blood_pressure_diastolic  INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            input_id             INTEGER NOT NULL,
            prediction           TEXT    NOT NULL,
            probability_none     REAL    NOT NULL,
            probability_insomnia REAL    NOT NULL,
            probability_apnea    REAL    NOT NULL,
            prediction_timestamp TEXT    NOT NULL,
            FOREIGN KEY (input_id) REFERENCES input_data(id)
        );
    """)
    conn.commit()
    print("Tables created.")

def populate_input_data(conn):
    occupations = [
        "Software Engineer", "Doctor", "Nurse", "Teacher",
        "Engineer", "Accountant", "Scientist", "Lawyer",
        "Sales Representative", "Salesperson",
    ]
    bmi_cats = ["Normal", "Normal Weight", "Overweight", "Obese"]
    genders  = ["Male", "Female"]

    rows = []
    for _ in range(N):
        stress = int(np.random.randint(3, 10))
        bmi    = np.random.choice(bmi_cats, p=[0.25, 0.25, 0.35, 0.15])
        sys_bp = int(110 + stress * 2 +
                     (10 if bmi == "Obese" else 0) +
                     np.random.randint(-5, 6))
        dia_bp = int(sys_bp * 0.62 + np.random.randint(-3, 4))

        rows.append((
            np.random.choice(genders),
            int(np.random.randint(27, 59)),
            np.random.choice(occupations),
            round(float(np.random.uniform(5.8, 8.5)), 1),
            int(np.random.randint(4, 10)),
            int(np.random.randint(30, 90)),
            stress,
            bmi,
            int(np.random.randint(62, 86)),
            int(np.random.randint(3000, 10000)),
            sys_bp,
            dia_bp,
        ))

    conn.executemany("""
        INSERT INTO input_data (
            gender, age, occupation, sleep_duration, quality_of_sleep,
            physical_activity_level, stress_level, bmi_category,
            heart_rate, daily_steps,
            blood_pressure_systolic, blood_pressure_diastolic
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
    print(f"Inserted {N} sample rows into input_data.")

if __name__ == "__main__":
    with sqlite3.connect(DB_PATH) as conn:
        create_tables(conn)
        # Only populate if table is empty
        count = conn.execute("SELECT COUNT(*) FROM input_data").fetchone()[0]
        if count == 0:
            populate_input_data(conn)
        else:
            print(f"input_data already has {count} rows — skipping population.")
    print(f"\nDatabase ready → {DB_PATH}")
