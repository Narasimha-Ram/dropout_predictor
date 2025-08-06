import pandas as pd
import numpy as np
import random

np.random.seed(42)
n_students = 1000

data = {
    "student_id": range(1, n_students + 1),
    "age": np.random.randint(18, 25, n_students),
    "gender": np.random.choice(["Male", "Female"], n_students),
    "attendance_rate": np.random.uniform(50, 100, n_students),
    "avg_assignment_score": np.random.uniform(40, 100, n_students),
    "study_hours_per_week": np.random.uniform(0, 40, n_students),
    "parental_education": np.random.choice(["High School", "Bachelor", "Master", "PhD"], n_students),
    "financial_aid": np.random.choice([0, 1], n_students)
}

dropout_probability = (
    (100 - data["attendance_rate"]) * 0.02 +
    (60 - data["avg_assignment_score"]) * 0.03 +
    (10 - data["study_hours_per_week"]) * 0.04
)
dropout_probability = np.clip(dropout_probability, 0, 1)
data["dropout"] = [1 if random.random() < p else 0 for p in dropout_probability]

df = pd.DataFrame(data)
df.to_csv("data/student_data.csv", index=False)
print("✅ Data generated and saved to data/student_data.csv")
