import pandas as pd
import numpy as np
import os

INPUT = r"D:\adaptive_biomedical_ml\data\processed\enose_sensor_features.csv"
OUTPUT = r"D:\adaptive_biomedical_ml\data\processed\enose_noisy_environment.csv"

df = pd.read_csv(INPUT)

sensors = [f"S{i}" for i in range(1, 9)]

np.random.seed(42)

# ---------------------------------------------------------
# Simulate limited / imperfect sensing
# ---------------------------------------------------------

for sensor in sensors:

    # Sensor reliability for each sample
    reliability = np.random.uniform(0.65, 1.0, len(df))

    for feature in ["mean", "std", "min", "max", "range"]:

        col = f"{sensor}_{feature}"

        values = df[col].values.astype(float)

        # Noise increases when reliability decreases
        scale = np.std(values) * 0.15

        noise = np.random.normal(
            0,
            scale * (1 - reliability + 0.05),
            len(values)
        )

        values_noisy = values + noise

        # Random measurement degradation
        degraded = reliability < 0.75
        values_noisy[degraded] += np.random.normal(
            0,
            scale * 0.5,
            degraded.sum()
        )

        df[f"{col}_noisy"] = values_noisy

    df[f"{sensor}_reliability"] = reliability


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

df.to_csv(OUTPUT, index=False)

print("\nNOISY ENVIRONMENT CREATED")
print("-------------------------")
print("Input :", INPUT)
print("Output:", OUTPUT)
print("Shape :", df.shape)

print("\nExample reliability:")
print(df[[f"S{i}_reliability" for i in range(1, 9)]].head())
