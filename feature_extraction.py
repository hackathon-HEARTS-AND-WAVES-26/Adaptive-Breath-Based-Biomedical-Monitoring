import os
import pandas as pd
import numpy as np

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\adaptive_biomedical_ml"
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

INPUT_FILE = os.path.join(
    PROCESSED_DIR,
    "enose_combined.csv"
)

OUTPUT_FILE = os.path.join(
    PROCESSED_DIR,
    "enose_sensor_features.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading e-nose dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Dataset shape: {df.shape}")


# ============================================================
# SENSOR GROUPS
# ============================================================

sensors = [
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
    "S7",
    "S8"
]


# ============================================================
# CREATE SENSOR-LEVEL FEATURES
# ============================================================

feature_data = pd.DataFrame(index=df.index)

for sensor in sensors:

    sensor_columns = [
        col for col in df.columns
        if col.startswith(sensor + "_")
    ]

    print(
        f"{sensor}: {len(sensor_columns)} raw measurement columns"
    )

    # Convert to numeric
    sensor_values = df[sensor_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    feature_data[f"{sensor}_mean"] = sensor_values.mean(
        axis=1,
        skipna=True
    )

    feature_data[f"{sensor}_std"] = sensor_values.std(
        axis=1,
        skipna=True
    )

    feature_data[f"{sensor}_min"] = sensor_values.min(
        axis=1,
        skipna=True
    )

    feature_data[f"{sensor}_max"] = sensor_values.max(
        axis=1,
        skipna=True
    )

    feature_data[f"{sensor}_range"] = (
        feature_data[f"{sensor}_max"]
        -
        feature_data[f"{sensor}_min"]
    )


# ============================================================
# ADD GROUP LABEL
# ============================================================

feature_data["group"] = df["group"].values


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

feature_data = feature_data.replace(
    [np.inf, -np.inf],
    np.nan
)

feature_data = feature_data.dropna(
    how="all",
    subset=[
        f"{sensor}_mean"
        for sensor in sensors
    ]
)


# ============================================================
# SAVE
# ============================================================

feature_data.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("FEATURE EXTRACTION COMPLETE")
print("========================================")

print(
    f"\nOutput file:\n{OUTPUT_FILE}"
)

print(
    f"\nOutput shape: {feature_data.shape}"
)

print("\nColumns:")

for column in feature_data.columns:
    print(" ", column)

print("\nGroup distribution:")
print(
    feature_data["group"].value_counts()
)

print("\nPreview:")
print(
    feature_data.head()
)
