import pandas as pd
import numpy as np
import os

# =========================================================
# PATHS
# =========================================================

INPUT = r"D:\adaptive_biomedical_ml\data\processed\bidmc_features.csv"

OUTPUT = (
    r"D:\adaptive_biomedical_ml\data\processed"
    r"\bidmc_temporal_features.csv"
)

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(INPUT)

print("Input shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# =========================================================
# CREATE TEMPORAL-LIKE FEATURES
# =========================================================
#
# IMPORTANT:
# Our current bidmc_features.csv contains summary features
# for each recording rather than the complete waveform.
#
# Therefore we create a first temporal-adaptation proxy
# using variability and signal-quality indicators.
#
# Later we can use the raw waveform for a stronger version.
# =========================================================

np.random.seed(42)

temporal = pd.DataFrame()

temporal["resp_mean"] = df["resp_mean"]
temporal["resp_std"] = df["resp_std"]
temporal["resp_range"] = df["resp_range"]

temporal["ppg_mean"] = df["ppg_mean"]
temporal["ppg_std"] = df["ppg_std"]
temporal["ppg_range"] = (
    df["ppg_max"] - df["ppg_min"]
)

# =========================================================
# DERIVED FEATURES
# =========================================================

# Respiratory variability
temporal["resp_variability"] = (
    temporal["resp_std"]
    / (temporal["resp_mean"].abs() + 1e-6)
)

# PPG variability
temporal["ppg_variability"] = (
    temporal["ppg_std"]
    / (temporal["ppg_mean"].abs() + 1e-6)
)

# Combined physiological variability
temporal["physiological_variability"] = (
    temporal["resp_variability"]
    +
    temporal["ppg_variability"]
) / 2


# =========================================================
# SIMULATE MONITORING STATES
# =========================================================
#
# 0 = STABLE
# 1 = CHANGING
# 2 = HIGH_VARIABILITY
#
# These are experimental states for adaptive sampling,
# NOT clinical diagnoses.
# =========================================================

q1 = temporal["physiological_variability"].quantile(0.50)
q2 = temporal["physiological_variability"].quantile(0.80)

def classify_state(value):

    if value < q1:
        return "STABLE"

    elif value < q2:
        return "CHANGING"

    else:
        return "HIGH_VARIABILITY"


temporal["monitoring_state"] = (
    temporal["physiological_variability"]
    .apply(classify_state)
)

# =========================================================
# ASSIGN TARGET SAMPLING POLICY
# =========================================================
#
# Stable           -> LOW
# Changing         -> MEDIUM
# High variability -> HIGH
#
# This is our first supervised target.
# =========================================================

sampling_policy = {
    "STABLE": "LOW_RATE",
    "CHANGING": "MEDIUM_RATE",
    "HIGH_VARIABILITY": "HIGH_RATE"
}

temporal["sampling_action"] = (
    temporal["monitoring_state"]
    .map(sampling_policy)
)

# =========================================================
# SAVE
# =========================================================

os.makedirs(
    os.path.dirname(OUTPUT),
    exist_ok=True
)

temporal.to_csv(
    OUTPUT,
    index=False
)

print("\n================================")
print("TEMPORAL DATASET CREATED")
print("================================")

print("Output:", OUTPUT)
print("Shape :", temporal.shape)

print("\nMonitoring states:")
print(
    temporal["monitoring_state"]
    .value_counts()
)

print("\nSampling actions:")
print(
    temporal["sampling_action"]
    .value_counts()
)

print("\nPreview:")
print(
    temporal.head(10)
)
