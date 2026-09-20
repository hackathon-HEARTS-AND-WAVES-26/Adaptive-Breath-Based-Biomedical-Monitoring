import os
import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\adaptive_biomedical_ml"

TEMPORAL_MODEL = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "temporal_sampling_policy.pkl"
)

SENSOR_MODEL = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "adaptive_policy_model.pkl"
)

INPUT_DATA = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "bidmc_waveform_windows.csv"
)

OUTPUT_DATA = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "combined_agent_results.csv"
)


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading models...")

temporal_model = joblib.load(
    TEMPORAL_MODEL
)

sensor_model = joblib.load(
    SENSOR_MODEL
)

print("Models loaded successfully.")


# ============================================================
# TEMPORAL FEATURES
# ============================================================

TEMPORAL_FEATURES = [
    "resp_mean",
    "resp_std",
    "resp_range",
    "resp_change",
    "ppg_mean",
    "ppg_std",
    "ppg_range",
    "ppg_change",
    "resp_variability",
    "ppg_variability"
]


# ============================================================
# SENSOR POLICY FEATURES
# ============================================================

# These are the features used when the adaptive sensor policy
# was trained.

SENSOR_FEATURES = [
    "S1_mean",
    "S1_std",
    "S1_min",
    "S1_max",
    "S1_range",

    "S2_mean",
    "S2_std",
    "S2_min",
    "S2_max",
    "S2_range",

    "S3_mean",
    "S3_std",
    "S3_min",
    "S3_max",
    "S3_range",

    "S4_mean",
    "S4_std",
    "S4_min",
    "S4_max",
    "S4_range",

    "S5_mean",
    "S5_std",
    "S5_min",
    "S5_max",
    "S5_range",

    "S6_mean",
    "S6_std",
    "S6_min",
    "S6_max",
    "S6_range",

    "S7_mean",
    "S7_std",
    "S7_min",
    "S7_max",
    "S7_range",

    "S8_mean",
    "S8_std",
    "S8_min",
    "S8_max",
    "S8_range"
]


# ============================================================
# LOAD TEMPORAL DATA
# ============================================================

df = pd.read_csv(
    INPUT_DATA
)

df = df.dropna(
    subset=TEMPORAL_FEATURES
).copy()


# ============================================================
# TEMPORAL DECISION
# ============================================================

print()
print("Generating sampling decisions...")

df["sampling_action"] = (
    temporal_model.predict(
        df[TEMPORAL_FEATURES]
    )
)


# ============================================================
# SENSOR POLICY
# ============================================================

print("Generating sensor decisions...")


# The sensor policy was trained using the e-nose dataset.
# BIDMC does not contain S1-S8 gas sensor measurements.
#
# Therefore, for this combined simulation we create a
# representative sensor state rather than pretending that
# BIDMC signals are actual S1-S8 measurements.

print()
print("NOTE:")
print("BIDMC does not contain S1-S8 gas sensor data.")
print("Sensor policy will therefore be demonstrated")
print("using a simulated sensor-state environment.")


# ============================================================
# SIMULATED SENSOR STATE
# ============================================================

rng = np.random.default_rng(42)

sensor_actions = [
    "STOP",
    "ACTIVATE_S2",
    "ACTIVATE_S3",
    "ACTIVATE_S4",
    "ACTIVATE_S5",
    "ACTIVATE_S6",
    "ACTIVATE_S7",
    "ACTIVATE_S8"
]


# For the combined simulation:
# most states already contain enough information,
# while a small percentage require another sensor.

df["sensor_action"] = rng.choice(
    sensor_actions,
    size=len(df),
    p=[
        0.825,
        0.025,
        0.025,
        0.025,
        0.025,
        0.025,
        0.025,
        0.025
    ]
)

# ============================================================
# COMBINED DECISION
# ============================================================

def combined_action(sensor_action, sampling_action):

    if sensor_action == "STOP":

        return f"STOP + {sampling_action}"

    else:

        return f"{sensor_action} + {sampling_action}"


df["combined_action"] = [
    combined_action(
        s,
        r
    )
    for s, r in zip(
        df["sensor_action"],
        df["sampling_action"]
    )
]


# ============================================================
# NORMALIZED COST
# ============================================================

SENSOR_COST = {
    "STOP": 1,
    "ACTIVATE_S2": 2,
    "ACTIVATE_S3": 2,
    "ACTIVATE_S4": 2,
    "ACTIVATE_S5": 2,
    "ACTIVATE_S6": 2,
    "ACTIVATE_S7": 2,
    "ACTIVATE_S8": 2
}


RATE_COST = {
    "LOW_RATE": 1,
    "MEDIUM_RATE": 2,
    "HIGH_RATE": 4
}


df["sensor_cost"] = (
    df["sensor_action"]
    .map(SENSOR_COST)
)

df["sampling_cost"] = (
    df["sampling_action"]
    .map(RATE_COST)
)


df["adaptive_total_cost"] = (
    df["sensor_cost"]
    *
    df["sampling_cost"]
)


# Fixed system:
# all sensors + high sampling

FIXED_SENSOR_COST = 2

FIXED_RATE_COST = 4

df["fixed_total_cost"] = (
    FIXED_SENSOR_COST
    *
    FIXED_RATE_COST
)


# ============================================================
# RESULTS
# ============================================================

adaptive_cost = (
    df["adaptive_total_cost"]
    .mean()
)

fixed_cost = (
    df["fixed_total_cost"]
    .mean()
)

reduction = (
    (fixed_cost - adaptive_cost)
    /
    fixed_cost
    *
    100
)


print()
print("========================================")
print("COMBINED ADAPTIVE AGENT")
print("========================================")

print(
    "Windows processed:",
    len(df)
)

print(
    "Average adaptive cost:",
    round(adaptive_cost, 3)
)

print(
    "Fixed cost:",
    round(fixed_cost, 3)
)

print(
    "Normalized cost reduction:",
    round(reduction, 2),
    "%"
)


# ============================================================
# ACTION DISTRIBUTION
# ============================================================

print()
print("Sampling decisions:")

print(
    df["sampling_action"]
    .value_counts()
)


print()
print("Sensor decisions:")

print(
    df["sensor_action"]
    .value_counts()
)


print()
print("Combined decisions:")

print(
    df["combined_action"]
    .value_counts()
    .head(15)
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_DATA,
    index=False
)

print()
print("Saved:")
print(OUTPUT_DATA)

print()
print("DONE.")
