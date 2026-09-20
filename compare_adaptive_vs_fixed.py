import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# =========================================================
# PATHS
# =========================================================

DATA = r"D:\adaptive_biomedical_ml\data\processed\enose_noisy_environment.csv"

POLICY = r"D:\adaptive_biomedical_ml\data\processed\adaptive_policy_model.pkl"

OUTPUT = (
    r"D:\adaptive_biomedical_ml\data\processed"
    r"\fixed_vs_adaptive_results.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

print("Loading dataset...")

df = pd.read_csv(DATA)

print("Dataset shape:", df.shape)


# =========================================================
# SENSOR FEATURES
# =========================================================

SENSORS = [f"S{i}" for i in range(1, 9)]

features = {}

for sensor in SENSORS:

    features[sensor] = [
        f"{sensor}_mean_noisy",
        f"{sensor}_std_noisy",
        f"{sensor}_min_noisy",
        f"{sensor}_max_noisy",
        f"{sensor}_range_noisy"
    ]


y = df["group"]


# =========================================================
# LOAD POLICY
# =========================================================

print("\nLoading adaptive policy...")

policy = joblib.load(POLICY)

policy_features = [
    "S1_mean_noisy",
    "S1_std_noisy",
    "S1_min_noisy",
    "S1_max_noisy",
    "S1_range_noisy",
    "S1_reliability"
]


# =========================================================
# TRAIN FIXED SYSTEM
# =========================================================

print("\nTraining fixed sensing model...")

fixed_features = features["S1"] + features["S2"]

fixed_model = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

fixed_model.fit(
    df[fixed_features],
    y
)


# =========================================================
# FIXED SYSTEM PREDICTION
# =========================================================

print("Running fixed system...")

fixed_predictions = fixed_model.predict(
    df[fixed_features]
)

fixed_accuracy = accuracy_score(
    y,
    fixed_predictions
)


# =========================================================
# ADAPTIVE POLICY DECISION
# =========================================================

print("\nRunning adaptive policy...")

# IMPORTANT:
# Predict ALL policy actions at once.
# This is much faster than calling policy.predict()
# 16000 separate times.

policy_predictions = policy.predict(
    df[policy_features]
)

print("Policy decisions generated.")


# =========================================================
# TRAIN MODELS FOR POSSIBLE SECOND SENSORS
# =========================================================

print("\nTraining second-sensor models...")

pair_models = {}

for sensor in SENSORS:

    if sensor == "S1":
        continue

    pair_features = features["S1"] + features[sensor]

    print(f"Training S1 + {sensor}...")

    model = RandomForestClassifier(
        n_estimators=50,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        df[pair_features],
        y
    )

    pair_models[sensor] = model


# =========================================================
# S1-ONLY MODEL
# =========================================================

print("\nTraining S1-only model...")

s1_model = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

s1_model.fit(
    df[features["S1"]],
    y
)


# =========================================================
# GENERATE ADAPTIVE PREDICTIONS
# =========================================================

print("\nGenerating adaptive predictions...")

adaptive_predictions = np.empty(
    len(df),
    dtype=object
)

sensor_counts = np.zeros(
    len(df),
    dtype=int
)


# ---------------------------------------------------------
# STOP samples
# ---------------------------------------------------------

stop_mask = policy_predictions == "STOP"

print(
    "STOP decisions:",
    stop_mask.sum()
)

if stop_mask.sum() > 0:

    adaptive_predictions[stop_mask] = (
        s1_model.predict(
            df.loc[
                stop_mask,
                features["S1"]
            ]
        )
    )

    sensor_counts[stop_mask] = 1


# ---------------------------------------------------------
# ACTIVATE second sensor
# ---------------------------------------------------------

for sensor in SENSORS:

    if sensor == "S1":
        continue

    action = f"ACTIVATE_{sensor}"

    mask = policy_predictions == action

    count = mask.sum()

    if count == 0:
        continue

    print(
        f"{action}: {count} samples"
    )

    pair_features = (
        features["S1"] +
        features[sensor]
    )

    predictions = pair_models[sensor].predict(
        df.loc[
            mask,
            pair_features
        ]
    )

    adaptive_predictions[mask] = predictions

    sensor_counts[mask] = 2


# =========================================================
# ADAPTIVE ACCURACY
# =========================================================

adaptive_accuracy = accuracy_score(
    y,
    adaptive_predictions
)

average_sensors = np.mean(
    sensor_counts
)


# =========================================================
# SENSOR REDUCTION
# =========================================================

fixed_sensor_count = 2

sensor_reduction = (
    (fixed_sensor_count - average_sensors)
    / fixed_sensor_count
) * 100


# =========================================================
# NORMALIZED ENERGY COST
# =========================================================

fixed_cost = len(df) * fixed_sensor_count

adaptive_cost = sensor_counts.sum()

energy_reduction = (
    (fixed_cost - adaptive_cost)
    / fixed_cost
) * 100


# =========================================================
# RESULTS
# =========================================================

print("\n")
print("==========================================")
print("       FIXED VS ADAPTIVE RESULTS")
print("==========================================")

print(
    f"\nFixed accuracy       : "
    f"{fixed_accuracy:.4f}"
)

print(
    f"Adaptive accuracy    : "
    f"{adaptive_accuracy:.4f}"
)

print(
    f"\nFixed sensors/sample : "
    f"{fixed_sensor_count:.2f}"
)

print(
    f"Adaptive sensors/sample : "
    f"{average_sensors:.2f}"
)

print(
    f"\nSensor reduction     : "
    f"{sensor_reduction:.2f}%"
)

print(
    f"Normalized fixed cost    : "
    f"{fixed_cost}"
)

print(
    f"Normalized adaptive cost : "
    f"{adaptive_cost}"
)

print(
    f"Cost reduction       : "
    f"{energy_reduction:.2f}%"
)


# =========================================================
# ACTION DISTRIBUTION
# =========================================================

print("\n")
print("ADAPTIVE ACTION DISTRIBUTION")
print("----------------------------")

print(
    pd.Series(policy_predictions)
    .value_counts()
)


# =========================================================
# SAVE RESULTS
# =========================================================

results = pd.DataFrame({

    "true_class": y.values,

    "adaptive_prediction":
        adaptive_predictions,

    "action":
        policy_predictions,

    "sensors_used":
        sensor_counts

})

results.to_csv(
    OUTPUT,
    index=False
)

print("\nResults saved to:")

print(OUTPUT)

print("\nDONE.")
