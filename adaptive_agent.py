import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\adaptive_biomedical_ml"

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "enose_sensor_features.csv"
)

# ============================================================
# SETTINGS
# ============================================================

SENSORS = [f"S{i}" for i in range(1, 9)]

FEATURES = [
    "mean",
    "std",
    "min",
    "max",
    "range"
]

# Sensors have equal cost for this first experiment.
# We can replace these with actual hardware energy costs later.
SENSOR_COST = {
    "S1": 1,
    "S2": 1,
    "S3": 1,
    "S4": 1,
    "S5": 1,
    "S6": 1,
    "S7": 1,
    "S8": 1
}

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("ADAPTIVE SENSOR AGENT")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

# ============================================================
# ENCODE TARGET
# ============================================================

encoder = LabelEncoder()

y = encoder.fit_transform(df["group"])

print("\nClasses:")

for i, name in enumerate(encoder.classes_):
    print(f"{i}: {name}")

# ============================================================
# CREATE SENSOR FEATURE GROUPS
# ============================================================

sensor_features = {}

for sensor in SENSORS:

    sensor_features[sensor] = [
        f"{sensor}_{feature}"
        for feature in FEATURES
    ]

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

indices = np.arange(len(df))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# ============================================================
# TRAIN MODELS FOR SENSOR STATES
# ============================================================

print("\nTraining sensor-state models...")

models = {}

# ------------------------------------------------------------
# Single-sensor models
# ------------------------------------------------------------

for sensor in SENSORS:

    features = sensor_features[sensor]

    X = df[features].copy()

    X = X.fillna(X.median())

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X.iloc[train_idx],
        y[train_idx]
    )

    models[(sensor,)] = model


# ------------------------------------------------------------
# Two-sensor models
# ------------------------------------------------------------

for i in range(len(SENSORS)):

    for j in range(i + 1, len(SENSORS)):

        s1 = SENSORS[i]
        s2 = SENSORS[j]

        features = (
            sensor_features[s1]
            + sensor_features[s2]
        )

        X = df[features].copy()

        X = X.fillna(X.median())

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X.iloc[train_idx],
            y[train_idx]
        )

        models[
            tuple(sorted([s1, s2]))
        ] = model


print(
    f"Models available: {len(models)}"
)

# ============================================================
# HELPER
# ============================================================

def predict_state(row_index, active_sensors):

    active_sensors = tuple(
        sorted(active_sensors)
    )

    model = models.get(active_sensors)

    if model is None:
        return None, 0

    features = []

    for sensor in active_sensors:
        features.extend(
            sensor_features[sensor]
        )

    X = df.loc[
        [row_index],
        features
    ].copy()

    X = X.fillna(
        df[features].median()
    )

    probabilities = model.predict_proba(X)[0]

    prediction = int(
        np.argmax(probabilities)
    )

    confidence = float(
        np.max(probabilities)
    )

    return prediction, confidence


# ============================================================
# ADAPTIVE POLICY
# ============================================================

def adaptive_decision(row_index):

    # --------------------------------------------------------
    # Start with S1.
    # In the real device this would represent the
    # low-power initial sensing stage.
    # --------------------------------------------------------

    active = ["S1"]

    prediction, confidence = predict_state(
        row_index,
        active
    )

    # --------------------------------------------------------
    # If confidence is high enough, stop.
    # --------------------------------------------------------

    if confidence >= 0.95:

        return {
            "prediction": prediction,
            "confidence": confidence,
            "sensors_used": active.copy(),
            "sensor_count": len(active),
            "cost": sum(
                SENSOR_COST[s]
                for s in active
            ),
            "action": "STOP"
        }

    # --------------------------------------------------------
    # Otherwise evaluate every possible second sensor.
    # --------------------------------------------------------

    best_sensor = None
    best_confidence = confidence

    for candidate in SENSORS:

        if candidate in active:
            continue

        new_sensors = [
            "S1",
            candidate
        ]

        new_prediction, new_confidence = (
            predict_state(
                row_index,
                new_sensors
            )
        )

        # Select the sensor that gives
        # the highest confidence.

        if new_confidence > best_confidence:

            best_confidence = new_confidence
            best_sensor = candidate
            best_prediction = new_prediction

    # --------------------------------------------------------
    # If another sensor helps, activate it.
    # --------------------------------------------------------

    if best_sensor is not None:

        active.append(best_sensor)

        return {
            "prediction": best_prediction,
            "confidence": best_confidence,
            "sensors_used": active.copy(),
            "sensor_count": len(active),
            "cost": sum(
                SENSOR_COST[s]
                for s in active
            ),
            "action": f"ACTIVATE_{best_sensor}"
        }

    # --------------------------------------------------------
    # Otherwise stop.
    # --------------------------------------------------------

    return {
        "prediction": prediction,
        "confidence": confidence,
        "sensors_used": active.copy(),
        "sensor_count": len(active),
        "cost": sum(
            SENSOR_COST[s]
            for s in active
        ),
        "action": "STOP"
    }


# ============================================================
# RUN AGENT
# ============================================================

print("\n" + "=" * 60)
print("RUNNING ADAPTIVE AGENT")
print("=" * 60)

results = []

for row_index in test_idx:

    result = adaptive_decision(
        row_index
    )

    true_label = y[row_index]

    results.append({

        "sample_index": row_index,

        "true_class": encoder.inverse_transform(
            [true_label]
        )[0],

        "predicted_class": encoder.inverse_transform(
            [result["prediction"]]
        )[0],

        "confidence": result["confidence"],

        "sensor_count": result["sensor_count"],

        "sensors_used": ",".join(
            result["sensors_used"]
        ),

        "cost": result["cost"],

        "action": result["action"],

        "correct": (
            result["prediction"]
            == true_label
        )
    })


results_df = pd.DataFrame(results)

# ============================================================
# RESULTS
# ============================================================

accuracy = results_df["correct"].mean()

average_sensors = (
    results_df["sensor_count"].mean()
)

average_cost = (
    results_df["cost"].mean()
)

print("\n" + "=" * 60)
print("ADAPTIVE AGENT RESULTS")
print("=" * 60)

print(
    f"\nAccuracy: {accuracy:.4f}"
)

print(
    f"Average sensors activated: "
    f"{average_sensors:.2f}"
)

print(
    f"Average sensing cost: "
    f"{average_cost:.2f}"
)

print("\nAction distribution:")

print(
    results_df["action"]
    .value_counts()
)

print("\nSensor usage:")

print(
    results_df["sensors_used"]
    .value_counts()
    .head(15)
)

print("\nSample decisions:")

print(
    results_df.head(20).to_string(
        index=False
    )
)

# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "adaptive_agent_results.csv"
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nResults saved to:")

print(OUTPUT_PATH)

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
