import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "adaptive_labels.csv"
)

# ============================================================
# SETTINGS
# ============================================================

SENSORS = [f"S{i}" for i in range(1, 9)]

FEATURES_PER_SENSOR = [
    "mean",
    "std",
    "min",
    "max",
    "range"
]

# Cost of activating each sensor.
# For now we use equal cost.
# Later these can be replaced by real hardware energy costs.
SENSOR_COST = {
    "S1": 1.0,
    "S2": 1.0,
    "S3": 1.0,
    "S4": 1.0,
    "S5": 1.0,
    "S6": 1.0,
    "S7": 1.0,
    "S8": 1.0
}

# Minimum confidence considered sufficient to stop.
STOP_CONFIDENCE = 0.90

# Minimum improvement required to justify another sensor.
MIN_INFORMATION_GAIN = 0.02

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

# ============================================================
# ENCODE TARGET
# ============================================================

encoder = LabelEncoder()

y = encoder.fit_transform(df["group"])

print("\nTarget classes:")

for i, cls in enumerate(encoder.classes_):
    print(f"{i}: {cls}")

# ============================================================
# CREATE SENSOR FEATURE LIST
# ============================================================

sensor_features = {}

for sensor in SENSORS:

    sensor_features[sensor] = [
        f"{sensor}_{feature}"
        for feature in FEATURES_PER_SENSOR
    ]

# ============================================================
# TRAIN A MODEL FOR EVERY POSSIBLE SENSOR COMBINATION
# ============================================================

print("\n" + "=" * 60)
print("TRAINING SENSOR-COMBINATION MODELS")
print("=" * 60)

# We need models representing different sensing states.
#
# Example:
#
# Model S1
# Model S1+S2
# Model S1+S3
# Model S1+S4
# ...
#
# These models allow us to ask:
#
# "If I activate another sensor, how much does my
# confidence improve?"

combination_models = {}

# ------------------------------------------------------------
# Split once so all models use the same samples
# ------------------------------------------------------------

indices = np.arange(len(df))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# ============================================================
# TRAIN ALL SINGLE-SENSOR MODELS
# ============================================================

for sensor in SENSORS:

    features = sensor_features[sensor]

    X = df[features].copy()
    X = X.fillna(X.median())

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X.iloc[train_idx],
        y[train_idx]
    )

    combination_models[(sensor,)] = model

# ============================================================
# TRAIN ALL TWO-SENSOR MODELS
# ============================================================

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
            n_estimators=150,
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X.iloc[train_idx],
            y[train_idx]
        )

        combination_models[
            tuple(sorted([s1, s2]))
        ] = model

# ============================================================
# TRAIN THREE-SENSOR MODELS
# ============================================================

# These are useful when the agent needs to evaluate
# a third sensor after already activating two.

from itertools import combinations

for combo in combinations(SENSORS, 3):

    features = []

    for sensor in combo:
        features.extend(sensor_features[sensor])

    X = df[features].copy()
    X = X.fillna(X.median())

    model = RandomForestClassifier(
        n_estimators=120,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X.iloc[train_idx],
        y[train_idx]
    )

    combination_models[
        tuple(sorted(combo))
    ] = model

print(
    f"\nModels trained: {len(combination_models)}"
)

# ============================================================
# HELPER FUNCTION
# ============================================================

def get_probability(row_index, active_sensors):

    active_sensors = tuple(sorted(active_sensors))

    if len(active_sensors) == 0:
        return None

    model = combination_models.get(active_sensors)

    if model is None:
        return None

    features = []

    for sensor in active_sensors:
        features.extend(sensor_features[sensor])

    X = df.loc[[row_index], features].copy()

    X = X.fillna(
        df[features].median()
    )

    probabilities = model.predict_proba(X)[0]

    return probabilities


# ============================================================
# GENERATE ADAPTIVE LABELS
# ============================================================

print("\n" + "=" * 60)
print("GENERATING ADAPTIVE SENSOR LABELS")
print("=" * 60)

adaptive_rows = []

# ------------------------------------------------------------
# We start with S1 as the initial low-cost sensor.
#
# Later, in the real hardware implementation, this can
# represent the always-on / wake-up sensor.
# ------------------------------------------------------------

INITIAL_SENSOR = "S1"

for row_index in range(len(df)):

    active = [INITIAL_SENSOR]

    initial_prob = get_probability(
        row_index,
        active
    )

    if initial_prob is None:
        continue

    current_confidence = float(
        np.max(initial_prob)
    )

    best_action = "STOP"
    best_gain = 0.0

    # --------------------------------------------------------
    # If confidence is already high:
    # STOP
    # --------------------------------------------------------

    if current_confidence >= STOP_CONFIDENCE:

        best_action = "STOP"

    else:

        # ----------------------------------------------------
        # Try every inactive sensor
        # ----------------------------------------------------

        for candidate in SENSORS:

            if candidate in active:
                continue

            new_active = sorted(
                active + [candidate]
            )

            new_prob = get_probability(
                row_index,
                new_active
            )

            if new_prob is None:
                continue

            new_confidence = float(
                np.max(new_prob)
            )

            confidence_gain = (
                new_confidence
                - current_confidence
            )

            # Information gain minus sensor cost.
            #
            # Cost is scaled here so the model doesn't
            # automatically activate every available sensor.

            score = (
                confidence_gain
                - 0.01 * SENSOR_COST[candidate]
            )

            if score > best_gain:
                best_gain = score
                best_action = candidate

        # ----------------------------------------------------
        # If improvement isn't meaningful, STOP
        # ----------------------------------------------------

        if best_gain < MIN_INFORMATION_GAIN:

            best_action = "STOP"

    adaptive_rows.append({

        "sample_index": row_index,

        "group": df.loc[
            row_index,
            "group"
        ],

        "initial_sensor": INITIAL_SENSOR,

        "initial_confidence": current_confidence,

        "next_sensor": best_action,

        "information_gain": best_gain,

        "num_active_sensors": 1

    })

# ============================================================
# CREATE OUTPUT DATAFRAME
# ============================================================

adaptive_df = pd.DataFrame(
    adaptive_rows
)

# ============================================================
# SAVE
# ============================================================

adaptive_df.to_csv(
    OUTPUT_PATH,
    index=False
)

# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 60)
print("ADAPTIVE LABEL GENERATION COMPLETE")
print("=" * 60)

print(
    f"\nOutput shape: {adaptive_df.shape}"
)

print("\nAction distribution:")

print(
    adaptive_df[
        "next_sensor"
    ].value_counts()
)

print("\nAverage initial confidence:")

print(
    adaptive_df[
        "initial_confidence"
    ].mean()
)

print("\nAverage information gain:")

print(
    adaptive_df[
        "information_gain"
    ].mean()
)

print("\nPreview:")

print(
    adaptive_df.head(20).to_string(
        index=False
    )
)

print("\nSaved to:")

print(OUTPUT_PATH)
