import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =========================================================
# SETTINGS
# =========================================================

INPUT = r"D:\adaptive_biomedical_ml\data\processed\enose_noisy_environment.csv"

OUTPUT_MODEL = r"D:\adaptive_biomedical_ml\data\processed\adaptive_policy_model.pkl"

np.random.seed(42)

SENSORS = [f"S{i}" for i in range(1, 9)]

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(INPUT)

print("Dataset:", df.shape)

# Dataset class
y = df["group"]

# =========================================================
# FEATURES FROM INITIAL SENSOR
# =========================================================

# We initially activate S1.
#
# The policy receives:
#   - S1 measurements
#   - S1 reliability
#
# Then it decides:
#
#   0 = STOP
#   1 = ACTIVATE_S2
#   2 = ACTIVATE_S3
#   ...
#   7 = ACTIVATE_S8
#
# =========================================================

initial_features = [
    "S1_mean_noisy",
    "S1_std_noisy",
    "S1_min_noisy",
    "S1_max_noisy",
    "S1_range_noisy",
    "S1_reliability"
]

X_initial = df[initial_features].copy()

# =========================================================
# TRAIN A STATE CLASSIFIER
# =========================================================

state_features = []

for sensor in SENSORS:

    for feature in ["mean", "std", "min", "max", "range"]:
        state_features.append(f"{sensor}_{feature}_noisy")

X_state = df[state_features]

X_train, X_test, y_train, y_test = train_test_split(
    X_state,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

state_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

state_model.fit(X_train, y_train)

print("\nState classifier trained.")

# =========================================================
# GENERATE ACTION LABELS
# =========================================================

print("\nGenerating adaptive action labels...")

actions = []

candidate_sensors = [f"S{i}" for i in range(2, 9)]

# Models using S1 + another sensor
pair_models = {}

for sensor in candidate_sensors:

    features = []

    for s in ["S1", sensor]:
        for feature in ["mean", "std", "min", "max", "range"]:
            features.append(f"{s}_{feature}_noisy")

    model = RandomForestClassifier(
        n_estimators=50,
        random_state=42,
        n_jobs=-1
    )

    model.fit(df[features], y)

    pair_models[sensor] = (model, features)


# S1-only model
s1_features = [
    f"S1_{feature}_noisy"
    for feature in ["mean", "std", "min", "max", "range"]
]

s1_model = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

s1_model.fit(df[s1_features], y)


# =========================================================
# DECIDE BEST ACTION
# =========================================================

for i in range(len(df)):

    row = df.iloc[[i]]

    # Current confidence
    s1_prob = s1_model.predict_proba(
        row[s1_features]
    )[0]

    current_confidence = np.max(s1_prob)

    # If already highly confident, stop.
    if current_confidence >= 0.90:
        actions.append("STOP")
        continue

    best_sensor = None
    best_confidence = current_confidence

    # Check which additional sensor gives
    # the largest confidence improvement.
    for sensor in candidate_sensors:

        model, features = pair_models[sensor]

        prob = model.predict_proba(
            row[features]
        )[0]

        confidence = np.max(prob)

        if confidence > best_confidence:
            best_confidence = confidence
            best_sensor = sensor

    if best_sensor is None:
        actions.append("STOP")
    else:
        actions.append(f"ACTIVATE_{best_sensor}")


df["action"] = actions

# =========================================================
# ACTION DISTRIBUTION
# =========================================================

print("\nACTION DISTRIBUTION")
print("-------------------")

print(df["action"].value_counts())

# =========================================================
# TRAIN POLICY MODEL
# =========================================================

policy_features = [
    "S1_mean_noisy",
    "S1_std_noisy",
    "S1_min_noisy",
    "S1_max_noisy",
    "S1_range_noisy",
    "S1_reliability"
]

X_policy = df[policy_features]
y_policy = df["action"]

X_train, X_test, y_train, y_test = train_test_split(
    X_policy,
    y_policy,
    test_size=0.2,
    random_state=42,
    stratify=y_policy
)

policy_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)

policy_model.fit(X_train, y_train)

pred = policy_model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("\nPOLICY MODEL RESULTS")
print("--------------------")

print("Accuracy:", round(accuracy, 4))

print("\nClassification report:")
print(classification_report(y_test, pred))

# =========================================================
# SAVE MODEL
# =========================================================

import joblib

joblib.dump(
    policy_model,
    OUTPUT_MODEL
)

print("\nPolicy model saved:")
print(OUTPUT_MODEL)

# =========================================================
# SAMPLE DECISIONS
# =========================================================

print("\nSAMPLE ADAPTIVE DECISIONS")
print("-------------------------")

samples = X_test.iloc[:10]

for idx, sample in samples.iterrows():

    action = policy_model.predict(
        sample.to_frame().T
    )[0]

    confidence = np.max(
        policy_model.predict_proba(
            sample.to_frame().T
        )[0]
    )

    print(
        f"Sample {idx}: "
        f"{action} | "
        f"policy confidence = {confidence:.3f}"
    )
