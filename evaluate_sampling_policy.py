import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import accuracy_score, classification_report


# =========================================================
# PATHS
# =========================================================

DATA = (
    r"D:\adaptive_biomedical_ml\data\processed"
    r"\bidmc_temporal_features.csv"
)

MODEL = (
    r"D:\adaptive_biomedical_ml\data\processed"
    r"\sampling_policy_model.pkl"
)

OUTPUT = (
    r"D:\adaptive_biomedical_ml\data\processed"
    r"\sampling_policy_results.csv"
)


# =========================================================
# LOAD
# =========================================================

print("Loading dataset...")

df = pd.read_csv(DATA)

model = joblib.load(MODEL)


# =========================================================
# FEATURES
# =========================================================

features = [
    "resp_mean",
    "resp_std",
    "resp_range",
    "ppg_mean",
    "ppg_std",
    "ppg_range",
    "resp_variability",
    "ppg_variability",
    "physiological_variability"
]

X = df[features]

y_true = df["sampling_action"]


# =========================================================
# PREDICT
# =========================================================

print("\nRunning adaptive sampling agent...")

y_pred = model.predict(X)


# =========================================================
# ACCURACY
# =========================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)


print("\n===================================")
print("ADAPTIVE SAMPLING RESULTS")
print("===================================")

print(
    f"\nPolicy accuracy: {accuracy:.4f}"
)


print("\nClassification report:")

print(
    classification_report(
        y_true,
        y_pred
    )
)


# =========================================================
# SAMPLING COST MODEL
# =========================================================
#
# Normalized rates:
#
# LOW    = 1 unit
# MEDIUM = 2 units
# HIGH   = 4 units
#
# These are NORMALIZED costs for simulation.
# They are NOT actual sensor power measurements.
# =========================================================

sampling_cost = {
    "LOW_RATE": 1,
    "MEDIUM_RATE": 2,
    "HIGH_RATE": 4
}


# =========================================================
# FIXED HIGH-RATE SYSTEM
# =========================================================

fixed_cost = len(df) * sampling_cost["HIGH_RATE"]


# =========================================================
# ADAPTIVE SYSTEM
# =========================================================

adaptive_cost = sum(
    sampling_cost[action]
    for action in y_pred
)


# =========================================================
# SAVINGS
# =========================================================

cost_reduction = (
    (fixed_cost - adaptive_cost)
    / fixed_cost
) * 100


average_fixed = (
    fixed_cost / len(df)
)

average_adaptive = (
    adaptive_cost / len(df)
)


print("\nSAMPLING COST")
print("-------------")

print(
    f"Fixed high-rate cost     : "
    f"{fixed_cost}"
)

print(
    f"Adaptive cost            : "
    f"{adaptive_cost}"
)

print(
    f"Fixed cost/sample        : "
    f"{average_fixed:.2f}"
)

print(
    f"Adaptive cost/sample     : "
    f"{average_adaptive:.2f}"
)

print(
    f"Normalized cost reduction: "
    f"{cost_reduction:.2f}%"
)


# =========================================================
# ACTION DISTRIBUTION
# =========================================================

print("\nPREDICTED SAMPLING ACTIONS")
print("--------------------------")

print(
    pd.Series(y_pred)
    .value_counts()
)


# =========================================================
# SAVE RESULTS
# =========================================================

results = df.copy()

results["predicted_action"] = y_pred

results["sampling_cost"] = [
    sampling_cost[action]
    for action in y_pred
]


results.to_csv(
    OUTPUT,
    index=False
)


print("\nResults saved:")
print(OUTPUT)

print("\nDONE.")
