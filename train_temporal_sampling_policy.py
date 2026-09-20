import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


BASE_DIR = r"D:\adaptive_biomedical_ml"

INPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "bidmc_waveform_windows.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "temporal_sampling_policy.pkl"
)

RESULT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "temporal_sampling_results.csv"
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_PATH)

print("Dataset:", df.shape)
print("Recordings:", df["recording"].nunique())


# ============================================================
# MODEL INPUT FEATURES
# IMPORTANT:
# physiological_change is NOT included.
# ============================================================

FEATURES = [
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
# CLEAN
# ============================================================

df = df.dropna(
    subset=FEATURES + ["physiological_change"]
).copy()

print("After cleaning:", df.shape)


# ============================================================
# CREATE TARGET
# ============================================================

# physiological_change is used ONLY to create the
# experimental sampling target.
#
# It is NOT given to the Random Forest.

change = df["physiological_change"]

q50 = change.quantile(0.50)
q80 = change.quantile(0.80)

print()
print("Target thresholds:")
print("LOW/MEDIUM threshold :", q50)
print("MEDIUM/HIGH threshold:", q80)


def create_action(value):

    if value <= q50:
        return "LOW_RATE"

    elif value <= q80:
        return "MEDIUM_RATE"

    else:
        return "HIGH_RATE"


df["sampling_action"] = change.apply(
    create_action
)


print()
print("Target distribution:")
print(
    df["sampling_action"].value_counts()
)


# ============================================================
# GROUPED TRAIN/TEST SPLIT
# ============================================================

X = df[FEATURES]

y = df["sampling_action"]

groups = df["recording"]


splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
)


X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]


print()
print("Training recordings:",
      groups.iloc[train_idx].nunique())

print("Testing recordings:",
      groups.iloc[test_idx].nunique())


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

print()
print("Training Random Forest...")


model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(
    X_train,
    y_train
)


# ============================================================
# TEST
# ============================================================

prediction = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    prediction
)


print()
print("========================================")
print("TEMPORAL POLICY RESULTS")
print("========================================")

print(
    f"Test accuracy: {accuracy:.4f}"
)


print()
print("Classification report:")

print(
    classification_report(
        y_test,
        prediction
    )
)


print()
print("Confusion matrix:")

print(
    confusion_matrix(
        y_test,
        prediction,
        labels=[
            "LOW_RATE",
            "MEDIUM_RATE",
            "HIGH_RATE"
        ]
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print()
print("Feature importance:")

print(
    importance.to_string(
        index=False
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_PATH
)

print()
print("Model saved:")
print(MODEL_PATH)


# ============================================================
# COST EVALUATION
# ============================================================

results = df.iloc[test_idx].copy()

results["predicted_action"] = prediction


COST = {
    "LOW_RATE": 1,
    "MEDIUM_RATE": 2,
    "HIGH_RATE": 4
}


results["adaptive_cost"] = (
    results["predicted_action"]
    .map(COST)
)


# Fixed high-rate system
results["fixed_cost"] = 4


adaptive_cost = results["adaptive_cost"].mean()

fixed_cost = results["fixed_cost"].mean()

cost_reduction = (
    (fixed_cost - adaptive_cost)
    / fixed_cost
    * 100
)


print()
print("========================================")
print("ADAPTIVE SAMPLING COST")
print("========================================")

print(
    f"Fixed cost/sample    : {fixed_cost:.3f}"
)

print(
    f"Adaptive cost/sample : {adaptive_cost:.3f}"
)

print(
    f"Cost reduction       : {cost_reduction:.2f}%"
)


print()
print("Predicted actions:")

print(
    results["predicted_action"]
    .value_counts()
)


# ============================================================
# SAVE RESULTS
# ============================================================

results.to_csv(
    RESULT_PATH,
    index=False
)

print()
print("Results saved:")
print(RESULT_PATH)

print()
print("DONE.")