import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# =========================================================
# PATHS
# =========================================================

INPUT = (
    r"D:\adaptive_biomedical_ml\data\processed"
    r"\bidmc_temporal_features.csv"
)

OUTPUT = (
    r"D:\adaptive_biomedical_ml\data\processed"
    r"\sampling_policy_model.pkl"
)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(INPUT)

print("Dataset shape:", df.shape)


# =========================================================
# INPUT FEATURES
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

y = df["sampling_action"]


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# =========================================================
# TRAIN ML POLICY
# =========================================================

print("\nTraining adaptive sampling agent...")


model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


# =========================================================
# TEST
# =========================================================

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)


print("\n================================")
print("SAMPLING POLICY RESULTS")
print("================================")

print(
    "\nAccuracy:",
    round(accuracy, 4)
)


print("\nClassification report:")

print(
    classification_report(
        y_test,
        predictions
    )
)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)


print("\nFEATURE IMPORTANCE")
print("------------------")

print(importance)


# =========================================================
# SAVE MODEL
# =========================================================

joblib.dump(
    model,
    OUTPUT
)

print("\nModel saved:")
print(OUTPUT)


# =========================================================
# SAMPLE DECISIONS
# =========================================================

print("\nSAMPLE AGENT DECISIONS")
print("----------------------")

samples = X_test.head(10)

sample_predictions = model.predict(samples)

for i, prediction in enumerate(sample_predictions):

    print(
        f"Sample {i+1}: "
        f"{prediction}"
    )

print("\nDONE.")
