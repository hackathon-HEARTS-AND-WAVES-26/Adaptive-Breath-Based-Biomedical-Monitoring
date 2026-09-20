import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

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
    "sensor_analysis.csv"
)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 50)
print("LOADING E-NOSE FEATURE DATA")
print("=" * 50)

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

# ============================================================
# DEFINE SENSORS
# ============================================================

sensors = [f"S{i}" for i in range(1, 9)]

sensor_features = {}

for sensor in sensors:
    sensor_features[sensor] = [
        f"{sensor}_mean",
        f"{sensor}_std",
        f"{sensor}_min",
        f"{sensor}_max",
        f"{sensor}_range"
    ]

# ============================================================
# ENCODE TARGET
# ============================================================

encoder = LabelEncoder()

y = encoder.fit_transform(df["group"])

print("\nGroups:")
for original, encoded in zip(encoder.classes_, range(len(encoder.classes_))):
    print(f"  {original} -> {encoded}")

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

train_idx, test_idx = train_test_split(
    np.arange(len(df)),
    test_size=0.25,
    random_state=42,
    stratify=y
)

y_train = y[train_idx]
y_test = y[test_idx]

# ============================================================
# BASELINE: ALL SENSORS
# ============================================================

all_features = []

for sensor in sensors:
    all_features.extend(sensor_features[sensor])

X_all = df[all_features].fillna(df[all_features].median())

X_train = X_all.iloc[train_idx]
X_test = X_all.iloc[test_idx]

model_all = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model_all.fit(X_train, y_train)

pred_all = model_all.predict(X_test)

all_accuracy = accuracy_score(y_test, pred_all)

print("\n" + "=" * 50)
print("ALL-SENSOR BASELINE")
print("=" * 50)

print(f"Using sensors: {', '.join(sensors)}")
print(f"Accuracy: {all_accuracy:.4f}")

# ============================================================
# SINGLE SENSOR ANALYSIS
# ============================================================

print("\n" + "=" * 50)
print("SINGLE SENSOR ANALYSIS")
print("=" * 50)

results = []

for sensor in sensors:

    features = sensor_features[sensor]

    X = df[features].copy()

    X = X.fillna(X.median())

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results.append({
        "sensor": sensor,
        "num_sensors": 1,
        "sensors_used": sensor,
        "accuracy": accuracy,
        "accuracy_gain_vs_none": accuracy
    })

    print(f"{sensor}: {accuracy:.4f}")

# ============================================================
# TWO-SENSOR COMBINATIONS
# ============================================================

print("\n" + "=" * 50)
print("TWO-SENSOR COMBINATION ANALYSIS")
print("=" * 50)

for i in range(len(sensors)):
    for j in range(i + 1, len(sensors)):

        s1 = sensors[i]
        s2 = sensors[j]

        features = sensor_features[s1] + sensor_features[s2]

        X = df[features].copy()

        X = X.fillna(X.median())

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        model = RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        results.append({
            "sensor": f"{s1}+{s2}",
            "num_sensors": 2,
            "sensors_used": f"{s1},{s2}",
            "accuracy": accuracy,
            "accuracy_gain_vs_none": accuracy
        })

        print(f"{s1} + {s2}: {accuracy:.4f}")

# ============================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 50)
print("FEATURE IMPORTANCE")
print("=" * 50)

importance_df = pd.DataFrame({
    "feature": all_features,
    "importance": model_all.feature_importances_
})

importance_df["sensor"] = importance_df["feature"].str.extract(
    r"^(S[1-8])"
)

sensor_importance = (
    importance_df
    .groupby("sensor")["importance"]
    .sum()
    .sort_values(ascending=False)
)

print("\nSensor importance:")

for sensor, importance in sensor_importance.items():
    print(f"{sensor}: {importance:.4f}")

# ============================================================
# CORRELATION BETWEEN SENSOR MEANS
# ============================================================

print("\n" + "=" * 50)
print("SENSOR CORRELATION")
print("=" * 50)

mean_columns = [f"{sensor}_mean" for sensor in sensors]

correlation = df[mean_columns].corr()

print("\nCorrelation matrix:")
print(correlation.round(3))

# ============================================================
# SAVE SENSOR IMPORTANCE
# ============================================================

importance_output = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "sensor_importance.csv"
)

sensor_importance.reset_index().rename(
    columns={"importance": "importance"}
).to_csv(
    importance_output,
    index=False
)

# ============================================================
# SAVE COMBINATION RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["num_sensors", "accuracy"],
    ascending=[True, False]
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

# ============================================================
# TOP RESULTS
# ============================================================

print("\n" + "=" * 50)
print("BEST SENSOR COMBINATIONS")
print("=" * 50)

print(
    results_df[
        ["sensors_used", "num_sensors", "accuracy"]
    ].head(15).to_string(index=False)
)

print("\n" + "=" * 50)
print("ANALYSIS COMPLETE")
print("=" * 50)

print(f"\nSaved:")
print(OUTPUT_PATH)
print(importance_output)
