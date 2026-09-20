import os
import joblib
import pandas as pd

BASE = r"D:\adaptive_biomedical_ml"

sensor_model_path = os.path.join(
    BASE, "data", "processed", "adaptive_policy_model.pkl"
)

temporal_model_path = os.path.join(
    BASE, "data", "processed", "temporal_sampling_policy.pkl"
)

# Load models
sensor_model = joblib.load(sensor_model_path)
temporal_model = joblib.load(temporal_model_path)

print("\n=== MODELS LOADED ===")
print("Sensor model:", type(sensor_model))
print("Temporal model:", type(temporal_model))

# --------------------------------------------------
# SENSOR MODEL TEST
# --------------------------------------------------

sensor_features = [
    "S1_mean_noisy",
    "S1_std_noisy",
    "S1_min_noisy",
    "S1_max_noisy",
    "S1_range_noisy",
    "S1_reliability"
]

# Example input
sensor_data = pd.DataFrame([{
    "S1_mean_noisy": 0.5,
    "S1_std_noisy": 0.05,
    "S1_min_noisy": 0.4,
    "S1_max_noisy": 0.6,
    "S1_range_noisy": 0.2,
    "S1_reliability": 0.95
}])

sensor_prediction = sensor_model.predict(sensor_data)[0]

print("\n=== SENSOR AGENT ===")
print("Input:")
print(sensor_data)

print("\nML decision:")
print(sensor_prediction)

if hasattr(sensor_model, "predict_proba"):
    probabilities = sensor_model.predict_proba(sensor_data)[0]

    print("\nConfidence:")
    for cls, prob in zip(sensor_model.classes_, probabilities):
        print(f"{cls}: {prob:.4f}")


# --------------------------------------------------
# TEMPORAL MODEL TEST
# --------------------------------------------------

temporal_features = [
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

temporal_data = pd.DataFrame([{
    "resp_mean": 0.5,
    "resp_std": 0.05,
    "resp_range": 0.2,
    "resp_change": 0.01,
    "ppg_mean": 0.5,
    "ppg_std": 0.05,
    "ppg_range": 0.2,
    "ppg_change": 0.01,
    "resp_variability": 0.7,
    "ppg_variability": 0.14
}])

temporal_prediction = temporal_model.predict(temporal_data)[0]

print("\n=== TEMPORAL AGENT ===")
print("ML decision:")
print(temporal_prediction)

if hasattr(temporal_model, "predict_proba"):
    probabilities = temporal_model.predict_proba(temporal_data)[0]

    print("\nConfidence:")
    for cls, prob in zip(temporal_model.classes_, probabilities):
        print(f"{cls}: {prob:.4f}")


print("\n================================")
print("ADAPTIVE ML TEST COMPLETE")
print("================================")