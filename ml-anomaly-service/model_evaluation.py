
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load model and scaler
model = joblib.load("anomaly_detection.joblib")
scaler = joblib.load("scaler.joblib")

# Load dataset
df = pd.read_csv("resources_utilization_dataset.csv")

FEATURES = [
    "cpu_utilization",
    "memory_usage",
    "disk_io",
    "network_latency",
    "process_count",
    "thread_count",
    "context_switches",
    "cache_miss_rate",
    "temperature",
    "power_consumption",
    "uptime"
]

# Ground-truth labels
y_true = df["status"].astype(int).to_numpy()

# Prepare input
X = df[FEATURES]
X_scaled = scaler.transform(X)

# Measure inference time
start = time.perf_counter()

predictions = model.predict(X_scaled)

elapsed = time.perf_counter() - start

# Convert Isolation Forest labels:
# -1 = anomaly, 1 = normal
y_pred = np.where(predictions == -1, 1, 0)

# Metrics
total_records = len(df)
actual_anomalies = int(y_true.sum())
predicted_anomalies = int(y_pred.sum())

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("=" * 50)
print("SYSTEM LOGS ANOMALY DETECTION EVALUATION")
print("=" * 50)

print(f"Total records: {total_records}")
print(f"Actual anomalies: {actual_anomalies}")
print(f"Predicted anomalies: {predicted_anomalies}")

print(f"\nTotal inference time: {elapsed:.6f} seconds")
print(f"Average inference time: {(elapsed / total_records) * 1000:.6f} ms")
print(f"Throughput: {total_records / elapsed:.2f} records/sec")

print("\nClassification Metrics")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Normal", "Anomaly"],
        zero_division=0
    )
)