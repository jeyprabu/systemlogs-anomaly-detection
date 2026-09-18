
import os
import time
import gc
import csv
import platform
import threading
import tracemalloc

import joblib
import numpy as np
import pandas as pd
import psutil


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "anomaly_detection.joblib"
SCALER_PATH = "scaler.joblib"
DATASET_PATH = "resources_utilization_dataset.csv"

BATCH_SIZES = [100, 500, 1000, 5000, 10000]

# More repetitions improve timing stability.
# Three is a reasonable starting point for a basic laptop.
REPETITIONS = 3

# Small pause between tests to reduce interference.
PAUSE_SECONDS = 1


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


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_rss_mb(process):
    """Return resident memory usage in MB."""
    return process.memory_info().rss / (1024 * 1024)


def get_cpu_time(process):
    """Return total user + system CPU time in seconds."""
    cpu_times = process.cpu_times()
    return cpu_times.user + cpu_times.system


def measure_cpu_usage(process, wall_time, cpu_time):
    """
    Approximate process CPU utilization.

    A value of 100 means one logical CPU was fully utilized
    for the duration of the measurement.

    Values above 100 can occur when multiple threads execute
    concurrently.
    """
    if wall_time <= 0:
        return 0.0

    logical_cpus = psutil.cpu_count(logical=True) or 1

    return (cpu_time / wall_time) * 100 / logical_cpus


def run_inference(model, scaler, batch):
    """Scale the batch and perform Isolation Forest inference."""
    batch_scaled = scaler.transform(batch)
    predictions = model.predict(batch_scaled)

    return predictions


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

print("=" * 70)
print("SYSTEM LOGS ANOMALY DETECTION")
print("BATCH SCALING BENCHMARK")
print("=" * 70)

print(f"Platform: {platform.platform()}")
print(f"CPU: {platform.processor()}")
print(f"Logical CPUs: {psutil.cpu_count(logical=True)}")
print(f"Physical CPUs: {psutil.cpu_count(logical=False)}")
print(f"RAM: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB")
print()

print("Loading model...")
model = joblib.load(MODEL_PATH)

print("Loading scaler...")
scaler = joblib.load(SCALER_PATH)

print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

X = df[FEATURES].to_numpy(dtype=np.float64)

print(f"Dataset records: {len(X)}")
print(f"Input features: {len(FEATURES)}")
print()

if len(X) < max(BATCH_SIZES):
    raise ValueError(
        f"Dataset must contain at least {max(BATCH_SIZES)} records."
    )


# ============================================================
# WARM-UP
# ============================================================

print("Warming up model...")

warmup_batch = X[:100]
run_inference(model, scaler, warmup_batch)

gc.collect()

print("Warm-up completed.")
print()


# ============================================================
# BENCHMARK
# ============================================================

process = psutil.Process(os.getpid())

results = []

for batch_size in BATCH_SIZES:

    batch = X[:batch_size]

    print("-" * 70)
    print(f"Testing batch size: {batch_size} records")

    for repetition in range(1, REPETITIONS + 1):

        gc.collect()

        # Memory before test
        rss_before = get_rss_mb(process)

        # CPU time before test
        cpu_before = get_cpu_time(process)

        # Start Python allocation tracking
        tracemalloc.start()

        start = time.perf_counter()

        predictions = run_inference(
            model,
            scaler,
            batch
        )

        end = time.perf_counter()

        # Stop allocation tracking
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Measurements after test
        wall_time = end - start

        cpu_after = get_cpu_time(process)
        cpu_time = cpu_after - cpu_before

        rss_after = get_rss_mb(process)

        cpu_usage = measure_cpu_usage(
            process,
            wall_time,
            cpu_time
        )

        records_per_second = (
            batch_size / wall_time
            if wall_time > 0
            else 0
        )

        average_latency_ms = (
            wall_time / batch_size * 1000
            if batch_size > 0
            else 0
        )

        predicted_anomalies = int(
            np.sum(predictions == -1)
        )

        result = {
            "batch_size": batch_size,
            "repetition": repetition,
            "wall_time_seconds": wall_time,
            "average_latency_ms": average_latency_ms,
            "records_per_second": records_per_second,
            "process_cpu_time_seconds": cpu_time,
            "approx_process_cpu_percent": cpu_usage,
            "rss_before_mb": rss_before,
            "rss_after_mb": rss_after,
            "rss_delta_mb": rss_after - rss_before,
            "python_peak_memory_mb": peak_memory / (1024 * 1024),
            "predicted_anomalies": predicted_anomalies
        }

        results.append(result)

        print(
            f"Run {repetition}: "
            f"{wall_time:.6f}s | "
            f"{records_per_second:.2f} records/sec | "
            f"CPU {cpu_usage:.2f}% | "
            f"RSS {rss_after:.2f} MB | "
            f"Anomalies {predicted_anomalies}"
        )

        time.sleep(PAUSE_SECONDS)


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = "batch_scaling_results.csv"

with open(output_file, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(results)

summary = (
    results_df
    .groupby("batch_size")
    .agg(
        avg_time_seconds=(
            "wall_time_seconds",
            "mean"
        ),
        min_time_seconds=(
            "wall_time_seconds",
            "min"
        ),
        max_time_seconds=(
            "wall_time_seconds",
            "max"
        ),
        avg_latency_ms=(
            "average_latency_ms",
            "mean"
        ),
        avg_records_per_second=(
            "records_per_second",
            "mean"
        ),
        avg_cpu_percent=(
            "approx_process_cpu_percent",
            "mean"
        ),
        max_rss_mb=(
            "rss_after_mb",
            "max"
        ),
        max_python_peak_mb=(
            "python_peak_memory_mb",
            "max"
        ),
        predicted_anomalies=(
            "predicted_anomalies",
            "mean"
        )
    )
    .reset_index()
)

summary_file = "batch_scaling_summary.csv"
summary.to_csv(summary_file, index=False)

print()
print("=" * 70)
print("FINAL BENCHMARK SUMMARY")
print("=" * 70)

print(summary.to_string(index=False))

print()
print(f"Detailed results saved to: {output_file}")
print(f"Summary saved to: {summary_file}")
print()
print("Benchmark completed successfully.")