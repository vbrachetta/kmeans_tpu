"""
Benchmark comparing scikit-learn K-Means (CPU) vs kmeans_tpu (JAX, TPU)
for both random and K-Means++ initialisation.

Requires scikit-learn:
    pip install scikit-learn

Run from the project root as:
    python3 benchmarks/02_benchmark_sklearn.py
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "16"  # OpenBLAS thread limit due to compiled configuration

import time
import numpy as np
import jax
import jax.numpy as jnp
from sklearn.cluster import KMeans
from kmeans_tpu import kmeans

# --- Configuration ---
N_SAMPLES   = 100_000
N_FEATURES  = 32
K           = 8
N_ITER      = 100
N_RUNS      = 5
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
data_np = np.random.randn(N_SAMPLES, N_FEATURES).astype("float32")
print(f"Available CPU cores: {os.cpu_count()}")


def benchmark_sklearn(init: str, n_runs: int = N_RUNS) -> np.ndarray:
    sklearn_init = "k-means++" if init == "plusplus" else "random"
    print(f"\nBenchmarking scikit-learn on CPU (init={sklearn_init}) ...")
    times = []
    for run in range(n_runs):
        model = KMeans(
            n_clusters=K,
            init=sklearn_init,
            max_iter=N_ITER,
            n_init=1,
            random_state=RANDOM_SEED,
        )
        start = time.perf_counter()
        model.fit(data_np)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {run + 1}/{n_runs}: {elapsed:.4f}s")
    times = np.array(times)
    print(f"  Mean: {times.mean():.4f}s | Std: {times.std():.4f}s | "
          f"Min: {times.min():.4f}s | Max: {times.max():.4f}s")
    return times


def benchmark_tpu(init: str, n_runs: int = N_RUNS) -> np.ndarray:
    print(f"\nBenchmarking on TPU (init={init}) ...")
    with jax.default_device(jax.devices("tpu")[0]):
        data_jax = jnp.array(data_np)

        print("  Warming up (JIT compilation) ...")
        _ = kmeans(data_jax, k=K, n_iter=2, init=init)
        jax.effects_barrier()

        times = []
        for run in range(n_runs):
            start = time.perf_counter()
            _ = kmeans(data_jax, k=K, n_iter=N_ITER, init=init)
            jax.effects_barrier()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"  Run {run + 1}/{n_runs}: {elapsed:.4f}s")

    times = np.array(times)
    print(f"  Mean: {times.mean():.4f}s | Std: {times.std():.4f}s | "
          f"Min: {times.min():.4f}s | Max: {times.max():.4f}s")
    return times


def print_full_summary(sklearn_times: np.ndarray, tpu_times: np.ndarray, init: str) -> None:
    sklearn_init = "k-means++" if init == "plusplus" else "random"
    speedup = sklearn_times.mean() / tpu_times.mean()
    print("\n" + "=" * 60)
    print(f"FULL BENCHMARK SUMMARY — init={sklearn_init}")
    print("=" * 60)
    print(f"  Scikit-learn CPU mean time : {sklearn_times.mean():.4f}s")
    print(f"  kmeans_tpu TPU mean time   : {tpu_times.mean():.4f}s")
    print(f"  Speedup (TPU vs sklearn)   : {speedup:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    for init in ("random", "plusplus"):
        print(f"\n{'=' * 60}")
        print(f"Initialisation strategy: {init}")
        print(f"{'=' * 60}")
        sklearn_times = benchmark_sklearn(init=init)
        tpu_times     = benchmark_tpu(init=init)
        print_full_summary(sklearn_times, tpu_times, init=init)
