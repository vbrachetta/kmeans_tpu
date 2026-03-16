"""
CPU vs TPU benchmark for both random and K-Means++ initialisation strategies.
Run from the project root as:
    python3 benchmarks/01_benchmark_cpu_tpu.py
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
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


def benchmark(device: str, data_np: np.ndarray, k: int, n_iter: int, init: str, n_runs: int) -> np.ndarray:
    print(f"\nBenchmarking on {device.upper()} (init={init}) ...")
    with jax.default_device(jax.devices(device)[0]):
        data_jax = jnp.array(data_np)

        print("  Warming up (JIT compilation) ...")
        _ = kmeans(data_jax, k=k, n_iter=2, init=init)
        jax.effects_barrier()

        times = []
        for run in range(n_runs):
            start = time.perf_counter()
            _ = kmeans(data_jax, k=k, n_iter=n_iter, init=init)
            jax.effects_barrier()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"  Run {run + 1}/{n_runs}: {elapsed:.4f}s")

    times = np.array(times)
    print(f"  Mean: {times.mean():.4f}s | Std: {times.std():.4f}s | "
          f"Min: {times.min():.4f}s | Max: {times.max():.4f}s")
    return times


def print_summary(cpu_times: np.ndarray, tpu_times: np.ndarray, init: str) -> None:
    speedup = cpu_times.mean() / tpu_times.mean()
    print("\n" + "=" * 50)
    print(f"BENCHMARK SUMMARY — init={init}")
    print("=" * 50)
    print(f"  CPU mean time : {cpu_times.mean():.4f}s")
    print(f"  TPU mean time : {tpu_times.mean():.4f}s")
    print(f"  Speedup (TPU vs CPU): {speedup:.2f}x")
    print("=" * 50)


if __name__ == "__main__":
    for init in ("random", "plusplus"):
        print(f"\n{'=' * 50}")
        print(f"Initialisation strategy: {init}")
        print(f"{'=' * 50}")
        cpu_times = benchmark("cpu", data_np, k=K, n_iter=N_ITER, init=init, n_runs=N_RUNS)
        tpu_times = benchmark("tpu", data_np, k=K, n_iter=N_ITER, init=init, n_runs=N_RUNS)
        print_summary(cpu_times, tpu_times, init=init)
