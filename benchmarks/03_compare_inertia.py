"""
Comparison of inertia between kmeans_tpu and scikit-learn.
This script verifies that kmeans_tpu produces clustering quality
consistent with a reference implementation.

Requires scikit-learn:
    pip install scikit-learn

Run from the project root as:
    python3 benchmarks/03_compare_inertia.py

Note: On machines with more than 128 cores, an OpenBLAS warning may appear.
This is harmless and does not affect the results.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "16"  # OpenBLAS thread limit due to compiled configuration

import numpy as np
import jax.numpy as jnp
from sklearn.cluster import KMeans
from kmeans_tpu import kmeans, inertia

# --- Configuration ---
N_SAMPLES   = 100_000
N_FEATURES  = 32
K           = 8
N_ITER      = 100
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
data_np = np.random.randn(N_SAMPLES, N_FEATURES).astype("float32")
data    = jnp.array(data_np)

print(f"Dataset: {N_SAMPLES} samples, {N_FEATURES} features, {K} clusters\n")

for init in ("random", "plusplus"):
    sklearn_init = "k-means++" if init == "plusplus" else "random"

    # scikit-learn first
    model = KMeans(
        n_clusters=K,
        init=sklearn_init,
        max_iter=N_ITER,
        n_init=1,
        random_state=RANDOM_SEED,
    )
    model.fit(data_np)
    sklearn_inertia = model.inertia_

    # kmeans_tpu second
    centroids, assignments, n_iter = kmeans(data, k=K, init=init, n_iter=N_ITER)
    tpu_inertia = inertia(data, centroids, assignments)

    diff_pct = abs(tpu_inertia - sklearn_inertia) / sklearn_inertia * 100

    print(f"--- Init: {sklearn_init} ---")
    print(f"  kmeans_tpu inertia  : {tpu_inertia:.2f}")
    print(f"  scikit-learn inertia: {sklearn_inertia:.2f}")
    print(f"  Difference          : {diff_pct:.4f}%\n")
