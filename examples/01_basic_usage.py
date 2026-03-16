"""
Basic usage of kmeans_tpu with random initialisation (K-Means).
"""
import numpy as np
import jax.numpy as jnp
from kmeans_tpu import kmeans

# Generate synthetic data
np.random.seed(42)
data = jnp.array(np.random.randn(1_000, 4).astype("float32"))

# Run K-Means with random initialisation
centroids, assignments, n_iter = kmeans(data, k=3, init="random", verbose=True)

print(f"\nConverged in {n_iter} iterations")
print(f"Cluster counts: {np.bincount(np.array(assignments), minlength=3)}")
