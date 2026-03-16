"""
Clustering example using make_blobs and K-Means++ initialisation,
with direct comparison between true labels and recovered cluster
assignments, visualised as a scatter plot.

Requires matplotlib and scikit-learn:
    pip install matplotlib scikit-learn

Run from the examples folder as:
    python3 blobs_clustering.py
"""
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from sklearn.datasets import make_blobs
from kmeans_tpu import kmeans

# --- Generate well-separated clusters ---
data_np, true_labels = make_blobs(n_samples=300, n_features=2, centers=3, random_state=42)
data = jnp.array(data_np.astype("float32"))

# --- Run K-Means++ ---
centroids, assignments, n_iter = kmeans(data, k=3, verbose=True)

assignments_np = np.array(assignments)
centroids_np   = np.array(centroids)

print(f"\nConverged in {n_iter} iterations")
print(f"Cluster counts:    {np.bincount(assignments_np, minlength=3)}")
print(f"True label counts: {np.bincount(true_labels, minlength=3)}")

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(data_np[:, 0], data_np[:, 1], c=true_labels, s=20, alpha=0.7)
axes[0].set_title("True Labels")

axes[1].scatter(data_np[:, 0], data_np[:, 1], c=assignments_np, s=20, alpha=0.7)
axes[1].scatter(centroids_np[:, 0], centroids_np[:, 1], c="black", s=200, marker="X", zorder=5, label="Centroids")
axes[1].set_title(f"K-Means++ (converged in {n_iter} iterations)")
axes[1].legend()

plt.suptitle("kmeans_tpu — Clustering Results", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("blobs_clustering.png", dpi=150, bbox_inches="tight")
print("Plot saved to blobs_clustering.png")
