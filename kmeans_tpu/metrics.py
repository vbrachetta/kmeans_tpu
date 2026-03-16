"""
Clustering quality metrics for kmeans_tpu.

This module provides the inertia (within-cluster sum of squares) metric
to evaluate clustering quality and compare results against reference
implementations such as scikit-learn.
"""
import jax.numpy as jnp


def inertia(data: jnp.ndarray, centroids: jnp.ndarray, assignments: jnp.ndarray) -> float:
    """
    Compute the inertia (within-cluster sum of squares).

    Lower values indicate tighter, better-defined clusters.

    Parameters
    ----------
    data : jnp.ndarray
        Input data of shape (n_samples, n_features).
    centroids : jnp.ndarray
        Final centroids of shape (k, n_features).
    assignments : jnp.ndarray
        Cluster assignment for each data point, shape (n_samples,).

    Returns
    -------
    float
        Inertia value.
    """
    assigned_centroids = centroids[assignments]
    return float(jnp.sum(jnp.sum((data - assigned_centroids) ** 2, axis=-1)))
