import jax
import jax.numpy as jnp
import numpy as np
from functools import partial  # required for static_argnums compatibility


def init_centroids_random(data: jnp.ndarray, k: int, seed: int = 0) -> jnp.ndarray:
    """
    Randomly initialise k centroids from the data.

    Parameters
    ----------
    data : jnp.ndarray
        Input data of shape (n_samples, n_features). Must be float32 or bfloat16.
        float32 is recommended for numerical stability. bfloat16 may offer
        memory and speed advantages on TPU but with reduced precision, which
        can affect convergence on certain datasets.
    k : int
        Number of clusters.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    centroids : jnp.ndarray
        Initialised centroids of shape (k, n_features).
    """
    key = jax.random.PRNGKey(seed)
    indices = jax.random.choice(key, data.shape[0], shape=(k,), replace=False)
    return data[indices]


def init_centroids_plusplus(data: jnp.ndarray, k: int, seed: int = 0) -> jnp.ndarray:
    """
    Initialise centroids using the K-Means++ algorithm (Arthur & Vassilvitskii, 2007).

    Centroids are selected sequentially with probability proportional to the
    squared distance from the nearest already-chosen centroid, yielding better
    initialisation than random selection and typically faster convergence.

    Parameters
    ----------
    data : jnp.ndarray
        Input data of shape (n_samples, n_features). Must be float32 or bfloat16.
        float32 is recommended for numerical stability. bfloat16 may offer
        memory and speed advantages on TPU but with reduced precision, which
        can affect convergence on certain datasets.
    k : int
        Number of clusters.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    centroids : jnp.ndarray
        Initialised centroids of shape (k, n_features).

    References
    ----------
    Arthur, D. & Vassilvitskii, S. (2007). k-means++: The Advantages of Careful
    Seeding. Proceedings of the 18th Annual ACM-SIAM Symposium on Discrete
    Algorithms (SODA), 1027-1035.
    """
    key = jax.random.PRNGKey(seed)

    # Choose the first centroid uniformly at random
    key, subkey = jax.random.split(key)
    first_idx = jax.random.randint(subkey, shape=(), minval=0, maxval=data.shape[0])
    centroids = [data[first_idx]]

    for _ in range(k - 1):
        # Compute squared distances from each point to its nearest centroid
        centroid_stack = jnp.stack(centroids)  # (current_k, n_features)
        dists = jnp.sum(
            (data[:, None, :] - centroid_stack[None, :, :]) ** 2, axis=-1
        )  # (n_samples, current_k)
        min_dists = jnp.min(dists, axis=-1)  # (n_samples,)

        # Sample next centroid with probability proportional to squared distance
        probabilities = min_dists / jnp.sum(min_dists)
        key, subkey = jax.random.split(key)
        next_idx = jax.random.choice(subkey, data.shape[0], p=probabilities)
        centroids.append(data[next_idx])

    return jnp.stack(centroids)


@partial(jax.jit, static_argnums=(2,))
def _step(centroids: jnp.ndarray, data: jnp.ndarray, k: int):
    """
    Perform a single K-Means iteration (assignment + centroid update).

    Parameters
    ----------
    centroids : jnp.ndarray
        Current centroids of shape (k, n_features).
    data : jnp.ndarray
        Input data of shape (n_samples, n_features).
    k : int
        Number of clusters (static).

    Returns
    -------
    new_centroids : jnp.ndarray
        Updated centroids of shape (k, n_features).
    assignments : jnp.ndarray
        Cluster assignment for each data point, shape (n_samples,).
    """
    dists = jnp.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
    assignments = jnp.argmin(dists, axis=-1)
    new_centroids = jax.vmap(
        lambda c: jnp.sum(data * (assignments == c).astype(data.dtype)[:, None], axis=0)
                  / (jnp.sum((assignments == c).astype(data.dtype)) + 1e-8)
    )(jnp.arange(k))
    return new_centroids, assignments


def kmeans(
    data: jnp.ndarray,
    k: int,
    n_iter: int = 100,
    init: str = "plusplus",
    seed: int = 0,
    tol: float = 1e-4,
    verbose: bool = False,
):
    supported_dtypes = (jnp.float32, jnp.bfloat16)
    if data.dtype not in supported_dtypes:
        raise ValueError(
            f"Data must be float32 or bfloat16, got {data.dtype}. "
            f"Convert your data with: data.astype(jnp.float32)"
        )
    """
    Run K-Means clustering with a choice of initialisation strategy.

    Parameters
    ----------
    data : jnp.ndarray
        Input data of shape (n_samples, n_features).
    k : int
        Number of clusters.
    n_iter : int
        Maximum number of iterations. Default is 100.
    init : str
        Initialisation strategy: 'random' or 'plusplus' (default).
        'plusplus' uses the K-Means++ algorithm (Arthur & Vassilvitskii, 2007),
        which selects centroids with probability proportional to their squared
        distance from already-chosen centroids, yielding better and more
        reproducible results than random initialisation.
    seed : int
        Random seed for reproducibility. The same seed always produces the
        same initialisation. Change this to obtain different initialisations
        and assess sensitivity to starting conditions. Default is 0.
    tol : float
        Convergence tolerance on maximum centroid shift between iterations.
        The algorithm stops when the maximum shift falls below this value.
        Default is 1e-4.
    verbose : bool
        If True, print the maximum centroid shift every 10 iterations and
        report when convergence is reached. Default is False.

    Returns
    -------
    centroids : jnp.ndarray
        Final centroids of shape (k, n_features).
    assignments : jnp.ndarray
        Cluster assignment for each data point.
    n_iter_run : int
        Number of iterations run before convergence.

    Raises
    ------
    ValueError
        If an unrecognised initialisation strategy is provided.
    ValueError
        If the data dtype is not float32 or bfloat16.
    """
    if init == "random":
        centroids = init_centroids_random(data, k, seed=seed)
    elif init == "plusplus":
        centroids = init_centroids_plusplus(data, k, seed=seed)
    else:
        raise ValueError(f"Unknown initialisation strategy '{init}'. "
                         f"Choose 'random' or 'plusplus'.")

    for i in range(n_iter):
        new_centroids, assignments = _step(centroids, data, k)
        shift = float(jnp.max(jnp.linalg.norm(new_centroids - centroids, axis=-1)))
        centroids = new_centroids
        if verbose and i % 10 == 0:
            print(f"  Iter {i:>3d} | Max shift: {shift:.6f}")
        if shift < tol:
            if verbose:
                print(f"  Converged at iteration {i}.")
            break

    return centroids, assignments, i + 1
