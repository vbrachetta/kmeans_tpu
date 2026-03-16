<p align="center">
  <img src="assets/kmeans_tpu.svg" width="150" alt="kmeans_tpu logo">
</p>

# kmeans_tpu

**A JAX-based open-source package for TPU-accelerated clustering**

kmeans_tpu is an open-source Python package providing a JAX-based implementation
of K-Means and K-Means++ clustering optimised for Google Tensor Processing Units (TPUs).
By leveraging JAX's XLA compilation and vectorisation primitives (jit, vmap),
the package achieves substantial performance gains over standard CPU-based
implementations, demonstrated against a JAX CPU baseline and the optimised
multi-threaded library scikit-learn. Benchmark results on 100,000 samples using
Google TPU v4 and v6e hardware demonstrate speedups of up to 80x over a JAX CPU
baseline and up to 6x over scikit-learn. The package is designed with research
reproducibility in mind: all benchmark results are fully documented with hardware
and software specifications, and ready-to-run benchmarking scripts are provided,
including saved output files for TPU v4 and TPU v6e.

**Author:** Vincenzo Brachetta, University of Birmingham (UK)  
**Contact:** v.brachetta@bham.ac.uk

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Benchmarking](#benchmarking)
- [Project Structure](#project-structure)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [Licence](#licence)

---

## Requirements

- Python 3.10 or later
- Access to a Google TPU

To recreate the exact environment used for benchmarking:
```bash
pip install -r requirements.txt
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/kmeans_tpu.git
cd kmeans_tpu
```

### 2. Create and activate a virtual environment

It is strongly recommended to use a virtual environment to avoid conflicts
with system packages.
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Upgrade pip and setuptools
```bash
pip install --upgrade pip setuptools wheel
```

### 4. Install JAX with TPU support

JAX requires a specific installation step to enable TPU support:
```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### 5. Install the package

For standard installation:
```bash
pip install .
```

For development (editable) installation, which allows you to modify the
source code without reinstalling:
```bash
pip install -e ".[dev]"
```

### 6. Install directly from GitHub

If you do not wish to clone the repository, you can install directly from GitHub:
```bash
pip install git+https://github.com/vbrachetta/kmeans_tpu.git
```

### 7. Verify the installation
```bash
python -c "import kmeans_tpu; print(kmeans_tpu.__version__)"
python -c "import jax; print(jax.devices())"
```

The second command should report one or more TPU devices if JAX is correctly
configured.

---

## Usage

### K-Means with K-Means++ initialisation (default)
```python
import numpy as np
import jax.numpy as jnp
from kmeans_tpu import kmeans

np.random.seed(42)
data = jnp.array(np.random.randn(1_000, 4).astype("float32"))

centroids, assignments, n_iter = kmeans(data, k=3, verbose=True)
print(f"Converged in {n_iter} iterations")
```

### K-Means with random initialisation
```python
import numpy as np
import jax.numpy as jnp
from kmeans_tpu import kmeans

np.random.seed(42)
data = jnp.array(np.random.randn(1_000, 4).astype("float32"))

centroids, assignments, n_iter = kmeans(data, k=3, init="random", verbose=True)
print(f"Converged in {n_iter} iterations")
```

### Using your own data
```python
import jax.numpy as jnp
import numpy as np
from kmeans_tpu import kmeans

data = jnp.array(np.load("your_data.npy").astype("float32"))

centroids, assignments, n_iter = kmeans(
    data,
    k=8,
    init="plusplus",
    n_iter=100,
    tol=1e-4,
    seed=42,
    verbose=True,
)
print(f"Converged in {n_iter} iterations")
```

---

## Examples

Ready-to-run example scripts are provided in the [`examples/`](examples/) folder:

| Script | Description |
|---|---|
| `01_basic_usage.py` | K-Means and K-Means++ basic usage |
| `02_blobs.py` | Clustering with make_blobs and scatter plot output |

Run from the examples folder as:
```bash
python3 01_basic_usage.py
python3 02_blobs.py
```

---

## Benchmarking

All benchmarks were run on 100,000 samples, 32 features, 8 clusters,
100 iterations, 5 runs.

### Hardware Environment

| Component | TPU v4 | TPU v6e |
|---|---|---|
| CPU | AMD EPYC 7B12 | AMD EPYC 9B14 |
| CPU logical cores | 240 | 180 |
| OpenBLAS threads | 16 | 16 |
| RAM | 400 GiB | 708 GiB |
| TPU cores | 4 | 4 |

### Software Environment

| Component | Version |
|---|---|
| Python | 3.10.12 |
| JAX | 0.6.2 |
| Scikit-learn | 1.7.2 |
| OS (TPU v4) | Linux 5.19.0-1030-gcp (x86_64) |
| OS (TPU v6e) | Linux 6.8.0-1015-gcp (x86_64) |

### JAX CPU vs TPU

| Strategy | CPU v4 (s) | CPU v6e (s) | TPU v4 (s) | TPU v6e (s) | Speedup v4 | Speedup v6e |
|---|---|---|---|---|---|---|
| K-Means (random) | 4.359 | 2.235 | 0.055 | 0.052 | 79.79x | 43.08x |
| K-Means++ | 4.528 | 2.253 | 0.090 | 0.069 | 50.56x | 32.58x |

The lower speedup on TPU v6e reflects a stronger CPU baseline (AMD EPYC 9B14)
rather than inferior TPU performance — the TPU times are consistent across
both generations.

### Scikit-learn vs TPU

| Strategy | Sklearn v4 (s) | Sklearn v6e (s) | TPU v4 (s) | TPU v6e (s) | Speedup v4 | Speedup v6e |
|---|---|---|---|---|---|---|
| K-Means (random) | 0.360 | 0.225 | 0.056 | 0.052 | 6.41x | 4.30x |
| K-Means++ | 0.418 | 0.259 | 0.090 | 0.069 | 4.67x | 3.74x |

Scikit-learn was benchmarked with 16 OpenBLAS threads due to a compiled limit
of the available installation. Additional tests across 4, 8, and 16 threads
showed negligible variation in scikit-learn performance, suggesting the
bottleneck is memory bandwidth rather than compute. The performance gap
is therefore not an artefact of the thread count used.

### Clustering Quality

Inertia comparison between kmeans_tpu and scikit-learn on 100,000 samples.

**TPU v4**

| Strategy | kmeans_tpu | scikit-learn | Difference |
|---|---|---|---|
| K-Means (random) | 2,962,242 | 2,961,856 | 0.0130% |
| K-Means++ | 2,961,346 | 2,961,666 | 0.0108% |

**TPU v6e**

| Strategy | kmeans_tpu | scikit-learn | Difference |
|---|---|---|---|
| K-Means (random) | 2,962,191 | 2,961,855 | 0.0113% |
| K-Means++ | 2,961,417 | 2,961,709 | 0.0099% |

Inertia differences below 0.02% across both hardware configurations confirm
that clustering quality is numerically consistent with scikit-learn and
independent of the underlying hardware.

### Reproducing the benchmarks
```bash
bash benchmarks/run_benchmarks.sh
```

---

## Project Structure
```
kmeans_tpu/
├── kmeans_tpu/
│   ├── __init__.py
│   ├── algorithm.py
│   └── metrics.py
├── assets/
│   ├── src/
│   │   ├── kmeans_tpu.svg
│   │   └── kmeans_tpu_banner.svg
│   ├── kmeans_tpu.svg
│   └── kmeans_tpu_banner.svg
├── benchmarks/
│   ├── 01_benchmark_cpu_tpu.py
│   ├── 02_benchmark_sklearn.py
│   ├── 03_compare_inertia.py
│   ├── benchmark_results_v4.txt
│   ├── benchmark_results_v6e.txt
│   └── run_benchmarks.sh
├── examples/
│   ├── 01_basic_usage.py
│   ├── 02_blobs.py
│   └── blobs_clustering.png
├── CHANGELOG.md
├── CITATION.cff
├── LICENSE
├── README.md
├── pyproject.toml
├── requirements_v4.txt
└── requirements_v6e.txt
```

---

## Acknowledgements

This work was carried out at the University of Birmingham (UK). The author wishes
to thank **Prof. Mayorkinos Papaelias** for providing the time and support
that made this work possible.

TPU resources were provided through the
[Google TPU Research Cloud (TRC)](https://sites.research.google/trc/about/)
programme. The author gratefully acknowledges Google's support.

This work was developed with the assistance of Claude Sonnet 4.6 (Anthropic)
for code generation and documentation drafting. All code and content were
reviewed, validated, and edited by the author.

---

## Citation

If you use this software in your research, please cite it as follows:
```bibtex
@software{brachetta2026kmeans_tpu,
  author    = {Brachetta, Vincenzo},
  title     = {kmeans\_tpu: A JAX-based open-source framework for TPU-accelerated clustering},
  year      = {2026},
  publisher = {Zenodo},
  url       = {https://github.com/vbrachetta/kmeans_tpu},
  doi       = {10.5281/zenodo.19048768}
}
```

Or see the provided [`CITATION.cff`](CITATION.cff) file, which is recognised
automatically by GitHub and Zenodo.

---

## Licence

This project is licensed under the MIT Licence. See the [LICENSE](LICENSE)
file for details.
