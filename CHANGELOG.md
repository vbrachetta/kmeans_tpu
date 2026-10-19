# Changelog

## [0.1.1] — 2026-10-19

### Added
- Three Colab-compatible Jupyter notebooks in `examples/notebooks/`:
  - `01_getting_started.ipynb` — installation, basic usage, first clustering result
  - `02_kmeans_vs_kmeanspp.ipynb` — visual comparison of initialisation strategies
  - `03_bfloat16.ipynb` — bfloat16 support and memory comparison
- Compatibility note in README confirming support for JAX 0.7.2 and Python 3.13

### Changed
- Updated README with Colab badges, notebook section, and compatibility information
- Updated `CITATION.cff` abstract for consistency with README

## [0.1.0] — 2026-03-16

### Initial release

- K-Means clustering with random initialisation
- K-Means++ initialisation (Arthur & Vassilvitskii, 2007)
- CPU vs TPU benchmarking
- Scikit-learn comparison benchmarks
- Inertia metric for clustering quality evaluation
- Support for float32 and bfloat16 data types
- Ready-to-run examples and benchmarking scripts
