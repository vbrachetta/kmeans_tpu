#!/bin/bash
# Run all benchmarks for kmeans_tpu.
# Requires: pip install scikit-learn
#
# Note: Set OPENBLAS_THREADS to the number of threads available on your machine.

set -e

OPENBLAS_THREADS=16  # Change this if needed
export OPENBLAS_NUM_THREADS=$OPENBLAS_THREADS
OUTPUT="benchmark_results.txt"

{
    echo "kmeans_tpu Benchmarks — $(date)"
    echo ""
    echo "Hardware:"
    echo "  CPU model   : $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
    echo "  CPU cores   : $(nproc)"
    echo "  RAM         : $(free -h | awk '/^Mem:/ {print $2}')"
    echo ""
    echo "TPU:"
    python3 -c "import jax; [print(' ', d) for d in jax.devices()]"
    echo ""
    echo "Software:"
    echo "  OS          : $(uname -sr)"
    echo "  Python      : $(python3 --version)"
    echo "  JAX         : $(python3 -c 'import jax; print(jax.__version__)')"
    echo "  Scikit-learn: $(python3 -c 'import sklearn; print(sklearn.__version__)')"
    echo "  OPENBLAS    : $OPENBLAS_THREADS threads"
    echo ""
} | tee -a $OUTPUT

echo "kmeans_tpu Benchmarks — $(date)" | tee -a $OUTPUT

echo "1/3 Running CPU vs TPU benchmark..."
python3 01_benchmark_cpu_tpu.py | tee -a $OUTPUT

echo "2/3 Running scikit-learn vs TPU benchmark..."
python3 02_benchmark_sklearn.py | tee -a $OUTPUT

echo "3/3 Running inertia comparison kmeans_tpu vs scikit-learn..."
python3 03_compare_inertia.py | tee -a $OUTPUT

