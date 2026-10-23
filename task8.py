"""CPU matrix geometry: naive and 16/64-wide tiles, with real native threads."""

import argparse
import ctypes
import os
import statistics

# Limit the independent correctness reference; benchmark kernels never use BLAS.
for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np

from bench_common import header, logical_cores, native, run_workers, student_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.size < 1 or args.repeats < 1:
        parser.error("size and repeats must be positive")

    header("Task 8: CPU matrix multiplication geometry")
    n = args.size
    workers = logical_cores()
    print("GPU path: no NVIDIA GPU available; running the CPU alternative.")
    print(f"Matrix dimensions: {n} x {n}, dtype=float64; workers={workers}")
    print(f"Trials per geometry: {args.repeats}; report median wall-clock time")
    print("Timing: synchronized worker release through last completion; excludes pool startup/teardown, inputs, validation.")
    print("Naive loop order: i-j-k; tiled loop order: i-k-j within blocks.")
    print("Runtime changes reflect both blocking and loop locality; hardware cache misses are not measured.")

    rng = np.random.default_rng(student_seed())
    a = rng.random((n, n), dtype=np.float64)
    b = rng.random((n, n), dtype=np.float64)
    c = np.empty((n, n), dtype=np.float64)
    lib = native()
    double_pointer = ctypes.POINTER(ctypes.c_double)
    signature = [double_pointer, double_pointer, double_pointer] + [ctypes.c_size_t] * 3
    lib.geometry_naive.argtypes = signature
    lib.geometry_naive.restype = None
    lib.geometry_tiled.argtypes = signature + [ctypes.c_size_t]
    lib.geometry_tiled.restype = None
    a_pointer = a.ctypes.data_as(double_pointer)
    b_pointer = b.ctypes.data_as(double_pointer)
    c_pointer = c.ctypes.data_as(double_pointer)

    # The environment is set before NumPy import. threadpoolctl also handles
    # interpreters that loaded a BLAS library before this script was imported.
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        reference = a @ b
    else:
        with threadpool_limits(limits=1):
            reference = a @ b

    geometries = [("Non-tiled", None), ("16 x 16", 16), ("64 x 64", 64)]
    timings = {name: [] for name, _ in geometries}
    errors = {name: 0.0 for name, _ in geometries}

    def execute(tile):
        def work(worker):
            begin, end = n * worker // workers, n * (worker + 1) // workers
            arguments = (a_pointer, b_pointer, c_pointer, n, begin, end)
            if tile is None:
                lib.geometry_naive(*arguments)
            else:
                lib.geometry_tiled(*arguments, tile)

        elapsed_ms, _ = run_workers(workers, work)
        if not np.allclose(c, reference, rtol=1e-11, atol=1e-11):
            raise AssertionError(f"Incorrect matrix product for tile={tile}")
        return elapsed_ms, float(np.max(np.abs(c - reference)))

    # Rotate the order between trials to reduce systematic first/last bias.
    for trial in range(args.repeats):
        order = geometries[trial % 3:] + geometries[:trial % 3]
        for name, tile in order:
            elapsed_ms, error = execute(tile)
            timings[name].append(elapsed_ms)
            errors[name] = max(errors[name], error)

    baseline = statistics.median(timings["Non-tiled"])
    print(f"{'Geometry':<14} | {'Wall time (ms)':>15} | {'Speedup':>9} | {'Delta vs naive (ms)':>20}")
    print("-" * 70)
    for name, _ in geometries:
        elapsed_ms = statistics.median(timings[name])
        print(f"{name:<14} | {elapsed_ms:15.6f} | {baseline / elapsed_ms:8.3f}x | {elapsed_ms - baseline:20.6f}")
        print("  Raw trials (ms): " + ", ".join(f"{value:.6f}" for value in timings[name]))
        print(f"  Validation: PASS; maximum absolute error={errors[name]:.3e}")


if __name__ == "__main__":
    main()
