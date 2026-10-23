"""Task 2: native parallel reductions and Amdahl's law at P=0.95."""
import argparse
import ctypes
import math
import statistics

from bench_common import header, native, student_seed, logical_cores, run_workers
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=20_000_000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.size <= 0 or args.repeats <= 0:
        parser.error("--size and --repeats must be positive")
    header("TASK 2 - Multi-core scaling versus Amdahl's law")
    lib = native()
    ptr = ctypes.POINTER(ctypes.c_double)
    lib.numeric_sum.argtypes = [ptr, ctypes.c_size_t, ctypes.c_size_t]
    lib.numeric_sum.restype = ctypes.c_double
    values = np.random.default_rng(student_seed()).random(args.size, dtype=np.float64)
    data = values.ctypes.data_as(ptr)
    reference = lib.numeric_sum(data, 0, values.size)
    cores = logical_cores()
    counts = [1, 2, 4, cores, cores * 2]
    rows = []
    print(f"Elements: {values.size:,}; repetitions per worker count: {args.repeats}")
    print("Same native summation kernel at every worker count; final partials use math.fsum.")
    print("Timing: ready-worker release through final reduction; excludes pool startup,")
    print("compilation, input generation and correctness checks; median of measured runs.")
    print("P = 0.95; Amdahl(p) = 1 / (0.05 + 0.95 / p)")
    for workers in counts:
        def work(index):
            return lib.numeric_sum(data, index * values.size // workers,
                                   (index + 1) * values.size // workers)
        run_workers(workers, work)  # untimed warmup
        samples = []
        total = 0.0
        for _ in range(args.repeats):
            elapsed, total = run_workers(workers, work, finalize=math.fsum)
            samples.append(elapsed)
            assert math.isclose(total, reference, rel_tol=1e-11)
        rows.append((workers, statistics.median(samples)))
        print(f"Raw wall times p={workers} (ms): " + ", ".join(f"{t:.6f}" for t in samples))
    baseline = rows[0][1]
    print(f"{'Cores':>7} | {'Wall Time (ms)':>16} | {'Speedup':>10} | {'Amdahl Upper Bound':>19}")
    print("-" * 65)
    for workers, elapsed in rows:
        print(f"{workers:7d} | {elapsed:16.6f} | {baseline / elapsed:10.6f} | "
              f"{1 / (0.05 + 0.95 / workers):19.6f}")
    print(f"Verified sum (serial reference): {reference:.17g}")
    print("Verification: PASS; tiny rounding differences from partitioning are expected.")
    print("Amdahl assumes ideal hardware and a fixed parallel fraction; measured results")
    print("also reflect memory access, scheduling and the reported container CPU quota.")


if __name__ == "__main__":
    main()
