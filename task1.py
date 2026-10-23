"""Task 1: 20 million Student-ID-seeded doubles, strictly serial statistics."""
import argparse
import ctypes
import math
import time

from bench_common import header, native, student_seed
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=20_000_000)
    args = parser.parse_args()
    if args.size <= 0:
        parser.error("--size must be positive")
    header("TASK 1 - Hardware probe and serial baseline")
    lib = native()
    ptr = ctypes.POINTER(ctypes.c_double)
    lib.numeric_sum.argtypes = [ptr, ctypes.c_size_t, ctypes.c_size_t]
    lib.numeric_sum.restype = ctypes.c_double
    lib.numeric_squared_deviation.argtypes = [ptr, ctypes.c_size_t, ctypes.c_double]
    lib.numeric_squared_deviation.restype = ctypes.c_double

    started = time.perf_counter_ns()
    values = np.random.default_rng(student_seed()).random(args.size, dtype=np.float64)
    generated = time.perf_counter_ns()
    data = values.ctypes.data_as(ptr)
    total = lib.numeric_sum(data, 0, values.size)
    mean = total / values.size
    std = math.sqrt(lib.numeric_squared_deviation(data, values.size, mean) / values.size)
    finished = time.perf_counter_ns()
    # Correctness checks run after timing, using a different reduction implementation.
    assert math.isclose(total, float(np.sum(values)), rel_tol=1e-11)
    assert math.isclose(std, float(np.std(values)), rel_tol=1e-11)
    print(f"Elements: {values.size:,}; allocation: {values.nbytes:,} bytes")
    print("Reduction threads: 1; scalar native loops; population std (ddof=0)")
    print(f"Sum: {total:.17g}")
    print(f"Mean: {mean:.17g}")
    print(f"Standard deviation: {std:.17g}")
    print(f"Generation wall time (ms): {(generated - started) / 1e6:.6f}")
    print(f"Serial statistics wall time (ms): {(finished - generated) / 1e6:.6f}")
    print(f"Generation + statistics wall time (ms): {(finished - started) / 1e6:.6f}")
    print("Verification: PASS (independent NumPy sum and standard deviation)")


if __name__ == "__main__":
    main()
