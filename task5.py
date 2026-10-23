#!/usr/bin/env python3
"""Measure the sampled serial/parallel crossover for y = a*x + b.

The same compiled kernel serves both paths. ctypes releases the GIL, so the
worker threads execute native arithmetic concurrently. A persistent pool models
repeated vector operations; its one-time startup/shutdown is excluded, while
every measured parallel call includes task submission and completion waiting.
"""

from __future__ import annotations

import argparse
import ctypes
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter_ns

from bench_common import header, logical_cores, native
import numpy as np


def measure_call(function) -> int:
    start = perf_counter_ns()
    function()
    return perf_counter_ns() - start


def crossover_report(rows: list[tuple[int, float, float]]) -> None:
    transitions = []
    reversals = []
    for left, right in zip(rows, rows[1:]):
        left_slower = left[2] >= left[1]
        right_faster = right[2] < right[1]
        if left_slower and right_faster:
            transitions.append((left[0], right[0]))
        elif not left_slower and not right_faster:
            reversals.append((left[0], right[0]))

    print("\nProgrammatic crossover result (median end-to-end call latency):")
    if transitions:
        for small, large in transitions:
            print(
                f"  SLOWER at N={small:,}; FASTER at N={large:,}. "
                f"Sampled crossover bracket: ({small:,}, {large:,}]."
            )
    else:
        print("  No slower-to-faster crossover was observed among the tested sizes.")
    if reversals:
        for small, large in reversals:
            print(f"  Reverse transition: FASTER at {small:,}, SLOWER at {large:,}.")
    faster = [n for n, serial, parallel in rows if parallel < serial]
    if faster:
        print(f"  Smallest tested N with a parallel win: {min(faster):,}.")
    else:
        print("  Parallel execution was slower at every tested N.")
    print("  These are exact tested brackets, not an unmeasured exact integer threshold.")
    print("  Timings and the crossover can change with host load and CPU quota.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", type=int, nargs="+",
        default=[100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000],
    )
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 1 or any(size < 1 for size in args.sizes):
        parser.error("sizes and repeats must be positive")
    sizes = sorted(set(args.sizes))
    workers = logical_cores()

    header("Task 5: parallel overhead crossover")
    print(f"Workers: {workers} (all os.cpu_count() logical CPUs)")
    print(f"Kernel: float64 y[i] = 1.25*x[i] - 0.75; {args.repeats} measured repeats")
    print("Serial and parallel use the same native C kernel; ctypes releases the GIL.")
    print("Timer includes kernel call; parallel also includes dispatch and join.")
    print("Reusable pool startup/shutdown, allocation, warmup, validation are excluded.")
    print("Order alternates each repeat; displayed latency is the median.")

    lib = native()
    ptr = ctypes.POINTER(ctypes.c_double)
    lib.vector_affine.argtypes = [ptr, ptr, ctypes.c_size_t, ctypes.c_size_t,
                                  ctypes.c_double, ctypes.c_double]
    lib.vector_affine.restype = None
    a, b = 1.25, -0.75
    rows = []
    print(f"\n{'N':>12} | {'Serial (ms)':>14} | {'Parallel (ms)':>15} | {'Speedup':>9} | Result")
    print("-" * 80)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Force every worker to start before entering any timed measurements.
        startup = threading.Barrier(workers + 1)
        ready = [pool.submit(startup.wait) for _ in range(workers)]
        startup.wait()
        for future in ready:
            future.result()

        for n in sizes:
            x = np.linspace(-1.0, 1.0, n, dtype=np.float64)
            serial_y = np.zeros_like(x)
            parallel_y = np.zeros_like(x)
            x_ptr = x.ctypes.data_as(ptr)
            serial_ptr = serial_y.ctypes.data_as(ptr)
            parallel_ptr = parallel_y.ctypes.data_as(ptr)
            bounds = [(n * i // workers, n * (i + 1) // workers)
                      for i in range(workers)]

            def serial() -> None:
                lib.vector_affine(x_ptr, serial_ptr, 0, n, a, b)

            def parallel() -> None:
                futures = [pool.submit(lib.vector_affine, x_ptr, parallel_ptr,
                                       begin, end, a, b)
                           for begin, end in bounds]
                for future in futures:
                    future.result()

            serial()
            parallel()
            serial_times, parallel_times = [], []
            for repeat in range(args.repeats):
                if repeat % 2:
                    parallel_times.append(measure_call(parallel))
                    serial_times.append(measure_call(serial))
                else:
                    serial_times.append(measure_call(serial))
                    parallel_times.append(measure_call(parallel))

            expected = a * x + b
            np.testing.assert_allclose(serial_y, expected, rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(parallel_y, expected, rtol=1e-14, atol=1e-14)
            serial_ms = statistics.median(serial_times) / 1_000_000
            parallel_ms = statistics.median(parallel_times) / 1_000_000
            speedup = serial_ms / parallel_ms
            result = "FASTER" if parallel_ms < serial_ms else "SLOWER"
            rows.append((n, serial_ms, parallel_ms))
            print(f"{n:12,d} | {serial_ms:14.6f} | {parallel_ms:15.6f} | {speedup:9.4f} | {result}")
            print("  serial sample ms:   " + ", ".join(f"{t/1e6:.6f}" for t in serial_times))
            print("  parallel sample ms: " + ", ".join(f"{t/1e6:.6f}" for t in parallel_times))

    print("Validation: every output element matches the serial mathematical result.")
    crossover_report(rows)


if __name__ == "__main__":
    main()
