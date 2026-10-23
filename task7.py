"""Task 7: memory-copy versus transcendental compute scaling."""
import argparse
import ctypes
import statistics

from bench_common import header, native, student_seed, logical_cores, run_workers
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=30_000_000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.size <= 0 or args.repeats <= 0:
        parser.error("--size and --repeats must be positive")
    header("TASK 7 - Arithmetic intensity versus memory bandwidth")
    lib = native()
    ptr = ctypes.POINTER(ctypes.c_double)
    for name in ("numeric_copy", "numeric_trig"):
        kernel = getattr(lib, name)
        kernel.argtypes = [ptr, ptr, ctypes.c_size_t, ctypes.c_size_t]
        kernel.restype = None
    source = np.random.default_rng(student_seed()).random(args.size, dtype=np.float64)
    dest = np.empty_like(source)
    src_ptr, dst_ptr = source.ctypes.data_as(ptr), dest.ctypes.data_as(ptr)
    # First-touch outside timing, with the same initialization for both kernels.
    dest.fill(0.0)
    counts = list(dict.fromkeys([1, 2, logical_cores()]))
    print(f"Elements: {source.size:,}; bytes per array: {source.nbytes:,}")
    print(f"Repetitions: {args.repeats}; median wall time; preallocated/prefaulted arrays")
    print("Timing: ready-worker release to results collected; pool startup/teardown,")
    print("allocation, compilation, validation and warmup excluded for both kernels.")
    all_results = {}
    for title, kernel in (("A: Memory-bound copy B[i] = A[i]", lib.numeric_copy),
                          ("B: Compute-bound sin(A)*cos(A)+tan(A)", lib.numeric_trig)):
        rows = []
        print(f"\nKernel {title}")
        for workers in counts:
            def work(index):
                kernel(src_ptr, dst_ptr, index * source.size // workers,
                       (index + 1) * source.size // workers)
            run_workers(workers, work)
            samples = [run_workers(workers, work)[0] for _ in range(args.repeats)]
            # Check every value, in bounded chunks, outside timed regions.
            for begin in range(0, source.size, 1_000_000):
                a = source[begin:begin + 1_000_000]
                b = dest[begin:begin + 1_000_000]
                expected = a if kernel == lib.numeric_copy else np.sin(a) * np.cos(a) + np.tan(a)
                assert np.allclose(b, expected, rtol=1e-12, atol=1e-14)
            rows.append((workers, statistics.median(samples)))
            print(f"Raw wall times p={workers} (ms): " + ", ".join(f"{t:.6f}" for t in samples))
        t1 = rows[0][1]
        print(f"{'Cores':>7} | {'Wall Time (ms)':>16} | {'Speedup':>10} | {'Efficiency':>11} | {'GB/s*':>9}")
        for workers, elapsed in rows:
            speedup = t1 / elapsed
            bandwidth = 2 * source.nbytes / (elapsed * 1e6)
            print(f"{workers:7d} | {elapsed:16.6f} | {speedup:10.6f} | {speedup / workers:11.6f} | {bandwidth:9.3f}")
        all_results[title[0]] = rows
    memory, compute = all_results["A"], all_results["B"]
    memory_last_gain = memory[-2][1] / memory[-1][1] if len(memory) > 1 else 1.0
    compute_last_gain = compute[-2][1] / compute[-1][1] if len(compute) > 1 else 1.0
    print("\n* GB/s is effective read+write traffic, excluding write-allocate traffic.")
    print(f"Last worker-count step: copy improvement {memory_last_gain:.3f}x; compute {compute_last_gain:.3f}x.")
    if len(memory) < 2:
        print("Memory saturation conclusion: unidentifiable with only one exposed CPU.")
    elif memory_last_gain <= 1.10 and compute_last_gain >= 1.15:
        print("Kernel A stalled in scaling; this is consistent with memory-channel saturation.")
    else:
        print("No distinct memory-channel saturation plateau established for Kernel A.")
        print("The measurements do not justify claiming a RAM bandwidth limit was reached.")
    print("Kernel B performs more arithmetic per byte. Both kernels can also be limited")
    print("by CPU quota/scheduling. Timings alone cannot prove memory-channel saturation;")
    print("hardware performance counters would be needed to separate those causes.")
    print("Verification: PASS (all output elements checked outside timings)")


if __name__ == "__main__":
    main()
