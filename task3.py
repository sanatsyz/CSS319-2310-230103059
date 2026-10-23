#!/usr/bin/env python3
"""Compare adjacent and cache-line-padded counters using true CPU threads."""

import argparse
import ctypes

from bench_common import header, native, physical_cores, run_workers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=50_000_000)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    header("Task 3 - Cache line contention / false sharing")
    workers = physical_cores()
    lib = native()
    lib.contention_increment.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64]
    lib.contention_increment.restype = None
    line_bytes = 64

    def measure(stride):
        # Align the first counter so adjacent elements demonstrably share a line.
        allocation = ctypes.create_string_buffer(workers * stride * 8 + line_bytes)
        base = (ctypes.addressof(allocation) + line_bytes - 1) & ~(line_bytes - 1)
        counters = [ctypes.c_uint64.from_address(base + i * stride * 8)
                    for i in range(workers)]

        def work(index):
            lib.contention_increment(ctypes.byref(counters[index]), args.iterations)

        elapsed_ms, _ = run_workers(workers, work)
        observed = [counter.value for counter in counters]
        if observed != [args.iterations] * workers:
            raise RuntimeError(f"Incorrect final counts: {observed}")
        return elapsed_ms, observed

    print(f"Worker threads (detected physical cores): {workers}")
    print(f"Increments per thread: {args.iterations:,}")
    print("Native volatile uint64 loads/stores; identical loop in both layouts")
    print("Both arrays begin at a 64-byte boundary; padded stride = 64 bytes")
    print("Timer includes worker dispatch, barrier release, and completion")
    unpadded_ms, unpadded_counts = measure(1)
    padded_ms, padded_counts = measure(line_bytes // 8)
    print(f"{'Layout':<24} | {'Wall Time (ms)':>17} | Final counts")
    print(f"{'Unpadded Shared':<24} | {unpadded_ms:17.6f} | {unpadded_counts}")
    print(f"{'Padded (64-byte stride)':<24} | {padded_ms:17.6f} | {padded_counts}")
    print(f"Execution ratio (Unpadded / Padded): {unpadded_ms / padded_ms:.6f}x")
    if workers == 1:
        print("Only one physical core was detected; inter-core false sharing cannot be demonstrated.")
    elif unpadded_ms <= padded_ms:
        print("This run did not show an unpadded slowdown; placement and scheduling affect this measurement.")


if __name__ == "__main__":
    main()
