#!/usr/bin/env python3
"""Show lost read-modify-write updates and the cost of a mutex per update."""

import argparse
import ctypes

from bench_common import header, native, run_workers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=2_000_000)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    header("Task 6 - Synchronization cost and data loss")
    workers = 4
    expected = workers * args.iterations
    lib = native()
    lib.contention_create_counter.argtypes = []
    lib.contention_create_counter.restype = ctypes.c_void_p
    for name in ("contention_free_counter", "contention_reset_counter"):
        function = getattr(lib, name)
        function.argtypes = [ctypes.c_void_p]
        function.restype = None
    lib.contention_get_counter.argtypes = [ctypes.c_void_p]
    lib.contention_get_counter.restype = ctypes.c_uint64
    for name in ("contention_racy_increment", "contention_locked_increment"):
        function = getattr(lib, name)
        function.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        function.restype = None

    counter = lib.contention_create_counter()
    if not counter:
        raise RuntimeError("Could not allocate counter or initialize mutex")
    try:
        def measure(function):
            lib.contention_reset_counter(counter)
            elapsed_ms, _ = run_workers(workers, lambda _: function(counter, args.iterations))
            return elapsed_ms, lib.contention_get_counter(counter)

        print(f"Parallel native threads: {workers}")
        print(f"Increment attempts per thread: {args.iterations:,}")
        print("Unlocked: separate relaxed load and store, without an atomic read-modify-write")
        print("Individual accesses avoid C undefined behavior; the compound increment remains racy")
        print("Locked: pthread_mutex protects each complete read-modify-write")
        print("Timer includes worker dispatch, barrier release, and completion")
        unlocked_ms, actual = measure(lib.contention_racy_increment)
        locked_ms, corrected = measure(lib.contention_locked_increment)
        if not 0 <= actual <= expected:
            raise RuntimeError(f"Unlocked counter out of range: {actual}")
        if corrected != expected:
            raise RuntimeError(f"Mutex counter mismatch: {corrected} != {expected}")
        lost_percentage = 100.0 * (expected - actual) / expected
        print(f"Expected final value: {expected:,}")
        print(f"Unlocked actual final value: {actual:,}")
        print(f"Updates lost: {expected - actual:,} ({lost_percentage:.6f}%)")
        print(f"Unlocked wall time: {unlocked_ms:.6f} ms")
        print(f"Mutex actual final value: {corrected:,}")
        print(f"Mutex wall time: {locked_ms:.6f} ms")
        print(f"Synchronization time penalty: {locked_ms - unlocked_ms:.6f} ms")
        print(f"Mutex / Unlocked time ratio: {locked_ms / unlocked_ms:.6f}x")
        print(f"Synchronization overhead: {(locked_ms / unlocked_ms - 1) * 100:.6f}%")
        if actual == expected:
            print("No updates were lost in this run; this observation does not make the unlocked increment safe.")
    finally:
        lib.contention_free_counter(counter)


if __name__ == "__main__":
    main()
