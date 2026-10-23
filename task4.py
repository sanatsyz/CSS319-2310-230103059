#!/usr/bin/env python3
"""Compare static contiguous blocks with a dynamic queue for irregular sleeps."""

import argparse
import queue
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from bench_common import header


WORKERS = 4


def benchmark(strategy, items, chunk_size):
    """Return total time and actual worker completion offsets, in milliseconds."""
    work = queue.Queue()
    if strategy == "Dynamic":
        for begin in range(0, items, chunk_size):
            work.put(range(begin, min(begin + chunk_size, items)))
    start_gate = threading.Barrier(WORKERS + 1)

    def worker(index):
        processed = []
        start_gate.wait()
        if strategy == "Static":
            chunks = [range(items * index // WORKERS,
                            items * (index + 1) // WORKERS)]
            for block in chunks:
                for item in block:
                    time.sleep(0.0001 * (item % 20))
                    processed.append(item)
        else:
            while True:
                try:
                    block = work.get_nowait()
                except queue.Empty:
                    break
                for item in block:
                    time.sleep(0.0001 * (item % 20))
                    processed.append(item)
        finished = time.perf_counter_ns()
        return processed, finished

    started = time.perf_counter_ns()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(worker, index) for index in range(WORKERS)]
        start_gate.wait()
        results = [future.result() for future in futures]
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    completed = [item for processed, _ in results for item in processed]
    if sorted(completed) != list(range(items)):
        raise AssertionError("Every item must be processed exactly once")
    finish_ms = [(finished - started) / 1_000_000 for _, finished in results]
    return elapsed_ms, min(finish_ms), max(finish_ms), max(finish_ms) - min(finish_ms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, default=1000)
    parser.add_argument("--chunk", type=int, choices=(1, 5), default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.items < 1 or args.repeats < 1:
        parser.error("--items and --repeats must be positive")

    header("Task 4: scheduling under skew")
    print(f"Items: {args.items}; workers: {WORKERS}; dynamic chunk size: {args.chunk}")
    print("Work per item: sleep(0.0001 * (i % 20)) seconds")
    print("Timer includes worker creation, start barrier, execution and shutdown.")
    print("Queue/block preparation and correctness checks are outside the timer.")
    print("First/last finish are worker completion offsets from the timer start.")
    print("Strategy | Run | Wall time (ms) | First finish (ms) | Last finish (ms) | Delta (ms)")
    observations = {"Static": [], "Dynamic": []}
    for repeat in range(args.repeats):
        order = ("Static", "Dynamic") if repeat % 2 == 0 else ("Dynamic", "Static")
        for strategy in order:
            result = benchmark(strategy, args.items, args.chunk)
            observations[strategy].append(result)
            elapsed, first, last, delta = result
            print(f"{strategy:8} | {repeat + 1:3d} | {elapsed:14.6f} | "
                  f"{first:17.6f} | {last:16.6f} | {delta:10.6f}")
    static_ms = statistics.median(row[0] for row in observations["Static"])
    dynamic_ms = statistics.median(row[0] for row in observations["Dynamic"])
    print(f"Median static wall time (ms): {static_ms:.6f}")
    print(f"Median dynamic wall time (ms): {dynamic_ms:.6f}")
    print(f"Static / dynamic execution ratio: {static_ms / dynamic_ms:.6f}x")
    print("Validation: every item processed exactly once in every run (PASS).")
    print("The repeating 20-item pattern nearly balances equal blocks; a large "
          "dynamic-scheduling speedup is not guaranteed.")


if __name__ == "__main__":
    main()
