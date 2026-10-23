"""Hardware metadata, reproducible seeding and real native worker threads."""
from __future__ import annotations

import ctypes
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import threading
import time

# Set before any NumPy import so reference checks cannot spawn hidden BLAS pools.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

ROOT = Path(__file__).resolve().parent
_library = None


def student_id() -> str:
    value = os.environ.get("STUDENT_ID")
    if value is None:
        value = (ROOT / "student_id.txt").read_text()
    value = value.strip()
    if not value:
        raise ValueError("Set STUDENT_ID or fill student_id.txt with your Student ID")
    return value


def student_seed() -> int:
    value = student_id()
    return int(value) if value.isdecimal() else int.from_bytes(
        hashlib.sha256(value.encode()).digest()[:8], "little")


def logical_cores() -> int:
    return os.cpu_count() or 1


def physical_cores() -> int:
    import psutil
    return psutil.cpu_count(logical=False) or logical_cores()


def header(task_name: str) -> None:
    import numpy as np
    model = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    print(task_name)
    print(f"Student ID: {student_id()}")
    print(f"Student ID seed: {student_seed()}")
    print(f"Run UTC: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"OS: {platform.platform()}")
    print(f"CPU: {model}")
    print(f"Logical cores (os.cpu_count): {logical_cores()}")
    print(f"Physical cores reported by OS: {physical_cores()}")
    if hasattr(os, "sched_getaffinity"):
        print(f"CPU affinity: {sorted(os.sched_getaffinity(0))}")
    quota = Path("/sys/fs/cgroup/cpu.max")
    if quota.exists():
        amount, period = quota.read_text().split()
        if amount != "max":
            print(f"Container CPU quota: {int(amount) / int(period):.3f} CPU equivalents")
    print(f"Python: {platform.python_version()}; NumPy: {np.__version__}; dtype: float64")
    print("Native kernels: gcc -O3 -std=c11 -shared -fPIC -pthread (no fast-math)")
    print("Timings are actual wall-clock measurements on this environment.")


def native() -> ctypes.CDLL:
    """Build once per source/compiler hash; CDLL calls release Python's GIL."""
    global _library
    if _library is not None:
        return _library
    sources = sorted(ROOT.glob("native_*.c"))
    if not sources:
        raise RuntimeError("No native_*.c kernel source files found")
    compiler = os.environ.get("CC", "gcc")
    version = subprocess.check_output([compiler, "--version"])
    digest = hashlib.sha256(version + b"".join(s.read_bytes() for s in sources)).hexdigest()[:16]
    build = ROOT / ".build"
    build.mkdir(exist_ok=True)
    target = build / f"libkernels-{digest}.so"
    if not target.exists():
        temporary = build / f"kernels-{digest}-{os.getpid()}.so"
        subprocess.run([compiler, "-O3", "-std=c11", "-shared", "-fPIC", "-pthread",
                        *map(str, sources), "-lm", "-o", str(temporary)], check=True)
        temporary.replace(target)
    _library = ctypes.CDLL(str(target))
    return _library


def run_workers(count: int, fn, *, finalize=None):
    """Return (ms, results) for synchronized, genuinely concurrent CDLL work.

    Pool creation, thread startup and teardown are excluded. Timing includes
    releasing the ready workers, the kernel and waiting for all their results.
    Array allocation, compilation and input generation belong outside this call.
    An optional final reduction is executed inside the measured interval.
    """
    ready = threading.Semaphore(0)
    start = threading.Event()

    def execute(index):
        ready.release()
        start.wait()
        return fn(index)

    with ThreadPoolExecutor(max_workers=count) as pool:
        futures = [pool.submit(execute, index) for index in range(count)]
        for _ in range(count):
            ready.acquire()
        begun = time.perf_counter_ns()
        start.set()
        results = [future.result() for future in futures]
        if finalize is not None:
            results = finalize(results)
        elapsed_ms = (time.perf_counter_ns() - begun) / 1_000_000
    return elapsed_ms, results
