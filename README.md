# CPU parallelism laboratory: tasks 1-9

Each `taskN.py` is executable independently. The corresponding
`taskN_output.txt` is its unedited console output from a full-size run.

**Student ID: 230103059**, inferred from the configured Git email
`230103059@sdu.edu.kz`. Confirm that this is your ID. To change it, edit
`student_id.txt` and regenerate the outputs. An environment override also works:
`STUDENT_ID=your_id python3 task1.py`.

These measurements come from the execution environment, **not your laptop**.
The machine exposes three logical/physical CPUs but the container has a
two-CPU quota. Every output records OS, CPU, seed, timestamp and limits.
Results on a different CPU or under a different quota will change.

## Run

Requires Linux, Python 3.10+, GCC and the packages in `requirements.txt`:

```bash
python3 -m pip install -r requirements.txt
python3 task1.py
python3 run_submission.py
```

The runner executes tasks 1-8 **sequentially** to avoid benchmark interference,
prints their console output and saves the raw output files. Native C helpers
compile automatically into an ignored `.build/` directory. Python's `ctypes`
releases the GIL around these helpers, allowing real parallel CPU execution.
No NumPy/BLAS thread pool implements the measured kernels.

For full reproduction with the required task commits and audit:

```bash
python3 run_submission.py --commit
# After configuring your actual submission repository:
git remote add origin YOUR_REPOSITORY_URL
git push -u origin HEAD
```

`--commit --push` additionally pushes to the configured origin. The runner
stops on any failed task, validation, commit or push. On a repository with
existing submission commits, rerunning makes new commits only for changed files.
Use the instructor's repository and confirm the cutoff date/time zone yourself;
this project never changes or backdates commit timestamps.

## What each task shows

| Task | Work | Measurement |
| --- | --- | --- |
| 1 | 20,000,000 seeded float64 values | Serial sum and population standard deviation; generation and statistics times separately |
| 2 | Same input, workers `[1,2,4,max,max*2]` | Wall time, measured speedup and `1/(0.05+0.95/p)` |
| 3 | 50,000,000 increments per physical-core worker | Adjacent versus 64-byte-aligned padded counters; identical volatile loops |
| 4 | 1,000 prescribed irregular sleeps, 4 workers | Static blocks versus a dynamic chunk-1 queue; first/last completion delta |
| 5 | `y=1.25*x-0.75`, six required sizes | Same native kernel, serial versus a reused all-core pool; sampled crossover brackets |
| 6 | Four workers, 2,000,000 increments each | Lost updates from unprotected compound increments, then per-increment mutex correction |
| 7 | 30,000,000 elements | Copy versus `sin(x)*cos(x)+tan(x)`; speedup and efficiency |
| 8 | 1024 x 1024 matrix product | All-core CPU implementation, non-tiled versus 16 x 16 and 64 x 64 tiles |
| 9 | Git verification | `audit.txt` records the requested log command before the audit commit |

All outputs include correctness checks outside measured regions. CLI size and
repetition overrides exist for smoke checks; the committed outputs use the
assignment defaults. Tasks 2, 5, 7 and 8 print raw timing samples and medians;
task 4 prints each trial. Task 3 and task 6 report one full comparison each.

Timing boundaries are stated in each output. Ready-worker timing in tasks 2,
3, 6, 7 and 8 excludes pool startup/teardown and includes release/wait overhead.
Task 2 includes the final serial reduction. Task 4 includes thread creation and
shutdown. Task 5 measures dispatch and completion using a warmed persistent
pool, excluding its one-time startup. Input generation, compilation and
correctness checks are excluded from kernel benchmarks.

## Interpreting the results

- Floating-point summation is rounded; differently partitioned reductions can
  differ slightly. Sum correctness uses a tight relative tolerance.
- The exact scheduling formula repeats every 20 items and nearly balances
  static 250-item blocks; a large work-stealing improvement is not guaranteed.
- Six array sizes identify a measured **crossover bracket**, not an exact
  integer threshold. The program reports every observed transition and reversal.
- Task 6 uses separate relaxed atomic loads and stores in its unlocked case.
  Their combined read-modify-write is **not atomic**, so updates can be lost.
  Individual accesses are defined by C11 and introduce no lock or ordering
  between workers. This avoids undefined C data races. The corrected case uses
  `pthread_mutex` around the whole increment.
- Bandwidth saturation cannot be proven from time alone on a quota-limited
  virtual CPU. Task 7 reports any observed plateau and its limitations instead
  of inventing a hardware-counter measurement.
- No NVIDIA device is exposed here, so task 8 uses the required CPU alternative.
  Tiled loops also use a more cache-friendly loop order. Runtime differences
  therefore reflect both changes; cache-miss counters are not measured.
- `audit.txt` is generated immediately before its own commit, so it lists the
  preceding history. Verify the audit commit itself with `git log -1`.
