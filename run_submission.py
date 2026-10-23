"""Run full benchmarks sequentially and optionally commit/push the submission."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
TASKS = [
    ("feat: task 1 baseline execution", ["task1.py", "bench_common.py", "native_numeric.c",
        "student_id.txt", "requirements.txt", ".gitignore", "README.md", "run_submission.py"]),
    ("feat: task 2 amdahl scaling", ["task2.py"]),
    ("feat: task 3 false sharing profile", ["task3.py", "native_contention.c"]),
    ("feat: task 4 scheduling under skew", ["task4.py"]),
    ("feat: task 5 overhead crossover", ["task5.py", "native_vector.c"]),
    ("feat: task 6 race condition forensics", ["task6.py"]),
    ("feat: task 7 memory bandwidth saturation", ["task7.py"]),
    ("feat: task 8 compute geometry", ["task8.py", "native_geometry.c"]),
]


def git(*args, **kwargs):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, **kwargs)


def commit(message, paths):
    git("add", "--", *paths)
    changed = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT)
    if changed.returncode == 1:
        git("commit", "-m", message)
    elif changed.returncode != 0:
        raise RuntimeError("Could not inspect staged changes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", action="store_true", help="make the nine required commits")
    parser.add_argument("--push", action="store_true", help="push to origin after committing")
    args = parser.parse_args()
    if args.push and not args.commit:
        parser.error("--push requires --commit")
    if args.commit:
        subprocess.run(["git", "init", "-b", "main"], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL)
        if subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=ROOT).strip():
            raise RuntimeError("Commit or unstage existing staged changes before using --commit")

    environment = dict(os.environ, PYTHONUNBUFFERED="1")
    for index, (message, paths) in enumerate(TASKS, 1):
        output = ROOT / f"task{index}_output.txt"
        with output.open("w") as handle:
            process = subprocess.Popen([sys.executable, f"task{index}.py"], cwd=ROOT,
                env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
            returncode = process.wait()
        if returncode:
            raise RuntimeError(f"Task {index} failed; see {output}")
        if args.commit:
            commit(message, [*paths, output.name])

    if args.commit:
        tracked = set(subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines())
        for index in range(1, 9):
            for name in (f"task{index}.py", f"task{index}_output.txt"):
                if name not in tracked or not (ROOT / name).stat().st_size:
                    raise RuntimeError(f"Missing committed deliverable: {name}")
        with (ROOT / "audit.txt").open("w") as handle:
            git("log", '--pretty=format:%h - %cd : %s', "--date=iso", stdout=handle)
        commit("chore: final submission audit", ["audit.txt"])
        if args.push:
            git("push", "-u", "origin", "HEAD")


if __name__ == "__main__":
    main()
