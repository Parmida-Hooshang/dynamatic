#!/usr/bin/env python3
"""
-------------- AI-generated script for running tests
Measures how long Gurobi takes to solve the FPL22 buffer placement MILPs.

For each selected integration test, compiles it --runs times (default 2) with
set-clock-period --period (default 4 ns) and compile --buffer-algorithm fpl22
--all-lazy, and records the solver time of every MILP solved during buffer
placement. Each compilation is written as one row of the CSV.

The solver times come from the "TEMP DEBUG: MILP (...) solve time=X s" lines
printed by MILP<Result>::optimize in include/dynamatic/Support/MILP.h (wall
time of model->optimize()), so dynamatic-opt must be built with that print.

Tests run one after the other by default: running several at the same time
makes the solves compete for CPU and inflates the measured times.

Usage:
    python3 milp_runtime.py [--period 4] [--runs 2] [--csv results.csv]
                            [--tests gcd fir ... | --tests test_list.txt]
"""

import argparse
import csv
import os
import re
import shutil
import time
import traceback
from pathlib import Path

from sweep_clock_period import (DEFAULT_TESTS, DYNAMATIC_DIR, active_lock,
                                active_procs, expand_tests, kill_group,
                                parse_optimality, resolve_test, run_dynamatic)

CSV_FIELDS = ["test", "run", "clock_period_ns", "status", "num_milps",
              "gurobi_time_s", "per_milp_s", "optimality", "compile_time_s"]


def parse_solve_times(log: str):
    """Returns [(model name, seconds)] for every MILP solved, in order."""
    return [(Path(name).name or "unnamed", float(t)) for name, t in re.findall(
        r"MILP \((.*?)\) solve time=([\d.eE+-]+) s", log)]


def run_compile(src: Path, period: float, run: int, args) -> dict:
    test = src.stem
    out_dir = src.parent / "out"
    args.log_dir.mkdir(parents=True, exist_ok=True)

    # Start from a clean output directory.
    shutil.rmtree(out_dir, ignore_errors=True)

    start = time.time()
    result, log = run_dynamatic(
        [f"set-src {src}", f"set-clock-period {period:g}",
         "compile --buffer-algorithm fpl22"],
        args.compile_timeout)
    elapsed = time.time() - start
    (args.log_dir / f"{test}_cp{period:g}_run{run}.log").write_text(log)

    times = parse_solve_times(log)
    status = "ok" if result == "ok" else f"compile_{result}"
    if status == "ok" and not times:
        status = "no_milp_time_found"

    return {
        "test": test,
        "run": run,
        "clock_period_ns": f"{period:g}",
        "status": status,
        "num_milps": len(times),
        "gurobi_time_s": f"{sum(t for _, t in times):.3f}" if times else "",
        "per_milp_s": "; ".join(f"{name}={t:.3f}" for name, t in times),
        "optimality": parse_optimality(log),
        "compile_time_s": f"{elapsed:.1f}",
    }



def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--period", type=float, default=4.0, help="clock period (ns)")
    p.add_argument("--runs", type=int, default=2, help="compilations per test")
    p.add_argument("--compile-timeout", type=float, default=3600,
                   help="seconds allowed for one compilation")
    p.add_argument("--csv", type=Path, default=DYNAMATIC_DIR / "fpl22_milp_runtime_E_Buf.csv")
    p.add_argument("--log-dir", type=Path, default=DYNAMATIC_DIR / "fpl22_milp_runtime_logs")
    p.add_argument("--tests", nargs="+", default=DEFAULT_TESTS,
                   help="test names, test directories, .c sources, and/or "
                        "list files (e.g. test_list.txt) with one test per line")
    args = p.parse_args()

    sources = []
    for t in expand_tests(args.tests):
        src = resolve_test(t)
        if src is None:
            p.error(f"no source found for test '{t}'")
        if src not in sources:
            sources.append(src)
    print(f"Compiling {len(sources)} test(s), {args.runs} run(s) each, "
          f"CP = {args.period:g} ns")

    if not args.csv.exists() or args.csv.stat().st_size == 0:
        with open(args.csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    try:
        for src in sources:
            for run in range(1, args.runs + 1):
                print(f"[{src.stem}] run {run} ...", flush=True)
                try:
                    row = run_compile(src, args.period, run, args)
                except Exception as e:
                    traceback.print_exc()
                    row = {"test": src.stem, "run": run,
                           "clock_period_ns": f"{args.period:g}",
                           "status": f"error: {type(e).__name__}: {e}"}
                with open(args.csv, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)
                print(f"[{src.stem}] run {run} -> {row['status']}, "
                      f"gurobi={row.get('gurobi_time_s', '')} s "
                      f"({row.get('per_milp_s', '')}), "
                      f"compile={row.get('compile_time_s', '')} s", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted: killing running Dynamatic processes ...")
        with active_lock:
            procs = list(active_procs)
        for proc in procs:
            kill_group(proc)
        print(f"Partial results are in {args.csv}")
        os._exit(130)

    print(f"Results written to {args.csv}")


if __name__ == "__main__":
    main()
