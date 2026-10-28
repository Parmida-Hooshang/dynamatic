#!/usr/bin/env python3
"""
-------------- AI-generated script for running tests
Simulation-only clock-period sweep for the FPL22 buffer placement algorithm.

For each selected integration test and each clock period in
[--start, --end] (step --step), runs set-src -> set-clock-period ->
compile fpl22 -> write-hdl verilog -> simulate (no synthesis) and records the
number of clock cycles. Each trial is written as one row of the CSV:
test, clock_period_ns, status, cycles.

Failure handling (shared with sweep_clock_period.py):
  * Compilation and write-hdl + simulation run in separate Dynamatic sessions
    with --exit-on-failure and their own timeouts. If compilation fails or
    times out, simulation is skipped and the status is recorded.
  * On timeout, the whole process group is killed, including vsim.
  * Any unexpected Python error in a trial is recorded and the sweep goes on.
  * Ctrl-C kills every running Dynamatic/vsim process before exiting.

Usage:
    python3 sim_sweep_clock_period.py [--start 4] [--end 8] [--step 1]
                                      [--workers 2] [--csv results.csv]
                                      [--tests gcd fir ... | --tests test_list.txt]
"""

import argparse
import csv
import os
import shutil
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sweep_clock_period import (DEFAULT_TESTS, DYNAMATIC_DIR, active_lock,
                                active_procs, expand_tests, kill_group,
                                parse_cycles, resolve_test, run_dynamatic)

CSV_FIELDS = ["test", "clock_period_ns", "status", "cycles"]

csv_lock = threading.Lock()


def run_trial(src: Path, period: float, args, log_dir: Path) -> dict:
    test = src.stem
    out_dir = src.parent / "out"
    tag = f"{test}_cp{period:g}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{tag}.log"
    log_file.write_text("")

    # Start from a clean output directory so stale reports are never parsed.
    shutil.rmtree(out_dir, ignore_errors=True)

    setup = [f"set-src {src}", f"set-clock-period {period:g}"]
    stages = [
        ("compile", ["compile --buffer-algorithm fpl22"],
         args.compile_timeout),
        # write-hdl sets the HDL that simulate uses, so they share a session.
        ("sim", ["write-hdl", "simulate"], args.sim_timeout),
    ]

    status = "ok"
    for name, commands, timeout_s in stages:
        result, log = run_dynamatic(setup + commands, timeout_s)
        with open(log_file, "a") as f:
            f.write(log)
        if result == "ok" and name == "sim":
            cycles, sim_pass = parse_cycles(out_dir / "sim" / "report.txt")
            if cycles is None or not sim_pass:
                result = "failed"
        if result != "ok":
            status = f"{name}_{result}"
            break

    cycles, _ = parse_cycles(out_dir / "sim" / "report.txt")

    # Keep the simulation report since the next trial wipes out/.
    report = out_dir / "sim" / "report.txt"
    if report.exists():
        shutil.copy(report, log_dir / f"{tag}_sim_report.txt")

    return {
        "test": test,
        "clock_period_ns": f"{period:g}",
        "status": status,
        "cycles": cycles if cycles is not None else "",
    }


def append_row(csv_path: Path, row: dict):
    with csv_lock:
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)


def sweep_test(src: Path, periods, args):
    test = src.stem
    for period in periods:
        print(f"[{test}] running CP = {period:g} ns ...", flush=True)
        try:
            row = run_trial(src, period, args, args.log_dir)
        except Exception as e:
            traceback.print_exc()
            row = {"test": test, "clock_period_ns": f"{period:g}",
                   "status": f"error: {type(e).__name__}: {e}", "cycles": ""}
        append_row(args.csv, row)
        print(f"[{test}] CP = {period:g} ns -> {row['status']}, "
              f"cycles={row['cycles']}", flush=True)
    return test


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=float, default=4.0, help="first clock period (ns)")
    p.add_argument("--end", type=float, default=8.0, help="last clock period (ns)")
    p.add_argument("--step", type=float, default=1.0, help="period increment (ns)")
    p.add_argument("--workers", type=int, default=2,
                   help="tests run in parallel")
    p.add_argument("--compile-timeout", type=float, default=1800,
                   help="seconds allowed for compilation")
    p.add_argument("--sim-timeout", type=float, default=1800,
                   help="seconds allowed for write-hdl + simulation")
    p.add_argument("--csv", type=Path, default=DYNAMATIC_DIR / "fpl22_sim_E_Buf.csv")
    p.add_argument("--log-dir", type=Path, default=DYNAMATIC_DIR / "fpl22_sim_sweep_logs")
    p.add_argument("--tests", nargs="+", default=DEFAULT_TESTS,
                   help="test names, test directories, .c sources, and/or "
                        "list files (e.g. test_list.txt) with one test per line")
    args = p.parse_args()

    if args.step <= 0 or args.end < args.start:
        p.error("need --step > 0 and --end >= --start")
    periods = []
    n = 0
    while args.start + n * args.step <= args.end + 1e-9:
        periods.append(round(args.start + n * args.step, 3))
        n += 1

    sources = []
    for t in expand_tests(args.tests):
        src = resolve_test(t)
        if src is None:
            p.error(f"no source found for test '{t}'")
        if src not in sources:
            sources.append(src)
    print(f"Running {len(sources)} test(s) at periods "
          f"{', '.join(f'{x:g}' for x in periods)} ns")

    if not args.csv.exists() or args.csv.stat().st_size == 0:
        with open(args.csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    ex = ThreadPoolExecutor(max_workers=args.workers)
    try:
        futures = {ex.submit(sweep_test, src, periods, args): src.stem
                   for src in sources}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                fut.result()
                print(f"[{t}] done")
            except Exception as e:
                print(f"[{t}] raised an exception: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted: killing running Dynamatic/vsim processes ...")
        with active_lock:
            procs = list(active_procs)
        for proc in procs:
            kill_group(proc)
        print(f"Partial results are in {args.csv}")
        os._exit(130)
    ex.shutdown()

    print(f"Results written to {args.csv}")


if __name__ == "__main__":
    main()
