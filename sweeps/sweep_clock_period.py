#!/usr/bin/env python3
"""
-------------- AI-generated script for running tests
Clock-period sweep for the FPL22 buffer placement algorithm.

For each selected integration test, runs the full Dynamatic flow
(set-src -> set-clock-period -> compile fpl22 -> write-hdl verilog -> simulate
-> synthesize) starting at START_PERIOD and increasing the period by STEP until
post-place-and-route timing is met. After the first met period, EXTRA_MET_ROUNDS
further periods are tried. Each trial is written as one row of the CSV.

Failure handling:
  * Each trial runs in three stages (compile, write-hdl + simulate, synthesize),
    each in its own Dynamatic process with --exit-on-failure and its own
    timeout. A failed or timed-out stage skips the remaining stages of that
    trial, and its status is recorded in the CSV.
  * On timeout, the whole process group is killed, including vsim/Vivado.
  * At most --synth-jobs Vivado runs happen at the same time, so parallel
    tests do not run the machine out of memory.
  * A test is abandoned after --max-failures consecutive failed trials.
  * Any unexpected Python error in a trial is recorded and the sweep goes on.
  * Ctrl-C kills every running Dynamatic/Vivado process before exiting.
  * Resuming: trials already recorded in the CSV (same test and period) are
    not run again; their recorded rows are reused so the met-timing search
    continues where it stopped. Pass --no-resume to rerun everything.

Usage:
    python3 sweep_clock_period.py [--start 4] [--step 1] [--max-period 20]
                                  [--workers 2] [--csv results.csv]
                                  [--tests gcd fir ... | --tests test_list.txt]

By default, runs the tests in DEFAULT_RUNS at the periods listed there.
"""

import argparse
import csv
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DYNAMATIC_DIR = Path(__file__).resolve().parent.parent
DYNAMATIC_BIN = DYNAMATIC_DIR / "bin" / "dynamatic"
INTEGRATION_DIR = DYNAMATIC_DIR / "integration-test"

# A mix of control-heavy, memory-heavy, floating-point and nested-loop kernels.
DEFAULT_TESTS = [
    "fir",            # simple single loop, integer MAC
    "iir",            # loop-carried dependency
    "gcd",            # data-dependent control flow
    "matvec",         # 2-level nested loop
    "atax",           # multiple loop nests sharing arrays
    "bicg",           # two accumulations in the same loop
    "gaussian",       # triangular loop bounds
    "histogram",      # floating point + LSQ (memory dependencies)
    "sobel",          # stencil with many memory accesses
    "stencil_2d",     # 2D stencil
]

# Tests run by default, each with the clock periods (ns) to try. Entries with
# periods run exactly those periods; entries without any use the met-timing
# search (--start/--step/--max-period).
DEFAULT_RUNS = [
    "atax 4, 5, 6, 7, 8",
"fir 4, 5, 6",
"gaussian 5, 6, 7, 8, 9",
"gcd 6",
"gsum 4, 5, 6, 7, 8, 9, 10",
"iir 4, 5, 6",
"matvec 4, 5, 6",
"single_loop 4, 5, 6",
"while_loop_1 4, 6, 7, 8",
]

EXTRA_MET_ROUNDS = 2

CSV_FIELDS = [
    "test", "clock_period_ns", "status", "sim_pass", "cycles",
    "timing", "slack_ns", "registers", "slices", "slicel", "slicem", "luts",
    "lut_logic", "lut_memory", "f7_muxes", "f8_muxes", "bram_tiles", "dsps",
    "optimality", "elapsed_s",
]

csv_lock = threading.Lock()

# All running Dynamatic invocations, killed with their children on Ctrl-C.
active_procs = set()
active_lock = threading.Lock()


def parse_cycles(sim_report: Path):
    if not sim_report.exists():
        return None, False
    text = sim_report.read_text(errors="ignore")
    m = re.search(r"Latency\s*=\s*(\d+)\s*cycles", text)
    sim_pass = "C and VHDL outputs match" in text
    return (int(m.group(1)) if m else None), sim_pass


def parse_timing(timing_report: Path):
    """Returns ("MET"/"VIOLATED", slack in ns) or (None, None)."""
    if not timing_report.exists():
        return None, None
    text = timing_report.read_text(errors="ignore")
    m = re.search(r"Slack\s*\((MET|VIOLATED)\)\s*:\s*(-?[\d.]+)ns", text)
    if not m:
        return None, None
    return m.group(1), float(m.group(2))


def parse_registers(util_report: Path):
    if not util_report.exists():
        return None
    text = util_report.read_text(errors="ignore")
    m = re.search(r"\|\s*Slice Registers\s*\|\s*(\d+)\s*\|", text)
    return int(m.group(1)) if m else None


# CSV column -> row label in the "Used" column of Vivado's post-route
# utilization report. "registers" is the "Slice Registers" row.
UTILIZATION_ROWS = {
    "registers": "Slice Registers",
    "slices": "Slice",
    "slicel": "SLICEL",
    "slicem": "SLICEM",
    "luts": "Slice LUTs",
    "lut_logic": "LUT as Logic",
    "lut_memory": "LUT as Memory",
    "f7_muxes": "F7 Muxes",
    "f8_muxes": "F8 Muxes",
    "bram_tiles": "Block RAM Tile",
    "dsps": "DSPs",
}


def parse_utilization(util_report: Path) -> dict:
    """Returns {CSV column: used count} for UTILIZATION_ROWS; a value is ""
    when the report or its row is missing. Block RAM tiles can be fractional
    (a RAMB18 counts as half a tile)."""
    text = util_report.read_text(errors="ignore") if util_report.exists() else ""
    values = {}
    for column, label in UTILIZATION_ROWS.items():
        m = re.search(r"^\|\s*" + re.escape(label) + r"\s*\|\s*([\d.]+)\s*\|",
                      text, re.MULTILINE)
        values[column] = (float(m.group(1)) if "." in m.group(1)
                          else int(m.group(1))) if m else ""
    return values


def parse_optimality(log: str):
    """Returns "OPTIMAL" if every MILP solved during compilation reported an
    optimal status, "NON" if at least one did not, or "" if no MILP status was
    found in the log (relies on the TEMP DEBUG print in MILP.h)."""
    statuses = re.findall(r"TEMP DEBUG: MILP \(.*?\) status=(-?\d+) \(OPTIMAL=(-?\d+)",
                          log)
    if not statuses:
        return ""
    return "OPTIMAL" if all(s == opt for s, opt in statuses) else "NON"


def resolve_test(entry: str):
    """Maps a test name (e.g. "fir"), a test directory, or a .c source path to
    the test's C source. Relative paths are tried against the current directory
    and the Dynamatic root. Returns None if no source is found."""
    for base in (Path.cwd(), DYNAMATIC_DIR, INTEGRATION_DIR):
        path = (base / entry).resolve()
        if path.is_dir():
            path = path / f"{path.name}.c"
        if path.is_file() and path.suffix == ".c":
            return path
    return None


def expand_tests(entries):
    """Replaces every entry that is a list file (anything but a .c file) with
    the tests listed in it, one per line; blank lines and #-comments are
    skipped."""
    tests = []
    for entry in entries:
        path = Path(entry)
        if path.is_file() and path.suffix != ".c":
            for line in path.read_text().splitlines():
                line = line.split("#", 1)[0].strip()
                if line:
                    tests.append(line)
        else:
            tests.append(entry)
    return tests


def split_periods(entry: str):
    """Splits "name p1, p2, ..." into ("name", [p1, p2, ...]). Returns
    (entry, None) when the entry lists no periods."""
    parts = entry.split(None, 1)
    if len(parts) < 2:
        return entry, None
    try:
        periods = [float(x) for x in re.split(r"[,\s]+", parts[1].strip()) if x]
    except ValueError:
        return entry, None
    return parts[0], periods or None


def kill_group(proc: subprocess.Popen):
    """Terminates the process group of a Dynamatic invocation (Dynamatic +
    scripts + vsim/Vivado)."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        # Give the group up to 10s to exit before escalating.
        for _ in range(20):
            time.sleep(0.5)
            proc.poll()  # reap the leader so a zombie does not keep the group
            try:
                os.killpg(proc.pid, 0)
            except ProcessLookupError:
                return


def run_dynamatic(commands, timeout_s: float):
    """Runs one Dynamatic session in its own process group. Returns
    (result, log) with result one of "ok", "failed", "timeout"."""
    script = "\n".join(commands + ["exit"]) + "\n"
    proc = subprocess.Popen([str(DYNAMATIC_BIN), "--exit-on-failure"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            errors="ignore", cwd=DYNAMATIC_DIR,
                            start_new_session=True)
    with active_lock:
        active_procs.add(proc)
    try:
        log, _ = proc.communicate(script, timeout=timeout_s)
        result = "ok" if proc.returncode == 0 else "failed"
    except subprocess.TimeoutExpired:
        kill_group(proc)
        log, _ = proc.communicate()
        log += f"\n[SWEEP] stage killed after {timeout_s:.0f}s timeout\n"
        result = "timeout"
    finally:
        with active_lock:
            active_procs.discard(proc)
    return result, log or ""


def run_trial(src: Path, period: float, args, log_dir: Path,
              synth_slots: threading.Semaphore) -> dict:
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
        ("synth", ["synthesize"], args.synth_timeout),
    ]

    start = time.time()
    status = "ok"
    full_log = ""
    for name, commands, timeout_s in stages:
        if name == "synth":
            with synth_slots:
                result, log = run_dynamatic(setup + commands, timeout_s)
        else:
            result, log = run_dynamatic(setup + commands, timeout_s)
        full_log += log
        with open(log_file, "a") as f:
            f.write(log)

        if result == "ok" and name == "sim":
            cycles, sim_pass = parse_cycles(out_dir / "sim" / "report.txt")
            if cycles is None or not sim_pass:
                result = "failed"
        if result != "ok":
            status = f"{name}_{result}"
            break
    elapsed = time.time() - start

    cycles, sim_pass = parse_cycles(out_dir / "sim" / "report.txt")
    timing, slack = parse_timing(out_dir / "synth" / "timing_post_pr.rpt")
    utilization = parse_utilization(out_dir / "synth" / "utilization_post_pr.rpt")
    if status == "ok" and timing is None:
        status = "synth_failed"

    # Keep the reports of this trial since the next trial wipes out/.
    keep = log_dir / tag
    keep.mkdir(exist_ok=True)
    for rel in ["sim/report.txt", "synth/timing_post_pr.rpt",
                "synth/utilization_post_pr.rpt"]:
        f = out_dir / rel
        if f.exists():
            shutil.copy(f, keep / rel.replace("/", "_"))

    return {
        "test": test,
        "clock_period_ns": f"{period:g}",
        "status": status,
        "sim_pass": sim_pass,
        "cycles": cycles if cycles is not None else "",
        "timing": timing or "",
        "slack_ns": slack if slack is not None else "",
        **utilization,
        "optimality": parse_optimality(full_log),
        "elapsed_s": f"{elapsed:.0f}",
    }


def append_row(csv_path: Path, row: dict):
    with csv_lock:
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)


def run_and_record(src: Path, period: float, args, csv_path: Path,
                   log_dir: Path, synth_slots: threading.Semaphore) -> dict:
    test = src.stem
    print(f"[{test}] running CP = {period:g} ns ...", flush=True)
    try:
        row = run_trial(src, period, args, log_dir, synth_slots)
    except Exception as e:
        traceback.print_exc()
        row = {"test": test, "clock_period_ns": f"{period:g}",
               "status": f"error: {type(e).__name__}: {e}"}
    append_row(csv_path, row)
    print(f"[{test}] CP = {period:g} ns -> {row['status']}, "
          f"cycles={row.get('cycles', '')}, timing={row.get('timing', '')}, "
          f"slack={row.get('slack_ns', '')}, slices={row.get('slices', '')}, "
          f"luts={row.get('luts', '')}, regs={row.get('registers', '')}",
          flush=True)
    return row


def load_recorded(csv_path: Path) -> dict:
    """Returns {(test, period string): row} for the trials already in the CSV.
    Malformed rows (e.g. a row cut off by an interrupted run) are ignored."""
    if not csv_path.exists():
        return {}
    recorded = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                period = f"{float(row['clock_period_ns']):g}"
            except (TypeError, ValueError):
                continue
            recorded[(row["test"], period)] = row
    return recorded


def strip_trailing_nuls(csv_path: Path):
    """An interrupted run can leave NUL padding at the end of the CSV; remove
    it so the rows appended next are not glued to it."""
    if not csv_path.exists():
        return
    data = csv_path.read_bytes()
    stripped = data.rstrip(b"\0")
    if stripped != data:
        print(f"Removing {len(data) - len(stripped)} trailing NUL byte(s) "
              f"from {csv_path}")
        csv_path.write_bytes(stripped)


def run_or_reuse(src: Path, period: float, args, csv_path: Path,
                 log_dir: Path, synth_slots: threading.Semaphore,
                 recorded: dict) -> dict:
    """Returns the recorded row of this trial if there is one, otherwise runs
    the trial and records it."""
    row = recorded.get((src.stem, f"{period:g}"))
    if row is not None:
        print(f"[{src.stem}] CP = {period:g} ns already recorded "
              f"({row['status']}, timing={row.get('timing', '')}), skipping",
              flush=True)
        return row
    return run_and_record(src, period, args, csv_path, log_dir, synth_slots)


def sweep_test(src: Path, periods, args, csv_path: Path, log_dir: Path,
               synth_slots: threading.Semaphore, recorded: dict):
    test = src.stem
    if periods:
        # Fixed list of periods: run every one of them, no met-timing search.
        for period in periods:
            run_or_reuse(src, period, args, csv_path, log_dir, synth_slots,
                         recorded)
        return test

    period = args.start
    met_rounds = 0
    failures = 0
    while met_rounds < 1 + EXTRA_MET_ROUNDS:
        if met_rounds == 0 and period > args.max_period:
            print(f"[{test}] never met timing up to {args.max_period} ns, giving up")
            break
        row = run_or_reuse(src, period, args, csv_path, log_dir, synth_slots,
                           recorded)

        failures = failures + 1 if row["status"] != "ok" else 0
        if failures >= args.max_failures:
            print(f"[{test}] {failures} consecutive failed trials, giving up")
            break

        # Once timing has been met, every following trial counts as one of the
        # extra rounds, even if timing is violated again or the trial fails.
        if row.get("timing") == "MET" or met_rounds > 0:
            met_rounds += 1
        period += args.step
    return test


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=float, default=4.0, help="first clock period (ns)")
    p.add_argument("--step", type=float, default=1.0, help="period increment (ns)")
    p.add_argument("--max-period", type=float, default=20.0,
                   help="give up on a test if timing is not met by this period")
    p.add_argument("--workers", type=int, default=2,
                   help="tests run in parallel")
    p.add_argument("--synth-jobs", type=int, default=2,
                   help="maximum number of Vivado runs at the same time")
    p.add_argument("--compile-timeout", type=float, default=1800,
                   help="seconds allowed for compilation")
    p.add_argument("--sim-timeout", type=float, default=1800,
                   help="seconds allowed for write-hdl + simulation")
    p.add_argument("--synth-timeout", type=float, default=3600,
                   help="seconds allowed for synthesis")
    p.add_argument("--max-failures", type=int, default=3,
                   help="abandon a test after this many consecutive failed trials")
    p.add_argument("--csv", type=Path, default=DYNAMATIC_DIR / "fpl22_synth_sweep_E_Buf.csv")
    p.add_argument("--no-resume", action="store_true",
                   help="rerun trials already recorded in the CSV")
    p.add_argument("--log-dir", type=Path, default=DYNAMATIC_DIR / "fpl22_synth_sweep_logs")
    p.add_argument("--tests", nargs="+", default=DEFAULT_RUNS,
                   help="test names, test directories, .c sources, and/or "
                        "list files (e.g. test_list.txt) with one test per line; "
                        "an entry may be followed by the periods to run, e.g. "
                        "\"histogram 9, 10, 13\" (quoted on the command line)")
    args = p.parse_args()

    runs = {}  # source -> periods (None = met-timing search)
    for t in expand_tests(args.tests):
        name, periods = split_periods(t)
        src = resolve_test(name)
        if src is None:
            p.error(f"no source found for test '{name}'")
        if src not in runs:
            runs[src] = periods
    print(f"Running {len(runs)} test(s)")
    for src, periods in runs.items():
        print(f"  {src.stem}: " + (", ".join(f"{x:g}" for x in periods) + " ns"
                                   if periods else "met-timing search"))

    strip_trailing_nuls(args.csv)
    if not args.csv.exists() or args.csv.stat().st_size == 0:
        with open(args.csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()
    else:
        with open(args.csv, newline="") as f:
            header = next(csv.reader(f), [])
        if header != CSV_FIELDS:
            p.error(f"{args.csv} already exists with different columns; "
                    f"pass a new --csv path so rows are not misaligned")

    recorded = {} if args.no_resume else load_recorded(args.csv)
    if recorded:
        print(f"Reusing {len(recorded)} trial(s) already recorded in {args.csv}")

    synth_slots = threading.Semaphore(args.synth_jobs)
    ex = ThreadPoolExecutor(max_workers=args.workers)
    try:
        futures = {ex.submit(sweep_test, src, periods, args, args.csv,
                             args.log_dir, synth_slots, recorded): src.stem
                   for src, periods in runs.items()}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                fut.result()
                print(f"[{t}] done")
            except Exception as e:
                print(f"[{t}] raised an exception: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted: killing running Dynamatic/Vivado processes ...")
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
