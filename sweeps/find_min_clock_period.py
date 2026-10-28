#!/usr/bin/env python3
"""
-------------- AI-generated script for running tests
Binary search for the smallest clock period at which one integration test
meets timing with the FPL22 buffer placement algorithm.

Each probe runs the full Dynamatic flow (set-src -> set-clock-period ->
compile fpl22 -> write-hdl verilog -> simulate -> synthesize) and reads the
post-place-and-route timing report.

1. Bracket: starting at --start, halve the period until timing is violated
   and/or double it until timing is met.
2. Bisect the bracket until it is narrower than --precision.

Note: FPL22 buffers the circuit for the target period, so the circuit changes
from one probe to the next and "met" is not guaranteed to be monotonic in the
period. The result is the smallest MET period found by the search.

Usage:
    python3 find_min_clock_period.py integration-test/histogram
    python3 find_min_clock_period.py integration-test/memory/test_memory_16/test_memory_16.c --start 6
"""

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from sweep_clock_period import (DYNAMATIC_BIN, DYNAMATIC_DIR, parse_cycles,
                                parse_optimality, parse_registers,
                                parse_timing)

TRIAL_TIMEOUT_S = 3600


def resolve_source(path: Path) -> Path:
    """Accepts either the test directory or the C source inside it."""
    path = path.resolve()
    if path.is_dir():
        path = path / f"{path.name}.c"
    if not path.is_file() or path.suffix != ".c":
        sys.exit(f"error: no C source found at {path}")
    return path


def parse_data_path_delay(timing_report: Path):
    """Critical path length (ns) of the worst path in the timing report."""
    if not timing_report.exists():
        return None
    text = timing_report.read_text(errors="ignore")
    m = re.search(r"Data Path Delay:\s*([\d.]+)ns", text)
    return float(m.group(1)) if m else None


def run_probe(src: Path, period: float, log_dir: Path) -> dict:
    out_dir = src.parent / "out"
    # Start from a clean output directory so stale reports are never parsed.
    shutil.rmtree(out_dir, ignore_errors=True)

    commands = "\n".join([
        f"set-src {src}",
        f"set-clock-period {period:.3f}",
        "compile --buffer-algorithm fpl22",
        "write-hdl",
        "simulate",
        "synthesize",
        "exit",
    ]) + "\n"

    start = time.time()
    try:
        proc = subprocess.run([str(DYNAMATIC_BIN)], input=commands, text=True,
                              capture_output=True, cwd=DYNAMATIC_DIR,
                              timeout=TRIAL_TIMEOUT_S)
        log = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as e:
        log = e.stdout.decode(errors="ignore") if isinstance(e.stdout, bytes) \
            else (e.stdout or "")
    elapsed = time.time() - start

    tag = f"cp{period:.3f}"
    (log_dir / f"{tag}.log").write_text(log)
    # Keep the reports of this probe since the next probe wipes out/.
    keep = log_dir / tag
    keep.mkdir(exist_ok=True)
    for rel in ["sim/report.txt", "synth/timing_post_pr.rpt",
                "synth/utilization_post_pr.rpt"]:
        f = out_dir / rel
        if f.exists():
            shutil.copy(f, keep / rel.replace("/", "_"))

    timing_rpt = out_dir / "synth" / "timing_post_pr.rpt"
    cycles, sim_pass = parse_cycles(out_dir / "sim" / "report.txt")
    timing, slack = parse_timing(timing_rpt)
    return {
        "period": period,
        "timing": timing,
        "slack": slack,
        "critical_path": parse_data_path_delay(timing_rpt),
        "cycles": cycles,
        "sim_pass": sim_pass,
        "registers": parse_registers(out_dir / "synth" / "utilization_post_pr.rpt"),
        "optimality": parse_optimality(log),
        "elapsed": elapsed,
    }


def fmt(v, spec=""):
    return "-" if v is None or v == "" else format(v, spec)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("test", type=Path,
                   help="integration test directory or its .c source")
    p.add_argument("--start", type=float, default=4.0,
                   help="first clock period to probe (ns)")
    p.add_argument("--precision", type=float, default=0.001,
                   help="stop when the search interval is this narrow (ns)")
    p.add_argument("--min-period", type=float, default=0.5,
                   help="lower limit when searching for a violated period (ns)")
    p.add_argument("--max-period", type=float, default=64.0,
                   help="upper limit when searching for a met period (ns)")
    p.add_argument("--log-dir", type=Path, default=None,
                   help="where to keep per-probe logs/reports "
                        "(default: min_cp_logs/<test>)")
    args = p.parse_args()

    src = resolve_source(args.test)
    test = src.stem
    log_dir = args.log_dir or DYNAMATIC_DIR / "min_cp_logs" / test
    log_dir.mkdir(parents=True, exist_ok=True)

    probes = {}

    def probe(period: float) -> dict:
        period = round(period, 3)
        if period not in probes:
            print(f"[{test}] probing CP = {period:.3f} ns ...", flush=True)
            r = run_probe(src, period, log_dir)
            probes[period] = r
            print(f"[{test}]   {fmt(r['timing'])}, slack={fmt(r['slack'])} ns, "
                  f"critical path={fmt(r['critical_path'])} ns, "
                  f"cycles={fmt(r['cycles'])}, optimality={fmt(r['optimality'])}, "
                  f"{r['elapsed']:.0f}s", flush=True)
            if r["timing"] is None:
                sys.exit(f"error: no timing result at CP = {period:.3f} ns "
                         f"(see {log_dir / f'cp{period:.3f}.log'})")
        return probes[period]

    if not args.min_period <= args.start <= args.max_period:
        p.error("--start must be within [--min-period, --max-period]")

    # 1. Bracket: lo is VIOLATED, hi is MET. Halving/doubling is clamped to
    # [--min-period, --max-period].
    lo = hi = None
    if probe(args.start)["timing"] == "MET":
        hi = args.start
        while lo is None and hi > args.min_period:
            cand = max(hi / 2, args.min_period)
            if probe(cand)["timing"] == "MET":
                hi = cand
            else:
                lo = cand
        if lo is None:
            print(f"[{test}] timing met at --min-period = {hi:.3f} ns; "
                  f"not searching below it")
    else:
        lo = args.start
        while hi is None:
            if lo >= args.max_period:
                sys.exit(f"error: timing still violated at --max-period = "
                         f"{lo:.3f} ns; raise --max-period to keep searching")
            cand = min(lo * 2, args.max_period)
            if probe(cand)["timing"] == "MET":
                hi = cand
            else:
                lo = cand

    # 2. Bisect.
    while lo is not None and hi - lo > args.precision:
        mid = round((lo + hi) / 2, 3)
        if mid in (round(lo, 3), round(hi, 3)):
            break
        if probe(mid)["timing"] == "MET":
            hi = mid
        else:
            lo = mid

    best = min((r for r in probes.values() if r["timing"] == "MET"),
               key=lambda r: r["period"])

    print("\nAll probes:")
    print(f"{'CP (ns)':>9} {'timing':>9} {'slack':>8} {'crit path':>10} "
          f"{'cycles':>8} {'regs':>7} {'optimality':>10}")
    for r in sorted(probes.values(), key=lambda r: r["period"]):
        print(f"{r['period']:>9.3f} {fmt(r['timing']):>9} {fmt(r['slack']):>8} "
              f"{fmt(r['critical_path']):>10} {fmt(r['cycles']):>8} "
              f"{fmt(r['registers']):>7} {fmt(r['optimality']):>10}")

    print(f"\nSmallest MET clock period for {test}: {best['period']:.3f} ns")
    print(f"  clock cycles        : {fmt(best['cycles'])}"
          + ("" if best["sim_pass"] else "  (WARNING: simulation did not pass)"))
    print(f"  critical path delay : {fmt(best['critical_path'])} ns")
    print(f"  slack               : {fmt(best['slack'])} ns")
    print(f"  MILP optimality     : {fmt(best['optimality'])}")
    print(f"Per-probe logs and reports: {log_dir}")


if __name__ == "__main__":
    main()
