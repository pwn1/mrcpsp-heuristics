from __future__ import annotations
"""Run constructive heuristics on MRCPSP instances."""

import csv
import sys
import os
import glob
import time
import multiprocessing

from mm_parser import parse_psplib
from sgs import SGS_SCHEMES
from priority_rules import PRIORITY_RULES, get_priority_fn
from mode_rules import MODE_RULES, CONTEXT_AWARE_RULES, get_mode_fn
from validate import validate_schedule
from justification import justify

JUSTIFY = False


def _run_combo(project, sgs_name, pr_name, mr_name):
    """Build and run one heuristic combination. Applies JUSTIFY if set.
    Returns the resulting Schedule, or None if infeasible."""
    pr_fn = get_priority_fn(pr_name, project, sgs_name, mr_name)
    mr_fn = get_mode_fn(mr_name, project, sgs_name, pr_name)
    is_ca = mr_name in CONTEXT_AWARE_RULES
    schedule = SGS_SCHEMES[sgs_name](project, pr_fn, mr_fn,
                                     mode_is_context_aware=is_ca)
    if schedule is not None and JUSTIFY:
        schedule = justify(project, schedule)
    return schedule


def _iter_combos():
    for sgs_name in SGS_SCHEMES:
        for pr_name in PRIORITY_RULES:
            for mr_name in MODE_RULES:
                yield sgs_name, pr_name, mr_name


def run_all_combinations(filepath: str):
    """Run all heuristic combinations on one instance and print results."""
    project = parse_psplib(filepath)
    print(f"\nInstance: {os.path.basename(filepath)}")
    print(f"{'SGS':<10} {'Priority':<8} {'Mode Rule':<20} {'Makespan':>8}  Status")
    print("-" * 65)

    for sgs_name, pr_name, mr_name in _iter_combos():
        schedule = _run_combo(project, sgs_name, pr_name, mr_name)
        if schedule is None:
            ms_str, status = "-", "INFEASIBLE"
        else:
            errors = validate_schedule(project, schedule)
            ms_str = str(schedule.compute_makespan(project))
            status = f"INVALID ({len(errors)} errors)" if errors else "OK"
        print(f"{sgs_name:<10} {pr_name:<8} {mr_name:<20} {ms_str:>8}  {status}")


def run_best(filepath: str):
    """Run all heuristic combinations on one instance, return the best valid schedule."""
    project = parse_psplib(filepath)
    best_ms, best_schedule, best_combo = None, None, None

    for combo in _iter_combos():
        schedule = _run_combo(project, *combo)
        if schedule is None or validate_schedule(project, schedule):
            continue
        ms = schedule.compute_makespan(project)
        if best_ms is None or ms < best_ms:
            best_ms, best_schedule, best_combo = ms, schedule, combo

    return project, best_schedule, best_combo


def _param_contents(project, schedule, best_combo, source_path):
    """Build Essence Prime .param contents for one instance.

    The schedule is the best feasible heuristic solution found; the horizon
    is set to makespan - 1 so a solver re-using this param must improve on
    it. Solution start/mode assignments are embedded as $-comments (1-based
    indexing to match the rest of the param file)."""
    n = project.num_activities
    ms = schedule.compute_makespan(project)
    sgs_name, pr_name, mr_name = best_combo

    lines = ["language ESSENCE' 1.0"]
    lines.append(f"$ Generated from file {source_path}")
    lines.append(f"$ Best heuristic solution: makespan = {ms}, "
                 f"combo = {sgs_name}/{pr_name}/{mr_name}")
    lines.append("$ activity,mode,start (1-based indexing)")
    for i in range(n):
        m = schedule.mode_assignments[i] + 1
        s = schedule.start_times[i]
        lines.append(f"$ {i + 1},{m},{s}")

    lines.append(f"letting jobs = {n}")
    lines.append(f"letting horizon = {ms - 1}")
    lines.append(f"letting resourcesRenew = {project.num_renewable}")
    lines.append(f"letting resourcesNonrenew = {project.num_nonrenewable}")

    succs = [[s + 1 for s in project.activities[i].successors] for i in range(n)]
    lines.append(f"letting successors = {succs}")

    durs = [[m.duration for m in project.activities[i].modes] for i in range(n)]
    lines.append(f"letting durations = {durs}")

    usage = [
        [list(m.renewable_demands) + list(m.nonrenewable_demands)
         for m in project.activities[i].modes]
        for i in range(n)
    ]
    lines.append(f"letting resourceUsage = {usage}")

    caps = list(project.renewable_capacities) + list(project.nonrenewable_capacities)
    lines.append(f"letting resourceLimits = {caps}")

    return "\n".join(lines) + "\n"


def _gen_param_worker(filepath: str) -> tuple[str, int | None, str]:
    """Pool worker: run all combos, write <filepath>.param. Returns
    (filepath, makespan_or_None, status)."""
    try:
        project, schedule, combo = run_best(filepath)
    except Exception as e:
        return (filepath, None, f"ERROR: {e}")
    if schedule is None:
        return (filepath, None, "no feasible solution")
    out_path = filepath + ".param"
    with open(out_path, "w") as f:
        f.write(_param_contents(project, schedule, combo, filepath))
    return (filepath, schedule.compute_makespan(project), "OK")


def generate_param(path: str, workers: int = None):
    """Generate .param file(s) for one instance or every .mm file in a dir."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.mm")))
        if not files:
            print(f"No .mm files in {path}")
            return
    else:
        files = [path]

    if workers is None:
        workers = max(1, (os.cpu_count() or 1) - 1)

    t0 = time.time()
    if len(files) == 1:
        results = [_gen_param_worker(files[0])]
    else:
        with multiprocessing.Pool(workers) as pool:
            results = pool.map(_gen_param_worker, files)
    elapsed = time.time() - t0

    ok = sum(1 for _, _, s in results if s == "OK")
    print(f"Wrote {ok}/{len(files)} .param files in {elapsed:.2f}s")
    for fp, _, status in results:
        if status != "OK":
            print(f"  {os.path.basename(fp)}: {status}")


def _run_instance(filepath: str) -> tuple[str, dict[tuple, int | None]]:
    """Run all heuristic combinations on one instance. Returns
    (filename, {(sgs, priority, mode): makespan_or_None})."""
    project = parse_psplib(filepath)
    results = {}
    for combo in _iter_combos():
        schedule = _run_combo(project, *combo)
        results[combo] = (schedule.compute_makespan(project)
                          if schedule is not None else None)
    return os.path.basename(filepath), results


def _save_instance_results(all_results, output_path):
    """Save per-instance results to CSV for later analysis."""
    combo_keys = list(all_results[0][1].keys())
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance", "sgs", "priority", "mode", "makespan"])
        for fname, instance_results in all_results:
            for key in combo_keys:
                ms = instance_results[key]
                w.writerow([fname, *key, ms if ms is not None else ""])
    print(f"Per-instance results saved to {output_path}")


def run_benchmark(directory: str, workers: int = None):
    """Run all combinations across all instances in a directory."""
    files = sorted(glob.glob(os.path.join(directory, "*.mm")))
    if not files:
        print(f"No .mm files found in {directory}")
        return

    if workers is None:
        workers = max(1, (os.cpu_count() or 1) - 1)

    n_combos = len(SGS_SCHEMES) * len(PRIORITY_RULES) * len(MODE_RULES)
    print(f"Found {len(files)} instances in {directory}")
    print(f"Running {n_combos} combinations with {workers} workers")

    t0 = time.time()
    with multiprocessing.Pool(workers) as pool:
        all_results = pool.map(_run_instance, files)
    elapsed = time.time() - t0

    _save_instance_results(all_results,
                           os.path.join(directory, "..", "benchmark_results.csv"))

    # Aggregate: (sgs, priority, mode) -> list of makespans (one per instance)
    combo_keys = list(all_results[0][1].keys())
    results = {key: [] for key in combo_keys}
    for _, instance_results in all_results:
        for key in combo_keys:
            results[key].append(instance_results[key])

    # Summary
    print(f"\nCompleted in {elapsed:.2f}s")
    print(f"\n{'SGS':<10} {'Priority':<8} {'Mode Rule':<20} {'Avg MS':>8} {'Feasible':>8}/{len(files)}")
    print("-" * 65)

    ranked = []
    for key, makespans in results.items():
        feasible = [m for m in makespans if m is not None]
        if feasible:
            avg = sum(feasible) / len(feasible)
            ranked.append((key, avg, len(feasible)))

    ranked.sort(key=lambda x: x[1])
    for key, avg, count in ranked:
        sgs_name, priority_name, mode_name = key
        print(f"{sgs_name:<10} {priority_name:<8} {mode_name:<20} {avg:>8.1f} {count:>8}/{len(files)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py <instance.mm>              # run all combos on one instance")
        print("  python main.py --best <instance.mm>       # run all combos, print best")
        print("  python main.py --benchmark <directory> [--workers N]  # run on all instances")
        print("  python main.py --param <instance.mm>      # write Essence Prime .param file")
        print("  python main.py --param <directory> [--workers N]     # batch .param generation")
        print("  Add --justify to any command to enable double justification")
        sys.exit(1)

    if "--justify" in sys.argv:
        JUSTIFY = True
        sys.argv = [a for a in sys.argv if a != "--justify"]

    if sys.argv[1] == "--benchmark" and len(sys.argv) >= 3:
        directory = sys.argv[2]
        workers = None
        for i, arg in enumerate(sys.argv):
            if arg == "--workers" and i + 1 < len(sys.argv):
                workers = int(sys.argv[i + 1])
        run_benchmark(directory, workers=workers)
    elif sys.argv[1] == "--param" and len(sys.argv) >= 3:
        path = sys.argv[2]
        workers = None
        for i, arg in enumerate(sys.argv):
            if arg == "--workers" and i + 1 < len(sys.argv):
                workers = int(sys.argv[i + 1])
        generate_param(path, workers=workers)
    elif sys.argv[1] == "--best" and len(sys.argv) >= 3:
        filepath = sys.argv[2]
        project, schedule, combo = run_best(filepath)
        if schedule is None:
            print(f"No feasible solution found for {os.path.basename(filepath)}")
            sys.exit(1)
        print(f"Instance:  {os.path.basename(filepath)}")
        print(f"Makespan:  {schedule.compute_makespan(project)}")
        print(f"Best combo: {' / '.join(combo)}")
        sol_path = os.path.splitext(filepath)[0] + "_solution.csv"
        with open(sol_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["activity", "mode", "start"])
            for i in range(project.num_activities):
                w.writerow([i, schedule.mode_assignments[i], schedule.start_times[i]])
        print(f"Solution:  {sol_path}")
    else:
        run_all_combinations(sys.argv[1])
