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
from time_window_pruning import time_window_prunable, top_k_longest_paths
from lower_bounds import compute_lower_bound

JUSTIFY = False
PRUNE_MODES = False


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


def _param_contents(project, schedule, best_combo, source_path,
                    kept_modes=None, near_critical_paths=None):
    """Build Essence Prime .param contents for one instance.

    The schedule is the best feasible heuristic solution found; the horizon
    is set to makespan - 1 so a solver re-using this param must improve on
    it. Solution start/mode assignments are embedded as $-comments (1-based
    indexing to match the rest of the param file).

    If kept_modes is provided, it is a list (per activity) of original mode
    indices to retain (in original order). Pruned modes are dropped and
    surviving modes are renumbered 1..k in the output."""
    n = project.num_activities
    ms = schedule.compute_makespan(project)
    lb = compute_lower_bound(project)
    assert lb <= ms, f"unsound LB: lb={lb} > heuristic UB={ms}"
    sgs_name, pr_name, mr_name = best_combo

    if kept_modes is None:
        kept_modes = [list(range(len(act.modes))) for act in project.activities]
    remap = [{orig: new for new, orig in enumerate(keeps)} for keeps in kept_modes]
    total_pruned = sum(len(act.modes) - len(k) for act, k in zip(project.activities, kept_modes))

    lines = ["language ESSENCE' 1.0"]
    lines.append(f"$ Generated from file {source_path}")
    lines.append(f"$ Best heuristic solution: makespan = {ms}, "
                 f"combo = {sgs_name}/{pr_name}/{mr_name}")
    if total_pruned:
        lines.append(f"$ Pruned {total_pruned} mode(s) by time-window test "
                     f"(horizon = {ms})")
    lines.append("$ activity,mode,start (1-based indexing)")
    for i in range(n):
        orig_m = schedule.mode_assignments[i]
        m = remap[i][orig_m] + 1
        s = schedule.start_times[i]
        lines.append(f"$ {i + 1},{m},{s}")

    lines.append(f"letting jobs = {n}")
    lines.append(f"letting lowerBound = {lb}")
    lines.append(f"letting horizon = {ms - 1}")
    lines.append(f"letting resourcesRenew = {project.num_renewable}")
    lines.append(f"letting resourcesNonrenew = {project.num_nonrenewable}")

    succs = [[s + 1 for s in project.activities[i].successors] for i in range(n)]
    lines.append(f"letting successors = {succs}")

    durs = [[project.activities[i].modes[m].duration for m in kept_modes[i]]
            for i in range(n)]
    lines.append(f"letting durations = {durs}")

    usage = [
        [list(project.activities[i].modes[m].renewable_demands)
         + list(project.activities[i].modes[m].nonrenewable_demands)
         for m in kept_modes[i]]
        for i in range(n)
    ]
    lines.append(f"letting resourceUsage = {usage}")

    caps = list(project.renewable_capacities) + list(project.nonrenewable_capacities)
    lines.append(f"letting resourceLimits = {caps}")

    if near_critical_paths is not None:
        lengths = [length for length, _ in near_critical_paths]
        lines.append(f"$ Near-critical path lengths (shortest-mode sums): {lengths}")
        paths_1based = [[j + 1 for j in path] for _, path in near_critical_paths]
        lines.append(f"letting nearCriticalPaths = {paths_1based}")

    return "\n".join(lines) + "\n"


def _gen_param_worker(filepath: str) -> tuple[str, int | None, str]:
    """Pool worker: run all combos, write <filepath>.param. Returns
    (filepath, makespan_or_None, status). If PRUNE_MODES is set, drop modes
    that the time-window test (horizon = best makespan) proves infeasible."""
    try:
        project, schedule, combo = run_best(filepath)
    except Exception as e:
        return (filepath, None, f"ERROR: {e}")
    if schedule is None:
        return (filepath, None, "no feasible solution")
    kept_modes = None
    near_critical_paths = None
    if PRUNE_MODES:
        ms = schedule.compute_makespan(project)
        prunable = time_window_prunable(project, ms)
        kept_modes = [
            [i for i in range(len(act.modes)) if i not in set(prunable[j])]
            for j, act in enumerate(project.activities)
        ]
        shortest = [min(m.duration for m in act.modes) for act in project.activities]
        near_critical_paths = top_k_longest_paths(project, shortest, 5)
    out_path = filepath + ".param"
    with open(out_path, "w") as f:
        f.write(_param_contents(project, schedule, combo, filepath,
                                kept_modes, near_critical_paths))
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


def _load_ub_per_instance(results_csv: str) -> dict[str, int]:
    """Read benchmark_results.csv and return {instance: min_makespan}."""
    ubs: dict[str, int] = {}
    with open(results_csv) as f:
        r = csv.reader(f)
        next(r)  # header
        for row in r:
            inst, _, _, _, ms = row
            if not ms:
                continue
            ms_int = int(ms)
            cur = ubs.get(inst)
            if cur is None or ms_int < cur:
                ubs[inst] = ms_int
    return ubs


def _twp_worker(args):
    filepath, ub = args
    project = parse_psplib(filepath)
    pruned = time_window_prunable(project, ub)
    total = sum(len(act.modes) for act in project.activities)
    n_pruned = sum(len(r) for r in pruned)
    return os.path.basename(filepath), total, n_pruned, pruned, ub


def scan_time_window(directory: str, results_csv: str, workers: int = None):
    """Scan instances, prune modes by time-window test using UB from CSV."""
    files = sorted(glob.glob(os.path.join(directory, "*.mm")))
    if not files:
        print(f"No .mm files found in {directory}")
        return
    ubs = _load_ub_per_instance(results_csv)
    missing = [f for f in files if os.path.basename(f) not in ubs]
    if missing:
        print(f"Warning: {len(missing)} instances have no UB in {results_csv}; skipping")
        files = [f for f in files if os.path.basename(f) in ubs]

    if workers is None:
        workers = max(1, (os.cpu_count() or 1) - 1)

    args = [(f, ubs[os.path.basename(f)]) for f in files]
    t0 = time.time()
    if len(args) == 1:
        results = [_twp_worker(args[0])]
    else:
        with multiprocessing.Pool(workers) as pool:
            results = pool.map(_twp_worker, args)
    elapsed = time.time() - t0

    total_modes = sum(t for _, t, _, _, _ in results)
    total_pruned = sum(p for _, _, p, _, _ in results)
    hits = [(name, p, total, pr, ub)
            for name, total, p, pr, ub in results if p > 0]

    print(f"Scanned {len(files)} instances in {elapsed:.2f}s")
    print(f"Instances with prunable modes: "
          f"{len(hits)}/{len(files)} "
          f"({100.0 * len(hits) / len(files):.1f}%)")
    print(f"Removable mode entries: {total_pruned}/{total_modes} "
          f"({100.0 * total_pruned / total_modes:.2f}% of all task-mode pairs)")
    print()
    if hits:
        print(f"{'Instance':<20} {'UB':>4} {'Pruned':>6} {'Total':>6}  Per-activity (act:[modes])")
        print("-" * 75)
        for name, p, total, pr, ub in hits:
            per_act = ", ".join(f"{i}:{r}" for i, r in enumerate(pr) if r)
            print(f"{name:<20} {ub:>4} {p:>6} {total:>6}  {per_act}")


def check_lower_bounds(directory: str,
                       best_known_csv: str = "data/mmlib50_best_known.csv",
                       output_csv: str = "data/lb_check.csv"):
    """Compute compute_lower_bound on every .mm in directory, validate
    soundness against published lower bounds and best-known makespans, and
    report the quality distribution."""
    from lower_bounds import compute_lower_bound, critical_path_lb, resource_workload_lb

    files = sorted(glob.glob(os.path.join(directory, "*.mm")))
    if not files:
        print(f"No .mm files found in {directory}")
        return

    published = {}
    with open(best_known_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            published[row["instance"]] = (int(row["lower_bound"]),
                                          int(row["best_known"]))

    rows = []
    violations = []
    beats_pub_lb = []
    cp_binding = wl_binding = tied = 0
    t0 = time.time()
    for fp in files:
        name = os.path.basename(fp)
        project = parse_psplib(fp)
        lb0 = critical_path_lb(project)
        lb1 = resource_workload_lb(project)
        lb = max(lb0, lb1)
        if lb1 > lb0:
            wl_binding += 1
        elif lb0 > lb1:
            cp_binding += 1
        else:
            tied += 1
        pub_lb, pub_bk = published.get(name, (None, None))
        rows.append((name, lb0, lb1, lb, pub_lb, pub_bk))
        if pub_bk is not None and lb > pub_bk:
            violations.append((name, lb, pub_bk))
        if pub_lb is not None and lb > pub_lb:
            beats_pub_lb.append((name, lb, pub_lb))
    elapsed = time.time() - t0

    with open(output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance", "lb0_critical_path", "lb1_workload",
                    "lb_combined", "published_lb", "published_best_known"])
        for r in rows:
            w.writerow(r)

    print(f"Computed LB for {len(files)} instances in {elapsed:.2f}s "
          f"-> {output_csv}")
    print(f"Binding bound: LB0(critical path) on {cp_binding}, "
          f"LB1(workload) on {wl_binding}, tied on {tied}")

    if violations:
        print(f"\nSOUNDNESS VIOLATIONS vs best-known UB ({len(violations)}):")
        for name, lb, ref in violations[:20]:
            print(f"  {name}: lb={lb} > best_known={ref}")
        sys.exit(1)
    print("Soundness: 0 violations against best-known UB.")

    matched = [r for r in rows if r[4] is not None]
    if matched:
        gaps_lb = sorted(r[4] - r[3] for r in matched)
        gaps_bk = sorted(r[5] - r[3] for r in matched)
        equal_to_pub = sum(1 for g in gaps_lb if g == 0)
        weaker = sum(1 for g in gaps_lb if g > 0)
        tighter = sum(1 for g in gaps_lb if g < 0)
        n = len(gaps_lb)

        def stats(xs):
            return (sum(xs) / n, xs[n // 2], xs[int(0.9 * n)], xs[-1])

        m_lb, med_lb, p90_lb, max_lb = stats(gaps_lb)
        m_bk, med_bk, p90_bk, max_bk = stats(gaps_bk)
        print(f"\nQuality vs published LB ({n} instances):")
        print(f"  our_lb == published: {equal_to_pub} "
              f"({100.0 * equal_to_pub / n:.1f}%)")
        print(f"  our_lb >  published: {tighter} "
              f"({100.0 * tighter / n:.1f}%)  (we beat the literature LB)")
        print(f"  our_lb <  published: {weaker} "
              f"({100.0 * weaker / n:.1f}%)  (literature LB tighter)")
        print(f"  gap (published_lb - our_lb): "
              f"mean={m_lb:+.2f}, median={med_lb:+d}, p90={p90_lb:+d}, max={max_lb:+d}")
        print(f"  gap (best_known - our_lb):   "
              f"mean={m_bk:.2f}, median={med_bk}, p90={p90_bk}, max={max_bk}")


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
        print("  python main.py --prune-modes <directory> <results.csv> [--workers N] # report time-window-prunable modes")
        print("  python main.py --check-lb <directory>     # validate LB soundness and report quality")
        print("  Add --justify to any command to enable double justification")
        print("  Add --prune-modes to --param to drop time-window-infeasible modes from output")
        sys.exit(1)

    if "--justify" in sys.argv:
        JUSTIFY = True
        sys.argv = [a for a in sys.argv if a != "--justify"]

    if sys.argv[1] != "--prune-modes" and "--prune-modes" in sys.argv:
        PRUNE_MODES = True
        sys.argv = [a for a in sys.argv if a != "--prune-modes"]

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
    elif sys.argv[1] == "--prune-modes" and len(sys.argv) >= 4:
        directory = sys.argv[2]
        results_csv = sys.argv[3]
        workers = None
        for i, arg in enumerate(sys.argv):
            if arg == "--workers" and i + 1 < len(sys.argv):
                workers = int(sys.argv[i + 1])
        scan_time_window(directory, results_csv, workers=workers)
    elif sys.argv[1] == "--check-lb" and len(sys.argv) >= 3:
        check_lower_bounds(sys.argv[2])
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
