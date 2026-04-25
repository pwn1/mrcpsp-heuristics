"""Run full benchmark and analyse results against best known solutions."""

import csv
import glob
import os
import sys
import time

from mm_parser import parse_psplib
from sgs import SGS_SCHEMES
from priority_rules import PRIORITY_RULES, get_priority_fn
from mode_rules import MODE_RULES, CONTEXT_AWARE_RULES, get_mode_fn
from validate import validate_schedule


def load_best_known(path="data/mmlib50_best_known.csv"):
    bk = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            bk[row["instance"]] = {
                "lb": int(row["lower_bound"]),
                "best_known": int(row["best_known"]),
            }
    return bk


def run_benchmark(directory, max_instances=None):
    files = sorted(glob.glob(os.path.join(directory, "*.mm")))
    if max_instances is not None:
        files = files[:max_instances]
    n_files = len(files)
    n_combos = len(SGS_SCHEMES) * len(PRIORITY_RULES) * len(MODE_RULES)
    print(f"Running {n_combos} combinations on {n_files} instances "
          f"({n_combos * n_files} total runs)...")

    # Per-combo results
    combo_results = {
        (s, p, m): []
        for s in SGS_SCHEMES for p in PRIORITY_RULES for m in MODE_RULES
    }

    # Per-instance best
    best_per_instance = {}

    t0 = time.time()
    for fi, filepath in enumerate(files):
        if fi % max(1, n_files // 20) == 0:
            elapsed = time.time() - t0
            eta = (elapsed / (fi + 1)) * (n_files - fi - 1) if fi > 0 else 0
            print(f"  {fi}/{n_files} ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")
        fname = os.path.basename(filepath)
        project = parse_psplib(filepath)
        inst_best = None

        for sgs_name, sgs_fn in SGS_SCHEMES.items():
            for pr_name in PRIORITY_RULES:
                for mr_name in MODE_RULES:
                    pr_fn = get_priority_fn(pr_name, project, sgs_name, mr_name)
                    mr_fn = get_mode_fn(mr_name, project, sgs_name, pr_name)
                    is_ca = mr_name in CONTEXT_AWARE_RULES
                    schedule = sgs_fn(project, pr_fn, mr_fn,
                                      mode_is_context_aware=is_ca)
                    if schedule is not None:
                        ms = schedule.compute_makespan(project)
                        combo_results[(sgs_name, pr_name, mr_name)].append(ms)
                        if inst_best is None or ms < inst_best:
                            inst_best = ms
                    else:
                        combo_results[(sgs_name, pr_name, mr_name)].append(None)

        best_per_instance[fname] = inst_best

    total_time = time.time() - t0
    return combo_results, best_per_instance, n_files, n_combos, total_time


def analyse(combo_results, best_per_instance, bk, n_files, n_combos, total_time):
    lines = []

    def pr(s=""):
        lines.append(s)
        print(s)

    # Rank combos
    ranked = []
    for key, makespans in combo_results.items():
        feasible = [m for m in makespans if m is not None]
        if feasible:
            ranked.append((key, sum(feasible) / len(feasible), len(feasible)))
    ranked.sort(key=lambda x: x[1])

    # Virtual best
    vb_values = [v for v in best_per_instance.values() if v is not None]
    vb_avg = sum(vb_values) / len(vb_values)
    vb_count = len(vb_values)

    # Gaps to best known and lower bound
    gaps_bk = []
    gaps_lb = []
    matches_bk = 0
    matches_lb = 0
    for fname, ms in best_per_instance.items():
        if ms is None:
            continue
        bk_info = bk.get(fname, {})
        best_known = bk_info.get("best_known")
        lb = bk_info.get("lb")
        if best_known is not None:
            gap = (ms - best_known) / best_known * 100
            gaps_bk.append(gap)
            if ms == best_known:
                matches_bk += 1
        if lb is not None:
            gap = (ms - lb) / lb * 100
            gaps_lb.append(gap)
            if ms == lb:
                matches_lb += 1

    # --- Output ---
    pr(f"Completed in {total_time:.1f}s ({total_time / 60:.1f} min)")
    pr(f"Combinations: {n_combos}, Instances: {n_files}")
    pr()

    pr("## Top 10 Heuristic Combinations (by average makespan)")
    pr()
    pr("| Rank | SGS      | Priority     | Mode Rule        | Avg Makespan | Feasible |")
    pr("|------|----------|--------------|------------------|-------------|----------|")
    for i, (key, avg, count) in enumerate(ranked[:10]):
        sgs_name, pr_name, mr_name = key
        pr(f"| {i+1:<4} | {sgs_name:<8} | {pr_name:<12} | {mr_name:<16} | {avg:<11.1f} | {count}/{n_files}  |")
    pr(f"| **VB** | **(all)** | **(all)**     | **(min/instance)** | **{vb_avg:.1f}**     | **{vb_count}/{n_files}**  |")
    pr()

    pr("## Bottom 5 Heuristic Combinations")
    pr()
    pr("| Rank | SGS      | Priority     | Mode Rule        | Avg Makespan | Feasible |")
    pr("|------|----------|--------------|------------------|-------------|----------|")
    for i, (key, avg, count) in enumerate(ranked[-5:]):
        sgs_name, pr_name, mr_name = key
        pr(f"| {len(ranked)-4+i} | {sgs_name:<8} | {pr_name:<12} | {mr_name:<16} | {avg:<11.1f} | {count}/{n_files}  |")
    pr()

    pr("## Comparison with Best Known Solutions")
    pr()
    pr("| Metric | Value |")
    pr("|--------|-------|")
    pr(f"| Feasible solutions | {vb_count}/{n_files} |")
    pr(f"| Matches best known | {matches_bk}/{len(gaps_bk)} ({matches_bk/len(gaps_bk)*100:.1f}%) |")
    pr(f"| Avg gap to best known | {sum(gaps_bk)/len(gaps_bk):.1f}% |")
    pr(f"| Median gap to best known | {sorted(gaps_bk)[len(gaps_bk)//2]:.1f}% |")
    pr(f"| Min gap to best known | {min(gaps_bk):.1f}% |")
    pr(f"| Max gap to best known | {max(gaps_bk):.1f}% |")
    pr(f"| Matches lower bound | {matches_lb}/{len(gaps_lb)} ({matches_lb/len(gaps_lb)*100:.1f}%) |")
    pr(f"| Avg gap to lower bound | {sum(gaps_lb)/len(gaps_lb):.1f}% |")
    pr(f"| Median gap to lower bound | {sorted(gaps_lb)[len(gaps_lb)//2]:.1f}% |")
    pr()

    pr("### Gap to Best Known Distribution")
    pr()
    pr("| Gap range | Count | Percentage |")
    pr("|-----------|-------|------------|")
    brackets = [(0, 0), (0.1, 5), (5.1, 10), (10.1, 20), (20.1, 50), (50.1, 200)]
    for lo, hi in brackets:
        count = sum(1 for g in gaps_bk if lo <= g <= hi)
        label = "0% (exact match)" if lo == hi else f"{lo:.0f}-{hi:.0f}%"
        pr(f"| {label} | {count} | {count/len(gaps_bk)*100:.1f}% |")
    pr()

    # Best tie-breaker per primary priority rule (averaged across all mode rules)
    pr("## Best Tie-Breaker per Priority Rule")
    pr()
    pr("For each primary rule, the average makespan of its best and worst "
       "tie-breaker (averaged across all SGS and mode rule combinations).")
    pr()
    pr("| Primary | Best Avg | Best TB | Worst Avg | Worst TB | Spread |")
    pr("|---------|----------|---------|-----------|----------|--------|")
    primaries = ["LFT", "MSLK", "MTS", "GRPW", "WRUP", "SPT", "MIS", "GRD"]
    for prim in primaries:
        # Group by tie-breaker: average across all SGS + mode rule combos
        tb_avgs = {}
        for key, avg, count in ranked:
            pr_name = key[1]
            if pr_name.startswith(prim + "/"):
                tb = pr_name.split("/", 1)[1]
                tb_avgs.setdefault(tb, []).append(avg)
        if tb_avgs:
            tb_means = {tb: sum(v)/len(v) for tb, v in tb_avgs.items()}
            best_tb = min(tb_means, key=tb_means.get)
            worst_tb = max(tb_means, key=tb_means.get)
            spread = tb_means[worst_tb] - tb_means[best_tb]
            pr(f"| {prim} | {tb_means[best_tb]:.1f} | {prim}/{best_tb} | "
               f"{tb_means[worst_tb]:.1f} | {prim}/{worst_tb} | {spread:.1f} |")
    pr()

    # Best tie-breaker per primary mode rule (averaged across all priority rules)
    pr("## Best Tie-Breaker per Mode Rule")
    pr()
    pr("For each primary mode rule, the average makespan of its best and worst "
       "tie-breaker (averaged across all SGS and priority rule combinations).")
    pr()
    pr("| Primary Mode | Best Avg | Best TB | Worst Avg | Worst TB | Spread |")
    pr("|--------------|----------|---------|-----------|----------|--------|")
    mode_primaries = ["shortest_duration", "min_resource",
                      "earliest_start", "earliest_finish", "resource_fitting"]
    for prim in mode_primaries:
        # Group by tie-breaker: average across all SGS + priority combos
        tb_avgs = {}
        for key, avg, count in ranked:
            mr_name = key[2]
            if mr_name.startswith(prim + "/"):
                tb = mr_name.split("/", 1)[1]
                tb_avgs.setdefault(tb, []).append(avg)
        if tb_avgs:
            tb_means = {tb: sum(v)/len(v) for tb, v in tb_avgs.items()}
            best_tb = min(tb_means, key=tb_means.get)
            worst_tb = max(tb_means, key=tb_means.get)
            spread = tb_means[worst_tb] - tb_means[best_tb]
            pr(f"| {prim} | {tb_means[best_tb]:.1f} | {prim}/{best_tb} | "
               f"{tb_means[worst_tb]:.1f} | {prim}/{worst_tb} | {spread:.1f} |")
    pr()

    return lines


def main():
    max_instances = None
    directory = "data/MMLIB50"

    if "--test" in sys.argv:
        max_instances = 10
        print("*** TEST MODE: 10 instances only ***\n")

    combo_results, best_per_instance, n_files, n_combos, total_time = \
        run_benchmark(directory, max_instances)

    bk = load_best_known()
    lines = analyse(combo_results, best_per_instance, bk, n_files, n_combos, total_time)

    if "--test" not in sys.argv:
        mode = "a" if "--append" in sys.argv else "w"
        with open("RESULTS.md", mode) as f:
            if mode == "a":
                f.write("\n---\n\n")
            f.write("# MMLIB50 Benchmark Results\n\n")
            f.write(f"{n_files} instances, 52 activities (50 + source/sink), "
                    "3 modes per activity, 2 renewable and 2 non-renewable resources.\n\n")
            f.write(f"{n_combos} heuristic combinations: 2 SGS x "
                    f"{len(PRIORITY_RULES)} priority rules (with tie-breakers) x "
                    f"{len(MODE_RULES)} mode selection rules.\n\n")
            for line in lines:
                f.write(line + "\n")
        print(f"\nResults {'appended to' if mode == 'a' else 'written to'} RESULTS.md")


if __name__ == "__main__":
    main()
