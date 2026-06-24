"""Analyse benchmark results from CSV (produced by main.py --benchmark).

Computes VB, gap statistics, and random mode contribution from the saved
per-instance results without re-running heuristics.
"""

import csv
import sys


def load_best_known(path="data/mmlib50_best_known.csv"):
    bk = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            bk[row["instance"]] = {
                "lb": int(row["lower_bound"]),
                "best_known": int(row["best_known"]),
            }
    return bk


def load_results(path):
    """Load per-instance results CSV. Returns
    {instance: {(sgs, priority, mode): makespan_or_None}}."""
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            fname = row["instance"]
            key = (row["sgs"], row["priority"], row["mode"])
            ms = int(row["makespan"]) if row["makespan"] else None
            if fname not in data:
                data[fname] = {}
            data[fname][key] = ms
    return data


def analyse(results_path):
    data = load_results(results_path)
    bk = load_best_known()

    instances = sorted(data.keys())
    n_files = len(instances)

    # Count combos from first instance
    first = data[instances[0]]
    n_combos = len(first)

    # Per-instance VB
    vb = {}
    for fname, results in data.items():
        feasible_ms = [ms for ms in results.values() if ms is not None]
        vb[fname] = min(feasible_ms) if feasible_ms else None

    # Per-combo averages
    combo_totals = {}
    combo_counts = {}
    for fname, results in data.items():
        for key, ms in results.items():
            if key not in combo_totals:
                combo_totals[key] = 0
                combo_counts[key] = 0
            if ms is not None:
                combo_totals[key] += ms
                combo_counts[key] += 1

    ranked = []
    for key in combo_totals:
        if combo_counts[key] > 0:
            avg = combo_totals[key] / combo_counts[key]
            ranked.append((key, avg, combo_counts[key]))
    ranked.sort(key=lambda x: x[1])

    # VB stats
    vb_vals = [v for v in vb.values() if v is not None]
    vb_avg = sum(vb_vals) / len(vb_vals)
    feasible_count = len(vb_vals)

    # Gap stats
    gaps_bk, gaps_lb = [], []
    match_bk, match_lb = 0, 0
    for fname, ms in vb.items():
        if ms is None:
            continue
        info = bk.get(fname, {})
        best_known = info.get("best_known")
        lb = info.get("lb")
        if best_known is not None:
            gap = (ms - best_known) / best_known * 100
            gaps_bk.append(gap)
            if ms == best_known:
                match_bk += 1
        if lb is not None:
            gap_lb = (ms - lb) / lb * 100
            gaps_lb.append(gap_lb)
            if ms == lb:
                match_lb += 1

    # Print summary
    print(f"Instances: {n_files}, Combinations: {n_combos}")
    print(f"\nVB avg makespan: {vb_avg:.1f}")
    print(f"Feasible: {feasible_count}/{n_files}")
    print(f"Matches best known: {match_bk}/{len(gaps_bk)} ({match_bk/len(gaps_bk)*100:.1f}%)")
    print(f"Avg gap to BK: {sum(gaps_bk)/len(gaps_bk):.2f}%")
    print(f"Median gap to BK: {sorted(gaps_bk)[len(gaps_bk)//2]:.2f}%")
    print(f"Matches LB: {match_lb}/{len(gaps_lb)} ({match_lb/len(gaps_lb)*100:.1f}%)")
    print(f"Avg gap to LB: {sum(gaps_lb)/len(gaps_lb):.2f}%")
    print(f"Median gap to LB: {sorted(gaps_lb)[len(gaps_lb)//2]:.2f}%")
    print(f"Max gap to BK: {max(gaps_bk):.1f}%")

    # Top 10 and bottom 5
    print(f"\nTop 10:")
    for i, (key, avg, cnt) in enumerate(ranked[:10]):
        print(f"  {i+1}. {key[0]:<10} {key[1]:<12} {key[2]:<30} {avg:.1f}  {cnt}/{n_files}")
    print(f"\nBottom 5:")
    for i, (key, avg, cnt) in enumerate(ranked[-5:]):
        print(f"  {len(ranked)-4+i}. {key[0]:<10} {key[1]:<12} {key[2]:<30} {avg:.1f}  {cnt}/{n_files}")

    # Gap distribution
    print(f"\nGap-to-BK distribution:")
    brackets = [(0, 0), (0.01, 5), (5.01, 10), (10.01, 20), (20.01, 50), (50.01, 200)]
    for lo, hi in brackets:
        cnt = sum(1 for g in gaps_bk if lo <= g <= hi)
        label = "0% (exact)" if lo == hi else f"{lo:.0f}-{hi:.0f}%"
        print(f"  {label}: {cnt} ({cnt/len(gaps_bk)*100:.1f}%)")

    # Random mode contribution
    random_wins = 0
    only_random = 0
    for fname, results in data.items():
        best_ms = None
        has_random_winner = False
        has_nonrandom_winner = False
        for key, ms in results.items():
            if ms is None:
                continue
            if best_ms is None or ms < best_ms:
                best_ms = ms
                has_random_winner = key[2].startswith("random")
                has_nonrandom_winner = not key[2].startswith("random")
            elif ms == best_ms:
                if key[2].startswith("random"):
                    has_random_winner = True
                else:
                    has_nonrandom_winner = True
        if has_random_winner:
            random_wins += 1
            if not has_nonrandom_winner:
                only_random += 1

    print(f"\nRandom mode wins: {random_wins}/{feasible_count} "
          f"({random_wins/feasible_count*100:.1f}%) ties for best")
    print(f"Only random wins: {only_random}/{feasible_count} "
          f"({only_random/feasible_count*100:.1f}%) sole winner")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/benchmark_results.csv"
    analyse(path)
