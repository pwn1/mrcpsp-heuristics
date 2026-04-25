"""Count how often the 'random' mode rule contributes to the per-instance
best makespan across a subset of MMLIB50 instances.

For each instance, runs all combos and records (a) which combos tie for the
minimum makespan, (b) whether any 'random'-mode combo is among the winners,
(c) whether 'random' is strictly required (i.e. only random-mode combos win).
"""

import glob
import os
import sys
import time

from mm_parser import parse_psplib
from sgs import SGS_SCHEMES
from priority_rules import PRIORITY_RULES, get_priority_fn
from mode_rules import MODE_RULES, CONTEXT_AWARE_RULES, get_mode_fn


def run(directory, max_instances=100):
    files = sorted(glob.glob(os.path.join(directory, "*.mm")))[:max_instances]
    n = len(files)

    any_random_wins = 0
    only_random_wins = 0
    any_feasible = 0

    t0 = time.time()
    for fi, filepath in enumerate(files):
        if fi % max(1, n // 10) == 0:
            el = time.time() - t0
            eta = (el / (fi + 1)) * (n - fi - 1) if fi > 0 else 0
            print(f"  {fi}/{n} ({el:.0f}s elapsed, ~{eta:.0f}s remaining)",
                  flush=True)
        project = parse_psplib(filepath)

        best_ms = None
        winners = []
        for sgs_name, sgs_fn in SGS_SCHEMES.items():
            for pr_name in PRIORITY_RULES:
                for mr_name in MODE_RULES:
                    pr_fn = get_priority_fn(pr_name, project, sgs_name, mr_name)
                    mr_fn = get_mode_fn(mr_name, project, sgs_name, pr_name)
                    is_ca = mr_name in CONTEXT_AWARE_RULES
                    sch = sgs_fn(project, pr_fn, mr_fn, mode_is_context_aware=is_ca)
                    if sch is None:
                        continue
                    ms = sch.compute_makespan(project)
                    if best_ms is None or ms < best_ms:
                        best_ms = ms
                        winners = [(sgs_name, pr_name, mr_name)]
                    elif ms == best_ms:
                        winners.append((sgs_name, pr_name, mr_name))

        if best_ms is None:
            continue
        any_feasible += 1
        random_winners = [w for w in winners if w[2] == "random"]
        non_random_winners = [w for w in winners if w[2] != "random"]
        if random_winners:
            any_random_wins += 1
            if not non_random_winners:
                only_random_wins += 1

    total_time = time.time() - t0
    print(f"\nDone in {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Instances with feasible solution:        {any_feasible}/{n}")
    print(f"Instances where 'random' ties for best:  {any_random_wins}/{any_feasible} "
          f"({any_random_wins/any_feasible*100:.1f}%)")
    print(f"Instances where ONLY 'random' wins:      {only_random_wins}/{any_feasible} "
          f"({only_random_wins/any_feasible*100:.1f}%)")


if __name__ == "__main__":
    max_inst = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run("data/MMLIB50", max_instances=max_inst)
