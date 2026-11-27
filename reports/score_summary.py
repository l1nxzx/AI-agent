"""Prints a comparison table from reports/scores/*.json.   python reports/score_summary.py

Each scores file lists graded replies (group, prompt, reply, score 0/1/2). The summary is the mean score per
group and overall, plus the share of replies scored 2 (sensible and on-topic) and 0 (nonsense)."""
import glob
import json
import os
from collections import defaultdict

here = os.path.dirname(os.path.abspath(__file__))
runs = []
for path in sorted(glob.glob(os.path.join(here, "scores", "*.json"))):
    with open(path) as f:
        runs.append(json.load(f))
groups = sorted({it["group"] for r in runs for it in r["items"]})
print("| run | n | overall | share 2 | share 0 | " + " | ".join(groups) + " |")
print("|" + "---|" * (5 + len(groups)))
for r in runs:
    by = defaultdict(list)
    for it in r["items"]:
        by[it["group"]].append(it["score"])
    allv = [it["score"] for it in r["items"]]
    cells = [f"{sum(by[g]) / len(by[g]):.2f} ({len(by[g])})" if by.get(g) else "-" for g in groups]
    print(f"| {r['run']} | {len(allv)} | {sum(allv) / len(allv):.2f} | {allv.count(2) / len(allv):.0%} | "
          f"{allv.count(0) / len(allv):.0%} | " + " | ".join(cells) + " |")
