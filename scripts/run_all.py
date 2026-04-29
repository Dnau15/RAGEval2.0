"""Reproduce every CSV in the report from a clean checkout.

Order matters because some scripts read CSVs written by earlier ones
(``phase_a_n_required`` reads ``hybrid_test_comparison.csv`` from
``next_stage_paired_boot``; ``phase_b2_reranker`` reads the BioASQ
cache from ``phase_b1_bioasq``; ``phase_b3_router`` reuses BGE +
BGE-reranker models).

Usage
-----
    python scripts/run_all.py            # run every step
    python scripts/run_all.py phase_a    # only the Phase A steps
    python scripts/run_all.py phase_b1 phase_b2  # specific scripts

A full run on a Colab T4 takes roughly 2-3 hours.
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path


# Ordered list of (group, module name, brief description).
PIPELINE = [
    ("phase_a", "phase_a_first_stage",
     "NFCorpus 6-way + multi-dataset first-stage (Tab 3, 4)"),
    ("next_stage", "next_stage_paired_boot",
     "Per-split metrics, hybrid sweep, paired boot, subsampling (Tab 6, 12, 13)"),
    ("phase_a", "phase_a_n_required",
     "Hoeffding n_min table (Tab 11)"),
    ("next_stage", "next_stage_regression",
     "Vocab-gap features, correlations, stratifications, OLS (Tab 7, 14, 15)"),
    ("phase_b1", "phase_b1_bioasq",
     "BioASQ-subset retrieval + paired bootstrap (Tab 4 row, BioASQ-PB)"),
    ("phase_b2", "phase_b2_reranker",
     "BGE cross-encoder + MedCPT-CE reranking (Tab 5)"),
    ("phase_b3", "phase_b3_router",
     "Per-query router + train-test gap (Tab 8, 9, 13 router cols)"),
    ("phase_b4", "phase_b4_mirage",
     "Downstream PubMedQA + flan-t5 (Tab 16)"),
    ("phase_a", "phase_a_efficiency",
     "Index time, latency, throughput, memory (Tab 10) -- run last; reloads models"),
]


def _run(module_name: str) -> None:
    print("\n" + "=" * 78)
    print(f"  RUN: scripts/{module_name}.py")
    print("=" * 78)
    t0 = time.time()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    mod = importlib.import_module(module_name)
    mod.main()
    print(f"\n  done in {(time.time() - t0)/60:.1f} min")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        steps = [m for _, m, _ in PIPELINE]
    else:
        # Filter by group name or exact module name
        steps = []
        for group, mod, _ in PIPELINE:
            if group in args or mod in args:
                steps.append(mod)
        if not steps:
            print("No matching steps.  Available:")
            for group, mod, desc in PIPELINE:
                print(f"  [{group:<10}] {mod:<28} {desc}")
            sys.exit(1)

    print(f"Running {len(steps)} step(s):")
    for s in steps:
        print(f"  - {s}")

    t_start = time.time()
    for s in steps:
        try:
            _run(s)
        except Exception as exc:
            print(f"\n  FAILED: {s} -> {type(exc).__name__}: {exc}")
            print(f"  (continuing with remaining steps)")

    print(f"\nAll done in {(time.time() - t_start)/60:.1f} min total")


if __name__ == "__main__":
    main()
