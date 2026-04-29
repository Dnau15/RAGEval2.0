"""Run the full pipeline.

Order:
    1. analysis.py    -- writes hybrid_test_comparison.csv (needed by phase_a n_required)
    2. phase_a.py     -- first-stage retrieval, multi_dataset, n_required, efficiency
    3. phase_b.py     -- BioASQ, reranker, router, MIRAGE

A full run on a Colab T4 takes roughly 2-3 hours.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def run(label, fn):
    print("\n" + "=" * 70 + f"\n  {label}\n" + "=" * 70)
    t0 = time.time()
    fn()
    print(f"\n  done in {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    import analysis, phase_a, phase_b

    run("analysis.py: paired bootstrap, regression, stratifications", lambda: (
        (alpha := analysis.split_metrics_and_alpha()),
        analysis.boot_and_subsampling(alpha),
        analysis.regression_and_stratifications(),
    ))
    run("phase_a.py: first-stage + multi-dataset + n_required + efficiency",
        lambda: (
            phase_a.first_stage_and_multi_dataset(),
            phase_a.write_n_required(),
            phase_a.write_efficiency(),
        ))
    run("phase_b.py: BioASQ + reranker + router + MIRAGE", lambda: (
        phase_b.build_bioasq_subset(),
        phase_b.run_rerankers(),
        phase_b.run_router(),
        phase_b.run_mirage(),
    ))
