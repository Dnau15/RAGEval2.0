"""Hoeffding sample-size requirements (Table 11).

For each pairwise nDCG@10 gap, computes the minimum number of queries
needed to certify the gap at 95% confidence under Hoeffding.

Reads from ``nfcorpus_canonical.csv`` and ``hybrid_test_comparison.csv``;
writes ``n_required.csv``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rageval.utils import FEEDBACK2_DIR, NEXT_STAGE_DIR


DELTA = 0.05  # 1 - confidence


def n_required(gap: float) -> int:
    if abs(gap) < 1e-12:
        return 10**9
    return math.ceil(math.log(2 / DELTA) / (2 * gap ** 2))


def main() -> None:
    nf = {r["Method"]: float(r["nDCG@10"])
           for _, r in pd.read_csv(FEEDBACK2_DIR / "nfcorpus_canonical.csv").iterrows()}
    hyb = pd.read_csv(NEXT_STAGE_DIR / "hybrid_test_comparison.csv")
    hybrid = float(hyb.loc[hyb["method"] == "Hybrid", "nDCG@10"].iloc[0])
    dense = float(hyb.loc[hyb["method"] == "Dense", "nDCG@10"].iloc[0])
    bm25 = float(hyb.loc[hyb["method"] == "BM25", "nDCG@10"].iloc[0])

    rows = [
        ("BGE-small", "BM25", nf["BGE-small"] - nf["BM25"]),
        ("BGE-small", "E5-small", nf["BGE-small"] - nf["E5-small"]),
        ("Hybrid", "Dense", hybrid - dense),
        ("Dense", "BM25", dense - bm25),
    ]
    df = pd.DataFrame([
        {"A": a, "B": b, "observed_gap": round(g, 4),
         "hoeffding_n_required_95pct": n_required(g)}
        for a, b, g in rows
    ])
    out = FEEDBACK2_DIR / "n_required.csv"
    df.to_csv(out, index=False)
    print(df)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
