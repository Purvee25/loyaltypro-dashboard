"""
Export the trained Random Forest to JSON so the dashboard can score in the
browser, with no Flask process running.

The forest is small (100 shallow trees, ~8k nodes total), so shipping the split
thresholds is cheaper than hosting an inference server. Predictions are
identical to `app.py` — same trees, same averaging, same feature order.

Output: docs/model.json
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

MODEL_PATH = Path("rf_churn_model.pkl")
OUTPUT_PATH = Path("docs/model.json")

# Must match the training order used in app.py.
FEATURE_ORDER = [
    "Age", "Gender", "Location", "Tenure_Months", "Total_Spend",
    "Num_Purchases", "Last_Purchase_Days_Ago", "Satisfaction_Score",
    "Membership_Type", "Complaints", "Used_Discount", "Avg_Monthly_Spend",
]


def export_tree(tree) -> dict:
    """Flatten one sklearn tree into parallel arrays a JS walker can index."""
    t = tree.tree_
    # value[i] holds class counts at node i; store P(churn) for leaves only.
    proba = (t.value[:, 0, 1] / t.value[:, 0, :].sum(axis=1)).round(6)
    return {
        "feature": t.feature.tolist(),
        "threshold": [round(float(x), 6) for x in t.threshold],
        "left": t.children_left.tolist(),
        "right": t.children_right.tolist(),
        "proba": proba.tolist(),
    }


def main() -> None:
    with MODEL_PATH.open("rb") as handle:
        model = pickle.load(handle)

    payload = {
        "features": FEATURE_ORDER,
        "n_estimators": len(model.estimators_),
        "trees": [export_tree(estimator) for estimator in model.estimators_],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as handle:
        json.dump(payload, handle, separators=(",", ":"))

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"Wrote {OUTPUT_PATH} ({size_kb:.0f} KB, {payload['n_estimators']} trees)")

    # Parity check: the exported forest must match sklearn on random inputs.
    rng = np.random.default_rng(0)
    samples = np.column_stack([
        rng.uniform(18, 80, 500), rng.integers(0, 2, 500), rng.integers(0, 4, 500),
        rng.uniform(1, 60, 500), rng.uniform(1e3, 2e5, 500), rng.integers(1, 60, 500),
        rng.uniform(0, 365, 500), rng.integers(1, 6, 500), rng.integers(0, 3, 500),
        rng.integers(0, 5, 500), rng.integers(0, 2, 500), rng.uniform(100, 9000, 500),
    ])
    expected = model.predict_proba(samples)[:, 1]

    actual = []
    for row in samples:
        total = 0.0
        for tree in payload["trees"]:
            node = 0
            while tree["left"][node] != -1:
                node = (tree["left"][node] if row[tree["feature"][node]] <= tree["threshold"][node]
                        else tree["right"][node])
            total += tree["proba"][node]
        actual.append(total / payload["n_estimators"])

    max_delta = float(np.max(np.abs(np.array(actual) - expected)))
    print(f"Max deviation from sklearn over 500 samples: {max_delta:.2e}")
    if max_delta > 1e-6:
        raise SystemExit("Exported forest does not match sklearn.")


if __name__ == "__main__":
    main()
