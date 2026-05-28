import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = ["feature_1", "feature_2"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="data/processed/fruits_clustered.csv")
    parser.add_argument("--production", default="inference_logs/inferences.csv")
    parser.add_argument("--reports-dir", default="reports/monitoring")
    parser.add_argument("--psi-threshold", type=float, default=0.20)
    return parser.parse_args()


def psi(reference, production, bins=10):
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 3:
        edges = np.linspace(min(reference.min(), production.min()), max(reference.max(), production.max()), bins + 1)

    ref_counts, _ = np.histogram(reference, bins=edges)
    prod_counts, _ = np.histogram(production, bins=edges)

    ref_dist = np.maximum(ref_counts / max(ref_counts.sum(), 1), 1e-6)
    prod_dist = np.maximum(prod_counts / max(prod_counts.sum(), 1), 1e-6)
    return float(np.sum((prod_dist - ref_dist) * np.log(prod_dist / ref_dist)))


def main():
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    reference = pd.read_csv(args.reference)
    production_path = Path(args.production)
    if not production_path.exists():
        raise FileNotFoundError(
            f"No production inference log found at {production_path}. "
            "Call the deployed Space endpoint first, then export/copy inferences.csv here."
        )

    production = pd.read_csv(production_path)
    if production.empty:
        raise ValueError("Production inference log is empty.")

    feature_drift = {}
    alerts = []
    for feature in FEATURES:
        score = psi(reference[feature].astype(float), production[feature].astype(float))
        drift_detected = score >= args.psi_threshold
        feature_drift[feature] = {"psi": score, "drift_detected": drift_detected}
        if drift_detected:
            alerts.append(f"{feature} PSI={score:.4f} >= {args.psi_threshold}")

    cluster_reference = reference["cluster"].value_counts(normalize=True).sort_index().to_dict()
    cluster_production = production["cluster"].value_counts(normalize=True).sort_index().to_dict()

    report = {
        "reference_rows": int(len(reference)),
        "production_rows": int(len(production)),
        "psi_threshold": args.psi_threshold,
        "feature_drift": feature_drift,
        "cluster_distribution": {
            "reference": {str(k): float(v) for k, v in cluster_reference.items()},
            "production": {str(k): float(v) for k, v in cluster_production.items()},
        },
        "alerts": alerts,
    }

    json_path = reports_dir / "drift_report.json"
    html_path = reports_dir / "drift_report.html"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    html_path.write_text(
        "<html><body><h1>Fruits Drift Report</h1><pre>"
        + json.dumps(report, indent=2)
        + "</pre></body></html>",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
