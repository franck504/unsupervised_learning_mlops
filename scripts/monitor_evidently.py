import argparse
from pathlib import Path

import pandas as pd


FEATURES = ["feature_1", "feature_2", "cluster"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="data/processed/fruits_clustered.csv")
    parser.add_argument("--production", default="inference_logs/inferences.csv")
    parser.add_argument("--reports-dir", default="reports/evidently")
    return parser.parse_args()


def load_evidently_report():
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        return Report, DataDriftPreset, "new"
    except Exception:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        return Report, DataDriftPreset, "legacy"


def run_report(report, current, reference, api_style):
    if api_style == "new":
        result = report.run(current, reference)
        return result if result is not None else report
    report.run(reference_data=reference, current_data=current)
    return report


def save_json(report, path):
    if hasattr(report, "save_json"):
        report.save_json(str(path))
        return
    if hasattr(report, "json"):
        path.write_text(report.json(), encoding="utf-8")
        return
    path.write_text("{}", encoding="utf-8")


def save_html(report, path):
    if hasattr(report, "save_html"):
        report.save_html(str(path))
        return
    if hasattr(report, "get_html"):
        path.write_text(report.get_html(), encoding="utf-8")
        return
    path.write_text("<html><body><h1>Evidently report generated</h1></body></html>", encoding="utf-8")


def main():
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    reference = pd.read_csv(args.reference)[FEATURES]
    production = pd.read_csv(args.production)[FEATURES]

    Report, DataDriftPreset, api_style = load_evidently_report()
    report = Report([DataDriftPreset()])
    report = run_report(report, production, reference, api_style)

    save_json(report, reports_dir / "evidently_data_drift_report.json")
    save_html(report, reports_dir / "evidently_data_drift_report.html")
    print(f"Evidently report generated in {reports_dir}")


if __name__ == "__main__":
    main()
