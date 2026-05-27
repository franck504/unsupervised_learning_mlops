import argparse
import json
import os
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw/fruits.csv")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--reports-dir", default="reports")
    p.add_argument("--mlflow-uri", default="sqlite:///mlflow.db")
    p.add_argument("--experiment", default="fruits_usl_xai")
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=10)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    ensure_dirs(args.models_dir, args.reports_dir, "data/processed")

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run():
        df = pd.read_csv(args.input, header=None)
        df.columns = ["feature_1", "feature_2"]

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(df[["feature_1", "feature_2"]])

        results = []
        models = {}
        for k in range(args.k_min, args.k_max + 1):
            km = KMeans(n_clusters=k, random_state=args.random_state, n_init=20)
            labels = km.fit_predict(x_scaled)
            sil = silhouette_score(x_scaled, labels)
            ch = calinski_harabasz_score(x_scaled, labels)
            db = davies_bouldin_score(x_scaled, labels)
            results.append({"k": k, "silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db})
            models[k] = km

        metrics_df = pd.DataFrame(results)
        best_k = int(metrics_df.loc[metrics_df["silhouette"].idxmax(), "k"])
        kmeans = models[best_k]
        df["cluster"] = kmeans.labels_

        x = df[["feature_1", "feature_2"]]
        y = df["cluster"]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=args.test_size, random_state=args.random_state, stratify=y
        )

        proxy = RandomForestClassifier(n_estimators=300, random_state=args.random_state, class_weight="balanced")
        proxy.fit(x_train, y_train)
        pred = proxy.predict(x_test)

        acc = accuracy_score(y_test, pred)
        f1_macro = f1_score(y_test, pred, average="macro")
        f1_weighted = f1_score(y_test, pred, average="weighted")
        cm = confusion_matrix(y_test, pred)

        seeds = [0, 1, 2, 7, 21, 42, 99]
        labels_by_seed = {}
        stability_rows = []
        for s in seeds:
            km_s = KMeans(n_clusters=best_k, random_state=s, n_init=20)
            lbl = km_s.fit_predict(x_scaled)
            labels_by_seed[s] = lbl
            stability_rows.append(
                {
                    "seed": s,
                    "silhouette": silhouette_score(x_scaled, lbl),
                    "calinski_harabasz": calinski_harabasz_score(x_scaled, lbl),
                    "davies_bouldin": davies_bouldin_score(x_scaled, lbl),
                }
            )

        ari_pairs = []
        for i, s1 in enumerate(seeds):
            for s2 in seeds[i + 1 :]:
                ari_pairs.append(adjusted_rand_score(labels_by_seed[s1], labels_by_seed[s2]))

        stability_df = pd.DataFrame(stability_rows)

        processed_path = "data/processed/fruits_clustered.csv"
        df.to_csv(processed_path, index=False)

        scaler_path = os.path.join(args.models_dir, "scaler_fruits.joblib")
        kmeans_path = os.path.join(args.models_dir, "kmeans_fruits.joblib")
        proxy_path = os.path.join(args.models_dir, "proxy_rf_for_shap.joblib")
        joblib.dump(scaler, scaler_path)
        joblib.dump(kmeans, kmeans_path)
        joblib.dump(proxy, proxy_path)

        metrics_path = os.path.join(args.reports_dir, "k_selection_metrics.csv")
        stability_path = os.path.join(args.reports_dir, "stability_metrics.csv")
        cm_path = os.path.join(args.reports_dir, "proxy_confusion_matrix.csv")
        report_path = os.path.join(args.reports_dir, "proxy_classification_report.json")

        metrics_df.to_csv(metrics_path, index=False)
        stability_df.to_csv(stability_path, index=False)
        pd.DataFrame(cm).to_csv(cm_path, index=False)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(classification_report(y_test, pred, output_dict=True), f, indent=2)

        mlflow.log_params(
            {
                "k_min": args.k_min,
                "k_max": args.k_max,
                "best_k": best_k,
                "test_size": args.test_size,
                "random_state": args.random_state,
            }
        )
        mlflow.log_metrics(
            {
                "best_silhouette": float(metrics_df[metrics_df["k"] == best_k]["silhouette"].iloc[0]),
                "best_calinski_harabasz": float(metrics_df[metrics_df["k"] == best_k]["calinski_harabasz"].iloc[0]),
                "best_davies_bouldin": float(metrics_df[metrics_df["k"] == best_k]["davies_bouldin"].iloc[0]),
                "proxy_accuracy": float(acc),
                "proxy_f1_macro": float(f1_macro),
                "proxy_f1_weighted": float(f1_weighted),
                "stability_ari_mean": float(np.mean(ari_pairs)),
                "stability_ari_min": float(np.min(ari_pairs)),
                "stability_silhouette_std": float(stability_df["silhouette"].std()),
            }
        )

        for artifact in [
            processed_path,
            scaler_path,
            kmeans_path,
            proxy_path,
            metrics_path,
            stability_path,
            cm_path,
            report_path,
        ]:
            mlflow.log_artifact(artifact)

        summary = {
            "best_k": best_k,
            "proxy_accuracy": acc,
            "proxy_f1_macro": f1_macro,
            "proxy_f1_weighted": f1_weighted,
            "stability_ari_mean": float(np.mean(ari_pairs)),
            "stability_ari_min": float(np.min(ari_pairs)),
            "limitations": "SHAP explains the proxy model, not KMeans directly.",
        }

        with open(os.path.join(args.reports_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
