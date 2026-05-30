import os
import csv
import uuid
import joblib
import numpy as np
import pandas as pd
import gradio as gr
import shap
from datetime import datetime, timezone

MODELS_DIR = "models"
SCALER_PATH = os.path.join(MODELS_DIR, "scaler_fruits.joblib")
KMEANS_PATH = os.path.join(MODELS_DIR, "kmeans_fruits.joblib")
PROXY_PATH = os.path.join(MODELS_DIR, "proxy_rf_for_shap.joblib")
LOG_DIR = os.getenv("INFERENCE_LOG_DIR", "inference_logs")
LOG_PATH = os.path.join(LOG_DIR, "inferences.csv")


def load_models():
    scaler = joblib.load(SCALER_PATH)
    kmeans = joblib.load(KMEANS_PATH)
    proxy = joblib.load(PROXY_PATH)
    return scaler, kmeans, proxy


scaler, kmeans, proxy = load_models()
explainer = shap.TreeExplainer(proxy)
feature_names = ["feature_1", "feature_2"]


def log_inference(request_id, feature_1, feature_2, cluster, explanation):
    os.makedirs(LOG_DIR, exist_ok=True)
    file_exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "request_id",
                "feature_1",
                "feature_2",
                "cluster",
                "shap_feature_1",
                "shap_feature_2",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_id": request_id,
                "feature_1": feature_1,
                "feature_2": feature_2,
                "cluster": cluster,
                "shap_feature_1": explanation["feature_1"],
                "shap_feature_2": explanation["feature_2"],
            }
        )


def predict_and_explain(feature_1: float, feature_2: float):
    request_id = str(uuid.uuid4())
    x_df = pd.DataFrame([[feature_1, feature_2]], columns=feature_names)
    cluster = int(kmeans.predict(scaler.transform(x_df))[0])

    shap_values = explainer.shap_values(x_df)
    if isinstance(shap_values, list):
        class_index = int(np.where(proxy.classes_ == cluster)[0][0]) if cluster in proxy.classes_ else 0
        one_class = shap_values[class_index][0]
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            class_index = int(np.where(proxy.classes_ == cluster)[0][0]) if cluster in proxy.classes_ else 0
            one_class = arr[0, :, class_index]
        else:
            one_class = arr[0]

    explanation = {feature_names[i]: float(one_class[i]) for i in range(len(feature_names))}
    log_inference(request_id, feature_1, feature_2, cluster, explanation)

    # Human-readable SHAP explanation (text) - line by line
    lines = [f"**Cluster prédit: {cluster}**\n"]
    for name, val in explanation.items():
        sign = "+" if val >= 0 else ""
        direction = "pousse vers le cluster" if val >= 0 else "pousse contre le cluster"
        lines.append(f"**{name}:** {sign}{val:.3f}  \n*{direction}*\n")
    explanation_text = "".join(lines)

    # Cluster count from KMeans model (if available)
    cluster_count = getattr(kmeans, "n_clusters", None)

    # Return values for Gradio outputs: out_json, out_text, cluster_info
    out_json_dict = {
        "request_id": request_id,
        "cluster": cluster,
        "input": {"feature_1": feature_1, "feature_2": feature_2},
        "explanation": explanation,
        "explanation_text": explanation_text,
        "cluster_count": int(cluster_count) if cluster_count is not None else None,
    }
    cluster_info_md = f"🔍 **Clusters découverts:** {cluster_count if cluster_count else 'N/A'}"
    return out_json_dict, explanation_text, cluster_info_md


def get_inference_log():
    if not os.path.exists(LOG_PATH):
        return None
    return LOG_PATH


with gr.Blocks(title="Fruits - Apprentissage Non Supervisé & XAI") as demo:
    gr.Markdown("# 🍎 Fruits - Apprentissage Non Supervisé & XAI")
    with gr.Row():
        with gr.Column(scale=2):
            f1 = gr.Number(label="Caractéristique 1", value=25.0)
            f2 = gr.Number(label="Caractéristique 2", value=8.5)
            predict_btn = gr.Button("🔮 Prédire et Expliquer")
            dl_btn = gr.Button("📥 Télécharger les journaux")
        with gr.Column(scale=3):
            cluster_info = gr.Markdown(value=f"🔍 **Clusters découverts:** {getattr(kmeans, 'n_clusters', 'N/A')}")
            out_json = gr.JSON(label="Résultat (JSON)")
            out_text = gr.Markdown(label="Explication (texte)")
            log_file = gr.File(label="Journaux d'inférence (CSV)")

    # Wire buttons: keep API-compatible JSON output, and provide a readable text explanation
    predict_btn.click(
        fn=predict_and_explain,
        inputs=[f1, f2],
        outputs=[out_json, out_text, cluster_info],
        api_name="predict_and_explain",
    )

    dl_btn.click(
        fn=get_inference_log,
        inputs=[],
        outputs=log_file,
        api_name="download_inference_logs",
    )

if __name__ == "__main__":
    # On HF Spaces this will be served by the platform. For local testing:
    demo.launch(server_name="0.0.0.0", share=False, css=".gr-row{gap:12px}")
