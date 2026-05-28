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

    return {
        "request_id": request_id,
        "cluster": cluster,
        "input": {"feature_1": feature_1, "feature_2": feature_2},
        "explanation": explanation,
    }


with gr.Blocks(title="Fruits USL XAI") as demo:
    gr.Markdown("# Fruits Unsupervised Learning & XAI")
    with gr.Row():
        f1 = gr.Number(label="feature_1", value=25.0)
        f2 = gr.Number(label="feature_2", value=8.5)
    out = gr.JSON(label="Resultat")
    gr.Button("Predict and Explain").click(
        fn=predict_and_explain,
        inputs=[f1, f2],
        outputs=out,
        api_name="predict_and_explain",
    )

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
