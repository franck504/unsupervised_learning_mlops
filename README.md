# unsupervised_learning_mlops

Pipeline MLOps zero-cost pour clustering non supervise + XAI proxy SHAP.

## Structure

- `data/raw/fruits.csv`: dataset source
- `src/train.py`: pipeline d'entrainement (conversion notebook -> script)
- `dvc.yaml`: pipeline reproductible
- `models/`: artefacts modeles
- `reports/`: metriques et validations
- `deployment/`: app Gradio pour Hugging Face Space
- `.github/workflows/retrain_redeploy.yml`: auto-retrain + auto-redeploy

## Run local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py --input data/raw/fruits.csv --models-dir models --reports-dir reports
```

## Run via DVC

```bash
dvc repro
```

## DVC remote local

Le projet utilise un remote DVC local `localstore`.

```bash
dvc pull
dvc repro
dvc push
```

Dans cette version zero-cost, le remote est local au repo pour permettre la demonstration `dvc pull` dans GitHub Actions. Pour un usage production, remplacez-le par un remote cloud accessible par la CI.

## MLflow Registry local

L'entrainement enregistre les modeles dans le registry MLflow local (`mlflow.db`):
- `fruits-kmeans-clustering`
- `fruits-proxy-rf-shap`

Pour ouvrir MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

## Docker local

```bash
docker build -t fruits-mlops .
docker run --rm -v "$(pwd):/app" fruits-mlops
```

## Monitoring local des inférences

Le Space écrit les prédictions dans `inference_logs/inferences.csv` pendant son exécution.
Pour tester le monitoring localement, placez un fichier d'inférences dans ce chemin, puis lancez:

```bash
python scripts/monitor_drift.py \
  --reference data/processed/fruits_clustered.csv \
  --production inference_logs/inferences.csv \
  --reports-dir reports/monitoring
```

Sorties:
- `reports/monitoring/drift_report.json`
- `reports/monitoring/drift_report.html`

## Monitoring Evidently AI

Pour generer un rapport Evidently AI sur les donnees de production:

```bash
python scripts/monitor_evidently.py \
  --reference data/processed/fruits_clustered.csv \
  --production inference_logs/inferences.csv \
  --reports-dir reports/evidently
```

Sorties:
- `reports/evidently/evidently_data_drift_report.json`
- `reports/evidently/evidently_data_drift_report.html`

## GitHub Secrets requis

- `HF_TOKEN`: token Hugging Face avec permission write sur le Space
- `HF_SPACE_REPO`: URL du Space, ex: `https://huggingface.co/spaces/Franck504/fruits_unsupervised_learning_xai`

## Trigger CI

Le workflow se declenche sur push `main` si changement dans:
- `data/raw/**`
- `src/**`
- `dvc.yaml`
- `requirements.txt`

Il entraine, applique un quality gate, puis publie les nouveaux modeles vers le Space.
