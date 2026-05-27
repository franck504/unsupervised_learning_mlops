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

## Docker local

```bash
docker build -t fruits-mlops .
docker run --rm -v "$(pwd):/app" fruits-mlops
```

## GitHub Secrets requis

- `HF_TOKEN`: token Hugging Face avec permission write sur le Space
- `HF_SPACE_REPO`: URL du Space, ex: `https://huggingface.co/spaces/Franck504/fruits_usl_xai`

## Trigger CI

Le workflow se declenche sur push `main` si changement dans:
- `data/raw/**`
- `src/**`
- `dvc.yaml`
- `requirements.txt`

Il entraine, applique un quality gate, puis publie les nouveaux modeles vers le Space.
