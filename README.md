# Unsupervised Learning MLOps + XAI

Ce projet met en place un pipeline MLOps complet pour un apprentissage non supervise sur un dataset de fruits, avec interpretation XAI via SHAP, deploiement automatique sur Hugging Face Spaces, versionnement DVC, suivi MLflow et monitoring de drift.

## Objectif

L'objectif est de segmenter automatiquement des observations de fruits a partir de deux variables numeriques (`feature_1`, `feature_2`), puis d'expliquer pourquoi une observation est affectee a un cluster donne.

Le probleme est non supervise: il n'y a pas de label metier connu au depart. Le modele cherche donc des groupes naturels dans les donnees.

## Architecture

Flux global:

```text
git push
-> GitHub Actions
-> dvc pull
-> dvc repro
-> entrainement KMeans
-> validation metriques
-> MLflow Tracking + Registry
-> publication vers Hugging Face Space
-> redeploiement automatique
-> prediction + explication SHAP
-> logs d'inference
-> monitoring drift local + Evidently
```

Composants principaux:
- `DVC`: versionnement des donnees et artefacts.
- `MLflow`: tracking des experiences et registry local des modeles.
- `KMeans`: clustering non supervise.
- `RandomForestClassifier`: modele proxy utilise pour XAI.
- `SHAP`: explication des predictions du modele proxy.
- `GitHub Actions`: orchestration CI/CD.
- `Hugging Face Spaces`: deploiement de l'interface/API Gradio.
- `Evidently AI`: rapport de drift sur les donnees de production.

## Structure

- `data/raw/fruits.csv`: dataset source suivi par DVC.
- `data/processed/fruits_clustered.csv`: dataset enrichi avec les clusters.
- `src/train.py`: script d'entrainement principal.
- `dvc.yaml`: definition du pipeline DVC.
- `dvc.lock`: etat exact des donnees, dependances et sorties du pipeline.
- `models/`: modeles generes par le pipeline.
- `reports/`: metriques, validations et rapports.
- `deployment/app.py`: application Gradio pour Hugging Face Space.
- `scripts/publish_to_hf_space.sh`: publication automatique vers le Space.
- `scripts/monitor_drift.py`: monitoring PSI local.
- `scripts/monitor_evidently.py`: monitoring Evidently AI.
- `.github/workflows/retrain_redeploy.yml`: CI/CD auto-train + auto-deploy.

## Technique D'Entrainement

Le script [train.py](/home/franck/Documents/unsupervised_learning-XAI/unsupervised_learning_mlops/src/train.py) execute les etapes suivantes:

1. Chargement du dataset `fruits.csv`.
2. Attribution des noms de colonnes: `feature_1`, `feature_2`.
3. Standardisation avec `StandardScaler`.
4. Test de plusieurs valeurs de `k` pour `KMeans`.
5. Evaluation de chaque `k` avec:
- `silhouette`
- `calinski_harabasz`
- `davies_bouldin`
6. Selection du meilleur `k` selon le score silhouette.
7. Entrainement final du modele `KMeans`.
8. Export des labels dans `data/processed/fruits_clustered.csv`.

Le modele principal est donc:

```text
StandardScaler -> KMeans
```

Le pipeline genere:
- `models/scaler_fruits.joblib`
- `models/kmeans_fruits.joblib`
- `data/processed/fruits_clustered.csv`

## Validation Professionnelle

Le pipeline ne se contente pas d'entrainer un modele. Il valide aussi la qualite du resultat.

Metriques de clustering:
- `silhouette`: plus grand = meilleur.
- `calinski_harabasz`: plus grand = meilleur.
- `davies_bouldin`: plus petit = meilleur.

Stabilite:
- KMeans est relance avec plusieurs seeds.
- La stabilite est mesuree avec `Adjusted Rand Index`.

Fidelite du proxy:
- `accuracy`
- `F1 macro`
- `F1 weighted`

Le quality gate GitHub Actions exige:

```text
proxy_accuracy >= 0.90
proxy_f1_macro >= 0.90
stability_ari_mean >= 0.90
```

Si ces conditions ne sont pas respectees, le deploiement est bloque.

## XAI Avec SHAP

SHAP explique naturellement des modeles supervises. Or KMeans est un modele non supervise.

La strategie utilisee est donc une approche proxy:

1. KMeans genere des clusters.
2. Ces clusters deviennent des pseudo-labels.
3. Un `RandomForestClassifier` apprend a reproduire les clusters.
4. SHAP explique les decisions de ce RandomForest.

Le modele proxy est:

```text
feature_1, feature_2 -> RandomForestClassifier -> cluster
```

SHAP donne ensuite la contribution de chaque variable:
- contribution positive: la variable pousse vers le cluster predit.
- contribution negative: la variable pousse contre ce cluster.

Important:

SHAP explique le modele proxy, pas KMeans directement. C'est acceptable si la fidelite du proxy est bonne. C'est pourquoi le pipeline calcule `accuracy` et `F1` du proxy.

## MLflow Tracking Et Registry

L'entrainement enregistre:
- parametres
- metriques
- artefacts
- modeles

Le registry MLflow local contient:
- `fruits-kmeans-clustering`
- `fruits-proxy-rf-shap`

Pour ouvrir MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Puis ouvrir:

```text
http://127.0.0.1:5000
```

## DVC

Le projet utilise DVC pour versionner les donnees et les artefacts.

Commandes utiles:

```bash
dvc pull
dvc repro
dvc push
```

Dans cette version zero-cost, le remote DVC est local:

```text
dvc_remote/
```

Cela permet de demontrer un vrai `dvc pull` dans GitHub Actions sans compte cloud payant.

Limite:

Pour une production stricte, il faudrait remplacer ce remote local par un remote externe accessible par la CI, par exemple S3, GCS, Azure Blob ou un stockage compatible.

## CI/CD

Le workflow [retrain_redeploy.yml](/home/franck/Documents/unsupervised_learning-XAI/unsupervised_learning_mlops/.github/workflows/retrain_redeploy.yml) se declenche sur `push` vers `main` si un fichier surveille change.

Fichiers surveilles:
- `data/raw/**`
- `src/**`
- `deployment/**`
- `scripts/publish_to_hf_space.sh`
- `dvc.yaml`
- `requirements.txt`
- `.github/workflows/retrain_redeploy.yml`

Etapes CI:
1. Checkout du repo.
2. Installation Python.
3. Installation des dependances.
4. `dvc pull`.
5. `dvc repro`.
6. Quality gate.
7. Verification des secrets Hugging Face.
8. Publication vers Hugging Face Space.

Secrets GitHub requis:
- `HF_TOKEN`
- `HF_SPACE_REPO`

Exemple `HF_SPACE_REPO`:

```text
https://huggingface.co/spaces/Franck504/fruits_unsupervised_learning_xai
```

## Deploiement Hugging Face Space

Le script `scripts/publish_to_hf_space.sh` copie vers le Space:
- `deployment/app.py`
- `deployment/requirements.txt`
- `models/*.joblib`

Hugging Face Spaces redeploie automatiquement l'application apres chaque commit recu.

L'application expose:
- interface Gradio.
- endpoint `predict_and_explain`.
- endpoint `download_inference_logs`.

## Prediction Et Explication

Une prediction renvoie:

```json
{
  "request_id": "...",
  "cluster": 1,
  "input": {
    "feature_1": 25,
    "feature_2": 8.5
  },
  "explanation": {
    "feature_1": 0.35,
    "feature_2": 0.31
  }
}
```

Interpretation:
- `cluster`: groupe predit par KMeans.
- `explanation`: contributions SHAP calculees sur le proxy RandomForest.
- `request_id`: identifiant unique de l'inference.

## Logging D'Inference

Chaque appel au Space est journalise dans:

```text
inference_logs/inferences.csv
```

Colonnes:
- `timestamp`
- `request_id`
- `feature_1`
- `feature_2`
- `cluster`
- `shap_feature_1`
- `shap_feature_2`

Le bouton `Download inference logs` permet de recuperer ce fichier depuis l'interface Gradio.

## Monitoring Drift Local

Apres avoir telecharge `inferences.csv`, placer le fichier ici:

```text
inference_logs/inferences.csv
```

Puis lancer:

```bash
python scripts/monitor_drift.py \
  --reference data/processed/fruits_clustered.csv \
  --production inference_logs/inferences.csv \
  --reports-dir reports/monitoring
```

Sorties:
- `reports/monitoring/drift_report.json`
- `reports/monitoring/drift_report.html`

Le script compare les donnees de reference et les donnees de production avec PSI.

## Monitoring Evidently AI

Pour generer un rapport Evidently AI:

```bash
python scripts/monitor_evidently.py \
  --reference data/processed/fruits_clustered.csv \
  --production inference_logs/inferences.csv \
  --reports-dir reports/evidently
```

Sorties:
- `reports/evidently/evidently_data_drift_report.json`
- `reports/evidently/evidently_data_drift_report.html`

## Execution Locale

Sans Docker:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
dvc pull
dvc repro
```

Avec Docker:

```bash
docker build -t fruits-mlops .
docker run --rm -v "$(pwd):/app" fruits-mlops
```

## Scenario End-To-End

1. Un changement est fait dans le dataset ou le code.
2. Le developpeur fait `git push`.
3. GitHub Actions demarre.
4. DVC restaure les donnees avec `dvc pull`.
5. DVC relance le pipeline avec `dvc repro`.
6. `train.py` entraine KMeans et le proxy RandomForest.
7. MLflow logge les metriques et enregistre les modeles dans le registry.
8. Le quality gate valide les seuils.
9. Les modeles sont publies vers Hugging Face Space.
10. Le Space se redeploie automatiquement.
11. L'utilisateur fait des predictions.
12. Les predictions sont loggees.
13. Les logs peuvent etre analyses avec le monitoring local ou Evidently.

## Limites Actuelles

- Les variables `feature_1` et `feature_2` n'ont pas de signification metier documentee.
- Le remote DVC est local pour rester gratuit.
- Le monitoring est lance manuellement apres telechargement des logs.
- SHAP explique le modele proxy, pas KMeans directement.

## Ameliorations Possibles

- Remplacer le remote DVC local par un remote cloud.
- Stocker les logs d'inference dans une base externe.
- Planifier automatiquement le monitoring.
- Ajouter alerting webhook ou email.
- Ajouter une validation metier des clusters si un dictionnaire de donnees devient disponible.
