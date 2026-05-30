## Description du Projet

C'est un pipeline MLOps complet pour faire du clustering non supervisé (KMeans) sur des données de fruits avec explicabilité XAI via SHAP. Voici la structure :

## Objectif Principal

Segmenter automatiquement des observations (2 features numériques) en clusters, puis expliquer pourquoi chaque point est affecté à un cluster.

## Architecture Générale

🔧 Ce que vous devez savoir pour améliorer le projet

1. Pipeline d'Entraînement (`src/train.py`)
   - ✅ Teste automatiquement `k=2` à `k=10` (KMeans)
   - ✅ Sélectionne le meilleur `k` par *silhouette score*
   - ✅ Entraîne un Random Forest proxy (300 arbres) pour l'explication SHAP
   - ✅ Mesure 3 métriques : `silhouette`, `calinski_harabasz`, `davies_bouldin`
   - ✅ Valide la stabilité en relançant avec 7 seeds différents + calcul de l'ARI
   - À savoir : Les métriques de clustering sont loguées dans MLflow. Les rapports sont sauvegardés dans `reports/`.

2. Versionnement des Données (DVC)
   - Dataset source : `data/raw/fruits.csv` (suivi par DVC)
   - Dataset enrichi (avec clusters) : `data/processed/fruits_clustered.csv`
   - Remote DVC local : `dvc_remote/files/` (présent dans le dépôt pour démo locale)

3. MLflow Local
   - Chaque entraînement crée une run MLflow
   - Logger : paramètres, métriques, artefacts et modèles (scaler, KMeans, proxy RF)
   - Registry local avec alias "production"

4. Déploiement (Hugging Face Spaces)
   - Interface Gradio (`deployment/app.py`) : prend 2 inputs (`feature_1`, `feature_2`)
   - Retourne : cluster assigné + explication SHAP pour chaque feature
   - Les inférences sont loggées localement dans `inference_logs/inferences.csv`

5. Monitoring
   - Scripts : `scripts/monitor_drift.py` (PSI local), `scripts/monitor_evidently.py` (Evidently AI)
   - Les logs d'inférence permettent de déterminer si du drift est survenu

## Fichiers clés

- `src/train.py` — pipeline d'entraînement et génération des artefacts
- `data/raw/fruits.csv` — données brutes (DVC)
- `data/processed/fruits_clustered.csv` — données enrichies avec labels
- `models/` — modèles sauvegardés (`scaler`, `kmeans`, `proxy_rf`)
- `reports/` — rapports d'entraînement et monitoring
- `deployment/` — code pour le Space HF (`app.py`, `requirements.txt`)
- `inference_logs/inferences.csv` — journal d'inférence

## Utilisation rapide

1. Installer les dépendances :

```bash
pip install -r requirements.txt
```

2. Récupérer les données DVC :

```bash
dvc pull
```

3. Lancer l'entraînement complet :

```bash
python src/train.py
```

4. Déployer vers Hugging Face Space (exemple de script) :

```bash
./scripts/publish_to_hf_space.sh
```

5. Lancer le monitoring PSI local :

```bash
python scripts/monitor_drift.py --reference data/processed/fruits_clustered.csv --production inference_logs/inferences.csv --reports-dir reports/monitoring
```

