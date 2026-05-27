#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN is required}"
: "${HF_SPACE_REPO:?HF_SPACE_REPO is required, e.g. https://huggingface.co/spaces/Franck504/fruits_usl_xai}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

git clone "https://user:${HF_TOKEN}@${HF_SPACE_REPO#https://}" "$TMP_DIR/space"

mkdir -p "$TMP_DIR/space/models"
cp -f models/kmeans_fruits.joblib "$TMP_DIR/space/models/"
cp -f models/scaler_fruits.joblib "$TMP_DIR/space/models/"
cp -f models/proxy_rf_for_shap.joblib "$TMP_DIR/space/models/"

cp -f deployment/app.py "$TMP_DIR/space/app.py"
cp -f deployment/requirements.txt "$TMP_DIR/space/requirements.txt"

cd "$TMP_DIR/space"
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git add .
if git diff --cached --quiet; then
  echo "No changes to push to Space."
  exit 0
fi
git commit -m "CI: update models and app"
git push origin main
