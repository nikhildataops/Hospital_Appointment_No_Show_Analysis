#!/usr/bin/env bash
# Re-run the whole pipeline: clean -> EDA -> model, then launch the dashboard.
set -e
cd "$(dirname "$0")/notebooks"
export MPLBACKEND=Agg
for nb in 01_cleaning 02_eda 03_modeling; do
  echo ">>> Running $nb.ipynb"
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3000 "$nb.ipynb"
done
cd ..
echo ">>> Starting dashboard"
streamlit run dashboard/app.py
