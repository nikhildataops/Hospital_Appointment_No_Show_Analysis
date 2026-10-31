# Hospital Appointment Insights Dashboard

An interactive Streamlit dashboard for exploring the Kaggle Medical Appointment No Shows dataset.

## What's in this folder

- `app.py` — the complete dashboard, including data loading, filters, charts, record download, and the baseline-model summary.
- `appointments_public.csv` — analysis-ready data. Patient IDs and appointment IDs have been removed.
- `requirements.txt` — the Python packages used by the dashboard.
- `.streamlit/config.toml` — the dashboard's color theme.

## Run on your computer

1. Open a terminal in this folder.
2. Install the packages: `pip install -r requirements.txt`
3. Start the dashboard: `streamlit run app.py`
4. If Streamlit asks for an email in the terminal, press Enter to skip it for local use.

## Streamlit Community Cloud

The public GitHub repository should not contain the appointment CSV. It includes health attributes and appointment dates, even though patient and appointment IDs were removed. If the CSV is absent, the deployed app displays a notice instead of the dashboard.

To host a working dashboard, the CSV would need to be made available to the hosting service. Get explicit approval before publishing it with a public app. Streamlit's deployment flow uses a GitHub repository, branch, and `app.py` entry file; its app visibility follows repository visibility by default.

## About the analysis

The dashboard shows descriptive patterns. Its model tab reports a logistic-regression baseline evaluated on a stratified holdout set; it is included for learning and is not intended for clinical or operational decisions. Associations in these historical records do not establish cause and effect.

Dataset source: Kaggle, *Medical Appointment No Shows* (May 2016).
