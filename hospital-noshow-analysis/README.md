# Hospital Appointment & No-Show Analysis

An end-to-end data-science project: clean the data, explore it, test it statistically, predict which patients will miss their appointment, and explore everything in an interactive dashboard.

**Dataset:** [Medical Appointment No-Shows (Kaggle)](https://www.kaggle.com/datasets/joniarroba/noshowappointments) - 110,527 appointments from public hospitals in Vitoria, Brazil (Apr-Jun 2016).

## Headline results

| | |
|---|---|
| Overall no-show rate | **20.2%** (about 1 in 5) |
| Strongest factor | **Waiting time**: same-day 4.6% vs. waits over 2 weeks about 33% |
| Riskiest ages | Teenagers and young adults (about 25-27%); seniors 60-74 lowest (15%) |
| SMS paradox | SMS patients miss *more* overall (27.6% vs 16.7%) only because reminders go to long-wait bookings; within each waiting band they miss *less* |
| Best model | Tuned XGBoost, calibrated: **ROC-AUC 0.736**, PR-AUC 0.380 |
| Business value | Contacting the riskiest 20% of appointments reaches **38%** of all no-shows (random: 20%) |
| Robustness | Patient-grouped CV AUC 0.740; time-based split AUC 0.731 (no leakage or inflation) |

## Project structure

```
hospital-noshow-analysis/
├── data/
│   ├── raw/                     KaggleV2-May-2016.csv (original, untouched)
│   └── processed/               appointments_clean.csv, neighbourhood_summary.csv, test_predictions.csv
├── notebooks/
│   ├── 00_original_eda_reference.ipynb   your original analysis, kept for reference
│   ├── 01_cleaning.ipynb        quality checks, cleaning, feature engineering
│   ├── 02_eda.ipynb             EDA, SMS paradox, chi-square / Mann-Whitney tests, odds ratios
│   └── 03_modeling.ipynb        4 models, tuning, robustness, calibration, SHAP, business gains
├── src/
│   └── data_prep.py             shared cleaning + feature code (used by notebooks and dashboard)
├── models/
│   └── noshow_model.joblib      calibrated model used by the dashboard calculator
├── reports/
│   ├── final_report.pdf / .md   full written report
│   ├── figures/                 24 charts (PNG)
│   ├── model_metrics.json, model_comparison.csv, decile_table.csv, statistical_tests.csv
├── dashboard/
│   └── app.py                   Streamlit dashboard (7 tabs)
├── requirements.txt
├── run_all.sh                   re-runs everything and launches the dashboard
└── README.md
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run dashboard/app.py                          # open the dashboard (all outputs are already included)
# or rebuild everything from the raw CSV:
bash run_all.sh                                         # Windows: run the three notebooks in order, then streamlit
```

Run notebooks in order **01 -> 02 -> 03**. Notebook 03 takes several minutes (hyper-parameter search, cross-validation, SHAP).

## Dashboard features

| Tab | What you can do |
|---|---|
| **Overview** | KPIs (appointments, no-shows, rate, average wait), weekly trend, rate by waiting time and weekday |
| **Patient factors** | Age, gender, health flags, SMS paradox chart, age x waiting-time heatmap |
| **Neighbourhoods** | Ranked best / worst areas with a minimum-volume slider (small areas are noisy) |
| **Risk model** | Metrics, risk-decile chart, SHAP, **downloadable high-risk appointment list** |
| **Risk calculator** | Enter one appointment, get a calibrated probability, risk tier, suggested actions and a waiting-time "what-if" curve |
| **Impact simulator** | Enter your own costs and reminder effectiveness to estimate net yearly benefit of a targeted reminder programme vs. random contact |
| **Data** | Filtered data table, CSV download, data dictionary |

The sidebar filters (dates, neighbourhood, gender, age group, waiting time, SMS, scholarship) apply to the Overview, Patient factors, Neighbourhoods and Data tabs.

> **Using Power BI instead?** Import `data/processed/appointments_clean.csv` (one row per appointment, all engineered columns) and `data/processed/test_predictions.csv` (adds `risk_score` and `risk_tier`). The `missed` column is already 1 = did not attend.

## Methodology highlights

* **Target recoding:** the raw `No-show = Yes` means the patient *missed*. It is recoded to `missed` (1 = missed).
* **No data leakage:** patient-history features (`prior_appts`, `prior_noshows`) count only appointments that had already happened on the booking day. Neighbourhood is target-encoded *inside* the cross-validated pipeline.
* **Imbalanced classes (80/20):** class weights; models compared on ROC-AUC and PR-AUC, not accuracy.
* **Honest validation:** stratified hold-out test set, plus patient-grouped CV and a time-based split.
* **Calibrated probabilities**, so "30% risk" really means about 30%.
* **Statistics:** chi-square with Cramer's V, Mann-Whitney U with rank-biserial r, Wilson confidence intervals, and a multivariable logistic regression (odds ratios).

## Changes compared with the original notebook

Your original EDA (`00_original_eda_reference.ipynb`) is a good starting point. The new notebooks keep its questions and make a few corrections:

* **Age 0 is kept.** The original dropped `Age > 0` only, which removed 3,539 genuine infants (only 1 row, age -1, is invalid).
* **Age groups** used strict `<` / `>` on both sides, which skipped patients aged exactly 3, 18, 30 and 60. The new bins cover every age.
* **Waiting days** are now computed on the cleaned data in one place (`data_prep.py`), and the 5 impossible rows (appointment before booking) are removed.
* **Condition plots** now show the *no-show rate* by condition (the originals plotted average age).
* **New:** statistical tests, the SMS confounding analysis, patient history, the predictive model, and the dashboard.

## Limitations

* Only about six weeks of appointments, so patient history is shallow.
* No information on appointment type, doctor, reason, distance, weather or transport.
* Results are associations, not proof of cause. Testing whether SMS reminders *work* requires an A/B test.
* The model is a moderate predictor (AUC 0.74). Use it to **prioritise outreach and add support**, never to deny, delay or penalise care.

## Tools

Python 3.10+, pandas, scipy, statsmodels, scikit-learn, XGBoost, SHAP, matplotlib, seaborn, Plotly, Streamlit.
