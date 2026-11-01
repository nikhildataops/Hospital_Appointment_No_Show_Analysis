"""
Shared data-preparation code for the Hospital Appointment No-Show project.

Used by:
  - notebooks/01_cleaning.ipynb  (builds data/processed/appointments_clean.csv)
  - dashboard/app.py             (can rebuild features for the risk-scoring tool)

Key design choices
------------------
* The raw column `No-show` is confusing ("Yes" = patient did NOT come). We recode it to
  `missed` (1 = missed, 0 = attended).
* Age 0 is KEPT (3,539 rows are real infants). Only age < 0 is invalid.
* Patient-history features (`prior_appts`, `prior_noshows`) only use appointments whose
  date is strictly BEFORE the day the current appointment was booked. That is exactly
  the information a hospital would have at booking time, so the model has no data leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

AGE_BINS = [-1, 2, 12, 17, 29, 44, 59, 74, 200]
AGE_LABELS = ["Infant (0-2)", "Child (3-12)", "Teen (13-17)", "Young adult (18-29)",
              "Adult (30-44)", "Middle age (45-59)", "Senior (60-74)", "Elderly (75+)"]

WAIT_BINS = [-1, 0, 3, 7, 14, 30, 10_000]
WAIT_LABELS = ["Same day", "1-3 days", "4-7 days", "8-14 days", "15-30 days", "30+ days"]

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(raw: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Fix names/types, build the target and drop invalid rows."""
    df = raw.copy()
    n0 = len(df)

    df = df.rename(columns={
        "PatientId": "patient_id", "AppointmentID": "appointment_id", "Gender": "gender",
        "ScheduledDay": "scheduled_dt", "AppointmentDay": "appointment_date", "Age": "age",
        "Neighbourhood": "neighbourhood", "Scholarship": "scholarship",
        "Hipertension": "hypertension", "Diabetes": "diabetes", "Alcoholism": "alcoholism",
        "Handcap": "handicap", "SMS_received": "sms_received", "No-show": "no_show",
    })
    df["patient_id"] = df["patient_id"].astype("int64")
    df["scheduled_dt"] = pd.to_datetime(df["scheduled_dt"]).dt.tz_localize(None)
    df["appointment_date"] = pd.to_datetime(df["appointment_date"]).dt.tz_localize(None).dt.normalize()
    df["scheduled_date"] = df["scheduled_dt"].dt.normalize()

    # Target: 1 = patient did NOT show up
    df["missed"] = (df["no_show"] == "Yes").astype("int8")
    df = df.drop(columns="no_show")

    df["wait_days"] = (df["appointment_date"] - df["scheduled_date"]).dt.days

    bad_age = df["age"] < 0
    bad_wait = df["wait_days"] < 0
    df = df[~bad_age & ~bad_wait].copy()
    if verbose:
        print(f"Rows in: {n0:,} | invalid age (<0): {bad_age.sum()} | "
              f"appointment before booking: {bad_wait.sum()} | rows out: {len(df):,}")

    df["handicap_level"] = df["handicap"].clip(upper=3)  # levels 3-4 are tiny; merge them
    df["handicap"] = (df["handicap"] > 0).astype("int8")  # simple yes / no flag
    return df.reset_index(drop=True)


def _history_features(df: pd.DataFrame) -> pd.DataFrame:
    """For every appointment: how many earlier appointments did this patient have whose
    outcome was already known on the booking day, and how many of them were missed?"""
    prior_appts = np.zeros(len(df), dtype=np.int32)
    prior_miss = np.zeros(len(df), dtype=np.int32)

    multi = df["patient_id"].map(df["patient_id"].value_counts()) > 1
    idx_multi = df.index[multi]
    for _, g in df.loc[idx_multi].groupby("patient_id"):
        order = g["appointment_date"].values.argsort()
        appt_sorted = g["appointment_date"].values[order]
        miss_cum = np.concatenate([[0], np.cumsum(g["missed"].values[order])])
        # number of earlier appointments with appointment_date < this booking date
        pos = np.searchsorted(appt_sorted, g["scheduled_date"].values, side="left")
        prior_appts[g.index] = pos
        prior_miss[g.index] = miss_cum[pos]

    out = df.copy()
    out["prior_appts"] = prior_appts
    out["prior_noshows"] = prior_miss
    out["prior_noshow_rate"] = np.where(prior_appts > 0, prior_miss / np.maximum(prior_appts, 1), np.nan)
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age_group"] = pd.cut(out["age"], bins=AGE_BINS, labels=AGE_LABELS)
    out["wait_group"] = pd.cut(out["wait_days"], bins=WAIT_BINS, labels=WAIT_LABELS)
    out["same_day"] = (out["wait_days"] == 0).astype("int8")

    out["appointment_weekday"] = out["appointment_date"].dt.day_name()
    out["appointment_dow"] = out["appointment_date"].dt.dayofweek
    out["scheduled_weekday"] = out["scheduled_date"].dt.day_name()
    out["scheduled_hour"] = out["scheduled_dt"].dt.hour
    out["appointment_month"] = out["appointment_date"].dt.to_period("M").astype(str)
    out["appointment_week"] = out["appointment_date"].dt.to_period("W").dt.start_time

    out["num_conditions"] = (out[["hypertension", "diabetes", "alcoholism", "handicap"]].sum(axis=1)).astype("int8")
    out["is_female"] = (out["gender"] == "F").astype("int8")

    out = _history_features(out)
    out["has_history"] = (out["prior_appts"] > 0).astype("int8")
    return out


def build_dataset(raw_path: str, verbose: bool = True) -> pd.DataFrame:
    return add_features(clean(load_raw(raw_path), verbose=verbose))


# Columns fed to the ML models (all known at booking time -> no leakage)
NUMERIC_FEATURES = ["age", "wait_days", "scheduled_hour", "appointment_dow", "prior_appts",
                    "prior_noshows", "num_conditions"]
BINARY_FEATURES = ["is_female", "scholarship", "hypertension", "diabetes", "alcoholism",
                   "handicap", "sms_received", "same_day"]
CATEGORICAL_FEATURES = ["neighbourhood"]
