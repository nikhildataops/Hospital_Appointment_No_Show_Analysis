"""
Hospital Appointment No-Show Dashboard  (Streamlit)

Run from the project root:
    streamlit run dashboard/app.py

Tabs
  1. Overview          KPIs, trend, weekday, waiting time
  2. Patient factors   age, gender, conditions, SMS, age x wait heatmap
  3. Neighbourhoods    ranked clinics / areas with minimum-volume filter
  4. Risk model        performance, risk deciles, high-risk patient list (downloadable)
  5. Risk calculator   score a single appointment
  6. Impact simulator  what could a reminder programme save?
  7. Data              filtered data, download, data dictionary
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "appointments_clean.csv"
PRED = ROOT / "data" / "processed" / "test_predictions.csv"
MODEL = ROOT / "models" / "noshow_model.joblib"
METRICS = ROOT / "reports" / "model_metrics.json"
DECILES = ROOT / "reports" / "decile_table.csv"
FIGS = ROOT / "reports" / "figures"

BLUE, RED, GREY, GREEN, AMBER = "#2E86AB", "#E4572E", "#9AA5B1", "#4C9F70", "#F2A541"
AGE_ORDER = ["Infant (0-2)", "Child (3-12)", "Teen (13-17)", "Young adult (18-29)", "Adult (30-44)",
             "Middle age (45-59)", "Senior (60-74)", "Elderly (75+)"]
WAIT_ORDER = ["Same day", "1-3 days", "4-7 days", "8-14 days", "15-30 days", "30+ days"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

st.set_page_config(page_title="Hospital No-Show Dashboard", page_icon="🏥", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem;}
    div[data-testid="stMetric"] {background: rgba(46,134,171,.08); border-radius: 12px; padding: 14px 16px;}
    h1, h2, h3 {letter-spacing: -0.01em;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ data loading
@st.cache_data(show_spinner="Loading data ...")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=["scheduled_dt", "appointment_date", "scheduled_date", "appointment_week"])
    df["age_group"] = pd.Categorical(df["age_group"], AGE_ORDER, ordered=True)
    df["wait_group"] = pd.Categorical(df["wait_group"], WAIT_ORDER, ordered=True)
    return df


@st.cache_data
def load_predictions() -> pd.DataFrame | None:
    if not PRED.exists():
        return None
    p = pd.read_csv(PRED, parse_dates=["appointment_date", "scheduled_date"])
    p["risk_tier"] = pd.Categorical(p["risk_tier"], ["Low", "Medium", "High"], ordered=True)
    return p


@st.cache_resource
def load_model():
    if not MODEL.exists():
        return None
    try:
        return joblib.load(MODEL)
    except Exception as exc:  # e.g. model saved with a different xgboost / scikit-learn version
        st.sidebar.warning("Saved model could not be loaded (library version mismatch). "
                           "Run `python src/retrain_model.py` once, then restart the dashboard.")
        print("Model load failed:", exc)
        return None


@st.cache_data
def load_metrics():
    return json.loads(METRICS.read_text()) if METRICS.exists() else None


@st.cache_data
def load_deciles():
    return pd.read_csv(DECILES, index_col=0) if DECILES.exists() else None


if not DATA.exists():
    st.error("Cleaned data not found. Run `notebooks/01_cleaning.ipynb` first (it creates data/processed/appointments_clean.csv).")
    st.stop()

df_all = load_data()
preds = load_predictions()
bundle = load_model()
metrics = load_metrics()
deciles = load_deciles()


# ------------------------------------------------------------------ helpers
def rate(s: pd.Series) -> float:
    return float(s.mean()) if len(s) else float("nan")


def grouped_rate(d: pd.DataFrame, col: str, min_n: int = 1) -> pd.DataFrame:
    g = d.groupby(col, observed=True)["missed"].agg(rate="mean", appointments="size", no_shows="sum").reset_index()
    return g[g["appointments"] >= min_n]


def rate_bar(g: pd.DataFrame, x: str, base: float, title: str, horizontal: bool = False, height: int = 360):
    g = g.copy()
    g["pct"] = g["rate"] * 100
    if horizontal:
        fig = px.bar(g, y=x, x="pct", orientation="h", text=g["pct"].map("{:.1f}%".format),
                     hover_data={"appointments": ":,", "no_shows": ":,", "pct": ":.1f"})
        fig.add_vline(x=base * 100, line_dash="dash", line_color=RED, annotation_text=f"Average {base:.1%}")
    else:
        fig = px.bar(g, x=x, y="pct", text=g["pct"].map("{:.1f}%".format),
                     hover_data={"appointments": ":,", "no_shows": ":,", "pct": ":.1f"})
        fig.add_hline(y=base * 100, line_dash="dash", line_color=RED, annotation_text=f"Average {base:.1%}", annotation_position="top left")
    fig.update_traces(marker_color=BLUE, textposition="outside", cliponaxis=False)
    fig.update_layout(title=title, height=height, margin=dict(t=50, b=10, l=10, r=10),
                      xaxis_title=None if not horizontal else "No-show rate (%)",
                      yaxis_title="No-show rate (%)" if not horizontal else None, showlegend=False)
    return fig


def small_sample_note(d: pd.DataFrame):
    if len(d) < 300:
        st.warning(f"Only {len(d):,} appointments match the filters - rates may be unreliable.")


# ------------------------------------------------------------------ sidebar filters
st.sidebar.title("🏥 Filters")
dmin, dmax = df_all["appointment_date"].min().date(), df_all["appointment_date"].max().date()
date_range = st.sidebar.date_input("Appointment dates", (dmin, dmax), min_value=dmin, max_value=dmax)
if isinstance(date_range, tuple) and len(date_range) == 2:
    d0, d1 = date_range
else:
    d0, d1 = dmin, dmax

neigh_all = sorted(df_all["neighbourhood"].unique())
neigh_sel = st.sidebar.multiselect("Neighbourhood (empty = all)", neigh_all)
gender_sel = st.sidebar.multiselect("Gender", ["F", "M"], default=["F", "M"])
age_sel = st.sidebar.multiselect("Age group", AGE_ORDER, default=AGE_ORDER)
wait_sel = st.sidebar.multiselect("Waiting time", WAIT_ORDER, default=WAIT_ORDER)
sms_opt = st.sidebar.radio("SMS reminder", ["All", "Received", "Not received"], horizontal=True)
sch_opt = st.sidebar.radio("Welfare scholarship", ["All", "Enrolled", "Not enrolled"], horizontal=True)

mask = (df_all["appointment_date"].dt.date.between(d0, d1)
        & df_all["gender"].isin(gender_sel) & df_all["age_group"].isin(age_sel) & df_all["wait_group"].isin(wait_sel))
if neigh_sel:
    mask &= df_all["neighbourhood"].isin(neigh_sel)
if sms_opt != "All":
    mask &= df_all["sms_received"] == (1 if sms_opt == "Received" else 0)
if sch_opt != "All":
    mask &= df_all["scholarship"] == (1 if sch_opt == "Enrolled" else 0)
df = df_all[mask]

BASE = rate(df_all["missed"])
st.sidebar.markdown("---")
st.sidebar.caption(f"**{len(df):,}** of {len(df_all):,} appointments selected")
st.sidebar.caption("Data: Medical Appointment No-Shows (Kaggle), Vitoria, Brazil, Apr-Jun 2016")

st.title("Hospital Appointment & No-Show Dashboard")
st.caption("Who misses appointments, why, and what can we do about it?")

if df.empty:
    st.warning("No appointments match the current filters.")
    st.stop()

tabs = st.tabs(["📊 Overview", "👥 Patient factors", "📍 Neighbourhoods", "🤖 Risk model",
                "🧮 Risk calculator", "💰 Impact simulator", "🗂️ Data"])

# ================================================================== 1 OVERVIEW
with tabs[0]:
    small_sample_note(df)
    n, miss = len(df), int(df["missed"].sum())
    r = rate(df["missed"])
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Appointments", f"{n:,}")
    c2.metric("No-shows", f"{miss:,}")
    c3.metric("No-show rate", f"{r:.1%}", f"{(r - BASE) * 100:+.1f} pts vs all data", delta_color="inverse")
    c4.metric("Avg waiting time", f"{df['wait_days'].mean():.1f} days")
    c5.metric("Unique patients", f"{df['patient_id'].nunique():,}")

    st.markdown("#### Weekly trend")
    wk = df.groupby("appointment_week").agg(appointments=("missed", "size"), rate=("missed", "mean")).reset_index()
    wk = wk[wk["appointments"] >= 200]
    fig = go.Figure()
    fig.add_bar(x=wk["appointment_week"], y=wk["appointments"], name="Appointments", marker_color="#D6E6F0")
    fig.add_scatter(x=wk["appointment_week"], y=wk["rate"] * 100, name="No-show rate (%)", yaxis="y2",
                    mode="lines+markers", line=dict(color=RED, width=3))
    fig.update_layout(height=340, margin=dict(t=10, b=10), yaxis=dict(title="Appointments"),
                      yaxis2=dict(title="No-show rate (%)", overlaying="y", side="right", range=[0, max(30, wk["rate"].max() * 120)]),
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    with a:
        g = grouped_rate(df, "wait_group")
        st.plotly_chart(rate_bar(g, "wait_group", BASE, "No-show rate by waiting time"), width="stretch")
    with b:
        g = grouped_rate(df, "appointment_weekday", 20).set_index("appointment_weekday").reindex([d for d in DAYS]).dropna().reset_index()
        st.plotly_chart(rate_bar(g, "appointment_weekday", BASE, "No-show rate by weekday"), width="stretch")

    st.info("**Key insight:** waiting time is the strongest driver. Same-day appointments are rarely missed, "
            "while waits over two weeks are missed roughly 7x as often.")

# ================================================================== 2 PATIENT FACTORS
with tabs[1]:
    a, b = st.columns(2)
    with a:
        g = grouped_rate(df, "age_group")
        st.plotly_chart(rate_bar(g, "age_group", BASE, "No-show rate by age group"), width="stretch")
    with b:
        g = grouped_rate(df, "gender")
        g["gender"] = g["gender"].map({"F": "Female", "M": "Male"})
        st.plotly_chart(rate_bar(g, "gender", BASE, "No-show rate by gender"), width="stretch")

    st.markdown("#### Health and social factors")
    rows = []
    for col, label in [("scholarship", "Welfare scholarship"), ("hypertension", "Hypertension"), ("diabetes", "Diabetes"),
                       ("alcoholism", "Alcoholism"), ("handicap", "Handicap"), ("sms_received", "SMS received")]:
        for v in (0, 1):
            s = df[df[col] == v]["missed"]
            if len(s) >= 30:
                rows.append({"factor": label, "group": "Yes" if v else "No", "rate": s.mean() * 100, "n": len(s)})
    f = pd.DataFrame(rows)
    if not f.empty:
        fig = px.bar(f, y="factor", x="rate", color="group", barmode="group", orientation="h",
                     color_discrete_map={"No": GREY, "Yes": BLUE}, hover_data={"n": ":,"})
        fig.add_vline(x=BASE * 100, line_dash="dash", line_color=RED)
        fig.update_layout(height=420, xaxis_title="No-show rate (%)", yaxis_title=None, margin=dict(t=10),
                          legend=dict(orientation="h", y=1.08, title=None))
        st.plotly_chart(fig, width="stretch")
    st.caption("Chronic-condition patients are older, and older patients attend more, so raw condition effects are partly an age effect.")

    st.markdown("#### SMS reminders: the paradox")
    g = df.groupby(["wait_group", "sms_received"], observed=True)["missed"].agg(rate="mean", n="size").reset_index()
    g["SMS"] = g["sms_received"].map({0: "No SMS", 1: "SMS sent"})
    g["rate"] *= 100
    fig = px.bar(g, x="wait_group", y="rate", color="SMS", barmode="group", color_discrete_map={"No SMS": GREY, "SMS sent": BLUE},
                 hover_data={"n": ":,"}, category_orders={"wait_group": WAIT_ORDER})
    fig.update_layout(height=360, yaxis_title="No-show rate (%)", xaxis_title="Waiting time", margin=dict(t=10),
                      legend=dict(orientation="h", y=1.1, title=None))
    st.plotly_chart(fig, width="stretch")
    st.caption("Overall, SMS patients miss *more* - but only because reminders go mostly to long-wait appointments. "
               "Within each waiting band, SMS patients miss less. (Observational data: not proof of cause.)")

    st.markdown("#### Age x waiting time heatmap")
    hm = df.pivot_table(index="age_group", columns="wait_group", values="missed", aggfunc="mean", observed=True) * 100
    cnt = df.pivot_table(index="age_group", columns="wait_group", values="missed", aggfunc="size", observed=True)
    hm = hm.where(cnt >= 30)
    fig = px.imshow(hm, text_auto=".0f", color_continuous_scale="Reds", aspect="auto", labels=dict(color="No-show %"))
    fig.update_layout(height=420, margin=dict(t=10), xaxis_title="Waiting time", yaxis_title=None)
    st.plotly_chart(fig, width="stretch")
    st.caption("Cells with fewer than 30 appointments are left blank. The riskiest combination is teens / young adults with long waits.")

# ================================================================== 3 NEIGHBOURHOODS
with tabs[2]:
    c1, c2 = st.columns([1, 3])
    min_n = c1.slider("Minimum appointments", 50, 3000, 500, step=50,
                      help="Small neighbourhoods give noisy rates, so they are excluded from the ranking.")
    top_n = c1.slider("Show top / bottom", 5, 20, 10)
    g = grouped_rate(df, "neighbourhood", min_n).sort_values("rate")
    if g.empty:
        c2.warning("No neighbourhood reaches that minimum with the current filters. Lower the minimum.")
    else:
        sel = pd.concat([g.head(top_n), g.tail(top_n)]).drop_duplicates()
        sel["pct"] = sel["rate"] * 100
        fig = px.bar(sel, y="neighbourhood", x="pct", orientation="h", text=sel["pct"].map("{:.1f}%".format),
                     hover_data={"appointments": ":,", "no_shows": ":,", "pct": ":.1f"},
                     color="pct", color_continuous_scale=["#2E86AB", "#F2E9E4", "#E4572E"])
        fig.add_vline(x=BASE * 100, line_dash="dash", line_color="black")
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(height=max(380, 26 * len(sel)), margin=dict(t=10), coloraxis_showscale=False,
                          xaxis_title="No-show rate (%)", yaxis_title=None)
        c2.plotly_chart(fig, width="stretch")
        st.markdown(f"**{len(g)} neighbourhoods** have at least {min_n} appointments.")
        t = g.sort_values("rate", ascending=False).assign(rate=lambda x: (x["rate"] * 100).round(1))
        t = t.rename(columns={"neighbourhood": "Neighbourhood", "rate": "No-show %", "appointments": "Appointments", "no_shows": "No-shows"})
        st.dataframe(t, width="stretch", hide_index=True)

# ================================================================== 4 RISK MODEL
with tabs[3]:
    if metrics is None or preds is None:
        st.warning("Model outputs not found. Run `notebooks/03_modeling.ipynb` first.")
    else:
        st.markdown(f"**Model:** {metrics['best_model']} (probabilities calibrated) &nbsp;|&nbsp; trained on {metrics['n_train']:,} appointments, "
                    f"evaluated on {metrics['n_test']:,} held-out appointments")
        best = metrics["test_results"][metrics["best_model"]]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC-AUC", f"{best['ROC-AUC']:.3f}", help="0.5 = random guessing, 1.0 = perfect")
        c2.metric("PR-AUC", f"{best['PR-AUC']:.3f}", f"baseline {metrics['base_rate']:.3f}", delta_color="off")
        c3.metric("Riskiest 20% reach", f"{metrics['top20_capture']:.0%} of no-shows", help="Random contact would reach 20%")
        c4.metric("Top-decile lift", f"{metrics['top_decile_lift']:.1f}x")
        st.caption(f"Robustness: patient-grouped CV ROC-AUC {metrics['grouped_cv_auc']:.3f}; time-based split ROC-AUC {metrics['time_split_auc']:.3f}.")

        a, b = st.columns(2)
        with a:
            comp = pd.DataFrame(metrics["test_results"]).T[["ROC-AUC", "PR-AUC", "Recall", "Precision", "F1"]]
            st.markdown("##### Model comparison (test set, threshold 0.50)")
            st.dataframe(comp.round(3), width="stretch")
        with b:
            if deciles is not None:
                d = deciles.reset_index()
                fig = px.bar(d, x=d["decile"].astype(str), y=d["actual_rate"] * 100, text=(d["actual_rate"] * 100).round(1))
                fig.add_hline(y=metrics["base_rate"] * 100, line_dash="dash", line_color=RED)
                fig.update_traces(marker_color=BLUE, textposition="outside")
                fig.update_layout(height=330, margin=dict(t=40, b=10), title="Actual no-show rate by risk decile (1 = riskiest)",
                                  xaxis_title="Risk decile", yaxis_title="No-show rate (%)")
                st.plotly_chart(fig, width="stretch")

        fshap, fgain = FIGS / "17_shap_beeswarm.png", FIGS / "15_gains_lift.png"
        a, b = st.columns(2)
        if fshap.exists():
            a.image(str(fshap), caption="SHAP: what drives predicted risk", width="stretch")
        if fgain.exists():
            b.image(str(fgain), caption="Gains and lift", width="stretch")

        st.markdown("---")
        st.markdown("### 📋 High-risk appointment list")
        st.caption("Scores come from the held-out test set (the model never saw these appointments during training).")
        f1, f2, f3 = st.columns(3)
        tiers = f1.multiselect("Risk tier", ["High", "Medium", "Low"], default=["High"])
        min_risk = f2.slider("Minimum risk score", 0.0, 1.0, 0.0, 0.05)
        nb_f = f3.multiselect("Neighbourhood", sorted(preds["neighbourhood"].unique()))
        p = preds[preds["risk_tier"].isin(tiers) & (preds["risk_score"] >= min_risk)]
        if nb_f:
            p = p[p["neighbourhood"].isin(nb_f)]
        p = p.sort_values("risk_score", ascending=False)
        st.markdown(f"**{len(p):,}** appointments listed - actual no-show rate in this list: **{rate(p['missed']):.1%}**" if len(p) else "No appointments match.")
        show = p[["appointment_id", "patient_id", "appointment_date", "age", "gender", "neighbourhood", "wait_days",
                  "sms_received", "prior_noshows", "risk_score", "risk_tier", "missed"]].copy()
        show["risk_score"] = (show["risk_score"] * 100).round(1)
        show["appointment_date"] = show["appointment_date"].dt.date
        show = show.rename(columns={"risk_score": "risk %", "missed": "actually missed"})
        st.dataframe(show.head(1000), width="stretch", hide_index=True)
        st.download_button("⬇️ Download list as CSV", show.to_csv(index=False).encode(), "high_risk_appointments.csv", "text/csv")

# ================================================================== 5 RISK CALCULATOR
with tabs[4]:
    st.markdown("### Score a single appointment")
    if bundle is None or preds is None:
        st.warning("Model not available. Run `python src/retrain_model.py` (about 2 minutes), then restart the dashboard.")
    else:
        with st.form("calc"):
            c1, c2, c3 = st.columns(3)
            age = c1.number_input("Patient age", 0, 110, 30)
            gender = c1.selectbox("Gender", ["F", "M"])
            nbh = c1.selectbox("Neighbourhood", neigh_all, index=neigh_all.index("JARDIM CAMBURI") if "JARDIM CAMBURI" in neigh_all else 0)
            wait = c2.number_input("Days between booking and appointment", 0, 180, 14)
            dow = c2.selectbox("Appointment weekday", DAYS, index=1)
            hour = c2.slider("Hour the appointment was booked", 6, 21, 10)
            sms = c3.checkbox("SMS reminder will be sent", value=True)
            sch = c3.checkbox("Welfare scholarship (Bolsa Familia)")
            prior_a = c3.number_input("Previous appointments on record", 0, 100, 0)
            prior_m = c3.number_input("...of which missed", 0, 100, 0)
            hyp = c1.checkbox("Hypertension")
            dia = c2.checkbox("Diabetes")
            alc = c3.checkbox("Alcoholism")
            hand = c1.checkbox("Handicap")
            go_btn = st.form_submit_button("Calculate risk", type="primary")

        if go_btn:
            if prior_m > prior_a:
                st.error("Missed appointments cannot exceed previous appointments.")
            else:
                row = pd.DataFrame([{
                    "age": age, "wait_days": wait, "scheduled_hour": hour, "appointment_dow": DAYS.index(dow),
                    "prior_appts": prior_a, "prior_noshows": prior_m,
                    "num_conditions": int(hyp) + int(dia) + int(alc) + int(hand),
                    "is_female": int(gender == "F"), "scholarship": int(sch), "hypertension": int(hyp), "diabetes": int(dia),
                    "alcoholism": int(alc), "handicap": int(hand), "sms_received": int(sms), "same_day": int(wait == 0),
                    "neighbourhood": nbh}])[bundle["features"]]
                prob = float(bundle["model"].predict_proba(row)[:, 1][0])
                q_med, q_high = preds["risk_score"].quantile([0.60, 0.85])
                tier = "High" if prob >= q_high else "Medium" if prob >= q_med else "Low"
                color = {"High": RED, "Medium": AMBER, "Low": GREEN}[tier]

                a, b = st.columns([1, 1])
                with a:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=prob * 100, number={"suffix": "%"},
                        title={"text": f"Estimated no-show probability - <b>{tier} risk</b>"},
                        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                               "steps": [{"range": [0, q_med * 100], "color": "#E3F1E8"},
                                         {"range": [q_med * 100, q_high * 100], "color": "#FCEFD6"},
                                         {"range": [q_high * 100, 100], "color": "#F9DDD5"}]}))
                    fig.update_layout(height=300, margin=dict(t=60, b=0))
                    st.plotly_chart(fig, width="stretch")
                with b:
                    st.markdown(f"**Average patient in the data:** {metrics['base_rate']:.1%}  \n**This appointment:** {prob:.1%}  "
                                f"({prob / metrics['base_rate']:.1f}x the average)")
                    actions = {
                        "High": ["Call or WhatsApp the patient 1-2 days before", "Ask them to confirm - offer an easy reschedule option",
                                 "Consider a waitlist backup for this slot", "If the wait is long, try offering an earlier slot"],
                        "Medium": ["Send an SMS reminder 2 days and 1 day before", "Include a one-tap cancel / reschedule link"],
                        "Low": ["Standard reminder is enough"]}[tier]
                    st.markdown("**Suggested actions**")
                    for x in actions:
                        st.markdown(f"- {x}")
                st.caption("A statistical estimate for prioritising outreach - not a judgement about any individual. "
                           "Never use it to deny or delay care.")

                # what-if on waiting time
                st.markdown("##### What if the patient were seen sooner?")
                waits = [0, 1, 3, 7, 14, 21, 30, 60]
                what = pd.concat([row.assign(wait_days=w, same_day=int(w == 0)) for w in waits], ignore_index=True)
                pr = bundle["model"].predict_proba(what)[:, 1] * 100
                fig = px.line(x=waits, y=pr, markers=True, labels={"x": "Waiting days", "y": "Predicted no-show %"})
                fig.update_traces(line_color=BLUE)
                fig.add_vline(x=wait, line_dash="dot", line_color=RED, annotation_text="current")
                fig.update_layout(height=300, margin=dict(t=10))
                st.plotly_chart(fig, width="stretch")
                st.caption("Other inputs held constant. The model learned this pattern from the data; it does not prove that shortening a wait would cause attendance.")

# ================================================================== 6 IMPACT SIMULATOR
with tabs[5]:
    st.markdown("### What could a targeted reminder programme save?")
    st.caption("The model's ranking quality comes from real test data. **Everything in the left column is your assumption** - change it to match your hospital.")
    if deciles is None or metrics is None:
        st.warning("Run `notebooks/03_modeling.ipynb` first.")
    else:
        c1, c2 = st.columns([1, 2])
        with c1:
            appts = c1.number_input("Appointments per week", 100, 20000, 1000, step=100)
            base_r = c1.slider("Current no-show rate (%)", 5.0, 40.0, float(round(metrics["base_rate"] * 100, 1)), 0.5) / 100
            cost_slot = c1.number_input("Cost of one missed slot (currency)", 0, 10000, 60,
                                        help="Lost revenue + idle staff time. Use your own number.")
            cost_contact = c1.number_input("Cost of one extra contact (call/WhatsApp)", 0.0, 100.0, 1.5, step=0.5)
            eff = c1.slider("Share of contacted no-shows who then attend (%)", 0, 60, 20,
                            help="Published reminder studies often show modest effects. Be conservative.") / 100
            weeks = c1.slider("Weeks per year", 1, 52, 50)

        cum = np.r_[0.0, deciles["cum_capture_%"].values / 100]
        ks = np.linspace(0, 1, 101)
        capture = np.interp(ks, np.linspace(0, 1, 11), cum)
        reached = appts * base_r * capture
        saved = reached * eff
        contacts = appts * ks
        net = saved * cost_slot - contacts * cost_contact
        opt = int(np.argmax(net))

        with c2:
            fig = go.Figure()
            fig.add_scatter(x=ks * 100, y=net * weeks, mode="lines", name="Net benefit / year", line=dict(color=BLUE, width=3))
            fig.add_scatter(x=[ks[opt] * 100], y=[net[opt] * weeks], mode="markers", marker=dict(size=12, color=RED), name="Best share to contact")
            fig.update_layout(height=340, margin=dict(t=10), xaxis_title="% of appointments contacted (riskiest first)",
                              yaxis_title="Net benefit per year", legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig, width="stretch")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Best share to contact", f"{ks[opt]:.0%}")
        m2.metric("No-shows reached / week", f"{reached[opt]:.0f}", f"of {appts * base_r:.0f}", delta_color="off")
        m3.metric("Slots recovered / year", f"{saved[opt] * weeks:,.0f}")
        m4.metric("Net benefit / year", f"{net[opt] * weeks:,.0f}")
        random_net = (appts * base_r * ks[opt] * eff * cost_slot - appts * ks[opt] * cost_contact) * weeks
        st.caption(f"For comparison, contacting the same number of patients **at random** would net about {random_net:,.0f} per year. "
                   "The difference is the value of using the risk model.")

# ================================================================== 7 DATA
with tabs[6]:
    st.markdown("### Filtered data")
    cols = ["appointment_id", "patient_id", "gender", "age", "age_group", "neighbourhood", "scheduled_date", "appointment_date",
            "wait_days", "wait_group", "appointment_weekday", "scholarship", "hypertension", "diabetes", "alcoholism", "handicap",
            "sms_received", "prior_appts", "prior_noshows", "missed"]
    view = df[cols].copy()
    view["scheduled_date"] = view["scheduled_date"].dt.date
    view["appointment_date"] = view["appointment_date"].dt.date
    st.dataframe(view.head(2000), width="stretch", hide_index=True)
    st.caption(f"Showing the first 2,000 of {len(view):,} rows.")
    st.download_button("⬇️ Download filtered data (CSV)", view.to_csv(index=False).encode(), "filtered_appointments.csv", "text/csv")

    with st.expander("Data dictionary"):
        st.markdown("""
| Column | Meaning |
|---|---|
| `missed` | **1 = patient did NOT show up**, 0 = attended (recoded from the confusing raw `No-show`) |
| `wait_days` | Days between booking and appointment |
| `age_group`, `wait_group` | Banded versions of age and waiting time |
| `scholarship` | Enrolled in the Bolsa Familia welfare programme |
| `hypertension`, `diabetes`, `alcoholism`, `handicap` | Patient flags (1 = yes) |
| `sms_received` | At least one reminder SMS was sent |
| `prior_appts`, `prior_noshows` | Patient's earlier appointments / missed ones, known at booking time (no leakage) |
""")