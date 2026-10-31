"""Single-file Streamlit dashboard for the Kaggle appointment no-show dataset."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Hospital Appointment Insights",
    page_icon="🏥",
    layout="wide",
)

DATA_FILE = Path(__file__).with_name("appointments_public.csv")
COLORS = {"Attended": "#168C82", "Missed": "#E76F51", "blue": "#426A8C", "gold": "#E9B949"}

# Baseline results generated in the accompanying learning project using a stratified
# 80/20 split (random_state=42). They are shown for education, not decision-making.
MODEL_METRICS = {
    "roc_auc": 0.670,
    "recall_missed": 0.584,
    "precision_missed": 0.311,
    "f1_missed": 0.406,
    "test_records": 22106,
    "confusion_matrix": [[11879, 5763], [1858, 2606]],
}


@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Place appointments_public.csv beside app.py. Expected: {DATA_FILE}")
    frame = pd.read_csv(DATA_FILE)

    frame = frame.rename(
        columns={
            "Neighbourhood": "Neighborhood",
            "Hipertension": "Hypertension",
            "Handcap": "Handicap",
            "SMS_received": "SMS_Received",
            "No-show": "No_show",
        }
    ).copy()

    if "ScheduledDay" in frame.columns:
        frame["ScheduledDay"] = pd.to_datetime(frame["ScheduledDay"], utc=True).dt.tz_convert(None)
    if "AppointmentDay" not in frame.columns:
        raise ValueError("The CSV must contain an AppointmentDay column.")
    frame["AppointmentDay"] = pd.to_datetime(frame["AppointmentDay"], utc=True).dt.tz_convert(None)

    if "Wait_Days" not in frame.columns:
        if "ScheduledDay" not in frame.columns:
            raise ValueError("The CSV must contain ScheduledDay or Wait_Days.")
        frame["Wait_Days"] = (
            frame["AppointmentDay"].dt.normalize() - frame["ScheduledDay"].dt.normalize()
        ).dt.days

    if "No_show" not in frame.columns:
        raise ValueError("The CSV must contain the Kaggle No-show column.")
    if not pd.api.types.is_numeric_dtype(frame["No_show"]):
        frame["No_show"] = frame["No_show"].map({"No": 0, "Yes": 1})
    frame["No_show"] = pd.to_numeric(frame["No_show"], errors="coerce")

    required = {
        "Age", "Gender", "Neighborhood", "Scholarship", "Hypertension", "Diabetes",
        "Alcoholism", "Handicap", "SMS_Received", "No_show", "Wait_Days",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")

    frame = frame.drop(columns=["PatientId", "AppointmentID", "ScheduledDay"], errors="ignore")
    frame = frame.loc[frame["Age"] >= 0].drop_duplicates().copy()
    frame["Appointment_Weekday"] = frame["AppointmentDay"].dt.day_name()
    frame["Age_Group"] = pd.cut(
        frame["Age"],
        bins=[0, 18, 30, 45, 60, 121],
        right=False,
        labels=["0-17", "18-29", "30-44", "45-59", "60+"],
    ).astype("string")
    frame["Outcome"] = frame["No_show"].map({0: "Attended", 1: "Missed"})
    return frame


def no_show_chart(frame, group_column, label, title, category_order=None):
    summary = (
        frame.groupby(group_column, observed=True)["No_show"]
        .agg(Appointments="size", Rate="mean")
        .reset_index()
    )
    summary["No-show rate (%)"] = summary["Rate"] * 100
    chart = px.bar(
        summary,
        x=group_column,
        y="No-show rate (%)",
        color_discrete_sequence=[COLORS["blue"]],
        category_orders={group_column: category_order} if category_order else None,
        hover_data={"Appointments": True, "Rate": False},
        labels={group_column: label},
        title=title,
    )
    chart.update_layout(xaxis_title=label, yaxis_title="Missed appointments (%)")
    return chart


def binary_filter(frame, column, label):
    choice = st.sidebar.selectbox(label, ["All", "No", "Yes"], key=f"filter_{column}")
    if choice == "All":
        return frame
    return frame.loc[frame[column] == (1 if choice == "Yes" else 0)]


try:
    data = load_data()
except FileNotFoundError:
    st.title("Hospital Appointment Insights")
    st.warning(
        "The appointment data file is not included in this deployment. "
        "The dataset contains health attributes and dates, so it needs approval "
        "before it can be published with a public app."
    )
    st.stop()
except Exception as error:
    st.error(f"Could not load the appointment data: {error}")
    st.stop()

st.title("🏥 Hospital Appointment Insights")
st.caption(
    "Explore attendance patterns in the 2016 Kaggle dataset. "
    "Missed appointment means No-show = Yes."
)

# Sidebar filters
st.sidebar.title("Explore the data")
filtered = data.copy()

age_min, age_max = int(data["Age"].min()), int(data["Age"].max())
age_range = st.sidebar.slider("Age", age_min, age_max, (age_min, age_max))
filtered = filtered.loc[filtered["Age"].between(*age_range)]

genders = sorted(data["Gender"].dropna().unique().tolist())
selected_genders = st.sidebar.multiselect("Gender", genders, default=genders)
filtered = filtered.loc[filtered["Gender"].isin(selected_genders)]

neighborhoods = sorted(data["Neighborhood"].dropna().unique().tolist())
selected_neighborhoods = st.sidebar.multiselect(
    "Neighborhood", neighborhoods, default=neighborhoods
)
filtered = filtered.loc[filtered["Neighborhood"].isin(selected_neighborhoods)]

first_day, last_day = data["AppointmentDay"].min().date(), data["AppointmentDay"].max().date()
date_range = st.sidebar.date_input(
    "Appointment date", (first_day, last_day), min_value=first_day, max_value=last_day
)
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    filtered = filtered.loc[
        filtered["AppointmentDay"].dt.date.between(date_range[0], date_range[1])
    ]

wait_min, wait_max = int(data["Wait_Days"].min()), int(data["Wait_Days"].max())
wait_range = st.sidebar.slider("Days between scheduling and appointment", wait_min, wait_max, (wait_min, wait_max))
filtered = filtered.loc[filtered["Wait_Days"].between(*wait_range)]

for field, label in [
    ("Scholarship", "Bolsa Família scholarship"),
    ("Hypertension", "Hypertension"),
    ("Diabetes", "Diabetes"),
    ("Alcoholism", "Alcoholism"),
    ("SMS_Received", "SMS recorded"),
]:
    filtered = binary_filter(filtered, field, label)

handicap_values = sorted(data["Handicap"].dropna().astype(int).unique().tolist())
selected_handicap = st.sidebar.multiselect(
    "Recorded handicap level", handicap_values, default=handicap_values
)
filtered = filtered.loc[filtered["Handicap"].isin(selected_handicap)]
st.sidebar.caption("Patient and appointment ID fields are excluded from this dashboard and downloads.")

if filtered.empty:
    st.warning("No appointments match these filters. Adjust the selections in the sidebar.")
    st.stop()

# Summary cards
miss_rate = float(filtered["No_show"].mean())
valid_wait = filtered.loc[filtered["Wait_Days"] >= 0, "Wait_Days"]
avg_wait = float(valid_wait.mean()) if len(valid_wait) else float("nan")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Appointments", f"{len(filtered):,}")
k2.metric("No-show rate", f"{miss_rate:.1%}")
k3.metric("Attendance rate", f"{1 - miss_rate:.1%}")
k4.metric("Average valid wait", f"{avg_wait:.1f} days" if pd.notna(avg_wait) else "N/A")

overview, people, scheduling, indicators, locations, model = st.tabs(
    ["Overview", "Demographics", "Scheduling", "Health & social", "Neighborhoods", "Model"]
)

with overview:
    left, right = st.columns(2)
    with left:
        outcome_counts = filtered["Outcome"].value_counts().rename_axis("Outcome").reset_index(name="Appointments")
        pie = px.pie(
            outcome_counts,
            names="Outcome",
            values="Appointments",
            hole=0.58,
            color="Outcome",
            color_discrete_map=COLORS,
            title="Appointment outcomes",
        )
        st.plotly_chart(pie, use_container_width=True)
    with right:
        age_order = ["0-17", "18-29", "30-44", "45-59", "60+"]
        st.plotly_chart(no_show_chart(filtered, "Age_Group", "Age group", "No-show rate by age group", age_order), use_container_width=True)

    st.subheader("Waiting time")
    wait_data = filtered.loc[filtered["Wait_Days"] >= 0].copy()
    wait_data["Wait band"] = pd.cut(
        wait_data["Wait_Days"],
        bins=[-1, 0, 7, 14, 30, 60, float("inf")],
        labels=["Same day", "1–7 days", "8–14 days", "15–30 days", "31–60 days", "61+ days"],
    )
    wait_summary = wait_data.groupby("Wait band", observed=True)["No_show"].agg(
        Appointments="size", Rate="mean"
    ).reset_index()
    wait_summary["No-show rate (%)"] = wait_summary["Rate"] * 100
    wait_plot = px.bar(
        wait_summary,
        x="Wait band",
        y="No-show rate (%)",
        color_discrete_sequence=[COLORS["blue"]],
        hover_data={"Appointments": True, "Rate": False},
        title="No-show rate by waiting time",
    )
    st.plotly_chart(wait_plot, use_container_width=True)
    if (filtered["Wait_Days"] < 0).any():
        st.caption("Some source records have negative wait days; they are excluded from this chart.")

    safe_columns = [
        "Outcome", "Age", "Age_Group", "Gender", "Neighborhood", "AppointmentDay",
        "Appointment_Weekday", "Wait_Days", "Scholarship", "Hypertension", "Diabetes",
        "Alcoholism", "Handicap", "SMS_Received",
    ]
    with st.expander("View or download filtered appointment records"):
        st.dataframe(filtered[safe_columns].head(5000), use_container_width=True, height=360)
        st.download_button(
            "Download filtered records (IDs excluded)",
            filtered[safe_columns].to_csv(index=False).encode("utf-8"),
            file_name="filtered_appointment_records.csv",
            mime="text/csv",
        )

with people:
    left, right = st.columns(2)
    with left:
        st.plotly_chart(no_show_chart(filtered, "Gender", "Gender", "No-show rate by gender"), use_container_width=True)
    with right:
        age_counts = filtered.groupby("Age_Group", observed=True).size().rename("Appointments").reset_index()
        age_plot = px.bar(
            age_counts, x="Age_Group", y="Appointments",
            category_orders={"Age_Group": ["0-17", "18-29", "30-44", "45-59", "60+"]},
            color_discrete_sequence=[COLORS["gold"]], title="Appointment volume by age group",
        )
        st.plotly_chart(age_plot, use_container_width=True)

with scheduling:
    left, right = st.columns(2)
    with left:
        week_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        st.plotly_chart(no_show_chart(filtered, "Appointment_Weekday", "Weekday", "No-show rate by weekday", week_order), use_container_width=True)
    with right:
        st.plotly_chart(no_show_chart(filtered, "SMS_Received", "SMS recorded (0 = no, 1 = yes)", "No-show rate by SMS status", [0, 1]), use_container_width=True)
    st.info("SMS comparisons are descriptive. The dataset does not show whether a reminder was read or why it was sent.")

with indicators:
    rows = []
    for field in ["Scholarship", "Hypertension", "Diabetes", "Alcoholism"]:
        summary = filtered.groupby(field)["No_show"].agg(Appointments="size", Rate="mean").reset_index()
        for _, row in summary.iterrows():
            rows.append({
                "Indicator": field,
                "Status": "Yes" if int(row[field]) == 1 else "No",
                "Appointments": int(row["Appointments"]),
                "No-show rate (%)": float(row["Rate"] * 100),
            })
    indicator_plot = px.bar(
        pd.DataFrame(rows), x="Indicator", y="No-show rate (%)", color="Status",
        barmode="group", hover_data={"Appointments": True},
        color_discrete_map={"No": COLORS["blue"], "Yes": COLORS["Missed"]},
        title="No-show rates by recorded indicator",
    )
    st.plotly_chart(indicator_plot, use_container_width=True)
    handicap_summary = filtered.groupby("Handicap")["No_show"].agg(Appointments="size", Rate="mean").reset_index()
    handicap_summary["No-show rate (%)"] = handicap_summary["Rate"] * 100
    handicap_plot = px.bar(
        handicap_summary, x="Handicap", y="No-show rate (%)", hover_data={"Appointments": True},
        color_discrete_sequence=[COLORS["blue"]], title="No-show rate by recorded handicap level",
    )
    st.plotly_chart(handicap_plot, use_container_width=True)

with locations:
    minimum_size = st.slider("Minimum appointments per neighborhood", 1, 500, 100, 10)
    neighborhood_summary = filtered.groupby("Neighborhood")["No_show"].agg(Appointments="size", Rate="mean").reset_index()
    neighborhood_summary = neighborhood_summary.loc[neighborhood_summary["Appointments"] >= minimum_size].copy()
    neighborhood_summary["No-show rate (%)"] = neighborhood_summary["Rate"] * 100
    neighborhood_summary = neighborhood_summary.nlargest(20, "No-show rate (%)")
    if neighborhood_summary.empty:
        st.info("No neighborhood meets the selected minimum appointment count.")
    else:
        location_plot = px.bar(
            neighborhood_summary.sort_values("No-show rate (%)"),
            x="No-show rate (%)", y="Neighborhood", orientation="h",
            hover_data={"Appointments": True}, color_discrete_sequence=[COLORS["Missed"]],
            title="Neighborhoods with the highest observed no-show rates",
        )
        st.plotly_chart(location_plot, use_container_width=True)
        st.caption("Small groups can have unstable rates. This chart is descriptive, not a neighborhood rating.")

with model:
    st.write("A logistic-regression baseline was evaluated on a stratified 20% holdout set.")
    st.warning("Learning example only. This model is not intended for clinical or operational decisions.")
    a, b, c, d = st.columns(4)
    a.metric("ROC-AUC", f"{MODEL_METRICS['roc_auc']:.3f}")
    b.metric("Recall: missed", f"{MODEL_METRICS['recall_missed']:.3f}")
    c.metric("Precision: missed", f"{MODEL_METRICS['precision_missed']:.3f}")
    d.metric("F1: missed", f"{MODEL_METRICS['f1_missed']:.3f}")
    cm = MODEL_METRICS["confusion_matrix"]
    matrix = go.Figure(go.Heatmap(
        z=cm,
        x=["Predicted attended", "Predicted missed"],
        y=["Actual attended", "Actual missed"],
        text=[[str(value) for value in row] for row in cm],
        texttemplate="%{text}", colorscale="Blues", showscale=False,
    ))
    matrix.update_layout(title="Holdout confusion matrix", xaxis_title="Prediction", yaxis_title="Actual outcome")
    st.plotly_chart(matrix, use_container_width=True)
    st.caption(f"Evaluation records: {MODEL_METRICS['test_records']:,}. Missed appointments are the positive class.")

st.caption("Source: Kaggle Medical Appointment No Shows. Observed associations do not establish cause and effect.")
