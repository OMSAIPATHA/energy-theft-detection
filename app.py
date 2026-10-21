"""
Smart Energy Theft Detection - web app.

Run with:   streamlit run app.py
Then open the link it prints (usually http://localhost:8501).
"""
import hashlib
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import theft_core as tc

st.set_page_config(page_title="Energy Theft Detection | Analytics", page_icon="⚡", layout="wide")

# Professional, restrained utility-dashboard styling.
st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1500px;}
  h1 {letter-spacing: -0.035em; font-weight: 750;}
  h2, h3 {letter-spacing: -0.02em; font-weight: 680;}
  [data-testid="stMetric"] {background: linear-gradient(135deg, #ffffff 0%, #f5fafb 100%); border: 1px solid #dce7eb; padding: 16px 18px; border-radius: 14px; box-shadow: 0 3px 12px rgba(15, 45, 60, 0.04);}
  [data-testid="stMetricLabel"] {color: #536773; font-weight: 600;}
  [data-testid="stMetricValue"] {color: #102a43; font-weight: 750;}
  [data-testid="stTabs"] button {font-weight: 650;}
  [data-testid="stSidebar"] {background: #f4f8fa;}
  div[data-testid="stDataFrame"] {border: 1px solid #dce7eb; border-radius: 10px; overflow: hidden;}
</style>
""", unsafe_allow_html=True)


def show(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def risk_band(score):
    """Display band for a 0-1 risk score; this is not a separate model prediction."""
    if pd.isna(score):
        return "Unknown"
    if score >= 0.65:
        return "High"
    if score >= 0.35:
        return "Medium"
    return "Low"


def fig_risk_distribution(report, threshold):
    """Plot consumer risk scores and the currently configured alert threshold."""
    fig, ax = plt.subplots(figsize=(8, 3.4), constrained_layout=True)
    scores = pd.to_numeric(report["risk_score"], errors="coerce").dropna()
    if scores.empty:
        ax.text(0.5, 0.5, "No risk scores available", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.hist(scores, bins=np.linspace(0, 1, 21), color="#168c8c", edgecolor="white", alpha=0.92)
        ax.axvline(threshold, color="#e76f51", linewidth=2.2, linestyle="--",
                   label=f"Alert threshold: {threshold:.2f}")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Risk score")
        ax.set_ylabel("Number of consumers")
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="upper right")
    return fig


def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


# ------------------------------------------------------------------ cached steps
@st.cache_data(show_spinner=False)
def get_simulated(n, days, rate, seed):
    return tc.simulate_meter_data(n_consumers=n, days=days, theft_rate=rate, seed=seed)


@st.cache_data(show_spinner=False)
def get_uploaded(meter_bytes, label_bytes):
    raw = pd.read_csv(io.BytesIO(meter_bytes))
    labels = pd.read_csv(io.BytesIO(label_bytes)) if label_bytes else None
    return tc.check_uploaded(raw, labels)


@st.cache_data(show_spinner=False)
def get_results(source_key, threshold, contamination, _raw, _labels):
    # _raw/_labels are not hashed (too big); source_key identifies the data instead
    return tc.run_pipeline(_raw, _labels, threshold, contamination)


@st.cache_data(show_spinner=False)
def sample_csv():
    raw, labels = tc.simulate_meter_data(n_consumers=40, days=60, seed=7)
    return to_csv_bytes(raw), to_csv_bytes(labels[["consumer_id", "is_theft", "theft_type"]])


# ------------------------------------------------------------------ sidebar
st.sidebar.title("⚡ Settings")
source = st.sidebar.radio("Data source", ["Simulated data (demo)", "Upload my own CSV"])

with st.sidebar.form("settings"):
    if source == "Simulated data (demo)":
        n_cons = st.slider("Number of consumers", 100, 1000, 500, step=50)
        n_days = st.slider("Days of hourly readings", 60, 180, 90, step=10)
        theft_pct = st.slider("Consumers stealing (%)", 5, 30, 15)
        seed = st.number_input("Random seed", 0, 9999, 42, help="Change it to get a different dataset")
        meter_file = label_file = None
    else:
        meter_file = st.file_uploader("Meter readings CSV", type="csv",
                                      help="Columns: consumer_id, timestamp, energy_kwh "
                                           "(+ optional voltage, current, power_factor)")
        label_file = st.file_uploader("Theft labels CSV (optional)", type="csv",
                                      help="Columns: consumer_id, is_theft (0/1), optional theft_type")
    st.markdown("**Detection settings**")
    threshold = st.slider("Alert threshold (risk score)", 0.1, 0.9, 0.5, 0.05,
                          help="Used when theft labels are available")
    contamination = st.slider("Expected theft rate (%) - no-label mode", 1, 30, 10,
                              help="Used by the Isolation Forest when there are no labels") / 100
    tariff = st.number_input("Tariff (₹ per kWh)", 1.0, 20.0, 7.0, 0.5,
                             help="Used only to estimate revenue loss")
    st.form_submit_button("▶ Run analysis", type="primary")

if source == "Upload my own CSV":
    meter_csv, label_csv = sample_csv()
    st.sidebar.markdown("**Need a template?**")
    st.sidebar.download_button("Sample meter CSV", meter_csv, "sample_meter_readings.csv", "text/csv")
    st.sidebar.download_button("Sample labels CSV", label_csv, "sample_labels.csv", "text/csv")

# ------------------------------------------------------------------ load data
st.title("⚡ Smart Energy Theft Detection")
st.caption("Detects abnormal electricity consumption that may indicate power theft or meter tampering, "
           "using Python and machine learning.")

if source == "Simulated data (demo)":
    with st.spinner("Generating simulated smart-meter data..."):
        raw, labels = get_simulated(n_cons, n_days, theft_pct / 100, int(seed))
    source_key = f"sim-{n_cons}-{n_days}-{theft_pct}-{seed}"
    load_warnings = []
else:
    if meter_file is None:
        st.info("⬅ Upload a meter readings CSV in the sidebar and press **Run analysis**.\n\n"
                "Required columns: `consumer_id`, `timestamp`, `energy_kwh`. "
                "Optional: `voltage`, `current`, `power_factor` (enables the physics check). "
                "Readings should be hourly and cover at least ~45 days. "
                "Download the sample files in the sidebar to see the format.")
        st.stop()
    try:
        meter_bytes = meter_file.getvalue()
        label_bytes = label_file.getvalue() if label_file else None
        raw, labels, load_warnings = get_uploaded(meter_bytes, label_bytes)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        st.stop()
    source_key = "up-" + hashlib.md5(meter_bytes + (label_bytes or b"")).hexdigest()

with st.spinner("Cleaning data, extracting features and running the models... (first run takes ~10-20 s)"):
    res = get_results(source_key, threshold, contamination, raw, labels)

for w in load_warnings + res["warnings"]:
    st.warning(w)

rep, summ, daily, df = res["report"], res["summary"], res["daily"], res["df"]
supervised = res["mode"] == "supervised"
has_labels = "is_theft" in rep.columns
alerts = rep[rep.status == "SUSPICIOUS"]
normal = rep[rep.status == "NORMAL"]

tabs = st.tabs(["📊 Overview", "🧹 Data & cleaning", "🔍 Patterns", "🤖 Model performance",
                "🚨 Alerts", "👤 Consumer inspector", "ℹ️ How it works"])

# ================================================================== OVERVIEW
with tabs[0]:
    lost_kwh = alerts.est_unbilled_kwh.sum()
    c = st.columns(5)
    c[0].metric("Consumers analysed", f"{len(rep):,}")
    c[1].metric("Suspicious", f"{len(alerts):,}")
    c[2].metric("Flagged", f"{len(alerts) / len(rep):.1%}")
    c[3].metric("Est. unbilled energy (30 d)", f"{lost_kwh:,.0f} kWh")
    c[4].metric("Est. revenue loss (30 d)", f"₹{lost_kwh * tariff:,.0f}")

    st.markdown(f"**Model used:** {res['model_name']}  ·  **Data:** {summ['rows']:,} readings, "
                f"{summ['start']:%d %b %Y} – {summ['end']:%d %b %Y}")
    if has_labels:
        caught = int(((rep.is_theft == 1) & (rep.status == "SUSPICIOUS")).sum())
        false_alarms = int(((rep.is_theft == 0) & (rep.status == "SUSPICIOUS")).sum())
        st.success(f"Against the known labels: caught **{caught} of {int(rep.is_theft.sum())}** thieves, "
                   f"with **{false_alarms}** false alarm(s).")
    else:
        st.info("No theft labels provided - consumers are ranked by how unusual they look "
                "(Isolation Forest). Inspection results can be uploaded later as labels to train "
                "the more accurate supervised model.")

    left, right = st.columns([1.05, 1.25], gap="large")
    with left:
        st.subheader("Why consumers were flagged")
        if len(alerts):
            show(tc.fig_categories(alerts))
        else:
            st.info("No suspicious consumers at this threshold.")
        st.subheader("Consumer risk distribution")
        show(fig_risk_distribution(rep, threshold))
        st.caption("The dashed line is the configured alert threshold. Risk bands are a display aid, not a separate model output.")
    with right:
        st.subheader("Highest-risk consumers")
        st.caption("Prioritise these records for investigation; an alert is not proof of theft.")
        top_risk = alerts.sort_values("risk_score", ascending=False).head(10).copy()
        if not top_risk.empty:
            top_risk.insert(2, "risk_level", top_risk["risk_score"].map(risk_band))
            st.dataframe(top_risk[["consumer_id", "risk_score", "risk_level", "reasons"]],
                         hide_index=True, use_container_width=True,
                         column_config={"risk_score": st.column_config.ProgressColumn(
                             "Risk score", min_value=0.0, max_value=1.0, format="%.2f"),
                             "risk_level": st.column_config.TextColumn("Risk level")})
        else:
            st.info("No consumers are currently flagged.")
    st.caption("Unbilled energy is a rough estimate: for each suspicious consumer, the larger of "
               "(measured V×I×pf − billed) and (expected from own history − billed) over the last 30 days.")

# ================================================================== DATA
with tabs[1]:
    st.subheader("Dataset")
    c = st.columns(4)
    c[0].metric("Readings", f"{summ['rows']:,}")
    c[1].metric("Consumers", f"{summ['consumers']:,}")
    c[2].metric("Days", f"{daily.shape[1]}")
    c[3].metric("Voltage/current data", "Yes" if summ["has_electrical"] else "No")
    st.dataframe(raw.head(200), hide_index=True)
    st.caption("First 200 raw readings.")

    st.subheader("Cleaning steps")
    miss = summ["missing_before"]
    steps = pd.DataFrame({
        "Step": ["Missing energy readings", "Missing voltage readings",
                 "Negative (impossible) energy readings removed",
                 "Voltage outside 180-270 V removed", "Missing values after interpolation"],
        "Count": [int(miss.get("energy_kwh", 0)), int(miss.get("voltage", 0)),
                  summ["negative_removed"], summ["bad_voltage_removed"], summ["missing_after"]]})
    st.dataframe(steps, hide_index=True)
    st.caption("Missing or impossible values are filled by linear interpolation within each consumer's "
               "own time series. Genuine zero readings are kept - they can be evidence of theft.")
    st.subheader("Cleaned data summary")
    st.dataframe(df.drop(columns=["consumer_id"]).describe().round(2))

# ================================================================== PATTERNS
with tabs[2]:
    st.subheader("Daily consumption examples")
    if has_labels and set(tc.THEFT_TYPES) & set(rep.theft_type):
        ex = {"Normal": rep.query("is_theft == 0").consumer_id.iloc[-1]}
        for tt in tc.THEFT_TYPES:
            if (rep.theft_type == tt).any():
                ex[tt] = rep.query("theft_type == @tt").consumer_id.iloc[0]
        starts = (labels.set_index("consumer_id").theft_start_day.to_dict()
                  if labels is not None and "theft_start_day" in labels else {})
        st.caption("One normal consumer and one example of each theft type. Dashed line = theft starts.")
    else:
        ex = {"Normal (lowest risk)": rep.consumer_id.iloc[-1]}
        for i, cid in enumerate(alerts.consumer_id.head(4)):
            ex[f"Suspicious #{i + 1}"] = cid
        starts = {}
        st.caption("The lowest-risk consumer compared with the four highest-risk consumers.")
    show(tc.fig_daily_examples(daily, ex, starts))

    st.subheader("Physics check: billed vs measured energy")
    if summ["has_electrical"]:
        by = "theft_type" if has_labels and set(tc.THEFT_TYPES) & set(rep.theft_type) else "status"
        show(tc.fig_physics(rep, by))
        st.caption("Honest meters bill ≈ 100% of V×I×pf. Meter slowdown and frozen registers fall far "
                   "below; bypass and night cut-off hide load from the sensor too, so only the behaviour "
                   "features reveal them.")
    else:
        st.info("Not available - the data has no voltage/current columns.")

    st.subheader("Average feature values")
    grp = "is_theft" if has_labels else "status"
    tbl = rep.groupby(grp)[tc.FEATURES].mean().T
    tbl.columns = [{0: "Normal", 1: "Theft"}.get(cn, str(cn).title()) for cn in tbl.columns]
    tbl.insert(0, "Meaning", [tc.FEATURE_INFO[f] for f in tbl.index])
    st.dataframe(tbl.round(3))

# ================================================================== MODELS
with tabs[3]:
    if supervised:
        y_te, models = res["test"]["y"], res["test"]["models"]
        st.markdown(f"Consumers were split **70% training / 30% testing** "
                    f"({len(y_te)} test consumers, {int(y_te.sum())} thieves). All numbers below are on "
                    f"the test set - consumers the models never saw.")
        st.subheader("Model comparison")
        st.dataframe(res["metrics"], column_config={
            m: st.column_config.NumberColumn(m, format="%.3f") for m in res["metrics"].columns})
        st.caption("Precision = of flagged consumers, how many were really stealing. "
                   "Recall = of all thieves, how many were caught. ROC-AUC = how well the score ranks "
                   "thieves above honest consumers (1 = perfect, 0.5 = guessing).")

        st.subheader("Confusion matrices")
        show(tc.fig_confusion(models, y_te))
        a, b = st.columns(2)
        with a:
            st.subheader("ROC curves")
            show(tc.fig_roc(models, y_te))
        with b:
            st.subheader("Feature importance")
            show(tc.fig_importance(res["importance"]))
        st.subheader("Risk score distribution (Random Forest, test set)")
        show(tc.fig_risk_hist(models["Random Forest"][0], y_te == 1, threshold))

        if rep.theft_type.nunique() > 2:
            st.subheader("Detection rate by theft type (all consumers, cross-validated)")
            by_type = (rep.query("is_theft == 1").assign(caught=lambda d: d.status == "SUSPICIOUS")
                       .groupby("theft_type").caught.agg(caught="sum", total="count"))
            by_type["detection rate"] = (by_type.caught / by_type.total).map("{:.0%}".format)
            st.dataframe(by_type)
    else:
        st.info("Model accuracy can only be measured when **theft labels** are available "
                "(upload a labels CSV, or use the simulated data). Without labels the app uses the "
                "unsupervised Isolation Forest, which ranks consumers by how unusual they are.")
        st.subheader("Anomaly score distribution")
        show(tc.fig_risk_hist(rep.risk_score.values, (rep.status == "SUSPICIOUS").values,
                              bad_label="Flagged suspicious", good_label="Normal"))

# ================================================================== ALERTS
with tabs[4]:
    st.subheader(f"🚨 {len(alerts)} suspicious consumers")
    if len(alerts):
        f1, f2, f3 = st.columns([2, 1, 1])
        cats = f1.multiselect("Reason", sorted(alerts.category.unique()),
                              default=sorted(alerts.category.unique()))
        min_risk = f2.slider("Minimum risk", 0.0, 1.0, 0.0, 0.05)
        search = f3.text_input("Search consumer ID")
        view = alerts[alerts.category.isin(cats) & (alerts.risk_score >= min_risk)].copy()
        if search:
            view = view[view.consumer_id.astype(str).str.contains(search.strip(), case=False, na=False)]
        view = view.sort_values("risk_score", ascending=False)
        view.insert(2, "risk_level", view["risk_score"].map(risk_band))
        cols = ["consumer_id", "risk_score", "risk_level", "category", "reasons", "est_unbilled_kwh"]
        if has_labels:
            cols.append("theft_type")
        st.dataframe(view[cols], hide_index=True, height=470, use_container_width=True, column_config={
            "risk_score": st.column_config.ProgressColumn("Risk score", min_value=0.0, max_value=1.0, format="%.2f"),
            "risk_level": st.column_config.TextColumn("Risk level"),
            "est_unbilled_kwh": st.column_config.NumberColumn("Est. unbilled kWh (30 d)", format="%.0f"),
            "theft_type": st.column_config.TextColumn("Known label")})
        st.caption(f"Showing {len(view)} of {len(alerts)}.")
    d1, d2 = st.columns(2)
    d1.download_button("⬇ Download alerts (CSV)",
                       to_csv_bytes(alerts[["consumer_id", "risk_score", "status", "category", "reasons",
                                            "est_unbilled_kwh"]]), "alerts.csv", "text/csv")
    d2.download_button("⬇ Download all consumers with scores (CSV)",
                       to_csv_bytes(rep[["consumer_id", "risk_score", "status", "category", "reasons"]
                                        + tc.FEATURES]), "all_consumers.csv", "text/csv")

# ================================================================== INSPECTOR
with tabs[5]:
    names = {row.consumer_id: f"{row.consumer_id} - {row.status} (risk {row.risk_score:.2f})"
             for row in rep.itertuples()}
    pick = st.selectbox("Choose a consumer (sorted by risk, highest first)", list(names),
                        format_func=names.get)
    r = rep.set_index("consumer_id").loc[pick]
    c = st.columns(4)
    c[0].metric("Status", r.status)
    c[1].metric("Risk score", f"{r.risk_score:.2f}")
    c[2].metric("Est. unbilled (30 d)", f"{r.est_unbilled_kwh:,.0f} kWh")
    c[3].metric("True label", r.theft_type if has_labels else "unknown")
    if r.status == "SUSPICIOUS":
        st.error(f"**Why flagged:** {r.reasons}")
    else:
        st.success("No alert - consumption looks consistent with this consumer's own history.")

    start = None
    if labels is not None and "theft_start_day" in labels:
        start = labels.set_index("consumer_id").theft_start_day.get(pick)
    st.subheader("Daily consumption")
    show(tc.fig_consumer_daily(daily.loc[pick], start))
    st.subheader("Last 7 days, hourly")
    show(tc.fig_consumer_hourly(df[df.consumer_id == pick], summ["has_electrical"]))
    if summ["has_electrical"]:
        st.caption("If the orange (billed) line sits below the grey (measured) line, the meter is "
                   "billing less than the power actually drawn.")

    st.subheader("Feature values vs typical normal consumer")
    comp = pd.DataFrame({"This consumer": r[tc.FEATURES].astype(float),
                         "Typical normal (median)": normal[tc.FEATURES].median(),
                         "Meaning": [tc.FEATURE_INFO[f] for f in tc.FEATURES]})
    st.dataframe(comp.round(3))

# ================================================================== ABOUT
with tabs[6]:
    st.markdown("""
### How it works

**1. Data** - hourly smart-meter readings: voltage (V), current (I), power factor (pf) and billed energy (kWh).

**2. Cleaning** - impossible values removed, missing readings filled by interpolation.

**3. Features** - each consumer's last 30 days are compared with their own history, giving 11 numbers:

* **Physics check:** on an honest meter, billed kWh ≈ V × I × pf × time. A big gap means tampering.
* **Behaviour check:** lasting drop in usage, zero readings, unnaturally flat readings, change in night usage,
  whether low usage recovers (vacations do, theft doesn't).

**4. Models**

| Model | Needs labels? | Role |
|---|---|---|
| Isolation Forest | No | Flags unusual consumers - usable from day one |
| Logistic Regression | Yes | Simple baseline |
| Random Forest | Yes | Final model - learns from confirmed theft cases |

**5. Output** - every consumer gets a risk score (0-1), a status (NORMAL / SUSPICIOUS) and a reason an
inspector can check on site.

### Theft types in the simulated data

| Type | What happens | What gives it away |
|---|---|---|
| Meter slowdown | Meter register tampered | Billed ≪ V×I×pf |
| Line bypass | Wire hooked around the meter | Lasting drop vs own history |
| Night cut-off | Meter switched off at night | Zero readings at night |
| Flat reading | Register frozen / replaced | Constant readings |

### Limitations
* Simulated data gives optimistic results; real data is noisier.
* The physics check assumes V and I come from a sensor the consumer cannot tamper with.
* Alerts are leads for inspection, not proof of theft.
""")
