# Smart Energy Theft Detection - Web App

A browser-based dashboard for the Smart Energy Theft Detection project. It runs
the same pipeline as the notebook (cleaning → features → ML models → alerts) and
lets you explore the results interactively.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Your browser opens at **http://localhost:8501**. If it doesn't, open that link yourself.
The first analysis takes about 10–20 seconds; after that results are cached and
switching tabs or consumers is instant.

## Files

| File | What it does |
|---|---|
| `app.py` | The web interface: sidebar settings and the 7 tabs |
| `theft_core.py` | All analysis logic: data, cleaning, features, models, alerts, charts (no web code) |
| `requirements.txt` | Libraries needed |
| `.streamlit/config.toml` | Light theme and upload size limit (hidden folder; keep it) |

## What the app shows

| Tab | Contents |
|---|---|
| 📊 Overview | Consumers analysed, number suspicious, estimated unbilled energy and revenue loss, reasons chart, top-10 risk list |
| 🧹 Data & cleaning | Raw data preview, missing / impossible values found and fixed |
| 🔍 Patterns | Daily consumption of normal vs theft consumers, physics check, average feature values |
| 🤖 Model performance | Isolation Forest vs Logistic Regression vs Random Forest: metrics, confusion matrices, ROC curves, feature importance |
| 🚨 Alerts | Filterable list of suspicious consumers with reasons; CSV download |
| 👤 Consumer inspector | Pick any consumer: risk score, reason, daily and hourly charts, features vs a typical consumer |
| ℹ️ How it works | Short explanation of the method |

## Data sources (sidebar)

**Simulated data (demo):** choose the number of consumers, days and theft percentage, then press *Run analysis*.

**Upload my own CSV:**
- **Meter readings** (required): `consumer_id, timestamp, energy_kwh`. Optionally add
  `voltage, current, power_factor`; without them the physics check is switched off.
  Readings should be hourly and cover at least about 45 days.
- **Labels** (optional): `consumer_id, is_theft` (0/1), optionally `theft_type`.
  - With at least 10 theft and 10 normal labels, the supervised Random Forest is trained and accuracy is shown.
  - Otherwise the unsupervised Isolation Forest ranks consumers by how unusual they are.

Sample files in the right format can be downloaded from the sidebar.

## Settings

- **Alert threshold:** a consumer is flagged if its risk score is at or above this value (supervised mode).
- **Expected theft rate:** the share of consumers the Isolation Forest flags (no-label mode).
- **Tariff:** used only to convert unbilled kWh into an estimated ₹ loss.
