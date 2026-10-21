"""
theft_core.py - all the analysis logic of the Smart Energy Theft Detection project.

This file has NO web code. It is the same pipeline as the notebook, organised as
functions so the web app (app.py) can call them:

    load data  ->  preprocess  ->  build features  ->  run models  ->  alerts
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)

SEED = 42
EPS = 1e-6
RECENT_DAYS = 30
THEFT_TYPES = ["meter_slowdown", "line_bypass", "night_cutoff", "flat_reading"]
REQUIRED_COLS = ["consumer_id", "timestamp", "energy_kwh"]
ELECTRICAL_COLS = ["voltage", "current", "power_factor"]

FEATURES = ["drop_ratio", "mismatch_ratio", "mismatch_change", "zero_frac_recent",
            "zero_frac_change", "flatness", "night_share_change", "recovery_ratio",
            "daily_cv_recent", "trend_slope", "voltage_std"]

FEATURE_INFO = {
    "drop_ratio": "Recent avg daily kWh ÷ baseline (≪ 1 is suspicious)",
    "mismatch_ratio": "Billed kWh ÷ V×I×pf, recent (≪ 1 means tampering)",
    "mismatch_change": "Change in mismatch vs baseline",
    "zero_frac_recent": "Share of recent hours reading ~0",
    "zero_frac_change": "Change in zero-reading share vs baseline",
    "flatness": "Recent std ÷ baseline std (≪ 1 means frozen register)",
    "night_share_change": "Change in share of energy used 10 pm-6 am",
    "recovery_ratio": "Last week ÷ lowest recent week (vacations recover, theft doesn't)",
    "daily_cv_recent": "Day-to-day variation, recent",
    "trend_slope": "Normalised slope of daily kWh",
    "voltage_std": "Voltage variability",
}

# ---------------------------------------------------------------- chart style
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, INK, MUTED = "#a3a29c", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.dpi": 110, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GREY, "axes.labelcolor": MUTED, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
    "lines.linewidth": 2, "legend.frameon": False,
})


# ================================================================ 1. DATA
def simulate_meter_data(n_consumers=500, days=90, theft_rate=0.15,
                        vacation_rate=0.08, seed=SEED):
    """Simulated smart-meter data (same model as the notebook)."""
    rng = np.random.default_rng(seed)
    hours = days * 24
    ts = pd.date_range("2026-01-01", periods=hours, freq="h")
    hod, dow, t = ts.hour.values, ts.dayofweek.values, np.arange(hours)
    theft_start_day = days - RECENT_DAYS          # theft begins in the last 30 days

    profile = 0.4 + 0.5 * np.exp(-((hod - 8) / 2.0) ** 2) + 0.9 * np.exp(-((hod - 20) / 2.5) ** 2)
    weekend = np.where(dow >= 5, 1.1, 1.0)

    thieves = set(rng.choice(n_consumers, int(round(n_consumers * theft_rate)), replace=False))
    readings, labels = [], []

    for c in range(n_consumers):
        scale = rng.lognormal(0, 0.4)
        season = 1 + 0.15 * np.sin(2 * np.pi * t / hours + rng.uniform(0, 2 * np.pi))
        true_kw = scale * profile * weekend * season * rng.lognormal(0, 0.25, hours)

        if c not in thieves and rng.random() < vacation_rate:
            v0 = rng.integers(10, days - 12) * 24
            true_kw[v0:v0 + rng.integers(5, 13) * 24] *= 0.1

        voltage = rng.normal(230, 4, hours)
        pf = np.clip(rng.normal(0.92, 0.03, hours), 0.75, 1.0)
        line_kw, billed = true_kw.copy(), true_kw.copy()

        theft_type, start_day = "none", None
        if c in thieves:
            theft_type = rng.choice(THEFT_TYPES)
            start_day = int(theft_start_day + rng.integers(0, 10))
            m = t >= start_day * 24
            if theft_type == "meter_slowdown":
                billed[m] *= rng.uniform(0.3, 0.6)
            elif theft_type == "line_bypass":
                frac = rng.uniform(0.4, 0.7)
                line_kw[m] *= frac
                billed[m] *= frac
            elif theft_type == "night_cutoff":
                night = m & ((hod >= 22) | (hod < 6))
                line_kw[night] = 0
                billed[night] = 0
            elif theft_type == "flat_reading":
                billed[m] = np.median(true_kw) * rng.uniform(0.3, 0.5)

        current = line_kw * 1000 / (voltage * pf)
        billed[rng.random(hours) < 0.01] = np.nan
        voltage[rng.random(hours) < 0.005] = np.nan
        bad = rng.random(hours) < 0.001
        billed[bad] = -billed[bad]

        readings.append(pd.DataFrame({
            "consumer_id": f"C{c:04d}", "timestamp": ts,
            "voltage": voltage.round(1), "current": current.round(3),
            "power_factor": pf.round(3), "energy_kwh": billed.round(4)}))
        labels.append({"consumer_id": f"C{c:04d}", "is_theft": int(c in thieves),
                       "theft_type": theft_type, "theft_start_day": start_day})

    return pd.concat(readings, ignore_index=True), pd.DataFrame(labels)


def check_uploaded(raw, labels=None):
    """Validate user CSVs. Returns (raw, labels, list_of_warnings) or raises ValueError."""
    raw = raw.rename(columns=lambda c: c.strip().lower())
    missing = [c for c in REQUIRED_COLS if c not in raw.columns]
    if missing:
        raise ValueError(f"Meter file is missing column(s): {', '.join(missing)}. "
                         f"Required: {', '.join(REQUIRED_COLS)}")
    warnings_ = []
    raw["consumer_id"] = raw["consumer_id"].astype(str)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    if raw["timestamp"].isna().all():
        raise ValueError("Could not read any dates in the 'timestamp' column.")
    raw = raw.dropna(subset=["timestamp"])
    if not all(c in raw.columns for c in ELECTRICAL_COLS):
        warnings_.append("No voltage / current / power_factor columns - the physics check "
                         "(billed vs V×I×pf) is switched off; only behaviour features are used.")
    span = (raw.groupby("consumer_id").timestamp.max() - raw.groupby("consumer_id").timestamp.min()).dt.days
    if span.median() < 45:
        raise ValueError(f"Need at least ~45 days of readings per consumer (median here: {span.median():.0f}). "
                         "The last 30 days are compared with the history before them.")
    if labels is not None:
        labels = labels.rename(columns=lambda c: c.strip().lower())
        if not {"consumer_id", "is_theft"} <= set(labels.columns):
            warnings_.append("Labels file needs 'consumer_id' and 'is_theft' columns - ignored.")
            labels = None
        else:
            labels["consumer_id"] = labels["consumer_id"].astype(str)
            labels["is_theft"] = labels["is_theft"].astype(int)
            if "theft_type" not in labels:
                labels["theft_type"] = np.where(labels.is_theft == 1, "theft", "none")
    return raw, labels, warnings_


# ================================================================ 2. PREPROCESS
def preprocess(raw):
    """Clean the readings. Returns (clean_df, summary_dict)."""
    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["consumer_id", "timestamp"]).reset_index(drop=True)
    has_elec = all(c in df.columns for c in ELECTRICAL_COLS)
    cols = ["energy_kwh"] + (ELECTRICAL_COLS if has_elec else [])

    missing_before = df[cols].isna().sum()
    n_neg = int((df.energy_kwh < 0).sum())
    df.loc[df.energy_kwh < 0, "energy_kwh"] = np.nan
    n_bad_v = 0
    if has_elec:
        bad_v = (df.voltage < 180) | (df.voltage > 270)
        n_bad_v = int(bad_v.sum())
        df.loc[bad_v, "voltage"] = np.nan
    df[cols] = df.groupby("consumer_id")[cols].transform(lambda s: s.interpolate(limit_direction="both"))
    df[cols] = df[cols].fillna(0)

    if has_elec:
        df["computed_kwh"] = df.voltage * df.current * df.power_factor / 1000
    else:                                   # no physics check possible
        df["computed_kwh"] = df.energy_kwh
        df["voltage"] = 0.0

    summary = {"rows": len(df), "consumers": df.consumer_id.nunique(),
               "start": df.timestamp.min(), "end": df.timestamp.max(),
               "missing_before": missing_before, "negative_removed": n_neg,
               "bad_voltage_removed": n_bad_v,
               "missing_after": int(df[cols].isna().sum().sum()), "has_electrical": has_elec}
    return df, summary


def daily_matrix(df):
    """Rows = consumers, columns = days, values = kWh/day."""
    return (df.set_index("timestamp").groupby("consumer_id")["energy_kwh"]
              .resample("D").sum().unstack().fillna(0))


# ================================================================ 3. FEATURES
def _consumer_features(g, recent_days=RECENT_DAYS):
    cutoff = g.timestamp.max() - pd.Timedelta(days=recent_days)
    rec, base = g.timestamp > cutoff, g.timestamp <= cutoff
    e, comp = g.energy_kwh, g.computed_kwh

    d = g.set_index("timestamp").energy_kwh.resample("D").sum()
    d_rec = d[d.index > cutoff.normalize()]
    week = d_rec.rolling(7).mean().dropna()
    night = (g.timestamp.dt.hour >= 22) | (g.timestamp.dt.hour < 6)
    zero = e < 0.01

    def night_share(m): return e[m & night].sum() / (e[m].sum() + EPS)
    def mismatch(m):    return e[m].sum() / (comp[m].sum() + EPS)

    base_daily = e[base].sum() / max(base.sum() / 24, 1)
    rec_daily = e[rec].sum() / max(rec.sum() / 24, 1)
    x = np.arange(len(d))

    return pd.Series({
        "drop_ratio": rec_daily / (base_daily + EPS),
        "mismatch_ratio": mismatch(rec),
        "mismatch_change": mismatch(rec) - mismatch(base),
        "zero_frac_recent": zero[rec].mean(),
        "zero_frac_change": zero[rec].mean() - zero[base].mean(),
        "flatness": e[rec].std() / (e[base].std() + EPS),
        "night_share_change": night_share(rec) - night_share(base),
        "recovery_ratio": d_rec.tail(7).mean() / (week.min() + EPS) if len(week) else 1.0,
        "daily_cv_recent": d_rec.std() / (d_rec.mean() + EPS),
        "trend_slope": np.polyfit(x, d.values, 1)[0] / (d.mean() + EPS) if len(d) > 1 else 0.0,
        "voltage_std": g.voltage.std(),
        # used for the loss estimate (not model features)
        "_billed_recent": e[rec].sum(),
        "_measured_recent": comp[rec].sum(),
        "_expected_recent": base_daily * rec.sum() / 24,
    })


def build_features(df):
    feats = df.groupby("consumer_id").apply(_consumer_features).reset_index()
    feats[FEATURES] = feats[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    return feats


# ================================================================ 4. MODELS
def alert_reasons(r):
    reasons = []
    if r.mismatch_ratio < 0.85:
        reasons.append(f"billed energy is only {r.mismatch_ratio:.0%} of measured V×I×pf (meter tampering?)")
    if r.zero_frac_change > 0.10:
        reasons.append(f"{r.zero_frac_recent:.0%} of recent hours read zero (meter switched off?)")
    if r.flatness < 0.30:
        reasons.append("readings have become unnaturally flat (frozen register?)")
    if r.drop_ratio < 0.80 and r.recovery_ratio < 1.3 and r.mismatch_ratio >= 0.85:
        reasons.append(f"usage fell to {r.drop_ratio:.0%} of normal and has not recovered (possible bypass)")
    return "; ".join(reasons) or "unusual overall consumption pattern"


def reason_category(r):
    if r.mismatch_ratio < 0.85:
        return "Meter tampering (billed < V×I×pf)"
    if r.zero_frac_change > 0.10:
        return "Meter switched off (zero readings)"
    if r.flatness < 0.30:
        return "Frozen register (flat readings)"
    if r.drop_ratio < 0.80 and r.recovery_ratio < 1.3:
        return "Lasting usage drop (possible bypass)"
    return "Unusual overall pattern"


def run_models(feats, labels=None, threshold=0.5, contamination=0.10):
    """
    With labels  -> supervised mode: evaluate 3 models on a 70/30 split, and score
                    every consumer with a cross-validated Random Forest.
    Without      -> unsupervised mode: Isolation Forest only.
    Returns a dict with the per-consumer report and (if labelled) evaluation results.
    """
    out = {"mode": "unsupervised", "warnings": []}
    data = feats.copy()
    if labels is not None:
        data = data.merge(labels, on="consumer_id", how="left")
        if data.is_theft.isna().any():
            out["warnings"].append(f"{int(data.is_theft.isna().sum())} consumers have no label - "
                                   "they are treated as normal for training.")
        data["is_theft"] = data.is_theft.fillna(0).astype(int)
        data["theft_type"] = data.theft_type.fillna("none")
        n_pos, n_neg = int(data.is_theft.sum()), int((data.is_theft == 0).sum())
        if min(n_pos, n_neg) >= 10:
            out["mode"] = "supervised"
        else:
            out["warnings"].append(f"Only {n_pos} theft / {n_neg} normal labels - need at least 10 of each "
                                   "to train. Using the unsupervised Isolation Forest instead.")

    X = data[FEATURES].to_numpy(dtype=float)       # plain NumPy arrays: safe on every pandas version
    Xs_all = StandardScaler().fit_transform(X)

    # Isolation Forest on everyone (works without labels)
    iso_all = IsolationForest(n_estimators=300, contamination=contamination, random_state=SEED).fit(Xs_all)
    raw_s = -iso_all.score_samples(Xs_all)
    data["anomaly_score"] = (raw_s - raw_s.min()) / (raw_s.max() - raw_s.min() + EPS)

    if out["mode"] == "supervised":
        y = data.is_theft.to_numpy(dtype=int)
        ids = data.consumer_id.astype(str).to_numpy(dtype=object)
        X_tr, X_te, y_tr, y_te, id_tr, id_te = train_test_split(
            X, y, ids, test_size=0.30, stratify=y, random_state=SEED)
        sc = StandardScaler().fit(X_tr)
        X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)

        iso = IsolationForest(n_estimators=300, contamination=float(y.mean()), random_state=SEED).fit(X_tr_s)
        ref = -iso.score_samples(X_tr_s)
        iso_score = np.clip((-iso.score_samples(X_te_s) - ref.min()) / (ref.max() - ref.min()), 0, 1)
        logreg = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X_tr_s, y_tr)
        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                    random_state=SEED).fit(X_tr_s, y_tr)

        models = {
            "Isolation Forest (no labels)": (iso_score, (iso.predict(X_te_s) == -1).astype(int)),
            "Logistic Regression": (logreg.predict_proba(X_te_s)[:, 1],
                                    (logreg.predict_proba(X_te_s)[:, 1] >= threshold).astype(int)),
            "Random Forest": (rf.predict_proba(X_te_s)[:, 1],
                              (rf.predict_proba(X_te_s)[:, 1] >= threshold).astype(int)),
        }
        out["metrics"] = pd.DataFrame({
            name: {"Accuracy": accuracy_score(y_te, p),
                   "Precision": precision_score(y_te, p, zero_division=0),
                   "Recall": recall_score(y_te, p),
                   "F1-score": f1_score(y_te, p),
                   "ROC-AUC": roc_auc_score(y_te, s)}
            for name, (s, p) in models.items()}).T
        out["test"] = {"y": y_te, "ids": id_te, "models": models}
        out["importance"] = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()

        # Every consumer scored by a model that never saw it (5-fold)
        folds = min(5, int(min(y.sum(), (y == 0).sum())))
        final = make_pipeline(StandardScaler(), RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=SEED))
        data["risk_score"] = cross_val_predict(
            final, X, y, cv=StratifiedKFold(folds, shuffle=True, random_state=SEED),
            method="predict_proba")[:, 1]
        data["status"] = np.where(data.risk_score >= threshold, "SUSPICIOUS", "NORMAL")
        out["model_name"] = "Random Forest (cross-validated)"
    else:
        data["risk_score"] = data["anomaly_score"]
        data["status"] = np.where(iso_all.predict(Xs_all) == -1, "SUSPICIOUS", "NORMAL")
        out["model_name"] = "Isolation Forest (unsupervised)"

    data["reasons"] = ""
    data["category"] = ""
    sus = data.status == "SUSPICIOUS"
    data.loc[sus, "reasons"] = data[sus].apply(alert_reasons, axis=1)
    data.loc[sus, "category"] = data[sus].apply(reason_category, axis=1)

    # Estimated unbilled energy over the last 30 days (rough, for suspicious consumers):
    # the larger of (measured - billed) and (expected from own history - billed)
    gap = np.maximum(data._measured_recent - data._billed_recent,
                     data._expected_recent - data._billed_recent).clip(lower=0)
    data["est_unbilled_kwh"] = np.where(sus, gap, 0.0)

    out["report"] = data.sort_values("risk_score", ascending=False).reset_index(drop=True)
    return out


def run_pipeline(raw, labels=None, threshold=0.5, contamination=0.10):
    """Everything in one call. Returns dict used by the web app."""
    df, summary = preprocess(raw)
    feats = build_features(df)
    res = run_models(feats, labels, threshold, contamination)
    res.update({"df": df, "summary": summary, "daily": daily_matrix(df), "labels": labels})
    return res


# ================================================================ 5. CHARTS
def _finish(fig):
    fig.tight_layout()
    return fig


def fig_daily_examples(daily, examples, starts=None):
    """examples: {title: consumer_id}; starts: {consumer_id: theft start day} (optional)."""
    starts = starts or {}
    fig, axes = plt.subplots(len(examples), 1, figsize=(10, 1.8 * len(examples) + 0.4), sharex=True)
    axes = np.atleast_1d(axes)
    days = np.arange(daily.shape[1])
    for ax, (title, cid) in zip(axes, examples.items()):
        normal = title.lower().startswith("normal")
        ax.plot(days, daily.loc[cid].values, color=BLUE if normal else ORANGE)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("kWh/day")
        ax.set_title(f"{title}  ({cid})", loc="left")
        s = starts.get(cid)
        if s is not None and not pd.isna(s):
            ax.axvline(s, color=MUTED, linestyle="--", linewidth=1)
            ax.text(s + 0.7, ax.get_ylim()[1] * 0.82, "theft starts", color=MUTED, fontsize=9)
    axes[-1].set_xlabel("Day")
    return _finish(fig)


def fig_physics(report, by):
    """Box plot of billed/measured ratio, grouped by column `by`."""
    groups = [g for g in (["none"] + THEFT_TYPES if by == "theft_type" else ["NORMAL", "SUSPICIOUS"])
              if g in set(report[by])]
    groups += sorted(set(report[by]) - set(groups))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    bp = ax.boxplot([report.loc[report[by] == g, "mismatch_ratio"] for g in groups],
                    patch_artist=True, widths=0.5, medianprops=dict(color=INK))
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(BLUE if g in ("none", "NORMAL") else ORANGE)
        patch.set_alpha(0.85)
    ax.set_xticks(range(1, len(groups) + 1), ["normal" if g == "none" else g for g in groups])
    ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1)
    ax.set_ylabel("Billed / measured energy")
    ax.set_title("Physics check (last 30 days): honest meters bill ≈ 100% of V×I×pf", loc="left")
    return _finish(fig)


def fig_confusion(models, y_test):
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.5))
    for ax, (name, (_, pred)) in zip(np.atleast_1d(axes), models.items()):
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(cm.max(), 1) * 1.15)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() * 0.6 else INK)
        ax.set_xticks([0, 1], ["Normal", "Theft"])
        ax.set_yticks([0, 1], ["Normal", "Theft"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.grid(False)
        ax.set_title(name, loc="left", fontsize=10)
    return _finish(fig)


def fig_roc(models, y_test):
    fig, ax = plt.subplots(figsize=(6, 5))
    for (name, (score, _)), c in zip(models.items(), [BLUE, ORANGE, AQUA]):
        fpr, tpr, _ = roc_curve(y_test, score)
        ax.plot(fpr, tpr, color=c, label=f"{name}  (AUC = {roc_auc_score(y_test, score):.3f})")
    ax.plot([0, 1], [0, 1], color=GREY, linestyle="--", linewidth=1, label="Random guess")
    ax.set_xlabel("False positive rate (honest consumers flagged)")
    ax.set_ylabel("True positive rate (thieves caught)")
    ax.legend(loc="lower right", fontsize=9)
    return _finish(fig)


def fig_importance(imp):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(imp.index, imp.values, color=BLUE, height=0.6)
    for i, v in enumerate(imp.values):
        ax.text(v + imp.max() * 0.01, i, f"{v:.2f}", va="center", fontsize=9, color=MUTED)
    ax.set_xlabel("Importance (share of the Random Forest's decisions)")
    ax.grid(axis="y", visible=False)
    return _finish(fig)


def fig_risk_hist(scores, is_bad, threshold=None, bad_label="Actually theft", good_label="Actually normal"):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bins = np.linspace(0, 1, 21)
    ax.hist(scores[~is_bad], bins=bins, color=BLUE, alpha=0.85, label=good_label, rwidth=0.9)
    ax.hist(scores[is_bad], bins=bins, color=ORANGE, alpha=0.85, label=bad_label, rwidth=0.9)
    if threshold is not None:
        ax.axvline(threshold, color=INK, linestyle="--", linewidth=1)
        ax.text(threshold + 0.01, ax.get_ylim()[1] * 0.9, "alert threshold", fontsize=9, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Risk score")
    ax.set_ylabel("Number of consumers")
    ax.legend()
    return _finish(fig)


def fig_categories(alerts):
    counts = alerts.category.value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(counts) + 1))
    ax.barh(counts.index, counts.values, color=ORANGE, height=0.6)
    for i, v in enumerate(counts.values):
        ax.text(v + counts.max() * 0.01, i, str(v), va="center", fontsize=9, color=MUTED)
    ax.set_xlabel("Number of suspicious consumers")
    ax.grid(axis="y", visible=False)
    return _finish(fig)


def fig_consumer_daily(daily_row, start_day=None):
    fig, ax = plt.subplots(figsize=(10, 3.4))
    days = np.arange(len(daily_row))
    cut = len(daily_row) - RECENT_DAYS
    ax.axvspan(cut, len(daily_row) - 1, color="#f4f3ef", zorder=0)
    ax.text(cut + 0.5, 0.02, "recent window (last 30 days)", transform=ax.get_xaxis_transform(),
            fontsize=9, color=MUTED)
    ax.plot(days, daily_row.values, color=BLUE)
    base_mean = daily_row.values[:cut].mean()
    ax.axhline(base_mean, color=GREY, linestyle=":", linewidth=1.2)
    ax.text(1, base_mean * 1.03, f"baseline avg {base_mean:.1f} kWh/day", fontsize=9, color=MUTED,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    ax.set_ylim(bottom=0)
    if start_day is not None and not pd.isna(start_day):
        ax.axvline(start_day, color=ORANGE, linestyle="--", linewidth=1)
        ax.text(start_day + 0.7, ax.get_ylim()[1] * 0.92, "theft starts (true label)",
                color=ORANGE, fontsize=9, ha="left")
    ax.set_xlabel("Day")
    ax.set_ylabel("kWh/day")
    return _finish(fig)


def fig_consumer_hourly(cdf, has_electrical, days=7):
    last = cdf[cdf.timestamp > cdf.timestamp.max() - pd.Timedelta(days=days)]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    if has_electrical:
        ax.plot(last.timestamp, last.computed_kwh, color=GREY, linewidth=1.5, label="Measured (V×I×pf)")
    ax.plot(last.timestamp, last.energy_kwh, color=ORANGE if has_electrical else BLUE,
            linewidth=1.5, label="Billed by meter")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("kWh per hour")
    ax.legend(loc="upper left", ncol=2)
    fig.autofmt_xdate()
    return _finish(fig)
