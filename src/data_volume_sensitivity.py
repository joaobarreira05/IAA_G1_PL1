"""
Data Volume Sensitivity Analysis
=================================
Tests the best-tuned classical models at 3 different training volumes
to answer the teacher's question: should we retrain on full data or
keep volume consistent with tuning?

Scenarios:
  A — 15,000 rows  (tuning subsample — all models feasible)
  B — 80,000 rows  (medium — IF feasible, OC-SVM borderline, LOF slow)
  C — 395,000 rows (full training set — only IF feasible)

Author: Margarida Ribeiro (Deliverable 3)
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "processed_data")
TUNING_DIR = os.path.join(BASE_DIR, "..", "tuning_results")
os.makedirs(TUNING_DIR, exist_ok=True)

# ── Plotting style ───────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300})

# ── Best-tuned parameters (from grid search) ────────
BEST_PARAMS = {
    "Isolation Forest": {
        "class": IsolationForest,
        "params": {"n_estimators": 100, "contamination": 0.20,
                   "max_features": 0.75, "random_state": 42},
    },
    "One-Class SVM": {
        "class": OneClassSVM,
        "params": {"kernel": "rbf", "nu": 0.25, "gamma": "auto"},
    },
    "LOF": {
        "class": LocalOutlierFactor,
        "params": {"n_neighbors": 60, "contamination": 0.25,
                   "novelty": True, "n_jobs": -1},
    },
}

# ── Scenarios ────────────────────────────────────────
SCENARIOS = {
    "A — 15k (tuning subsample)":  15_000,
    "B — 80k (medium)":            80_000,
    "C — Full (~395k)":            None,      # None = use all rows
}

# Timeout per model fit (seconds) — prevents hanging on O(n²)
TIMEOUT_SECONDS = 300   # 5 minutes max per model

# Models that are O(n²) and should be skipped above a threshold
QUADRATIC_MODELS = {"One-Class SVM", "LOF"}
# Conservative upper bounds where O(n²) models become intractable
MAX_ROWS = {
    "One-Class SVM": 15_000,   # O(n²) training, ~5k was used during tuning
    "LOF":           50_000,   # O(n²) inference, slightly more tolerant
}


def load_full_data():
    """Load full pre-processed train/test splits."""
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv")).values
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
    X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv")).values
    y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()
    return X_train, y_train, X_test, y_test


def stratified_subsample(X, y, n, random_state=42):
    """Take a stratified subsample preserving class proportions."""
    rng = np.random.default_rng(random_state)
    normal_idx  = np.where(y == 0)[0]
    anomaly_idx = np.where(y == 1)[0]
    ratio = len(anomaly_idx) / len(y)
    n_anomaly = int(n * ratio)
    n_normal  = n - n_anomaly
    chosen = np.concatenate([
        rng.choice(normal_idx,  min(n_normal, len(normal_idx)),   replace=False),
        rng.choice(anomaly_idx, min(n_anomaly, len(anomaly_idx)), replace=False),
    ])
    rng.shuffle(chosen)
    return X[chosen], y[chosen]


def evaluate_model(name, X_train_sub, X_test, y_test):
    """Train best-tuned model and evaluate on test set. Returns dict."""
    info = BEST_PARAMS[name]
    est = info["class"](**info["params"])

    t0 = time.time()
    est.fit(X_train_sub)
    train_time = time.time() - t0

    y_raw  = est.predict(X_test)
    y_pred = np.where(y_raw == -1, 1, 0)

    f1   = f1_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    try:
        scores = -est.decision_function(X_test)
        auc = roc_auc_score(y_test, scores)
    except (AttributeError, ValueError):
        auc = float("nan")

    return {
        "F1": f1, "Recall": rec, "Precision": prec,
        "AUC-ROC": auc, "Train Time (s)": round(train_time, 2),
    }


def run_sensitivity_analysis():
    print("\n" + "=" * 70)
    print("  DATA VOLUME SENSITIVITY ANALYSIS")
    print("=" * 70)

    X_train_full, y_train_full, X_test, y_test = load_full_data()
    full_size = len(X_train_full)
    print(f"  Full training set: {full_size} rows")
    print(f"  Test set:          {len(X_test)} rows\n")

    all_rows = []

    for scenario_name, target_n in SCENARIOS.items():
        print("-" * 70)
        print(f"  SCENARIO: {scenario_name}")
        print("-" * 70)

        if target_n is None or target_n >= full_size:
            X_sub, y_sub = X_train_full, y_train_full
            actual_n = full_size
        else:
            X_sub, y_sub = stratified_subsample(
                X_train_full, y_train_full, target_n)
            actual_n = len(X_sub)

        print(f"  Training rows: {actual_n}  "
              f"(normal={np.sum(y_sub==0)}, anomaly={np.sum(y_sub==1)})")

        for model_name in BEST_PARAMS:
            # Check if model is feasible at this volume
            if model_name in MAX_ROWS and actual_n > MAX_ROWS[model_name]:
                print(f"    ✗ {model_name}: SKIPPED — O(n²) infeasible "
                      f"at {actual_n} rows (limit={MAX_ROWS[model_name]})")
                all_rows.append({
                    "Scenario": scenario_name,
                    "Training Rows": actual_n,
                    "Model": model_name,
                    "F1": None, "Recall": None, "Precision": None,
                    "AUC-ROC": None, "Train Time (s)": None,
                    "Status": "SKIPPED (O(n²) infeasible)",
                })
                continue

            print(f"    → {model_name}: training on {actual_n} rows…", end=" ")
            try:
                metrics = evaluate_model(model_name, X_sub, X_test, y_test)
                print(f"F1={metrics['F1']:.4f}  Rec={metrics['Recall']:.4f}  "
                      f"Prec={metrics['Precision']:.4f}  "
                      f"Time={metrics['Train Time (s)']}s")
                row = {
                    "Scenario": scenario_name,
                    "Training Rows": actual_n,
                    "Model": model_name,
                    "Status": "OK",
                }
                row.update(metrics)
                all_rows.append(row)
            except Exception as e:
                print(f"FAILED — {e}")
                all_rows.append({
                    "Scenario": scenario_name,
                    "Training Rows": actual_n,
                    "Model": model_name,
                    "F1": None, "Recall": None, "Precision": None,
                    "AUC-ROC": None, "Train Time (s)": None,
                    "Status": f"FAILED: {e}",
                })

    # ── Save results ─────────────────────────────────
    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(TUNING_DIR, "data_volume_sensitivity.csv")
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(df.to_string(index=False))
    print(f"\n✓ Saved → {out_csv}")

    # ── Plot ─────────────────────────────────────────
    plot_sensitivity(df)
    return df


def plot_sensitivity(df: pd.DataFrame):
    """Bar chart of F1 / Recall per model across scenarios."""
    df_ok = df[df["Status"] == "OK"].copy()
    if df_ok.empty:
        return

    models    = df_ok["Model"].unique()
    scenarios = df_ok["Scenario"].unique()
    palette   = {"Isolation Forest": "#2E86AB",
                 "One-Class SVM":    "#A23B72",
                 "LOF":              "#F18F01"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, metric in zip(axes, ["F1", "Recall"]):
        x = np.arange(len(scenarios))
        width = 0.25
        offset = 0

        for model in models:
            sub = df_ok[df_ok["Model"] == model]
            vals = []
            for s in scenarios:
                row = sub[sub["Scenario"] == s]
                vals.append(row[metric].values[0] if len(row) else 0)

            bars = ax.bar(x + offset * width, vals, width,
                          label=model, color=palette.get(model, "#999"),
                          edgecolor="black", linewidth=0.8)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                            f"{v:.3f}", ha="center", va="bottom",
                            fontsize=8.5, fontweight="bold")
            offset += 1

        ax.set_xticks(x + width)
        ax.set_xticklabels([s.split("—")[0].strip() for s in scenarios],
                           fontsize=10)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{metric} by Training Volume", fontsize=12,
                     fontweight="bold")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(axis="y", alpha=0.35)

    fig.suptitle("Data Volume Sensitivity — Best-Tuned Classical Models",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(TUNING_DIR, "data_volume_sensitivity.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"✓ Plot saved → {out}")


if __name__ == "__main__":
    run_sensitivity_analysis()
