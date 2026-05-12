"""
Hyperparameter Tuning for Classical Anomaly Detection Models
============================================================
Performs GridSearchCV / RandomizedSearchCV on:
  - Isolation Forest
  - One-Class SVM
  - Local Outlier Factor

Generates:
  - Heatmaps of the hyperparameter search space
  - Bar chart of best configurations vs baseline
  - Summary CSV with all candidate scores

Author: Margarida Ribeiro (Deliverable 3)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from itertools import product

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import f1_score, roc_auc_score, make_scorer
from sklearn.base import clone

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
#  PATHS
# ──────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "processed_data")
TUNING_DIR = os.path.join(BASE_DIR, "..", "tuning_results")
os.makedirs(TUNING_DIR, exist_ok=True)

# ──────────────────────────────────────────────
#  PLOTTING STYLE
# ──────────────────────────────────────────────
PALETTE = {
    "if":    "#2E86AB",   # steel blue
    "ocsvm": "#A23B72",   # raspberry
    "lof":   "#F18F01",   # amber
}
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
})


# ──────────────────────────────────────────────
#  DATA LOADING (with subsampling for speed)
# ──────────────────────────────────────────────
def load_data(subsample: int = 15_000, random_state: int = 42) -> tuple:
    """
    Loads pre-processed train/test splits.
    Subsamples to `subsample` rows for fast CV during tuning.
    Returns (X_train, y_train, X_test, y_test) as numpy arrays.
    """
    print(f"Loading data from {DATA_DIR}…")
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
    X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

    # Stratified subsample so tuning finishes in reasonable time
    rng = np.random.default_rng(random_state)
    normal_idx  = np.where(y_train == 0)[0]
    anomaly_idx = np.where(y_train == 1)[0]
    n_each = subsample // 2
    chosen = np.concatenate([
        rng.choice(normal_idx,  min(n_each, len(normal_idx)),  replace=False),
        rng.choice(anomaly_idx, min(n_each, len(anomaly_idx)), replace=False),
    ])
    rng.shuffle(chosen)
    X_sub = X_train.iloc[chosen].values
    y_sub = y_train[chosen]

    print(f"  Subsample shape : {X_sub.shape}  "
          f"(normal={np.sum(y_sub==0)}, anomaly={np.sum(y_sub==1)})")
    print(f"  Full test shape : {X_test.shape}")
    return X_sub, y_sub, X_test.values, y_test


# ──────────────────────────────────────────────
#  CUSTOM CV SCORER (works with unsupervised API)
# ──────────────────────────────────────────────
def score_unsupervised(estimator, X, y):
    """
    Fits the estimator on X_train fold then predicts on X_test fold.
    Maps sklearn's {-1,+1} output → {1,0} anomaly labels.
    Returns F1 score.
    """
    y_raw = estimator.predict(X)
    y_pred = np.where(y_raw == -1, 1, 0)
    return f1_score(y, y_pred, zero_division=0)


def cv_search(estimator_cls, param_grid: dict, X, y,
              n_splits: int = 3, random_state: int = 42) -> pd.DataFrame:
    """
    Manual Stratified-KFold grid-search for unsupervised estimators.
    Returns a DataFrame with one row per parameter combination.
    """
    skf     = StratifiedKFold(n_splits=n_splits, shuffle=True,
                               random_state=random_state)
    records = []

    candidates = list(ParameterGrid(param_grid))
    total      = len(candidates)
    print(f"  → {total} candidate configurations × {n_splits} folds …")

    for i, params in enumerate(candidates, 1):
        fold_f1s  = []
        fold_aucs = []

        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_val_true  = y[val_idx]

            est = estimator_cls(**params)
            est.fit(X_tr)

            y_raw  = est.predict(X_val)
            y_pred = np.where(y_raw == -1, 1, 0)
            fold_f1s.append(f1_score(y_val_true, y_pred, zero_division=0))

            try:
                scores = -est.decision_function(X_val)
                fold_aucs.append(roc_auc_score(y_val_true, scores))
            except (AttributeError, ValueError):
                fold_aucs.append(np.nan)

        row = dict(params)
        row["mean_f1"]  = float(np.mean(fold_f1s))
        row["std_f1"]   = float(np.std(fold_f1s))
        row["mean_auc"] = float(np.nanmean(fold_aucs))
        records.append(row)

        if i % max(1, total // 5) == 0 or i == total:
            print(f"    [{i}/{total}] last F1={row['mean_f1']:.4f}")

    df = pd.DataFrame(records).sort_values("mean_f1", ascending=False).reset_index(drop=True)
    return df


# ──────────────────────────────────────────────
#  ISOLATION FOREST TUNING
# ──────────────────────────────────────────────
IF_GRID = {
    "n_estimators":  [50, 100, 200],
    "contamination": [0.10, 0.15, 0.20, 0.25],
    "max_features":  [0.5, 0.75, 1.0],
    "random_state":  [42],
}

def tune_isolation_forest(X_train, y_train) -> pd.DataFrame:
    print("\n" + "="*60)
    print("ISOLATION FOREST – Hyperparameter Tuning")
    print("="*60)
    results = cv_search(IsolationForest, IF_GRID, X_train, y_train)
    out_path = os.path.join(TUNING_DIR, "if_tuning_results.csv")
    results.to_csv(out_path, index=False)
    print(f"\n✓ Results saved → {out_path}")
    print(f"  Best config : {results.iloc[0].to_dict()}")
    return results


def plot_if_heatmap(results: pd.DataFrame):
    """
    Heatmap: contamination (rows) × n_estimators (cols),
    colour = mean F1, one panel per max_features value.
    """
    max_features_vals = sorted(results["max_features"].unique())
    n_panels = len(max_features_vals)

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4),
                             sharey=True)
    if n_panels == 1:
        axes = [axes]

    vmin = results["mean_f1"].min()
    vmax = results["mean_f1"].max()

    for ax, mf in zip(axes, max_features_vals):
        sub = results[results["max_features"] == mf]
        pivot = sub.pivot_table(index="contamination",
                                columns="n_estimators",
                                values="mean_f1",
                                aggfunc="mean")
        sns.heatmap(
            pivot, ax=ax, annot=True, fmt=".3f",
            cmap="YlOrRd", vmin=vmin, vmax=vmax,
            linewidths=0.5, linecolor="white",
            cbar=(ax is axes[-1]),
            annot_kws={"size": 9},
        )
        ax.set_title(f"max_features = {mf}", fontsize=11, fontweight="bold")
        ax.set_xlabel("n_estimators", fontsize=10)
        ax.set_ylabel("contamination" if ax is axes[0] else "", fontsize=10)

    fig.suptitle(
        "Isolation Forest – Mean CV F1-Score over Hyperparameter Grid",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out = os.path.join(TUNING_DIR, "if_heatmap.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ IF heatmap saved → {out}")


def plot_if_contam_effect(results: pd.DataFrame):
    """Line plot: contamination vs mean F1 for each n_estimators."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for ne, grp in results.groupby("n_estimators"):
        agg = grp.groupby("contamination")["mean_f1"].mean().reset_index()
        ax.plot(agg["contamination"], agg["mean_f1"],
                marker="o", linewidth=2.2, markersize=7, label=f"n_est={ne}")

    ax.set_xlabel("contamination", fontsize=12)
    ax.set_ylabel("Mean CV F1-Score", fontsize=12)
    ax.set_title("Isolation Forest – Effect of Contamination on F1",
                 fontsize=13, fontweight="bold")
    ax.legend(title="n_estimators", fontsize=10)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    out = os.path.join(TUNING_DIR, "if_contamination_effect.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"✓ IF contamination effect plot saved → {out}")


# ──────────────────────────────────────────────
#  ONE-CLASS SVM TUNING
# ──────────────────────────────────────────────
OCSVM_GRID = {
    "kernel": ["rbf", "poly"],
    "nu":     [0.05, 0.10, 0.15, 0.20, 0.25],
    "gamma":  ["scale", "auto"],
}

def tune_ocsvm(X_train, y_train) -> pd.DataFrame:
    print("\n" + "="*60)
    print("ONE-CLASS SVM – Hyperparameter Tuning")
    print("="*60)
    # OC-SVM is O(n²); use a smaller subsample inside tuning
    rng = np.random.default_rng(0)
    n = min(5_000, len(X_train))
    idx = rng.choice(len(X_train), n, replace=False)
    X_sub, y_sub = X_train[idx], y_train[idx]
    print(f"  (OC-SVM sub-subsample: {n} rows for speed)")

    results = cv_search(OneClassSVM, OCSVM_GRID, X_sub, y_sub)
    out_path = os.path.join(TUNING_DIR, "ocsvm_tuning_results.csv")
    results.to_csv(out_path, index=False)
    print(f"\n✓ Results saved → {out_path}")
    print(f"  Best config : {results.iloc[0].to_dict()}")
    return results


def plot_ocsvm_heatmap(results: pd.DataFrame):
    """Heatmap: nu (rows) × gamma (cols), one panel per kernel."""
    kernels = sorted(results["kernel"].unique())
    fig, axes = plt.subplots(1, len(kernels), figsize=(6 * len(kernels), 4.5),
                             sharey=True)
    if len(kernels) == 1:
        axes = [axes]

    vmin = results["mean_f1"].min()
    vmax = results["mean_f1"].max()

    for ax, ker in zip(axes, kernels):
        sub = results[results["kernel"] == ker]
        pivot = sub.pivot_table(index="nu", columns="gamma",
                                values="mean_f1", aggfunc="mean")
        sns.heatmap(
            pivot, ax=ax, annot=True, fmt=".3f",
            cmap="RdPu", vmin=vmin, vmax=vmax,
            linewidths=0.5, linecolor="white",
            cbar=(ax is axes[-1]),
            annot_kws={"size": 9},
        )
        ax.set_title(f"kernel = {ker}", fontsize=11, fontweight="bold")
        ax.set_xlabel("gamma", fontsize=10)
        ax.set_ylabel("nu" if ax is axes[0] else "", fontsize=10)

    fig.suptitle(
        "One-Class SVM – Mean CV F1-Score over Hyperparameter Grid",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out = os.path.join(TUNING_DIR, "ocsvm_heatmap.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ OC-SVM heatmap saved → {out}")


# ──────────────────────────────────────────────
#  LOCAL OUTLIER FACTOR TUNING
# ──────────────────────────────────────────────
LOF_GRID = {
    "n_neighbors":   [5, 10, 20, 40, 60],
    "contamination": [0.10, 0.15, 0.20, 0.25],
    "novelty":       [True],
    "n_jobs":        [-1],
}

def tune_lof(X_train, y_train) -> pd.DataFrame:
    print("\n" + "="*60)
    print("LOCAL OUTLIER FACTOR – Hyperparameter Tuning")
    print("="*60)
    results = cv_search(LocalOutlierFactor, LOF_GRID, X_train, y_train)
    out_path = os.path.join(TUNING_DIR, "lof_tuning_results.csv")
    results.to_csv(out_path, index=False)
    print(f"\n✓ Results saved → {out_path}")
    print(f"  Best config : {results.iloc[0].to_dict()}")
    return results


def plot_lof_heatmap(results: pd.DataFrame):
    """Heatmap: contamination (rows) × n_neighbors (cols)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot = results.pivot_table(
        index="contamination", columns="n_neighbors",
        values="mean_f1", aggfunc="mean",
    )
    sns.heatmap(
        pivot, ax=ax, annot=True, fmt=".3f",
        cmap="YlGn",
        linewidths=0.5, linecolor="white",
        annot_kws={"size": 9},
    )
    ax.set_title("LOF – Mean CV F1-Score over Hyperparameter Grid",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("n_neighbors", fontsize=11)
    ax.set_ylabel("contamination", fontsize=11)
    plt.tight_layout()
    out = os.path.join(TUNING_DIR, "lof_heatmap.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"✓ LOF heatmap saved → {out}")


def plot_lof_neighbors_effect(results: pd.DataFrame):
    """Line plot: n_neighbors vs mean F1 per contamination value."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for cont, grp in results.groupby("contamination"):
        agg = grp.groupby("n_neighbors")["mean_f1"].mean().reset_index()
        ax.plot(agg["n_neighbors"], agg["mean_f1"],
                marker="s", linewidth=2.2, markersize=7,
                label=f"contam={cont}")
    ax.set_xlabel("n_neighbors", fontsize=12)
    ax.set_ylabel("Mean CV F1-Score", fontsize=12)
    ax.set_title("LOF – Effect of n_neighbors on F1",
                 fontsize=13, fontweight="bold")
    ax.legend(title="contamination", fontsize=10)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    out = os.path.join(TUNING_DIR, "lof_neighbors_effect.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"✓ LOF neighbors effect plot saved → {out}")


# ──────────────────────────────────────────────
#  BASELINE vs BEST COMPARISON
# ──────────────────────────────────────────────
BASELINE_PARAMS = {
    "Isolation Forest": {
        "class":  IsolationForest,
        "params": {"n_estimators": 100, "contamination": 0.20,
                   "max_features": 1.0, "random_state": 42},
    },
    "One-Class SVM": {
        "class":  OneClassSVM,
        "params": {"kernel": "rbf", "gamma": "auto", "nu": 0.20},
    },
    "LOF": {
        "class":  LocalOutlierFactor,
        "params": {"n_neighbors": 20, "contamination": 0.20,
                   "novelty": True, "n_jobs": -1},
    },
}


def evaluate_on_test(estimator_cls, params, X_train, X_test, y_test):
    """Fits estimator on full train subsample, evaluates on test."""
    est = estimator_cls(**params)
    est.fit(X_train)
    y_raw  = est.predict(X_test)
    y_pred = np.where(y_raw == -1, 1, 0)
    f1     = f1_score(y_test, y_pred, zero_division=0)
    try:
        scores = -est.decision_function(X_test)
        auc    = roc_auc_score(y_test, scores)
    except (AttributeError, ValueError):
        auc = float("nan")
    return f1, auc


def plot_baseline_vs_best(
        if_results, ocsvm_results, lof_results,
        X_train, X_test, y_train, y_test):
    """
    Bar chart comparing baseline F1 vs best-tuned F1 for all three models.
    """
    print("\n" + "="*60)
    print("Evaluating baseline vs best configurations on test set…")
    print("="*60)

    best_params = {
        "Isolation Forest": dict(if_results.iloc[0].drop(
            ["mean_f1", "std_f1", "mean_auc"]).dropna()),
        "One-Class SVM":   dict(ocsvm_results.iloc[0].drop(
            ["mean_f1", "std_f1", "mean_auc"]).dropna()),
        "LOF":             dict(lof_results.iloc[0].drop(
            ["mean_f1", "std_f1", "mean_auc", "novelty", "n_jobs"]).dropna()),
    }
    # Restore fixed params that were frozen in the grid
    best_params["LOF"]["novelty"] = True
    best_params["LOF"]["n_jobs"]  = -1
    # Cast types robustly
    for k in ["n_estimators", "n_neighbors"]:
        for m in best_params.values():
            if k in m:
                m[k] = int(m[k])
    if "random_state" in best_params["Isolation Forest"]:
        best_params["Isolation Forest"]["random_state"] = 42

    model_classes = {
        "Isolation Forest": IsolationForest,
        "One-Class SVM":    OneClassSVM,
        "LOF":              LocalOutlierFactor,
    }

    rows = []
    for name, baseline_info in BASELINE_PARAMS.items():
        cls = model_classes[name]
        f1_base, _ = evaluate_on_test(cls, baseline_info["params"],
                                       X_train, X_test, y_test)
        # Use best params found (may equal baseline for some)
        bp = best_params[name]
        if name == "One-Class SVM":
            # sub-subsample for OC-SVM speed
            rng = np.random.default_rng(0)
            idx = rng.choice(len(X_train), min(5000, len(X_train)), replace=False)
            f1_best, _ = evaluate_on_test(cls, bp, X_train[idx], X_test, y_test)
        else:
            f1_best, _ = evaluate_on_test(cls, bp, X_train, X_test, y_test)
        rows.append({"Model": name, "Baseline F1": f1_base, "Tuned F1": f1_best})
        print(f"  {name}: baseline={f1_base:.4f} | tuned={f1_best:.4f}")

    df_cmp = pd.DataFrame(rows)
    df_cmp.to_csv(os.path.join(TUNING_DIR, "baseline_vs_best.csv"), index=False)

    # ── Plot ──────────────────────────────────
    x      = np.arange(len(df_cmp))
    width  = 0.35
    colors = [PALETTE["if"], PALETTE["ocsvm"], PALETTE["lof"]]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars1 = ax.bar(x - width/2, df_cmp["Baseline F1"], width,
                   color=[c + "88" for c in colors],  # semitransparent
                   edgecolor=colors, linewidth=1.8, label="Baseline")
    bars2 = ax.bar(x + width/2, df_cmp["Tuned F1"], width,
                   color=colors, edgecolor="black", linewidth=1.0, label="Best Tuned")

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(df_cmp["Model"], fontsize=11)
    ax.set_ylabel("F1-Score (test set)", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title("Baseline vs. Best Tuned F1-Score per Model",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.35)

    fig.tight_layout()
    out = os.path.join(TUNING_DIR, "baseline_vs_best.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"\n✓ Comparison bar chart saved → {out}")
    return df_cmp


# ──────────────────────────────────────────────
#  AUC HEATMAP OVER SEARCH SPACE (extra visual)
# ──────────────────────────────────────────────
def plot_if_auc_heatmap(results: pd.DataFrame):
    """Same layout as F1 heatmap but coloured by AUC-ROC."""
    max_features_vals = sorted(results["max_features"].unique())
    n_panels = len(max_features_vals)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), sharey=True)
    if n_panels == 1:
        axes = [axes]

    vmin = results["mean_auc"].min()
    vmax = results["mean_auc"].max()

    for ax, mf in zip(axes, max_features_vals):
        sub   = results[results["max_features"] == mf]
        pivot = sub.pivot_table(index="contamination", columns="n_estimators",
                                values="mean_auc", aggfunc="mean")
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".3f",
                    cmap="Blues", vmin=vmin, vmax=vmax,
                    linewidths=0.5, linecolor="white",
                    cbar=(ax is axes[-1]), annot_kws={"size": 9})
        ax.set_title(f"max_features = {mf}", fontsize=11, fontweight="bold")
        ax.set_xlabel("n_estimators", fontsize=10)
        ax.set_ylabel("contamination" if ax is axes[0] else "", fontsize=10)

    fig.suptitle("Isolation Forest – Mean CV AUC-ROC over Hyperparameter Grid",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(TUNING_DIR, "if_auc_heatmap.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ IF AUC heatmap saved → {out}")


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data(subsample=15_000)

    # ── Isolation Forest ──
    if_results = tune_isolation_forest(X_train, y_train)
    plot_if_heatmap(if_results)
    plot_if_auc_heatmap(if_results)
    plot_if_contam_effect(if_results)

    # ── One-Class SVM ──
    ocsvm_results = tune_ocsvm(X_train, y_train)
    plot_ocsvm_heatmap(ocsvm_results)

    # ── LOF ──
    lof_results = tune_lof(X_train, y_train)
    plot_lof_heatmap(lof_results)
    plot_lof_neighbors_effect(lof_results)

    # ── Cross-model comparison ──
    plot_baseline_vs_best(
        if_results, ocsvm_results, lof_results,
        X_train, X_test, y_train, y_test,
    )

    print("\n" + "="*60)
    print("✅  Hyperparameter tuning complete!")
    print(f"📁  All outputs in: {os.path.abspath(TUNING_DIR)}")
    print("="*60)
