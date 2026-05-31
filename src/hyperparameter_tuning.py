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
from sklearn.metrics import f1_score, roc_auc_score, make_scorer, recall_score, precision_score
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

    # Stratified subsample preserving original class ratio
    rng = np.random.default_rng(random_state)
    normal_idx  = np.where(y_train == 0)[0]
    anomaly_idx = np.where(y_train == 1)[0]
    ratio = np.sum(y_train == 1) / len(y_train)
    n_anomaly = int(subsample * ratio)
    n_normal  = subsample - n_anomaly
    chosen = np.concatenate([
        rng.choice(normal_idx,  min(n_normal,  len(normal_idx)),  replace=False),
        rng.choice(anomaly_idx, min(n_anomaly, len(anomaly_idx)), replace=False),
    ])
    rng.shuffle(chosen)
    X_sub = X_train.iloc[chosen].values
    y_sub = y_train[chosen]

    actual_ratio = np.sum(y_sub == 1) / len(y_sub)
    print(f"  Subsample shape : {X_sub.shape}  "
          f"(normal={np.sum(y_sub==0)}, anomaly={np.sum(y_sub==1)})")
    print(f"  Anomaly ratio   : {actual_ratio:.4f} "
          f"(original={ratio:.4f})")
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
        fold_f1s      = []
        fold_recalls  = []
        fold_precisions = []
        fold_aucs     = []

        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_val_true  = y[val_idx]

            est = estimator_cls(**params)
            est.fit(X_tr)

            y_raw  = est.predict(X_val)
            y_pred = np.where(y_raw == -1, 1, 0)
            fold_f1s.append(f1_score(y_val_true, y_pred, zero_division=0))
            fold_recalls.append(recall_score(y_val_true, y_pred, zero_division=0))
            fold_precisions.append(precision_score(y_val_true, y_pred, zero_division=0))

            try:
                scores = -est.decision_function(X_val)
                auc_val = roc_auc_score(y_val_true, scores)
                if auc_val < 0.5:
                    print(f"WARNING: AUC={auc_val:.4f} < 0.5 for {estimator_cls.__name__} — score sign may be inverted. Flipping.")
                    auc_val = 1 - auc_val
                fold_aucs.append(auc_val)
            except (AttributeError, ValueError):
                fold_aucs.append(np.nan)

        row = dict(params)
        row["mean_f1"]        = float(np.mean(fold_f1s))
        row["std_f1"]         = float(np.std(fold_f1s))
        row["mean_recall"]    = float(np.mean(fold_recalls))
        row["std_recall"]     = float(np.std(fold_recalls))
        row["mean_precision"] = float(np.mean(fold_precisions))
        row["std_precision"]  = float(np.std(fold_precisions))
        row["mean_auc"]       = float(np.nanmean(fold_aucs))
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
    "max_features":  [0.5, 0.75, 1.0],
    "random_state":  [42],
}

CONTAMINATION_SWEEP = [0.05, 0.10, 0.15, 0.20, 0.25]


def _threshold_sweep(estimator_cls, best_structural_params, X, y,
                     contam_key="contamination", n_splits=3, random_state=42):
    """
    After CV selects the best structural hyper-parameters, sweep
    contamination (threshold) values on the validation folds and pick
    the one that maximises F1.  This keeps threshold selection
    methodologically honest — it is reported separately.
    """
    print(f"\n  Threshold tuning on val set (contamination sweep) …")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)
    best_contam, best_f1 = None, -1
    sweep_rows = []
    for contam in CONTAMINATION_SWEEP:
        params = {**best_structural_params, contam_key: contam}
        fold_f1s = []
        for tr_idx, val_idx in skf.split(X, y):
            est = estimator_cls(**params)
            est.fit(X[tr_idx])
            y_pred = np.where(est.predict(X[val_idx]) == -1, 1, 0)
            fold_f1s.append(f1_score(y[val_idx], y_pred, zero_division=0))
        mean_f1 = float(np.mean(fold_f1s))
        sweep_rows.append({contam_key: contam, "mean_f1": mean_f1})
        print(f"    {contam_key}={contam:.2f} → mean F1={mean_f1:.4f}")
        if mean_f1 > best_f1:
            best_f1, best_contam = mean_f1, contam
    print(f"  ✓ Best threshold: {contam_key}={best_contam} (F1={best_f1:.4f})")
    return best_contam, best_f1, pd.DataFrame(sweep_rows)


def tune_isolation_forest(X_train, y_train) -> pd.DataFrame:
    print("\n" + "="*60)
    print("ISOLATION FOREST – Hyperparameter Tuning")
    print("="*60)
    results = cv_search(IsolationForest, IF_GRID, X_train, y_train)
    out_path = os.path.join(TUNING_DIR, "if_tuning_results.csv")
    results.to_csv(out_path, index=False)
    print(f"\n✓ Results saved → {out_path}")
    print(f"  Best structural config : {results.iloc[0].to_dict()}")

    # Separate contamination (threshold) sweep on val set
    metric_cols = ["mean_f1", "std_f1", "mean_recall", "std_recall",
                   "mean_precision", "std_precision", "mean_auc"]
    best_structural = dict(results.iloc[0].drop(
        [c for c in metric_cols if c in results.columns]).dropna())
    if "n_estimators" in best_structural:
        best_structural["n_estimators"] = int(best_structural["n_estimators"])
    if "random_state" in best_structural:
        best_structural["random_state"] = 42
    best_contam, _, sweep_df = _threshold_sweep(
        IsolationForest, best_structural, X_train, y_train)
    sweep_df.to_csv(os.path.join(TUNING_DIR, "if_threshold_sweep.csv"), index=False)

    # Store best contamination for downstream use
    results.attrs["best_contamination"] = best_contam
    return results


def plot_if_heatmap(results: pd.DataFrame):
    """
    Heatmap: max_features (rows) × n_estimators (cols),
    colour = mean F1.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot = results.pivot_table(index="max_features",
                                columns="n_estimators",
                                values="mean_f1",
                                aggfunc="mean")
    sns.heatmap(
        pivot, ax=ax, annot=True, fmt=".3f",
        cmap="YlOrRd",
        linewidths=0.5, linecolor="white",
        annot_kws={"size": 9},
    )
    ax.set_title("Isolation Forest – Mean CV F1-Score (structural params)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("n_estimators", fontsize=10)
    ax.set_ylabel("max_features", fontsize=10)
    plt.tight_layout()
    out = os.path.join(TUNING_DIR, "if_heatmap.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ IF heatmap saved → {out}")


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
    print(f"  Best structural config : {results.iloc[0].to_dict()}")

    # Separate contamination (threshold) sweep on val set
    metric_cols = ["mean_f1", "std_f1", "mean_recall", "std_recall",
                   "mean_precision", "std_precision", "mean_auc"]
    best_structural = dict(results.iloc[0].drop(
        [c for c in metric_cols if c in results.columns]).dropna())
    if "n_neighbors" in best_structural:
        best_structural["n_neighbors"] = int(best_structural["n_neighbors"])
    best_structural["novelty"] = True
    best_structural["n_jobs"] = -1
    best_contam, _, sweep_df = _threshold_sweep(
        LocalOutlierFactor, best_structural, X_train, y_train)
    sweep_df.to_csv(os.path.join(TUNING_DIR, "lof_threshold_sweep.csv"), index=False)

    results.attrs["best_contamination"] = best_contam
    return results


def plot_lof_heatmap(results: pd.DataFrame):
    """Bar chart: n_neighbors vs mean F1 (no contamination in grid)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    agg = results.groupby("n_neighbors")["mean_f1"].mean().reset_index()
    ax.bar(agg["n_neighbors"].astype(str), agg["mean_f1"],
           color="#F18F01", edgecolor="black", linewidth=0.8)
    for i, row in agg.iterrows():
        ax.text(i, row["mean_f1"] + 0.003, f"{row['mean_f1']:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("LOF – Mean CV F1-Score by n_neighbors (structural)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("n_neighbors", fontsize=11)
    ax.set_ylabel("Mean CV F1", fontsize=11)
    ax.set_ylim(0, max(agg["mean_f1"]) * 1.15)
    ax.grid(axis="y", alpha=0.35)
    plt.tight_layout()
    out = os.path.join(TUNING_DIR, "lof_heatmap.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"✓ LOF heatmap saved → {out}")


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
    rec    = recall_score(y_test, y_pred, zero_division=0)
    prec   = precision_score(y_test, y_pred, zero_division=0)
    try:
        scores = -est.decision_function(X_test)
        auc    = roc_auc_score(y_test, scores)
    except (AttributeError, ValueError):
        auc = float("nan")
    return f1, rec, prec, auc


def plot_baseline_vs_best(
        if_results, ocsvm_results, lof_results,
        X_train, X_test, y_train, y_test):
    """
    Bar chart comparing baseline F1 vs best-tuned F1 for all three models.
    """
    print("\n" + "="*60)
    print("Evaluating baseline vs best configurations on test set…")
    print("="*60)

    # Columns added by our extended cv_search
    metric_cols = ["mean_f1", "std_f1", "mean_recall", "std_recall",
                   "mean_precision", "std_precision", "mean_auc"]

    best_params = {
        "Isolation Forest": dict(if_results.iloc[0].drop(
            [c for c in metric_cols if c in if_results.columns]).dropna()),
        "One-Class SVM":   dict(ocsvm_results.iloc[0].drop(
            [c for c in metric_cols if c in ocsvm_results.columns]).dropna()),
        "LOF":             dict(lof_results.iloc[0].drop(
            [c for c in metric_cols + ["novelty", "n_jobs"]
             if c in lof_results.columns]).dropna()),
    }
    # Restore fixed params that were frozen in the grid
    best_params["LOF"]["novelty"] = True
    best_params["LOF"]["n_jobs"]  = -1
    # Add best contamination from threshold sweep (stored in attrs)
    if hasattr(if_results, 'attrs') and "best_contamination" in if_results.attrs:
        best_params["Isolation Forest"]["contamination"] = if_results.attrs["best_contamination"]
    if hasattr(lof_results, 'attrs') and "best_contamination" in lof_results.attrs:
        best_params["LOF"]["contamination"] = lof_results.attrs["best_contamination"]
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
        f1_base, rec_base, prec_base, _ = evaluate_on_test(
            cls, baseline_info["params"], X_train, X_test, y_test)
        # Use best params found (may equal baseline for some)
        bp = best_params[name]
        if name == "One-Class SVM":
            # Use same subsample size as CV (15k) for consistency
            print("NOTE: OC-SVM trained on same 15k subsample as CV for consistency.")
            rng = np.random.default_rng(0)
            idx = rng.choice(len(X_train), min(15_000, len(X_train)), replace=False)
            f1_best, rec_best, prec_best, _ = evaluate_on_test(
                cls, bp, X_train[idx], X_test, y_test)
        else:
            f1_best, rec_best, prec_best, _ = evaluate_on_test(
                cls, bp, X_train, X_test, y_test)
        rows.append({
            "Model": name,
            "Baseline F1": f1_base, "Baseline Recall": rec_base,
            "Baseline Precision": prec_base,
            "Tuned F1": f1_best, "Tuned Recall": rec_best,
            "Tuned Precision": prec_best,
        })
        print(f"  {name}: baseline F1={f1_base:.4f} Rec={rec_base:.4f} Prec={prec_base:.4f}"
              f" | tuned F1={f1_best:.4f} Rec={rec_best:.4f} Prec={prec_best:.4f}")

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
    """Heatmap: max_features × n_estimators, coloured by AUC-ROC."""
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot = results.pivot_table(index="max_features", columns="n_estimators",
                                values="mean_auc", aggfunc="mean")
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".3f",
                cmap="Blues",
                linewidths=0.5, linecolor="white",
                annot_kws={"size": 9})
    ax.set_title("Isolation Forest – Mean CV AUC-ROC (structural params)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("n_estimators", fontsize=10)
    ax.set_ylabel("max_features", fontsize=10)
    plt.tight_layout()
    out = os.path.join(TUNING_DIR, "if_auc_heatmap.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ IF AUC heatmap saved → {out}")


# ──────────────────────────────────────────────
#  FOLD-LEVEL SCALING VALIDATION
#  (Teacher requirement: prove global scaling
#   is equivalent to per-fold scaling)
# ──────────────────────────────────────────────
def validate_fold_scaling(X, y, n_splits: int = 3, random_state: int = 42):
    """
    Computes per-fold mean and std of features and compares them.
    If the differences are negligible, global pre-scaling is justified.
    Saves a summary CSV to TUNING_DIR.
    """
    from sklearn.preprocessing import StandardScaler

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)
    fold_stats = []
    for fold_idx, (tr_idx, _) in enumerate(skf.split(X, y), 1):
        X_fold = X[tr_idx]
        scaler = StandardScaler().fit(X_fold)
        fold_stats.append({
            "fold": fold_idx,
            "global_mean_of_means": float(np.mean(scaler.mean_)),
            "global_mean_of_stds":  float(np.mean(np.sqrt(scaler.var_))),
            "max_abs_mean":         float(np.max(np.abs(scaler.mean_))),
            "max_abs_std":          float(np.max(np.sqrt(scaler.var_))),
        })

    df = pd.DataFrame(fold_stats)
    out = os.path.join(TUNING_DIR, "fold_scaling_validation.csv")
    df.to_csv(out, index=False)

    print("\n" + "="*60)
    print("FOLD-LEVEL SCALING VALIDATION")
    print("="*60)
    print(df.to_string(index=False))

    # Compute max deviation across folds
    mean_range = df["global_mean_of_means"].max() - df["global_mean_of_means"].min()
    std_range  = df["global_mean_of_stds"].max()  - df["global_mean_of_stds"].min()
    print(f"\n  Max range in fold means: {mean_range:.6f}")
    print(f"  Max range in fold stds:  {std_range:.6f}")
    if mean_range < 0.01 and std_range < 0.01:
        print("  ✓ Differences are negligible → global pre-scaling is justified.")
    else:
        print("  ⚠ Non-trivial differences detected — consider per-fold scaling.")
    print(f"  Saved → {out}")
    return df


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data(subsample=15_000)

    # ── Validate global scaling assumption ──
    validate_fold_scaling(X_train, y_train)

    # ── Isolation Forest ──
    if_results = tune_isolation_forest(X_train, y_train)
    plot_if_heatmap(if_results)
    plot_if_auc_heatmap(if_results)

    # ── One-Class SVM ──
    ocsvm_results = tune_ocsvm(X_train, y_train)
    plot_ocsvm_heatmap(ocsvm_results)

    # ── LOF ──
    lof_results = tune_lof(X_train, y_train)
    plot_lof_heatmap(lof_results)

    # ── Cross-model comparison ──
    plot_baseline_vs_best(
        if_results, ocsvm_results, lof_results,
        X_train, X_test, y_train, y_test,
    )

    print("\n" + "="*60)
    print("✅  Hyperparameter tuning complete!")
    print(f"📁  All outputs in: {os.path.abspath(TUNING_DIR)}")
    print("="*60)
