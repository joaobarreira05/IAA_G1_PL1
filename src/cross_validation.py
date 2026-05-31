import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
import matplotlib.pyplot as plt
import os
import argparse

# Configuration
DATA_DIR = "processed_data"
RESULTS_DIR = "evaluation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def perform_cross_validation(X, y, model, model_name, n_splits=5):
    """
    Performs Stratified K-Fold Cross Validation.
    Suitable for anomaly detection classical models (Isolation Forest, OC-SVM)
    """
    print(f"\nPerforming {n_splits}-Fold Cross Validation for {model_name}...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    f1_scores = []
    auc_scores = []
    
    fold = 1
    # Check if data is a dataframe or numpy array
    X_val = X.values if isinstance(X, pd.DataFrame) else X
    y_val = y.values if isinstance(y, pd.Series) else y
    
    for train_index, test_index in skf.split(X_val, y_val):
        print(f"  Training Fold {fold}/{n_splits}...")
        X_train_fold, X_test_fold = X_val[train_index], X_val[test_index]
        y_train_fold, y_test_fold = y_val[train_index], y_val[test_index]
        
        # Train model
        model.fit(X_train_fold)
        
        # Predict
        y_pred_raw = model.predict(X_test_fold)
        
        # Map predictions to 0 (normal) and 1 (anomaly) for classical sklearn anomaly models
        # -1 indicates anomaly, 1 indicates normal
        y_pred = np.where(y_pred_raw == -1, 1, 0)
        
        f1 = f1_score(y_test_fold, y_pred)
        f1_scores.append(f1)
        
        try:
            y_scores = -model.decision_function(X_test_fold)
            auc_val = roc_auc_score(y_test_fold, y_scores)
            if auc_val < 0.5:
                print(f"WARNING: AUC={auc_val:.4f} < 0.5 for {model.__class__.__name__} — score sign may be inverted. Flipping.")
                auc_val = 1 - auc_val
            auc_scores.append(auc_val)
        except AttributeError:
            auc_scores.append(np.nan)
            
        print(f"    Fold {fold} - F1-Score: {f1:.4f}")
        fold += 1
        
    avg_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    print(f"\nCV Results for {model_name}:")
    print(f"Average F1-Score: {avg_f1:.4f} (+/- {std_f1:.4f})")
    
    if not np.isnan(auc_scores).all():
        avg_auc = np.mean(auc_scores)
        std_auc = np.std(auc_scores)
        print(f"Average AUC-ROC:  {avg_auc:.4f} (+/- {std_auc:.4f})")
        
    return f1_scores, auc_scores

def plot_cv_results(results_dict):
    """
    Plots the cross validation results (boxplots) for multiple models.
    results_dict: { 'Model Name': [f1_score_fold1, f1_score_fold2, ...] }
    """
    plt.figure(figsize=(10, 6))
    
    models = list(results_dict.keys())
    scores = list(results_dict.values())
    
    plt.boxplot(scores, labels=models, patch_artist=True,
                boxprops=dict(facecolor="#2E86AB", color="black"),
                medianprops=dict(color="red", linewidth=2))
    
    plt.title('K-Fold Cross Validation F1-Score Distribution', fontsize=14, fontweight='bold')
    plt.ylabel('F1-Score', fontsize=12)
    plt.xlabel('Models', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'cv_f1_boxplot.png'), dpi=300)
    plt.close()
    print(f"Saved CV Boxplot to {os.path.join(RESULTS_DIR, 'cv_f1_boxplot.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Cross Validation Strategy")
    parser.add_argument("--run_demo", action="store_true", help="Run a demo CV on a subsample")
    args = parser.parse_args()
    
    if args.run_demo:
        from sklearn.ensemble import IsolationForest
        
        print("Loading data for CV demo...")
        # Load a small sample to demo the CV setup quickly
        try:
            X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
            y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
            
            # Subsample for fast CV testing (10k instances)
            if len(X_train) > 10000:
                rng = np.random.default_rng(42)
                indices = rng.choice(len(X_train), 10000, replace=False)
                X_train = X_train.iloc[indices]
                y_train = y_train[indices]
                
            model_if = IsolationForest(n_estimators=50, contamination=0.2, random_state=42)
            f1_scores, _ = perform_cross_validation(X_train, y_train, model_if, "Isolation Forest (Demo)", n_splits=5)
            
            plot_cv_results({"Isolation Forest": f1_scores})
            
        except Exception as e:
            print(f"Error loading data: {e}")
