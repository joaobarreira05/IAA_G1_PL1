"""
Complete Evaluation Pipeline - Trains all models and generates comprehensive results
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from modeling import (
    load_data, train_isolation_forest, train_ocsvm, train_lof,
    train_autoencoder, train_autoencoder_bad
)
from evaluation import EvaluationFramework, save_summary_tables, plot_model_comparison

# Configuration
DATA_DIR = "processed_data"
MODELS_DIR = "models"
RESULTS_DIR = "evaluation_results"

os.makedirs(RESULTS_DIR, exist_ok=True)

def run_complete_evaluation():
    """Run complete evaluation pipeline"""
    
    print("\n" + "="*80)
    print(" "*20 + "COMPLETE ANOMALY DETECTION EVALUATION PIPELINE")
    print("="*80)
    
    # Load data
    print("\n📊 Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(DATA_DIR)
    print(f"✓ Data loaded successfully")
    print(f"  Train set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"  Train labels: {np.bincount(y_train)} (Normal, Anomaly)")
    print(f"  Test labels: {np.bincount(y_test)} (Normal, Anomaly)")
    
    # Dictionary to store all results
    all_results = {}
    
    # ========== 1. ISOLATION FOREST ==========
    print("\n" + "-"*80)
    print("Model 1/6: ISOLATION FOREST")
    print("-"*80)
    y_pred_if, y_scores_if = train_isolation_forest(X_train, X_test)
    
    eval_if = EvaluationFramework(
        y_true=y_test, 
        y_score=y_scores_if, 
        y_pred=y_pred_if,
        model_name="Isolation Forest"
    )
    all_results["Isolation Forest"] = eval_if.run_complete_evaluation()
    
    # ========== 2. ONE-CLASS SVM ==========
    print("\n" + "-"*80)
    print("Model 2/6: ONE-CLASS SVM")
    print("-"*80)
    y_pred_ocsvm, y_scores_ocsvm = train_ocsvm(X_train, X_test, y_train)
    
    eval_ocsvm = EvaluationFramework(
        y_true=y_test,
        y_score=y_scores_ocsvm,
        y_pred=y_pred_ocsvm,
        model_name="One-Class SVM"
    )
    all_results["One-Class SVM"] = eval_ocsvm.run_complete_evaluation()
    
    # ========== 3. LOCAL OUTLIER FACTOR ==========
    print("\n" + "-"*80)
    print("Model 3/6: LOCAL OUTLIER FACTOR")
    print("-"*80)
    y_pred_lof, y_scores_lof = train_lof(X_train, X_test)
    
    eval_lof = EvaluationFramework(
        y_true=y_test,
        y_score=y_scores_lof,
        y_pred=y_pred_lof,
        model_name="Local Outlier Factor"
    )
    all_results["Local Outlier Factor"] = eval_lof.run_complete_evaluation()
    
    # ========== 4. AUTOENCODER (PRIMARY) ==========
    print("\n" + "-"*80)
    print("Model 4/6: AUTOENCODER (PRIMARY - Normal Data)")
    print("-"*80)
    y_pred_ae, y_scores_ae, loss_ae = train_autoencoder(X_train, X_test, y_train)
    
    eval_ae = EvaluationFramework(
        y_true=y_test,
        y_score=y_scores_ae,
        y_pred=y_pred_ae,
        model_name="Autoencoder"
    )
    all_results["Autoencoder"] = eval_ae.run_complete_evaluation()
    
    # ========== 5. AUTOENCODER (SECONDARY - Bad Data) ==========
    print("\n" + "-"*80)
    print("Model 5/6: AUTOENCODER (SECONDARY - Anomaly Data)")
    print("-"*80)
    y_pred_ae2, y_scores_ae2, loss_ae2 = train_autoencoder_bad(X_train, X_test, y_train)
    
    eval_ae2 = EvaluationFramework(
        y_true=y_test,
        y_score=y_scores_ae2,
        y_pred=y_pred_ae2,
        model_name="Second Autoencoder"
    )
    all_results["Second Autoencoder"] = eval_ae2.run_complete_evaluation()
    
    # ========== 6. ENSEMBLE AUTOENCODER ==========
    print("\n" + "-"*80)
    print("Model 6/6: ENSEMBLE AUTOENCODER (Combination)")
    print("-"*80)
    print("Creating ensemble prediction (OR logic)...")
    y_pred_ensemble = (y_pred_ae | y_pred_ae2).astype(int)
    y_scores_ensemble = y_scores_ae + y_scores_ae2
    
    eval_ensemble = EvaluationFramework(
        y_true=y_test,
        y_score=y_scores_ensemble,
        y_pred=y_pred_ensemble,
        model_name="Ensemble Autoencoder"
    )
    all_results["Ensemble Autoencoder"] = eval_ensemble.run_complete_evaluation()
    
    # ========== GENERATE SUMMARY TABLES ==========
    print("\n" + "="*80)
    print("GENERATING SUMMARY TABLES")
    print("="*80)
    
    df_original, df_balanced, df_per_class = save_summary_tables(all_results)
    
    # ========== GENERATE COMPARISON PLOTS ==========
    print("\n" + "="*80)
    print("GENERATING COMPARISON PLOTS")
    print("="*80)
    
    for scenario in ['original', 'balanced']:
        for metric in ['F1-Score', 'Precision', 'Recall', 'AUC-ROC', 'Balanced_Accuracy']:
            try:
                plot_model_comparison(all_results, metric=scenario, metric_key=metric)
            except Exception as e:
                print(f"⚠ Could not plot {metric} for {scenario}: {str(e)}")
    
    # ========== LOSS-EPOCH COMPARISON ==========
    print("\n" + "="*80)
    print("LOSS-EPOCH ANALYSIS")
    print("="*80)
    
    # Plot both autoencoders loss on same figure for comparison
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(loss_ae)+1), loss_ae, marker='o', linewidth=2.5, markersize=5, 
             color='#2E86AB', label='Primary AE (Normal Data)')
    plt.xlabel('Epoch', fontsize=11, fontweight='bold')
    plt.ylabel('Loss (MSE)', fontsize=11, fontweight='bold')
    plt.title('Primary Autoencoder Loss', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(loss_ae2)+1), loss_ae2, marker='s', linewidth=2.5, markersize=6, 
             color='#A23B72', label='Secondary AE (Bad Data)')
    plt.xlabel('Epoch', fontsize=11, fontweight='bold')
    plt.ylabel('Loss (MSE)', fontsize=11, fontweight='bold')
    plt.title('Secondary Autoencoder Loss', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'autoencoder_loss_comparison.png'), dpi=300)
    plt.close()
    print("✓ Loss comparison plot saved")
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print("\n📊 ORIGINAL DISTRIBUTION (Realistic):")
    print(df_original.to_string(index=False))
    
    print("\n📊 BALANCED SUBSET (50-50):")
    print(df_balanced.to_string(index=False))
    
    print("\n📊 PER-CLASS ANALYSIS:")
    print(df_per_class.to_string(index=False))
    
    # Find best models
    print("\n" + "="*80)
    print("🏆 BEST PERFORMERS")
    print("="*80)
    
    best_f1_orig = df_original.loc[df_original['F1-Score'].idxmax()]
    print(f"\nBest F1-Score (Original): {best_f1_orig['Model']} ({best_f1_orig['F1-Score']:.4f})")
    
    best_auc = df_original.loc[df_original['AUC-ROC'].idxmax()]
    print(f"Best AUC-ROC (Original): {best_auc['Model']} ({best_auc['AUC-ROC']:.4f})")
    
    best_balanced = df_balanced.loc[df_balanced['F1-Score'].idxmax()]
    print(f"Best F1-Score (Balanced): {best_balanced['Model']} ({best_balanced['F1-Score']:.4f})")
    
    print("\n" + "="*80)
    print("✅ Complete evaluation finished!")
    print(f"📁 Results saved to: {os.path.abspath(RESULTS_DIR)}/")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_complete_evaluation()
