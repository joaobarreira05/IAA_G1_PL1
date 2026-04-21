"""
Comprehensive Evaluation Framework for Anomaly Detection Models
Handles imbalanced dataset, multi-perspective analysis, and detailed metrics
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, f1_score,
    precision_score, recall_score, roc_curve, auc, balanced_accuracy_score
)
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "processed_data"
RESULTS_DIR = "evaluation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class EvaluationFramework:
    """Multi-perspective evaluation for imbalanced anomaly detection"""
    
    def __init__(self, y_true, y_score=None, y_pred=None, model_name="Model"):
        """
        Args:
            y_true: True labels (0: Normal, 1: Anomaly)
            y_score: Continuous scores (probabilities/distances)
            y_pred: Binary predictions (0 or 1)
            model_name: Name for identification
        """
        self.y_true = y_true
        self.y_score = y_score
        self.y_pred = y_pred if y_pred is not None else (y_score > 0.5).astype(int)
        self.model_name = model_name
        self.results = {}
        
    def evaluate_original_distribution(self):
        """Evaluate on original imbalanced test set"""
        print(f"\n{'='*60}")
        print(f"Scenario 1: ORIGINAL DISTRIBUTION (Imbalanced)")
        print(f"{'='*60}")
        print(f"Class distribution: {np.bincount(self.y_true)}")
        
        metrics = {
            'Precision': precision_score(self.y_true, self.y_pred, zero_division=0),
            'Recall': recall_score(self.y_true, self.y_pred, zero_division=0),
            'F1-Score': f1_score(self.y_true, self.y_pred, zero_division=0),
            'Balanced_Accuracy': balanced_accuracy_score(self.y_true, self.y_pred),
        }
        
        if self.y_score is not None:
            metrics['AUC-ROC'] = roc_auc_score(self.y_true, self.y_score)
        else:
            metrics['AUC-ROC'] = None
            
        for key, value in metrics.items():
            if value is not None:
                print(f"{key:.<25} {value:.4f}")
            
        self.results['original'] = metrics
        return metrics
    
    def evaluate_balanced_subset(self):
        """Evaluate on balanced subset (50% anomaly, 50% normal)"""
        print(f"\n{'='*60}")
        print(f"Scenario 2: BALANCED SUBSET (50-50)")
        print(f"{'='*60}")
        
        # Get indices of each class
        normal_idx = np.where(self.y_true == 0)[0]
        anomaly_idx = np.where(self.y_true == 1)[0]
        
        # Balance: use all anomalies + equal number of normals
        n_anomalies = len(anomaly_idx)
        normal_idx_balanced = np.random.choice(normal_idx, size=n_anomalies, replace=False)
        balanced_idx = np.concatenate([normal_idx_balanced, anomaly_idx])
        
        y_true_bal = self.y_true[balanced_idx]
        y_pred_bal = self.y_pred[balanced_idx]
        y_score_bal = self.y_score[balanced_idx] if self.y_score is not None else None
        
        print(f"Balanced subset size: {len(balanced_idx)}")
        print(f"Class distribution: {np.bincount(y_true_bal)}")
        
        metrics = {
            'Precision': precision_score(y_true_bal, y_pred_bal, zero_division=0),
            'Recall': recall_score(y_true_bal, y_pred_bal, zero_division=0),
            'F1-Score': f1_score(y_true_bal, y_pred_bal, zero_division=0),
            'Balanced_Accuracy': balanced_accuracy_score(y_true_bal, y_pred_bal),
        }
        
        if y_score_bal is not None:
            metrics['AUC-ROC'] = roc_auc_score(y_true_bal, y_score_bal)
        else:
            metrics['AUC-ROC'] = None
            
        for key, value in metrics.items():
            if value is not None:
                print(f"{key:.<25} {value:.4f}")
                
        self.results['balanced'] = metrics
        return metrics
    
    def evaluate_per_class(self):
        """Per-class analysis"""
        print(f"\n{'='*60}")
        print(f"Scenario 3: PER-CLASS ANALYSIS")
        print(f"{'='*60}")
        
        # Get TP, FP, FN, TN
        cm = confusion_matrix(self.y_true, self.y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        recall_normal = tn / (tn + fp) if (tn + fp) > 0 else 0
        recall_anomaly = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision_anomaly = tp / (tp + fp) if (tp + fp) > 0 else 0
        precision_normal = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        metrics = {
            'Recall_Normal': recall_normal,
            'Recall_Anomaly': recall_anomaly,
            'Precision_Normal': precision_normal,
            'Precision_Anomaly': precision_anomaly,
        }
        
        print(f"Normal  Class => Recall: {recall_normal:.4f}, Precision: {precision_normal:.4f}")
        print(f"Anomaly Class => Recall: {recall_anomaly:.4f}, Precision: {precision_anomaly:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        print(f"  True Positives:  {tp}")
        
        self.results['per_class'] = metrics
        return metrics
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix heatmap"""
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Anomaly'],
                    yticklabels=['Normal', 'Anomaly'],
                    cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"{self.model_name.lower().replace(' ', '_')}_cm.png"), dpi=300)
        plt.close()
        print(f"✓ Confusion matrix saved")
    
    def plot_roc_curve(self):
        """Plot ROC curve"""
        if self.y_score is None:
            print("⚠ No scores available for ROC curve")
            return
        
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_name}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"{self.model_name.lower().replace(' ', '_')}_roc.png"), dpi=300)
        plt.close()
        print(f"✓ ROC curve saved")
    
    def plot_score_distribution(self):
        """Plot distribution of scores"""
        if self.y_score is None:
            print("⚠ No scores available")
            return
        
        plt.figure(figsize=(10, 6))
        plt.hist(self.y_score[self.y_true == 0], bins=50, alpha=0.6, label='Normal', color='blue')
        plt.hist(self.y_score[self.y_true == 1], bins=50, alpha=0.6, label='Anomaly', color='red')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.title(f'Score Distribution - {self.model_name}')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"{self.model_name.lower().replace(' ', '_')}_scores.png"), dpi=300)
        plt.close()
        print(f"✓ Score distribution saved")
    
    def run_complete_evaluation(self):
        """Run all evaluation scenarios and plots"""
        print(f"\n\n{'#'*60}")
        print(f"# COMPLETE EVALUATION: {self.model_name}")
        print(f"{'#'*60}")
        
        self.evaluate_original_distribution()
        self.evaluate_balanced_subset()
        self.evaluate_per_class()
        
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.plot_score_distribution()
        
        return self.results


def create_summary_table(all_results, metric='original'):
    """Create summary table across all models"""
    summary = []
    
    for model_name, results in all_results.items():
        scenario_results = results.get(metric, {})
        row = {'Model': model_name}
        row.update(scenario_results)
        summary.append(row)
    
    df = pd.DataFrame(summary)
    return df


def save_summary_tables(all_results):
    """Save all summary tables"""
    # Original distribution
    df_original = create_summary_table(all_results, 'original')
    df_original.to_csv(os.path.join(RESULTS_DIR, 'results_original_distribution.csv'), index=False)
    print("\n✓ Original distribution results saved")
    print(df_original.to_string(index=False))
    
    # Balanced subset
    df_balanced = create_summary_table(all_results, 'balanced')
    df_balanced.to_csv(os.path.join(RESULTS_DIR, 'results_balanced_subset.csv'), index=False)
    print("\n✓ Balanced subset results saved")
    print(df_balanced.to_string(index=False))
    
    # Per-class analysis
    df_per_class = create_summary_table(all_results, 'per_class')
    df_per_class.to_csv(os.path.join(RESULTS_DIR, 'results_per_class.csv'), index=False)
    print("\n✓ Per-class results saved")
    print(df_per_class.to_string(index=False))
    
    return df_original, df_balanced, df_per_class


def plot_model_comparison(all_results, metric='original', metric_key='F1-Score'):
    """Plot comparison of all models for a specific metric"""
    models = []
    scores = []
    
    for model_name, results in all_results.items():
        scenario_results = results.get(metric, {})
        if metric_key in scenario_results and scenario_results[metric_key] is not None:
            models.append(model_name)
            scores.append(scenario_results[metric_key])
    
    plt.figure(figsize=(12, 6))
    colors = ['#d62728' if s < 0.5 else '#ff7f0e' if s < 0.65 else '#2ca02c' for s in scores]
    bars = plt.bar(models, scores, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel(metric_key, fontsize=12, fontweight='bold')
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.title(f'Model Comparison: {metric_key} ({metric.upper()})', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim([0, 1.0])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'comparison_{metric_key.lower()}_{metric}.png'), dpi=300)
    plt.close()
    print(f"✓ Model comparison plot saved: {metric_key} ({metric})")


if __name__ == "__main__":
    print("Evaluation framework ready!")
    print(f"Results will be saved to: {RESULTS_DIR}/")
