# 📑 Complete Evaluation Pipeline - File Index

**Generated**: April 19, 2026  
**Location**: `/src/evaluation_results/` and `/`  
**Status**: ✅ Complete and Ready for Report

---

## 📊 MAIN REPORT SECTIONS

### For Your Report (Margarida):
1. **[MARGARIDA_EVALUATION_SECTION.md](../MARGARIDA_EVALUATION_SECTION.md)**
   - Complete technical section (~3000 words)
   - Covers evaluation strategy, methodology, mathematical formulas
   - Per-model analysis with interpretations
   - Production recommendations
   - **Sections**: 5.1 - 5.10
   - **Use in your report**: Chapter 5 or after Gonçalo's model selection

2. **[QUICK_REFERENCE.md](./QUICK_REFERENCE.md)** (This file)
   - 1-page executive summary
   - Key findings and recommendations
   - Best for: Intro slide or executive summary

3. **[EVALUATION_SUMMARY.md](./EVALUATION_SUMMARY.md)**
   - Comprehensive overview with all results
   - Best for: Detailed technical appendix

---

## 📈 RESULT FILES (CSV)

### Detailed Metrics Tables:
- **[results_original_distribution.csv](./results_original_distribution.csv)**
  - 6 models × 5 metrics (Precision, Recall, F1, Balanced Acc, AUC-ROC)
  - Main results on realistic imbalanced distribution
  
- **[results_balanced_subset.csv](./results_balanced_subset.csv)**
  - Same 6 models × 5 metrics on synthetic 50-50 balanced subset
  - Shows capability when bias is removed
  
- **[results_per_class.csv](./results_per_class.csv)**
  - Per-class performance (Normal vs Anomaly)
  - 6 models × 4 metrics (Recall_Normal, Recall_Anomaly, Precision_Normal, Precision_Anomaly)
  - Critical for cybersecurity assessment

---

## 📉 LOSS HISTORY (CSV)

- **[autoencoder_loss_history.csv](../models/autoencoder_loss_history.csv)**
  - Epoch-by-epoch loss for primary autoencoder
  - 30 rows (epochs) × 2 columns (Epoch, Loss)
  - Used to determine optimal training duration

- **[autoencoder_bad_loss_history.csv](../models/autoencoder_bad_loss_history.csv)**
  - Epoch-by-epoch loss for secondary autoencoder
  - 10 rows (epochs) × 2 columns
  - Shows faster convergence on anomaly data

---

## 📊 VISUALIZATION FILES

### Loss Analysis: [CRITICAL FOR EPOCH OPTIMIZATION]
- **[autoencoder_loss_comparison.png](./autoencoder_loss_comparison.png)**
  - Side-by-side loss curves for both autoencoders
  - Shows convergence behavior
  - Use in: Training & Hyperparameters section

- **[\<models\>/autoencoder_loss_history.png](../models/autoencoder_loss_history.png)**
  - Primary AE loss trajectory (30 epochs)
  - Blue line with circle markers
  
- **[\<models\>/autoencoder_bad_loss_history.png](../models/autoencoder_bad_loss_history.png)**
  - Secondary AE loss trajectory (10 epochs)
  - Purple line with square markers

### Model Comparison Plots (10 total):

#### Original Distribution (Realistic):
- **[comparison_f1-score_original.png](./comparison_f1-score_original.png)** — F1-Score bars
- **[comparison_auc-roc_original.png](./comparison_auc-roc_original.png)** — AUC-ROC bars
- **[comparison_recall_original.png](./comparison_recall_original.png)** — Recall bars
- **[comparison_precision_original.png](./comparison_precision_original.png)** — Precision bars
- **[comparison_balanced_accuracy_original.png](./comparison_balanced_accuracy_original.png)** — Balanced Acc bars

#### Balanced Subset (Debiased):
- **[comparison_f1-score_balanced.png](./comparison_f1-score_balanced.png)**
- **[comparison_auc-roc_balanced.png](./comparison_auc-roc_balanced.png)**
- **[comparison_recall_balanced.png](./comparison_recall_balanced.png)**
- **[comparison_precision_balanced.png](./comparison_precision_balanced.png)**
- **[comparison_balanced_accuracy_balanced.png](./comparison_balanced_accuracy_balanced.png)**

### Per-Model Diagnostic Plots (18 total - 3 per model):

#### Isolation Forest:
- **[isolation_forest_cm.png](./isolation_forest_cm.png)** — Confusion matrix heatmap
- **[isolation_forest_roc.png](./isolation_forest_roc.png)** — ROC curve
- **[isolation_forest_scores.png](./isolation_forest_scores.png)** — Score distribution histogram

#### One-Class SVM:
- **[one-class_svm_cm.png](./one-class_svm_cm.png)**
- **[one-class_svm_roc.png](./one-class_svm_roc.png)**
- **[one-class_svm_scores.png](./one-class_svm_scores.png)**

#### Local Outlier Factor:
- **[local_outlier_factor_cm.png](./local_outlier_factor_cm.png)**
- **[local_outlier_factor_roc.png](./local_outlier_factor_roc.png)**
- **[local_outlier_factor_scores.png](./local_outlier_factor_scores.png)**

#### Autoencoder (Primary):
- **[autoencoder_cm.png](./autoencoder_cm.png)**
- **[autoencoder_roc.png](./autoencoder_roc.png)**
- **[autoencoder_scores.png](./autoencoder_scores.png)**

#### Second Autoencoder (Anomaly-Trained):
- **[second_autoencoder_cm.png](./second_autoencoder_cm.png)**
- **[second_autoencoder_roc.png](./second_autoencoder_roc.png)**
- **[second_autoencoder_scores.png](./second_autoencoder_scores.png)**

#### Ensemble Autoencoder (Best):
- **[ensemble_autoencoder_cm.png](./ensemble_autoencoder_cm.png)** ⭐
- **[ensemble_autoencoder_roc.png](./ensemble_autoencoder_roc.png)** ⭐
- **[ensemble_autoencoder_scores.png](./ensemble_autoencoder_scores.png)** ⭐

---

## 🐍 PYTHON SCRIPTS

### New Evaluation Framework:
- **[src/evaluation.py](../src/evaluation.py)**
  - `EvaluationFramework` class for multi-perspective evaluation
  - Methods:
    - `evaluate_original_distribution()` — Realistic imbalanced evaluation
    - `evaluate_balanced_subset()` — Debiased 50-50 evaluation
    - `evaluate_per_class()` — Per-class analysis
    - `plot_confusion_matrix()` — CM heatmap
    - `plot_roc_curve()` — ROC curve
    - `plot_score_distribution()` — Score histogram
    - `run_complete_evaluation()` — Execute all above
  - Utilities:
    - `create_summary_table()` — Generate result tables
    - `save_summary_tables()` — Save all CSV results
    - `plot_model_comparison()` — Generate comparison plots

### Modified Model Training:
- **[src/modeling.py](../src/modeling.py)** (updated)
  - `train_autoencoder()` — Now saves loss history + plots + epoch analysis
  - `train_autoencoder_bad()` — Now saves loss history + plots
  - Returns `(y_pred, y_scores, loss_history)` tuples
  - Generates PNG plots for each autoencoder

### Complete Evaluation Pipeline:
- **[src/run_complete_evaluation.py](../src/run_complete_evaluation.py)**
  - Main execution script
  - Trains all 6 models sequentially
  - Applies EvaluationFramework to each
  - Generates all 31 visualization files
  - Saves all 3 CSV result files
  - Outputs console summary

---

## 📋 USAGE INSTRUCTIONS

### To Generate Results (Already Done):
```bash
cd /home/margarida/Desktop/3ano/IAA/IAA_G1_PL1
source venv/bin/activate
cd src
python3 run_complete_evaluation.py
```

### To View Results:
```bash
# All results are in:
src/evaluation_results/

# Quick overview:
cat QUICK_REFERENCE.md

# Detailed analysis:
cat EVALUATION_SUMMARY.md

# Technical section for report:
cat ../MARGARIDA_EVALUATION_SECTION.md
```

### To Integrate into Report:
1. **Copy** `MARGARIDA_EVALUATION_SECTION.md` content to your report as Chapter/Section 5
2. **Include images** from `evaluation_results/` in your results section
3. **Reference** tables from CSV files
4. **Copy** loss curves and model comparisons

---

## 🎯 RECOMMENDATIONS FOR YOUR REPORT

### What to Include:

**Section 5: Evaluation Strategy and Results**
1. Multi-perspective evaluation framework (3 scenarios)
2. Table: Original distribution results
3. Plot: Autoencoder loss-epoch comparison ⭐
4. Table: Per-class analysis
5. Plot: AUC-ROC comparison (original)
6. Discussion: Why ensemble wins
7. Production deployment recommendations

**Appendix (Optional)**
1. All 18 per-model diagnostic plots
2. All 5 comparison metric plots
3. Loss history CSV files

### Key Figures to Show:
- ✅ `autoencoder_loss_comparison.png` — Shows convergence optimization
- ✅ `comparison_f1-score_original.png` — Final rankings
- ✅ `comparison_auc-roc_original.png` — Discrimination ability
- ✅ `ensemble_autoencoder_roc.png` — Best model ROC
- ✅ `results_original_distribution.csv` (as table)

---

## 📊 Quick Statistics

| Item | Count | Status |
|------|-------|--------|
| Models Evaluated | 6 | ✓ Complete |
| Evaluation Perspectives | 3 | ✓ Complete |
| Metrics per Perspective | 5 | ✓ Complete |
| Visualization Files | 31 | ✓ Generated |
| Result CSV Files | 3 | ✓ Generated |
| Test Set Instances | 84,837 | ✓ Used |
| Training Duration | ~15 min | ✓ Completed |

---

## ✅ Checklist for Report Integration

- [ ] Read `MARGARIDA_EVALUATION_SECTION.md` (your writing base)
- [ ] Copy section 5.1-5.10 to report document
- [ ] Include loss comparison plot
- [ ] Include results table (CSV → Table)
- [ ] Include AUC-ROC comparison plot
- [ ] Include per-class analysis table
- [ ] Add recommendations section
- [ ] Cite loss convergence findings for epochs
- [ ] Reference evaluation limitations
- [ ] Add future work suggestions

---

## 🔗 File Navigation

```
3ano/IAA/IAA_G1_PL1/
├── MARGARIDA_EVALUATION_SECTION.md ← START HERE (Your Report Foundation)
├── src/
│   ├── evaluation.py (✅ New - Evaluation framework)
│   ├── modeling.py (✅ Modified - Loss tracking)
│   ├── run_complete_evaluation.py (✅ New - Main runner)
│   ├── evaluation_results/ (✅ All outputs here)
│   │   ├── QUICK_REFERENCE.md
│   │   ├── EVALUATION_SUMMARY.md
│   │   ├── results_*.csv (3 files)
│   │   ├── *_cm.png (6 confusion matrices)
│   │   ├── *_roc.png (6 ROC curves)
│   │   ├── *_scores.png (6 distributions)
│   │   ├── comparison_*.png (10 metrics)
│   │   ├── autoencoder_loss_comparison.png
│   │   └── FILE_INDEX.md (this file)
│   └── models/
│       ├── autoencoder_loss_history.png
│       ├── autoencoder_bad_loss_history.png
│       └── autoencoder_*_loss_history.csv (2 files)
```

---

**Last Updated**: April 19, 2026  
**Margarida's Task**: ✅ COMPLETE  
**Ready for Report**: ✅ YES
