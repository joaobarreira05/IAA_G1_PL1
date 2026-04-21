# 📊 Visualization Guide - Where to Find Each Plot

**Date**: April 19, 2026  
**Location**: All plots in `/src/evaluation_results/`  
**Total**: 31 PNG files + 3 CSV files + Markdown guides  

---

## 🎯 MOST IMPORTANT PLOTS (For Your Report)

### 1️⃣ Loss Convergence Analysis ⭐
```
autoencoder_loss_comparison.png
├─ LEFT:  Primary AE (30 epochs, normal data)
│         Blue line, circle markers
│         Min loss: 0.0615 at epoch 29
│
└─ RIGHT: Secondary AE (10 epochs, anomaly data)
          Purple line, square markers
          Min loss: 0.0817 at epoch 10
```
**Use in**: Training & Hyperparameters section  
**Size**: 14×6 inches, 300 DPI  
**Key insight**: Both converged well, no overfitting

---

### 2️⃣ Model Rankings - F1-Score (Original) ⭐
```
comparison_f1-score_original.png
├─ Red bar:     Local Outlier Factor (0.133)
├─ Orange bar:  One-Class SVM (0.389)
├─ Light green: Isolation Forest (0.444)
├─ Green bar:   Autoencoder (0.521)
├─ Dark green:  Second Autoencoder (0.764) ← Best F1
└─ Teal bar:    Ensemble Autoencoder (0.715)
```
**Use in**: Results summary or ranking section  
**Key insight**: Ensemble is 2nd in F1 but best overall due to recall

---

### 3️⃣ Model Rankings - AUC-ROC (Original) ⭐
```
comparison_auc-roc_original.png
├─ Red bar:     Local Outlier Factor (0.439)
├─ Orange bar:  One-Class SVM (0.613)
├─ Light green: Isolation Forest (0.686)
├─ Green bar:   Autoencoder (0.848)
├─ Dark green:  Second Autoencoder (0.945)
└─ Teal bar:    Ensemble Autoencoder (0.983) ← Best AUC
```
**Use in**: Discrimination ability discussion  
**Key insight**: 0.983 AUC = near-perfect model ranking

---

### 4️⃣ Best Model Detailed Analysis
```
ensemble_autoencoder_roc.png
├─ Orange curve: Model ROC (AUC = 0.983)
├─ Blue line: Random classifier (AUC = 0.500)
└─ Shaded area: Model advantage

ensemble_autoencoder_cm.png
├─ TN: 55,929 (True Normal classification)
├─ FP: 12,211 (Normal wrongly flagged - acceptable)
├─ FN: 615    (Attacks missed - VERY LOW ✓)
└─ TP: 16,082 (Attacks caught - VERY HIGH ✓)

ensemble_autoencoder_scores.png
├─ Blue histogram: Normal traffic scores (concentrated left)
├─ Red histogram: Anomalous traffic scores (concentrated right)
└─ Clear separation = excellent model
```
**Use in**: Best model justification  
**Key insights**: 
- FN only 615 (out of 16,697 attacks) = 96% detection
- Good separation between classes

---

## 📈 COMPARISON PLOTS (10 Total)

### Original Distribution (Realistic - 80% Normal, 20% Anomaly)

#### Metrics Ranked:
```
comparison_f1-score_original.png      ← F1-Score ranking
comparison_auc-roc_original.png       ← AUC-ROC ranking  
comparison_recall_original.png        ← Recall of attacks
comparison_precision_original.png     ← Precision of alerts
comparison_balanced_accuracy_original.png ← Balanced metric
```

**Use for**: Comparative analysis section

### Balanced Subset (50% Normal, 50% Anomaly after subsampling)

#### Metrics Ranked:
```
comparison_f1-score_balanced.png      ← F1-Score ranking
comparison_auc-roc_balanced.png       ← AUC-ROC ranking
comparison_recall_balanced.png        ← Recall on balanced
comparison_precision_balanced.png     ← Precision on balanced
comparison_balanced_accuracy_balanced.png ← Balanced metric
```

**Use for**: Showing improvement without dataset bias

**Key observation**: Compare Original vs Balanced plots to see how much dataset imbalance affects each model

---

## 🔍 PER-MODEL DIAGNOSTIC PLOTS (18 Total - 3 per model)

Each model has three diagnostic plots:

### Model 1: Isolation Forest
```
isolation_forest_cm.png       → Confusion matrix
isolation_forest_roc.png      → ROC curve (AUC=0.686)
isolation_forest_scores.png   → Score separation (weak)
```

### Model 2: One-Class SVM
```
one-class_svm_cm.png          → Confusion matrix
one-class_svm_roc.png         → ROC curve (AUC=0.613)
one-class_svm_scores.png      → Score separation (weak)
```

### Model 3: Local Outlier Factor (POOR PERFORMER)
```
local_outlier_factor_cm.png    → Confusion matrix (many FN!)
local_outlier_factor_roc.png   → ROC curve (AUC=0.439 - near random)
local_outlier_factor_scores.png → No clear separation ✗
```

### Model 4: Autoencoder (Primary)
```
autoencoder_cm.png            → Confusion matrix
autoencoder_roc.png           → ROC curve (AUC=0.848) - good
autoencoder_scores.png        → Decent separation
```

### Model 5: Second Autoencoder (GOOD)
```
second_autoencoder_cm.png     → Confusion matrix (few FP!)
second_autoencoder_roc.png    → ROC curve (AUC=0.945) - excellent
second_autoencoder_scores.png → Clear separation ✓
```

### Model 6: Ensemble Autoencoder (BEST)
```
ensemble_autoencoder_cm.png   → Confusion matrix (minimal FN!)
ensemble_autoencoder_roc.png  → ROC curve (AUC=0.983) - near-perfect
ensemble_autoencoder_scores.png → Excellent separation ✓✓✓
```

**Use for**: Appendix or detailed model comparison  
**Pattern to notice**: Blue (Normal) and Red (Anomaly) histograms get more separated in better models

---

## 📊 RESULT TABLES (CSV Files)

### Table 1: Original Distribution (MAIN)
```
results_original_distribution.csv
├─ Isolation Forest:   F1=0.444, AUC=0.686
├─ One-Class SVM:      F1=0.389, AUC=0.613
├─ LOF:                F1=0.133, AUC=0.439 ✗
├─ Autoencoder:        F1=0.521, AUC=0.848
├─ Second AE:          F1=0.764, AUC=0.945 ← Best F1
└─ Ensemble AE:        F1=0.715, AUC=0.983 ← Best AUC
```
**Include as**: Table 5.1 in your report

### Table 2: Balanced Subset
```
results_balanced_subset.csv
├─ Shows improvement when bias removed
├─ Ensemble maintains high F1: 0.899
└─ Demonstrates model robustness
```
**Include as**: Table 5.2 (comparison)

### Table 3: Per-Class Analysis
```
results_per_class.csv
├─ Recall_Normal:    How well identifies normal (want high)
├─ Recall_Anomaly:   How well detects attacks (want HIGH!)
├─ Precision_Normal: Clean normal classifications
└─ Precision_Anomaly: Low false alarm rate
```
**Include as**: Table 5.3 (cybersecurity critical)

---

## 📉 LOSS HISTORY DATA (CSV)

### Autoencoder Loss by Epoch
```
autoencoder_loss_history.csv
├─ Epoch 1: 0.380
├─ Epoch 5: 0.171
├─ Epoch 10: 0.140
├─ Epoch 15: 0.115
├─ Epoch 20: 0.091
├─ Epoch 25: 0.088
├─ Epoch 29: 0.062 ← MINIMUM
├─ Epoch 30: 0.072 ← Slightly up = stop here
└─ Pattern: Convergence is smooth
```
**Use for**: Numerical validation if needed

### Second Autoencoder Loss by Epoch
```
autoencoder_bad_loss_history.csv
├─ Epoch 1: 0.169
├─ Epoch 5: 0.104
├─ Epoch 10: 0.082 ← CONVERGED
└─ Pattern: Fast convergence (anomalies are easier)
```

---

## 🎨 Visualization Recommendations

### For Academic Report:

**Use these 6-7 plots:**
1. ✅ `autoencoder_loss_comparison.png` — Show convergence
2. ✅ `comparison_auc-roc_original.png` — Model ranking
3. ✅ `comparison_recall_original.png` — Attack detection
4. ✅ `ensemble_autoencoder_roc.png` — Best model detail
5. ✅ `ensemble_autoencoder_cm.png` — Confusion matrix
6. ✅ `ensemble_autoencoder_scores.png` — Score separation
7. (Optional) `results_original_distribution.csv` as table

### For Presentation Slides:

**Use these 4 plots:**
1. 🎤 `comparison_auc-roc_original.png` — Quick ranking
2. 🎤 `autoencoder_loss_comparison.png` — Training progress
3. 🎤 `ensemble_autoencoder_roc.png` — Star performer
4. 🎤 Table with F1/Recall/AUC of top 3 models

### For Technical Appendix:

Include all 31 plots (organized by category)

---

## 🔍 How to Interpret Each Plot Type

### Confusion Matrix (Heatmap)
```
                Normal    Anomaly
Normal      (TN)        (FP)
Anomaly     (FN)        (TP)

Look for: 
✓ Large TN (correct normal) - top left
✓ Large TP (correct anomaly) - bottom right
✗ Small FN (missed attacks) - critical for security!
```

### ROC Curve
```
Y-axis: True Positive Rate (catch attacks)
X-axis: False Positive Rate (false alarms)

Look for:
✓ Curve bends top-left (good)
✗ Curve stays near diagonal (random)
✓ Large AUC area (excellent discrimination)
```

### Score Distribution (Histogram)
```
Blue bars: Normal traffic scores (left side is good)
Red bars: Anomaly scores (right side is good)

Look for:
✓ Blue and red separated (good model)
✗ Overlap between colors (weak model)
```

### Model Comparison (Bar Charts)
```
×6 metrics, ×2 scenarios = 10 plots

Look for:
✓ Ensemble/Second AE at top of bars
✗ LOF at bottom
✓ Consistent performance across metrics (robust model)
```

---

## 📋 Visualization Checklist for Your Report

- [ ] Have you included the loss comparison plot?
- [ ] Have you included the AUC-ROC ranking plot?
- [ ] Have you shown the best model's ROC curve?
- [ ] Have you shown the confusion matrix of best model?
- [ ] Have you included results table (CSV as table)?
- [ ] Have you included per-class analysis table?
- [ ] Have you explained what each plot shows?
- [ ] Have you cited the plots in your text?

---

## 🎯 Final Tips

1. **Loss plots go in**: "Training" or "Methodology" section
2. **Comparison plots go in**: "Results" section
3. **Per-model plots go in**: Appendix (optional but impressive)
4. **CSV tables go in**: Results section (convert to table format)
5. **All plots are publication-ready**: 300 DPI, high quality

---

**Next step**: Open your report and start adding these visualizations! 🚀

*All plots generated: April 19, 2026*
