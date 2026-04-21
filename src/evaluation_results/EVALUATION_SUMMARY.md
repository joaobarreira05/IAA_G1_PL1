# 📊 Evaluation Results - Anomaly Detection Models

**Data**: April 19, 2026  
**Dataset**: CIC-IDS2017 (565k stratified sample)  
**Test Set**: 84,837 instances (68,140 Normal | 16,697 Anomaly)  
**Evaluation Framework**: Multi-perspective (Original, Balanced, Per-Class)

---

## 🎯 Executive Summary

### Best Overall Model: **🏆 Ensemble Autoencoder**
- **F1-Score** (Original): **0.7149** (2nd best - balanced approach)
- **Recall** (Original): **0.9632** ⭐ (Catches 96% of attacks!)
- **AUC-ROC** (Original): **0.9830** ⭐ (Excellent discrimination)
- **Balanced Accuracy**: **0.8920** (Very good on both classes)

### Alternative: **Second Autoencoder (Anomaly-Trained)**
- **F1-Score** (Original): **0.7642** ⭐ (Best F1-score)
- **Precision** (Original): **0.7581** (Fewer false alarms)
- **AUC-ROC** (Original): **0.9451** (Excellent)
- **Better for**: Reducing false positives

---

## 📈 Results: Original Distribution (Imbalanced - Realistic)

This reflects real-world conditions: ~80% normal traffic, ~20% anomalous.

| Model | Precision | Recall | F1-Score | Balanced Acc | AUC-ROC |
|-------|-----------|--------|----------|--------------|---------|
| **Ensemble Autoencoder** | 0.5684 | **0.9632** | 0.7149 | **0.8920** | **0.9830** |
| **Second Autoencoder** | **0.7581** | 0.7704 | **0.7642** | 0.8551 | 0.9451 |
| Autoencoder | 0.5171 | 0.5255 | 0.5213 | 0.7027 | 0.8482 |
| Isolation Forest | 0.4375 | 0.4500 | 0.4436 | 0.6541 | 0.6857 |
| One-Class SVM | 0.3804 | 0.3983 | 0.3891 | 0.6196 | 0.6129 |
| Local Outlier Factor | 0.1308 | 0.1342 | 0.1325 | 0.4579 | 0.4386 |

### Key Observations:
- ✅ **Top 3 models are autoencoders** - deep learning captures non-linear patterns
- ✅ **Ensemble provides highest Recall** - optimal for cybersecurity (detect attacks)
- ⚠️ **LOF performs poorly** - high dimensionality (70 features) is problematic
- ⚠️ **OC-SVM moderate** - expensive (only used 10% subset to fit in memory)

---

## 📊 Results: Balanced Subset (50-50 - Best Performance Scenario)

Subsampled to 33,394 instances (16,697 normal + 16,697 anomaly). Shows model capability without dataset bias.

| Model | Precision | Recall | F1-Score | Balanced Acc | AUC-ROC |
|-------|-----------|--------|----------|--------------|---------|
| **Ensemble Autoencoder** | 0.8428 | 0.9632 | **0.8990** | 0.8917 | 0.9832 |
| **Second Autoencoder** | **0.9285** | 0.7704 | 0.8421 | 0.8556 | 0.9452 |
| Autoencoder | 0.8189 | 0.5255 | 0.6402 | 0.7046 | 0.8492 |
| Isolation Forest | 0.7586 | 0.4500 | 0.5649 | 0.6534 | 0.6850 |
| One-Class SVM | 0.7202 | 0.3983 | 0.5129 | 0.6218 | 0.6179 |
| Local Outlier Factor | 0.3805 | 0.1342 | 0.1984 | 0.4579 | 0.4382 |

### Key Observations:
- ✅ **Ensemble achieves 89.9% F1-Score** - dramatic improvement
- ✅ **Second Autoencoder: 92.85% Precision** - very few false alarms
- 📌 **Original dataset imbalance reduces performance** - realistic but challenging

---

## 👥 Per-Class Analysis (Understanding Each Class)

Critical for cybersecurity: We want HIGH Recall for Anomalies (detect attacks!)

| Model | Recall_Normal | Recall_Anomaly | Precision_Normal | Precision_Anomaly |
|-------|---------------|----------------|------------------|-------------------|
| **Ensemble Autoencoder** | 0.8208 | **0.9632** | 0.9891 | 0.5684 |
| **Second Autoencoder** | 0.9398 | 0.7704 | 0.9435 | **0.7581** |
| Autoencoder | 0.8798 | 0.5255 | 0.8833 | 0.5171 |
| Isolation Forest | 0.8582 | 0.4500 | 0.8643 | 0.4375 |
| One-Class SVM | 0.8410 | 0.3983 | 0.8508 | 0.3804 |
| Local Outlier Factor | 0.7815 | 0.1342 | 0.7865 | 0.1308 |

### Interpretation:
- **Ensemble**: Detects 96.32% of attacks (best for security) but 43% false alarm rate
- **Second AE**: Detects 77% of attacks with only 24% false alarm rate (better for operational teams)
- **Isolation Forest**: Only catches 45% of attacks (insufficient for security)

---

## 📉 Autoencoder Loss-Epoch Analysis

### Primary Autoencoder (Trained on Normal Data)
- **Minimum Loss**: Epoch 29 (0.0615)
- **Suggested Optimal**: ~29 epochs
- **Trend**: Steady decrease until epoch 24, then slight oscillation
- **Conclusion**: 30 epochs is appropriate, no need for more

### Secondary Autoencoder (Trained on Anomaly Data)
- **Minimum Loss**: Epoch 10 (0.0817) 
- **Suggested Optimal**: All 10 epochs used effectively
- **Trend**: Rapid decrease, converges quickly with less data (77k anomalies vs 318k normal)
- **Conclusion**: 10 epochs is sufficient, anomalies are easier to reconstruct

### Combined Loss Plot
Generated: `autoencoder_loss_comparison.png`

---

## 🎯 Model Recommendations

### For **Maximum Security** (Detect all attacks, tolerate false alarms):
→ **Use Ensemble Autoencoder**
- Catches **96.32%** of attacks
- ~57% precision (acceptable for alerting)
- Best for: IDS/IPS systems with human review

### For **Operational Efficiency** (Balance detection and alerts):
→ **Use Second Autoencoder** 
- Catches **77% of attacks**
- 76% precision (fewer operator interruptions)
- Best for: Automated response systems

### For **Speed/Simplicity** (Non-deep learning):
→ **Use Isolation Forest**
- Catches **45% of attacks** (insufficient!)
- Fast training/inference
- Only viable if combined with signature detection

### ❌ NOT Recommended:
- **Local Outlier Factor**: Poor performance (13% recall)
- **One-Class SVM**: Moderate performance, expensive
- **Single Primary Autoencoder**: Good (85% AUC) but lower recall than ensemble

---

## 📁 Generated Artifacts

### Results Tables (CSV):
- `results_original_distribution.csv` - Main evaluation results
- `results_balanced_subset.csv` - Balanced subset performance
- `results_per_class.csv` - Per-class analysis

### Visualizations:

#### Loss Analysis:
- `autoencoder_loss_comparison.png` - Both AE loss curves over epochs

#### Model Comparisons (2 scenarios × 5 metrics = 10 plots):
- `comparison_f1-score_original.png`
- `comparison_auc-roc_original.png`
- `comparison_recall_original.png`
- `comparison_precision_original.png`
- `comparison_balanced_accuracy_original.png`
- `comparison_f1-score_balanced.png`
- `comparison_auc-roc_balanced.png`
- `comparison_recall_balanced.png`
- `comparison_precision_balanced.png`
- `comparison_balanced_accuracy_balanced.png`

#### Per-Model Diagnostic Plots (6 models × 3 plots = 18 plots):
Each model has:
1. **Confusion Matrix** (`*_cm.png`)
2. **ROC Curve** (`*_roc.png`)
3. **Score Distribution** (`*_scores.png`)

Example: 
- `ensemble_autoencoder_cm.png`
- `ensemble_autoencoder_roc.png`
- `ensemble_autoencoder_scores.png`

---

## 🔍 Methodology Notes

### Multi-Perspective Evaluation:
1. **Original Distribution**: Realistic (80% normal, 20% anomaly) - reflects production
2. **Balanced Subset**: Removes bias - shows true model capability
3. **Per-Class Analysis**: Separates performance by class - crucial for imbalanced data

### Threshold Selection Strategy:
- **Primary AE**: 80th percentile of reconstruction error
- **Secondary AE**: 20th percentile (reverse logic)
- **Ensemble**: Logical OR of both predictions

### Why Not SMOTE/Oversampling?
- Dataset is naturally imbalanced (realistic)
- Anomaly detection must work with real distributions
- Oversampling would give false sense of capability

---

## 📝 Next Steps (Optional)

1. **Hyperparameter Tuning**: Optimize contamination parameter, thresholds
2. **Further Epochs**: Test if 40-50 epochs improves AE performance
3. **Different Architectures**: Deeper/wider encoders, attention layers
4. **Ensemble Methods**: Weighted voting instead of OR logic
5. **Real-World Validation**: Test on newer attack types not in training data

---

**Prepared**: April 19, 2026  
**Evaluation Time**: ~15 minutes (6 models trained and evaluated)  
**Results Location**: `/src/evaluation_results/`
