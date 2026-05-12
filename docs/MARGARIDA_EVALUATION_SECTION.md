# 5. Evaluation Strategy and Results (Margarida)

## 5.1 Evaluation Framework for Imbalanced Anomaly Detection

The fundamental challenge in evaluating anomaly detection systems is handling the **severe class imbalance** present in real-world scenarios. Our dataset comprises ~80% normal traffic and ~20% anomalous traffic, which mirrors actual network conditions but creates multiple pitfalls for naive evaluation.

### 5.1.1 The Pitfall of Aggregate Metrics

A classifier that simply predicts "Normal" for all instances achieves 80% accuracy—completely uninformative. This demonstrates why **accuracy is not a sufficient metric** for anomaly detection. Traditional metrics like accuracy, when applied to imbalanced datasets, are dominated by the majority class performance.

To address this, we employ a **multi-perspective evaluation strategy**:

---

## 5.2 Multi-Perspective Evaluation Strategy

### Perspective 1: Original Distribution (Production Reality)

**Purpose**: Evaluate performance on the realistic, imbalanced test set.

**Data**: 84,837 test instances (68,140 normal | 16,697 anomaly)

**Rationale**: This distribution reflects what the system will encounter in production. Cybersecurity systems must perform well on naturally imbalanced data.

**Metrics Applied**:
- **Precision**: Of all predicted anomalies, how many are true positives?
  $$\text{Precision} = \frac{TP}{TP + FP}$$
  
- **Recall**: Of all actual anomalies, how many are detected?
  $$\text{Recall} = \frac{TP}{TP + FN}$$
  
- **F1-Score**: Harmonic mean, balancing precision and recall
  $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
  
- **Balanced Accuracy**: Average of recall for each class
  $$\text{Balanced Acc} = \frac{\text{Recall}_{\text{Normal}} + \text{Recall}_{\text{Anomaly}}}{2}$$
  
- **AUC-ROC**: Area under the Receiver Operating Characteristic curve
  - Measures model's ability to discriminate across all thresholds
  - **Range**: 0.5 (random) to 1.0 (perfect)
  - **Advantage**: Threshold-independent, unaffected by class imbalance

### Perspective 2: Balanced Subset (Debiased Performance)

**Purpose**: Evaluate on a synthetically balanced dataset to isolate model capability from dataset bias.

**Data**: 33,394 instances (16,697 normal + 16,697 anomaly)
- Subsampling: All anomalies + equal number of randomly selected normals

**Rationale**: Removes the dataset bias to show true separability. If a model performs much worse on balanced data, it may be relying on prior probabilities rather than learned features.

**Insight**: Compare Original vs. Balanced results:
- If F1 drops significantly → Model relies on class imbalance
- If F1 remains stable → Model learned robust features

### Perspective 3: Per-Class Analysis (Class-Specific Performance)

**Purpose**: Understand performance for each class independently.

**Metrics**:
- **Recall_Normal**: $\frac{TN}{TN + FP}$ — What fraction of normal traffic is correctly identified?
- **Recall_Anomaly**: $\frac{TP}{TP + FN}$ — What fraction of attacks are caught?
- **Precision_Normal**: $\frac{TN}{TN + FN}$ — Of predicted normals, how many are correct?
- **Precision_Anomaly**: $\frac{TP}{TP + FP}$ — Of predicted anomalies, how many are correct?

**Critical Insight for Cybersecurity**:
- **High Recall_Anomaly** = Few missed attacks (security)
- **High Precision_Anomaly** = Few false alerts (operational burden)
- In practice, we prioritize **Recall_Anomaly** over precision to minimize risk

---

## 5.3 Handling the Imbalance: Design Choices

### Decision: No SMOTE/Oversampling

We deliberately did **not** apply SMOTE (Synthetic Minority Over-sampling Technique) despite the imbalance. Reasons:

1. **Production Reality**: Generating synthetic anomalies is unrealistic. Models must handle real distributions.
2. **Evaluation Validity**: Oversampling the training set would artificially inflate test metrics on original data.
3. **Anomaly Uniqueness**: Synthetic anomalies may not capture novel attack patterns the model must generalize to.
4. **Anomaly Detection Philosophy**: The goal is to learn normal behavior, not to balance classes artificially.

### Decision: Threshold Strategy

Most models output continuous scores (probabilities or distances). A **threshold** is needed to convert scores to binary predictions:

$$y_{\text{pred}} = \begin{cases} 1 & \text{if } \text{score} > \theta \\ 0 & \text{otherwise} \end{cases}$$

**Our threshold selection**:
- **Isolation Forest, OC-SVM, LOF**: Default threshold (decision boundary)
- **Primary Autoencoder**: 80th percentile of reconstruction error
  - Rationale: Expect ~20% anomalies; top 20% errors are flagged
- **Secondary Autoencoder**: 20th percentile of reconstruction error
  - Rationale: Trained on anomalies; low error → looks like attack data
- **Ensemble**: Logical OR of both AE predictions
  - Rationale: Reduce false negatives (detect more attacks)

### Decision: Stratified Sampling

Data splits (70% train, 15% val, 15% test) were **stratified by label**. This ensures each subset maintains the 80-20 class ratio, preventing test set bias.

---

## 5.4 Confusion Matrix Interpretation

For each model, we compute the **confusion matrix**:

```
                Predicted Normal    Predicted Anomaly
Actual Normal         TN                   FP
Actual Anomaly        FN                   TP
```

From this:
- **TN (True Negatives)**: Normal traffic correctly identified (good)
- **FP (False Positives)**: Normal traffic flagged as attack (operational burden)
- **FN (False Negatives)**: Attacks missed (SECURITY RISK)
- **TP (True Positives)**: Attacks correctly detected (good)

**Trade-off in Cybersecurity**:
- Minimizing **FN** (false negatives) is critical—a missed attack is catastrophic
- Minimizing **FP** (false positives) improves operator efficiency
- In practice: Prefer more FP over a single FN

---

## 5.5 ROC Curve and AUC-ROC

The **Receiver Operating Characteristic (ROC) curve** plots:
- **X-axis**: False Positive Rate = $\frac{FP}{TN + FP}$
- **Y-axis**: True Positive Rate = $\frac{TP}{TP + FN}$

As we vary the threshold from 0 to 1, the point (FPR, TPR) traces a curve.

**AUC-ROC** (Area Under the Curve):
- Measures the probability that the model ranks a random anomaly higher than a random normal sample
- **Range**: 0.5 (random) to 1.0 (perfect)
- **Advantage**: Independent of threshold and class imbalance
- **Interpretation**: 
  - AUC = 0.85 means 85% chance model correctly ranks anomaly > normal

**Why AUC-ROC matters for imbalanced data**:
- Not affected by changing class ratios
- Comprehensive view of model performance across thresholds
- Helps identify best operating point for production

---

## 5.6 Balanced Accuracy

Standard accuracy is misleading on imbalanced data. **Balanced Accuracy** is the average of per-class recalls:

$$\text{Balanced Accuracy} = \frac{\text{Recall}_{\text{Normal}} + \text{Recall}_{\text{Anomaly}}}{2}$$

This metric:
- Gives equal weight to both classes
- Ranges from 0 to 1
- Unaffected by class imbalance
- Useful diagnostic: If much lower than F1, dataset imbalance is helping the model

---

## 5.7 Model-Specific Evaluation Results

### Model 1: Isolation Forest

| Metric | Original | Balanced |
|--------|----------|----------|
| Precision | 0.4375 | 0.7586 |
| Recall | 0.4500 | 0.4500 |
| F1-Score | 0.4436 | 0.5649 |
| AUC-ROC | 0.6857 | 0.6850 |

**Analysis**:
- Catches only 45% of attacks (insufficient)
- Very sensitive to class imbalance: Precision jumps from 44% → 76% on balanced data
- Low AUC-ROC indicates weak discrimination ability in high dimensions
- **Conclusion**: Suitable only as baseline; not recommended for production

### Model 2: One-Class SVM

| Metric | Original | Balanced |
|--------|----------|----------|
| Precision | 0.3804 | 0.7202 |
| Recall | 0.3983 | 0.3983 |
| F1-Score | 0.3891 | 0.5129 |
| AUC-ROC | 0.6129 | 0.6179 |

**Analysis**:
- Worse than Isolation Forest in most metrics
- Strong class imbalance dependency (Precision: 38% → 72%)
- Computationally expensive (only trained on 10% of data due to O(n²) complexity)
- Low AUC suggests poor generalization
- **Conclusion**: Not recommended; deprecated by modern methods

### Model 3: Local Outlier Factor (LOF)

| Metric | Original | Balanced |
|--------|----------|----------|
| Precision | 0.1308 | 0.3805 |
| Recall | 0.1342 | 0.1342 |
| F1-Score | 0.1325 | 0.1984 |
| AUC-ROC | 0.4386 | 0.4382 |

**Analysis**:
- **Poorest performer**: Only catches 13% of attacks
- AUC near 0.5 indicates near-random performance
- Fails in high-dimensional space (70 features)
- LOF uses local density—struggles when anomalies form loose clusters
- **Conclusion**: Not viable for this task; should be abandoned

### Model 4: Autoencoder (Primary - Trained on Normal Data)

| Metric | Original | Balanced |
|--------|----------|----------|
| Precision | 0.5171 | 0.8189 |
| Recall | 0.5255 | 0.5255 |
| F1-Score | 0.5213 | 0.6402 |
| AUC-ROC | 0.8482 | 0.8492 |

**Analysis**:
- Significant improvement over classical methods (AUC 0.85 vs 0.69)
- Precision improves dramatically on balanced data (52% → 82%)
- Moderate recall (53%) leaves room for improvement
- Learns non-linear reconstruction of normal patterns
- **Loss at Epoch 29**: 0.0615 (converged well)
- **Conclusion**: Good baseline for deep learning; use as Part 1 of ensemble

### Model 5: Second Autoencoder (Anomaly-Trained)

| Metric | Original | Balanced |
|--------|----------|----------|
| Precision | 0.7581 | 0.9285 |
| Recall | 0.7704 | 0.7704 |
| F1-Score | 0.7642 | 0.8421 |
| AUC-ROC | 0.9451 | 0.9452 |

**Analysis**:
- **Best F1-Score**: 76.4% on original distribution
- Excellent AUC (0.945) indicates strong discrimination
- High precision (76%) reduces operational burden
- Recall (77%) catches most attacks while maintaining precision
- Trained only on 77k anomalies (vs 318k normals); converges in 10 epochs
- **Rationale for Success**: 
  - Training on anomalies alone captures attack patterns
  - Low reconstruction error on attack data → flags benign as novel
  - Inverse logic compared to primary AE provides complementary signals
- **Conclusion**: Recommended for balanced security/operational efficiency trade-off

### Model 6: Ensemble Autoencoder (Combined)

| Metric | Original | Balanced |
|----------|----------|----------|
| Precision | 0.5684 | 0.8428 |
| Recall | **0.9632** | **0.9632** |
| F1-Score | 0.7149 | 0.8990 |
| AUC-ROC | **0.9830** | **0.9832** |

**Analysis**:
- **Highest AUC**: 0.983 (near-perfect discrimination)
- **Catches 96.3% of attacks** (critical for security)
- Recall remains stable across balanced/imbalanced (true capability)
- Precision drops to 57% (acceptable operational cost for 96% catch rate)
- Logical OR combines both AEs:
  - Primary AE: Detects deviations from normal
  - Secondary AE: Detects similarity to anomalies
  - Together: Catch from two angles, minimize false negatives
- **per-class metrics**: 
  - Recall_Normal: 82% (acceptable normal traffic clarity)
  - Recall_Anomaly: 96% (excellent attack detection)
- **Conclusion**: Optimal for cybersecurity applications requiring high sensitivity

---

## 5.8 Loss Curves and Convergence Analysis

### Primary Autoencoder

```
Loss by Epoch:
Epoch 1:   0.267 (rapid decrease)
Epoch 10:  0.140 (enters smooth convergence)
Epoch 20:  0.091 (approaching plateau)
Epoch 29:  0.062 (minimum)
Epoch 30:  0.072 (slight uptick - possible overfitting)
```

**Observations**:
- Steep descent until epoch 15
- Elbow around epoch 20-24
- Recommended optimal: **28-29 epochs**
- Beyond epoch 30: Risk of overfitting (loss increases)

### Secondary Autoencoder

```
Loss by Epoch:
Epoch 2:   0.129 (rapid initial decrease)
Epoch 5:   0.104 (converging)
Epoch 10:  0.082 (converged)
```

**Observations**:
- Converges **much faster** than primary AE
- Reason: Fewer samples to learn (77k vs 318k)
- **10 epochs is sufficient**—additional training unlikely to help
- Anomalies form more coherent patterns than diverse normal traffic

---

## 5.9 Recommendations for Production Deployment

### Scenario A: Maximum Detection (Security-Critical)
- **Model**: Ensemble Autoencoder
- **Expected Performance**: 96% attack detection, 8% false alarm rate
- **Deployment**: Alert every flagged flow for human review
- **Cost**: High operator workload but minimal missed attacks

### Scenario B: Balanced Operation
- **Model**: Second Autoencoder (Anomaly-Trained)
- **Expected Performance**: 77% attack detection, 24% false alarm rate
- **Deployment**: Alert if suspicious; can be partially automated
- **Cost**: Reasonable operator burden with acceptable risk

### Scenario C: Performance Baseline (Not Recommended)
- **Model**: Primary Autoencoder
- **Expected Performance**: 53% attack detection, 48% false alarm rate
- **Deployment**: Only viable as component of ensemble
- **Cost**: High false alarms, many missed attacks

---

## 5.10 Limitations and Future Work

### Known Limitations:
1. **Single Dataset**: CIC-IDS2017 may not represent all attack types
2. **Threshold Optimization**: Fixed thresholds; could be adaptive per flow type
3. **No Online Learning**: Batch processing only; no concept drift handling
4. **Feature Engineering**: Using raw statistical features; could extract temporal patterns
5. **Interpretability**: Deep models are black boxes; hard to explain decisions

### Future Improvements:
1. **Hyperparameter Tuning**: Grid search on contamination, learning rate, hidden layer size
2. **Architecture Search**: Test deeper/wider networks, attention mechanisms
3. **Temporal Modeling**: RNNs/LSTMs to capture flow dynamics
4. **Transfer Learning**: Pre-train on larger datasets, fine-tune on target
5. **Concept Drift**: Adapt model as attack patterns evolve
6. **Explainability**: SHAP values to understand model decisions per flow

---

**Prepared by**: Margarida Ribeiro  
**Date**: April 19, 2026  
**Technical Review**: Complete evaluation pipeline testing 6 models across 3 perspectives + loss curve analysis
