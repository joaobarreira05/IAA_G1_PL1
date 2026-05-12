# Analysis and Model Selection (Gonçalo)

## Models Explored and Justification for Selection

In this phase, we implemented and evaluated four distinct anomaly detection algorithms to identify malicious network traffic within the CIC-IDS2017 dataset.

| Model | F1-Score | AUC-ROC | Justification / Notes |
| :--- | :---: | :---: | :--- |
| **Autoencoder (PyTorch)** | **0.5718** | **0.8812** | **Best Performance.** Captures non-linear dependencies. |
| **Isolation Forest** | 0.4402 | 0.6901 | Fast and robust, but lacks the depth of the AE. |
| **One-Class SVM** | 0.3890 | 0.6181 | Computationally expensive; moderate performance. |
| **Local Outlier Factor** | 0.1371 | 0.4423 | Poor performance due to high dimensionality. |

### Justification:
The **Autoencoder** was selected as our primary model. By training the network to minimize reconstruction error on benign traffic, it learned to recognize normal patterns with high precision. Malicious flows, which deviate from these patterns, result in significantly higher reconstruction errors. The AUC-ROC of **0.88** demonstrates a strong ability to distinguish between classes compared to traditional distance-based (LOF) or partition-based (Isolation Forest) methods.

## Adjustments since Deliverable 1

*   **Shift to Anomaly Detection**: We moved from a supervised classification approach to a semi-supervised **Anomaly Detection** strategy. This allows the system to potentially detect "zero-day" attacks that were not present in the training labels.
*   **Data Handling**: Given the large size of the dataset (1.5GB+), we implemented a stratified sampling strategy (20%) in the pipeline to ensure computational feasibility while maintaining the statistical distribution of different attack types.
*   **Modular Execution**: The modeling pipeline was refactored to allow independent execution of models via command-line arguments, facilitating rapid iteration and hyperparameter tuning for specific algorithms.
