import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse

# --- Configuration ---
DATA_DIR = "processed_data"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data(data_dir=DATA_DIR):
    print(f"Loading data from {data_dir}...")
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_val = pd.read_csv(os.path.join(data_dir, "X_val.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_val = pd.read_csv(os.path.join(data_dir, "y_val.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def evaluate_model(name, y_true, y_pred, y_scores=None):
    """
    Evaluates the model and prints metrics.
    Note: y_pred and y_true should follow (0: Normal, 1: Anomaly)
    """
    print(f"\n{'='*10} {name} Evaluation {'='*10}")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(MODELS_DIR, f"{name.lower().replace(' ', '_')}_cm.png"))
    plt.close()
    
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores) if y_scores is not None else "N/A"
    
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC:  {auc if isinstance(auc, str) else f'{auc:.4f}'}")
    
    return {"Model": name, "F1-Score": f1, "AUC-ROC": auc}

# --- Models ---

def train_isolation_forest(X_train, X_test):
    print("\nTraining Isolation Forest...")
    # contamination is the expected proportion of outliers in the dataset
    # In X_train, we saw about 19.46% of anomalies.
    clf = IsolationForest(n_estimators=100, contamination=0.2, random_state=42, n_jobs=-1)
    clf.fit(X_train)
    
    # Predict: 1 for inliers, -1 for outliers
    y_pred_raw = clf.predict(X_test)
    y_scores = -clf.decision_function(X_test) # Higher score = more anomalous
    
    # Map to our labels (0: Normal, 1: Anomaly)
    y_pred = np.where(y_pred_raw == -1, 1, 0)
    
    return y_pred, y_scores

def train_ocsvm(X_train, X_test):
    print("\nTraining One-Class SVM (this might take a while)...")
    # Using a subset if too slow, but let's try with full focus first or sampled
    # One-Class SVM is O(n^2), so for 280k rows it's very slow.
    # Sampling 10% for training if it's too slow.
    X_train_sub = X_train.sample(frac=0.1, random_state=42) if len(X_train) > 50000 else X_train
    
    clf = OneClassSVM(kernel='rbf', gamma='auto', nu=0.2)
    clf.fit(X_train_sub)
    
    y_pred_raw = clf.predict(X_test)
    y_scores = -clf.decision_function(X_test)
    
    y_pred = np.where(y_pred_raw == -1, 1, 0)
    
    return y_pred, y_scores

def train_lof(X_train, X_test):
    print("\nTraining Local Outlier Factor (Novelty Detection)...")
    # LOF for novelty detection requires fit on X_train
    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.2, novelty=True, n_jobs=-1)
    clf.fit(X_train)
    
    y_pred_raw = clf.predict(X_test)
    y_scores = -clf.decision_function(X_test)
    
    y_pred = np.where(y_pred_raw == -1, 1, 0)
    
    return y_pred, y_scores

# --- Autoencoder (PyTorch) ---

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder(X_train, X_test, y_train):
    print("\nTraining Autoencoder (PyTorch)...")
    # For AE Anomaly Detection, we usually train ONLY on Normal data
    X_train_normal = X_train[y_train == 0].values
    
    input_dim = X_train_normal.shape[1]
    model = Autoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_data = torch.FloatTensor(X_train_normal)
    train_loader = DataLoader(TensorDataset(train_data), batch_size=64, shuffle=True)
    
    num_epochs = 30
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            output = model(data[0])
            loss = criterion(output, data[0])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch+1) % 2 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.6f}")
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_data = torch.FloatTensor(X_test.values)
        test_output = model(test_data)
        # Reconstruction error as anomaly score
        mse = nn.functional.mse_loss(test_output, test_data, reduction='none').mean(dim=1).numpy()
    
    # Threshold selection (e.g., 95th percentile)
    threshold = np.percentile(mse, 80) # Adjust based on expected contamination
    y_pred = (mse > threshold).astype(int)
    
    return y_pred, mse

def train_autoencoder_bad(X_train, X_test, y_train):
    print("\nTraining Second Autoencoder (trained on Anomalous data)...")
    # For this AE, we train ONLY on Anomalous (bad) data
    X_train_bad = X_train[y_train == 1].values
    
    input_dim = X_train_bad.shape[1]
    model = Autoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_data = torch.FloatTensor(X_train_bad)
    train_loader = DataLoader(TensorDataset(train_data), batch_size=64, shuffle=True)
    
    num_epochs = 10
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            output = model(data[0])
            loss = criterion(output, data[0])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch+1) % 2 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.6f}")
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_data = torch.FloatTensor(X_test.values)
        test_output = model(test_data)
        # Reconstruction error as anomaly score
        mse = nn.functional.mse_loss(test_output, test_data, reduction='none').mean(dim=1).numpy()
    
    # Threshold selection: Since it's trained on Anomalies, LOW error means Anomaly.
    # We want to flag samples that look like the 'bad' data.
    # If we assume ~20% of data is anomalous, we take the 20% with LOWEST error.
    threshold = np.percentile(mse, 20) 
    y_pred = (mse < threshold).astype(int)
    
    # For AUC calculation, y_scores should be higher for anomalies.
    # Since low MSE = anomaly here, we can use 1/MSE or -MSE.
    # We use -mse so that lower mse (anomaly) results in a higher score.
    return y_pred, -mse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Anomaly Detection Models")
    parser.add_argument("--model", type=str, choices=["if", "ocsvm", "lof", "ae", "ae2", "all"], default="all",
                        help="Model to run: if (Isolation Forest), ocsvm (One-Class SVM), lof (LOF), ae (Autoencoder), ae2 (Ensemble Primary+Secondary AE), or all")
    args = parser.parse_args()

    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    
    results = []
    
    if args.model in ["if", "all"]:
        # 1. Isolation Forest
        y_pred_if, y_scores_if = train_isolation_forest(X_train, X_test)
        results.append(evaluate_model("Isolation Forest", y_test, y_pred_if, y_scores_if))
    
    if args.model in ["ocsvm", "all"]:
        # 2. OC-SVM
        y_pred_ocsvm, y_scores_ocsvm = train_ocsvm(X_train, X_test)
        results.append(evaluate_model("One-Class SVM", y_test, y_pred_ocsvm, y_scores_ocsvm))
    
    if args.model in ["lof", "all"]:
        # 3. LOF
        y_pred_lof, y_scores_lof = train_lof(X_train, X_test)
        results.append(evaluate_model("Local Outlier Factor", y_test, y_pred_lof, y_scores_lof))
    
    # Autoencoders section (ae and ae2 can be dependent)
    y_pred_ae, y_scores_ae = None, None
    y_pred_ae2, y_scores_ae2 = None, None
    
    if args.model in ["ae", "ae2", "all"]:
        # 4. Primary Autoencoder
        y_pred_ae, y_scores_ae = train_autoencoder(X_train, X_test, y_train)
        results.append(evaluate_model("Autoencoder", y_test, y_pred_ae, y_scores_ae))
    
    if args.model in ["ae2", "all"]:
        # 5. Second Autoencoder (Bad Traffic)
        y_pred_ae2, y_scores_ae2 = train_autoencoder_bad(X_train, X_test, y_train)
        results.append(evaluate_model("Second Autoencoder", y_test, y_pred_ae2, y_scores_ae2))
        
        # 6. Ensemble (Primary + Second)
        print("\nCreating Ensemble Autoencoder (In conjunction)...")
        # Logical OR for predictions to reduce False Negatives
        y_pred_ensemble = (y_pred_ae | y_pred_ae2).astype(int)
        
        # Combined score for AUC: Sum of normalized scores (simple sum since they are same scale)
        # Note: y_scores_ae2 is already -mse, which is higher for anomalies
        y_scores_ensemble = y_scores_ae + y_scores_ae2
        
        results.append(evaluate_model("Ensemble Autoencoder", y_test, y_pred_ensemble, y_scores_ensemble))
    
    # Summary Table
    if results:
        summary_df = pd.DataFrame(results)
        print("\n" + "="*20)
        print("Final Model Comparison")
        print("="*20)
        print(summary_df.to_string(index=False))
        
        # Save results (append if file exists and we are running a single model)
        summary_file = os.path.join(MODELS_DIR, "model_comparison.csv")
        if os.path.exists(summary_file) and args.model != "all":
            existing_results = pd.read_csv(summary_file)
            # Remove existing row for this model name if it exists to avoid duplicates
            existing_results = existing_results[existing_results['Model'] != results[0]['Model']]
            summary_df = pd.concat([existing_results, summary_df], ignore_index=True)
            
        summary_df.to_csv(summary_file, index=False)
        print(f"\nResults saved to {summary_file}")
