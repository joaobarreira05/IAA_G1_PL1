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

def train_ocsvm(X_train, X_test, y_train):
    print("\nTraining One-Class SVM (this might take a while)...")
    # Stratified 15k subsample consistent with CV tuning
    if len(X_train) > 15_000:
        y_arr = y_train.values.ravel() if hasattr(y_train, 'values') else np.asarray(y_train).ravel()
        rng = np.random.default_rng(42)
        normal_idx  = np.where(y_arr == 0)[0]
        anomaly_idx = np.where(y_arr == 1)[0]
        ratio = len(anomaly_idx) / len(y_arr)
        n_anomaly = int(15_000 * ratio)
        n_normal  = 15_000 - n_anomaly
        chosen = np.concatenate([
            rng.choice(normal_idx,  min(n_normal,  len(normal_idx)),  replace=False),
            rng.choice(anomaly_idx, min(n_anomaly, len(anomaly_idx)), replace=False),
        ])
        X_train_sub = X_train.iloc[chosen] if hasattr(X_train, 'iloc') else X_train[chosen]
        print(f"NOTE: OC-SVM trained on 15k stratified subsample (consistent with CV tuning).")
    else:
        X_train_sub = X_train
    
    clf = OneClassSVM(kernel='rbf', gamma='auto', nu=0.2)
    clf.fit(X_train_sub)
    
    y_pred_raw = clf.predict(X_test)
    y_scores = -clf.decision_function(X_test)
    
    y_pred = np.where(y_pred_raw == -1, 1, 0)
    
    return y_pred, y_scores

def train_lof(X_train, X_test, y_train=None):
    print("\nTraining Local Outlier Factor (Novelty Detection)...")
    
    # Subsampling to 15k for speed (LOF is O(N^2) complexity and extremely slow on 395k rows)
    if len(X_train) > 15_000:
        if y_train is not None:
            # We want to train on a subsample of normal data since novelty=True
            y_arr = y_train.values.ravel() if hasattr(y_train, 'values') else np.asarray(y_train).ravel()
            normal_idx = np.where(y_arr == 0)[0]
            rng = np.random.default_rng(42)
            chosen = rng.choice(normal_idx, min(15_000, len(normal_idx)), replace=False)
            X_train_sub = X_train.iloc[chosen] if hasattr(X_train, 'iloc') else X_train[chosen]
            print(f"NOTE: LOF trained on 15k clean normal subsample (speed optimization and novelty detection requirement).")
        else:
            X_train_sub = X_train.sample(n=15_000, random_state=42)
            print(f"NOTE: LOF trained on 15k random subsample for speed.")
    else:
        if y_train is not None:
            y_arr = y_train.values.ravel() if hasattr(y_train, 'values') else np.asarray(y_train).ravel()
            normal_idx = np.where(y_arr == 0)[0]
            X_train_sub = X_train.iloc[normal_idx] if hasattr(X_train, 'iloc') else X_train[normal_idx]
        else:
            X_train_sub = X_train

    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.2, novelty=True, n_jobs=-1)
    clf.fit(X_train_sub)
    
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

def train_autoencoder(X_train, X_val, X_test, y_train, y_val):
    print("\nTraining Autoencoder (PyTorch)...")
    # For AE Anomaly Detection, we usually train ONLY on Normal data
    X_train_normal = X_train[y_train == 0].values
    X_val_normal = X_val[y_val == 0].values
    
    input_dim = X_train_normal.shape[1]
    model = Autoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_data = torch.FloatTensor(X_train_normal)
    train_loader = DataLoader(TensorDataset(train_data), batch_size=64, shuffle=True)
    
    val_data_tensor = torch.FloatTensor(X_val_normal)
    val_loader = DataLoader(TensorDataset(val_data_tensor), batch_size=256, shuffle=False)
    
    max_epochs = 100
    patience = 5
    min_delta = 1e-6
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    model.train()
    loss_history = []
    val_loss_history = []
    
    import copy
    
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            output = model(data[0])
            loss = criterion(output, data[0])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * data[0].size(0)
        avg_loss = epoch_loss / len(X_train_normal)
        loss_history.append(avg_loss)
        
        # Validation loss for early stopping
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                output = model(data[0])
                loss = criterion(output, data[0])
                epoch_val_loss += loss.item() * data[0].size(0)
        avg_val_loss = epoch_val_loss / len(X_val_normal)
        val_loss_history.append(avg_val_loss)
        
        if (epoch+1) % 2 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{max_epochs}], Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
        # Check early stopping
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best Val Loss: {best_val_loss:.6f}")
            break
            
    # Restore best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    num_epochs = len(loss_history)
    
    # Plot loss history
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), loss_history, marker='o', linewidth=2, markersize=4, color='#2E86AB', label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_loss_history, marker='x', linewidth=2, markersize=4, color='#D36582', label='Val Loss')
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
    plt.title('Autoencoder Training & Validation Loss over Epochs', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "autoencoder_loss_history.png"), dpi=300)
    plt.close()
    print(f"✓ Loss history plot saved")
    
    # Save loss data for analysis
    loss_df = pd.DataFrame({
        'Epoch': range(1, num_epochs+1), 
        'Train_Loss': loss_history,
        'Val_Loss': val_loss_history
    })
    loss_df.to_csv(os.path.join(MODELS_DIR, "autoencoder_loss_history.csv"), index=False)
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        # Compute threshold on validation set (X_val) to avoid test leakage
        val_data = torch.FloatTensor(X_val.values)
        val_output = model(val_data)
        val_mse = nn.functional.mse_loss(val_output, val_data, reduction='none').mean(dim=1).numpy()
        
        # Now evaluate on test set
        test_data = torch.FloatTensor(X_test.values)
        test_output = model(test_data)
        # Reconstruction error as anomaly score
        mse = nn.functional.mse_loss(test_output, test_data, reduction='none').mean(dim=1).numpy()
    
    # Threshold selection on X_val
    threshold = np.percentile(val_mse, 80) # Adjust based on expected contamination
    print(f"AE Threshold chosen from validation set: {threshold:.6f}")
    y_pred = (mse > threshold).astype(int)
    
    return y_pred, mse, loss_history

def train_autoencoder_bad(X_train, X_val, X_test, y_train, y_val):
    print("\nTraining Second Autoencoder (trained on Anomalous data)...")
    # For this AE, we train ONLY on Anomalous (bad) data
    X_train_bad = X_train[y_train == 1].values
    X_val_bad = X_val[y_val == 1].values
    
    input_dim = X_train_bad.shape[1]
    model = Autoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_data = torch.FloatTensor(X_train_bad)
    train_loader = DataLoader(TensorDataset(train_data), batch_size=64, shuffle=True)
    
    val_data_tensor = torch.FloatTensor(X_val_bad)
    val_loader = DataLoader(TensorDataset(val_data_tensor), batch_size=256, shuffle=False)
    
    max_epochs = 100
    patience = 5
    min_delta = 1e-6
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    model.train()
    loss_history = []
    val_loss_history = []
    
    import copy
    
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            output = model(data[0])
            loss = criterion(output, data[0])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * data[0].size(0)
        avg_loss = epoch_loss / len(X_train_bad)
        loss_history.append(avg_loss)
        
        # Validation loss for early stopping
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                output = model(data[0])
                loss = criterion(output, data[0])
                epoch_val_loss += loss.item() * data[0].size(0)
        avg_val_loss = epoch_val_loss / len(X_val_bad)
        val_loss_history.append(avg_val_loss)
        
        if (epoch+1) % 2 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{max_epochs}], Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
        # Check early stopping
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best Val Loss: {best_val_loss:.6f}")
            break
            
    # Restore best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    num_epochs = len(loss_history)
    
    # Plot loss history
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), loss_history, marker='s', linewidth=2, markersize=5, color='#A23B72', label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_loss_history, marker='x', linewidth=2, markersize=5, color='#F18F01', label='Val Loss')
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
    plt.title('Second Autoencoder (Bad Traffic) Training & Validation Loss over Epochs', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "autoencoder_bad_loss_history.png"), dpi=300)
    plt.close()
    print(f"✓ Loss history plot saved")
    
    # Save loss data for analysis
    loss_df = pd.DataFrame({
        'Epoch': range(1, num_epochs+1), 
        'Train_Loss': loss_history,
        'Val_Loss': val_loss_history
    })
    loss_df.to_csv(os.path.join(MODELS_DIR, "autoencoder_bad_loss_history.csv"), index=False)
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        # Compute threshold on validation set (X_val) to avoid test leakage
        val_data = torch.FloatTensor(X_val.values)
        val_output = model(val_data)
        val_mse = nn.functional.mse_loss(val_output, val_data, reduction='none').mean(dim=1).numpy()
        
        # Now evaluate on test set
        test_data = torch.FloatTensor(X_test.values)
        test_output = model(test_data)
        # Reconstruction error as anomaly score
        mse = nn.functional.mse_loss(test_output, test_data, reduction='none').mean(dim=1).numpy()
    
    # Threshold selection on X_val: Since it's trained on Anomalies, LOW error means Anomaly.
    # If we assume ~20% of data is anomalous, we take the 20% with LOWEST error on X_val.
    threshold = np.percentile(val_mse, 20)
    print(f"Secondary AE Threshold chosen from validation set: {threshold:.6f}")
    y_pred = (mse < threshold).astype(int)
    
    # For AUC calculation, y_scores should be higher for anomalies.
    # Since low MSE = anomaly here, we can use 1/MSE or -MSE.
    # We use -mse so that lower mse (anomaly) results in a higher score.
    return y_pred, -mse, loss_history

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
        y_pred_ocsvm, y_scores_ocsvm = train_ocsvm(X_train, X_test, y_train)
        results.append(evaluate_model("One-Class SVM", y_test, y_pred_ocsvm, y_scores_ocsvm))
    
    if args.model in ["lof", "all"]:
        # 3. LOF
        y_pred_lof, y_scores_lof = train_lof(X_train, X_test, y_train)
        results.append(evaluate_model("Local Outlier Factor", y_test, y_pred_lof, y_scores_lof))
    
    # Autoencoders section (ae and ae2 can be dependent)
    y_pred_ae, y_scores_ae = None, None
    y_pred_ae2, y_scores_ae2 = None, None
    
    if args.model in ["ae", "ae2", "all"]:
        # 4. Primary Autoencoder
        y_pred_ae, y_scores_ae = train_autoencoder(X_train, X_val, X_test, y_train, y_val)
        results.append(evaluate_model("Autoencoder", y_test, y_pred_ae, y_scores_ae))
    
    if args.model in ["ae2", "all"]:
        # 5. Second Autoencoder (Bad Traffic)
        y_pred_ae2, y_scores_ae2 = train_autoencoder_bad(X_train, X_val, X_test, y_train, y_val)
        results.append(evaluate_model("Second Autoencoder", y_test, y_pred_ae2, y_scores_ae2))
        
        # 6. Ensemble (Primary + Second)
        print("\nCreating Ensemble Autoencoder (In conjunction)...")
        # Logical OR for predictions to reduce False Negatives
        y_pred_ensemble = (y_pred_ae | y_pred_ae2).astype(int)
        
        # Combined score for AUC: Sum of min-max normalized scores to prevent cancellation and scale dominance
        y_scores_ae_min = y_scores_ae.min()
        y_scores_ae_max = y_scores_ae.max()
        y_scores_ae_norm = (y_scores_ae - y_scores_ae_min) / (y_scores_ae_max - y_scores_ae_min + 1e-8)
        
        y_scores_ae2_min = y_scores_ae2.min()
        y_scores_ae2_max = y_scores_ae2.max()
        y_scores_ae2_norm = (y_scores_ae2 - y_scores_ae2_min) / (y_scores_ae2_max - y_scores_ae2_min + 1e-8)
        
        y_scores_ensemble = y_scores_ae_norm + y_scores_ae2_norm
        
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
