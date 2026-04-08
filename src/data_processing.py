import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import glob

def clean_column_names(df):
    """Remove leading/trailing whitespaces from column names."""
    df.columns = df.columns.str.strip()
    return df

def load_and_sample_data(data_dir, sample_frac=0.2, random_state=42):
    """
    Reads all CSVs in the given directory, combines them,
    and takes a stratified sample to reduce data size while maintaining class distribution.
    """
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    print(f"Found {len(all_files)} CSV files in {data_dir}.")
    
    df_list = []
    for f in all_files:
        print(f"Loading {os.path.basename(f)}...")
        # Use low_memory=False to prevent mixed types warnings for some columns
        df_chunk = pd.read_csv(f, low_memory=False)
        df_chunk = clean_column_names(df_chunk)
        df_list.append(df_chunk)
        
    print("Combining all files...")
    full_df = pd.concat(df_list, axis=0, ignore_index=True)
    
    print(f"Original dataset shape: {full_df.shape}")
    
    # 1. Limpeza de Missing/Infinity Values
    print("Cleaning missing (NaN) and Infinity values...")
    full_df = full_df.replace([np.inf, -np.inf], np.nan)
    initial_rows = full_df.shape[0]
    full_df = full_df.dropna()
    dropped_rows = initial_rows - full_df.shape[0]
    print(f"Dropped {dropped_rows} rows containing NaN/Inf values.")

    # 2. Amostragem Estratificada
    # Para garantir que temos amostras suficientes mas que não rebentamos com a memória
    # o sample_frac permite reduzir o dataset original (1GB) para algo mais maneável.
    print(f"Sampling {sample_frac*100}% of the data stratified by 'Label'...")
    try:
        _, sample_df = train_test_split(full_df, test_size=sample_frac, 
                                        stratify=full_df['Label'], random_state=random_state)
    except ValueError as e:
        print(f"Stratification failed (probably some rare classes). Falling back to random sample: {e}")
        sample_df = full_df.sample(frac=sample_frac, random_state=random_state)
        
    print(f"Sampled dataset shape: {sample_df.shape}")
    return sample_df

def feature_engineering_and_split(df, output_dir="processed_data", random_state=42):
    """
    Applies feature selection, scaling, and Train/Val/Test splitting.
    Saves outputs securely for offline modeling.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. Label Encoding (Benign = 0, Attack = 1)
    # Como é deteção de anomalias, "BENIGN" é a nossa classe normal.
    print("Encoding labels...")
    df['Is_Anomaly'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    
    # Remover o label original e strings
    y = df['Is_Anomaly']
    X = df.drop(columns=['Label', 'Is_Anomaly'])
    
    # 4. Feature Selection básica: remover colunas com variância zero
    print("Removing zero-variance features...")
    variances = X.var(numeric_only=True)
    constant_features = variances[variances == 0].index
    if len(constant_features) > 0:
        print(f"Dropped {len(constant_features)} constant features: {list(constant_features)}")
        X = X.drop(columns=constant_features)
    
    # Check for any remaining non-numeric columns and drop them (or encode)
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        print(f"Dropping non-numeric columns: {list(non_numeric_cols)}")
        X = X.drop(columns=non_numeric_cols)
        
    # 5. Split Dataset (70% Train, 15% Val, 15% Test)
    # Para Anomaly Detection, muitas vezes queremos avaliar o modelo num test set 
    # equilibrado ou garantir que no train temos essencialmente benignos. 
    # Aqui fazemos um split padrão estratificado, cabendo aos Modelos (Margarida) 
    # decidir se filtram anomalias no X_train.
    print("Splitting data (70% Train, 15% Val, 15% Test)...")
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=random_state)
    
    # O train_test_split no X_temp com test_size relativo (15/85 = ~0.1764)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=(0.15/0.85), stratify=y_temp, random_state=random_state)
    
    # 6. Feature Scaling (StandardScaler)
    # Fundamental para modelos baseados em distâncias, fitted APENAS no treino
    print("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    # 7. Gravar em disco
    print("Saving processed datasets to disk...")
    # Guardar X
    X_train_scaled.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_val_scaled.to_csv(os.path.join(output_dir, "X_val.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    
    # Guardar y
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_val.to_csv(os.path.join(output_dir, "y_val.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    
    print("🎉 Data processing complete. Pipeline finished successfully!")
    print("-" * 50)
    print(f"X_train size: {X_train_scaled.shape}")
    print(f"X_val size:   {X_val_scaled.shape}")
    print(f"X_test size:  {X_test_scaled.shape}")
    print(f"Anomaly % in Train: {(y_train.mean() * 100):.2f}%")
    print("-" * 50)

if __name__ == "__main__":
    archive_path = "archive"
    processed_path = "processed_data"
    
    print("🚀 Starting Data Processing Pipeline...")
    # 20% do CIC-IDS2017 é aprox 500 mil instâncias, perfeito para correr em portáteis offline
    df_sample = load_and_sample_data(archive_path, sample_frac=0.20)
    feature_engineering_and_split(df_sample, output_dir=processed_path)
