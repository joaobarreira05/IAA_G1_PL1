import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_DIR = "processed_data"
IMAGES_DIR = "deliverable3/images"
os.makedirs(IMAGES_DIR, exist_ok=True)

def generate_eda():
    print("Loading scaled data for EDA plots...")
    try:
        X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
        y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # To avoid memory and visual clutter, sample 5000 instances for PCA / Correlations
    print("Sampling data for clean visualizations...")
    np.random.seed(42)
    sample_size = min(5000, len(X_train))
    indices = np.random.choice(len(X_train), sample_size, replace=False)
    
    X_sample = X_train.iloc[indices]
    y_sample = y_train[indices]

    # Plot 1: Feature Correlation Heatmap (Top 15 most variant features)
    print("Generating Feature Correlation Heatmap...")
    variances = X_sample.var()
    top_features = variances.nlargest(15).index
    
    plt.figure(figsize=(12, 10))
    corr_matrix = X_sample[top_features].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                vmax=1, vmin=-1, center=0, square=True, linewidths=.5)
    plt.title('Correlation Matrix of Top 15 High-Variance Features', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'feature_correlation.png'), dpi=300)
    plt.close()
    
    # Plot 2: Boxplots of a few specific features grouped by class
    print("Generating Feature Distributions (Boxplots)...")
    # Taking 4 random features from the top ones
    features_to_plot = top_features[:4]
    
    plt.figure(figsize=(14, 10))
    for i, feature in enumerate(features_to_plot, 1):
        plt.subplot(2, 2, i)
        df_plot = pd.DataFrame({feature: X_sample[feature], 'Class': ['Anomaly' if y == 1 else 'Benign' for y in y_sample]})
        sns.boxplot(x='Class', y=feature, data=df_plot, palette=['#2E86AB', '#F24236'])
        plt.title(f'Distribution of {feature}', fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'feature_boxplots.png'), dpi=300)
    plt.close()
    
    print(f"EDA graphics generated successfully in {IMAGES_DIR}/!")

if __name__ == "__main__":
    generate_eda()
