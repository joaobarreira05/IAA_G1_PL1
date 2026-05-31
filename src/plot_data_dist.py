import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

os.makedirs('evaluation_results', exist_ok=True)

def load_label_col(path):
    df = pd.read_csv(path)
    assert len(df.columns) == 1 or 'Is_Anomaly' in df.columns, \
        f"Unexpected columns in {path}: {list(df.columns)}"
    return df.iloc[:, 0]

try:
    y_train = load_label_col('processed_data/y_train.csv')
    y_val   = load_label_col('processed_data/y_val.csv')
    y_test  = load_label_col('processed_data/y_test.csv')
except Exception as e:
    print("Could not load processed data:", e)
    exit(1)

# Plot 1: Dataset Split
splits = ['Train (70%)', 'Validation (15%)', 'Test (15%)']
sizes = [len(y_train), len(y_val), len(y_test)]

plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=splits, autopct='%1.1f%%', colors=['#2E86AB', '#F24236', '#A23B72'], 
        startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
plt.title('Dataset Split Proportions', fontsize=14, fontweight='bold')
plt.savefig('evaluation_results/dataset_split.png', dpi=300)
plt.close()

# Plot 2: Class Distribution across splits
data = {
    'Split': ['Train', 'Train', 'Validation', 'Validation', 'Test', 'Test'],
    'Class': ['Benign (0)', 'Anomaly (1)'] * 3,
    'Count': [
        sum(y_train == 0), sum(y_train == 1),
        sum(y_val == 0), sum(y_val == 1),
        sum(y_test == 0), sum(y_test == 1)
    ]
}
df_dist = pd.DataFrame(data)

plt.figure(figsize=(10, 6))
sns.barplot(x='Split', y='Count', hue='Class', data=df_dist, palette=['#2E86AB', '#F24236'])
plt.title('Class Distribution Across Splits', fontsize=14, fontweight='bold')
plt.xlabel('Dataset Split', fontsize=12, fontweight='bold')
plt.ylabel('Number of Instances', fontsize=12, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
for p in plt.gca().patches:
    plt.gca().annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=10)
plt.legend(title='Traffic Type')
plt.tight_layout()
plt.savefig('evaluation_results/class_distribution.png', dpi=300)
plt.close()

print("Distribution plots generated.")
