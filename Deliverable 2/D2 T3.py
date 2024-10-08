# Fadi Haider
# Kaleb Chhoa
# D2 T3.py

import pandas as pd
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the breast cancer dataset
data = load_breast_cancer()

# Create a dataframe with the features and target (typeofcancer)
df = pd.DataFrame(data.data, columns=data.feature_names)
df['typeofcancer'] = data.target

# Map target values to meaningful labels
df['typeofcancer'] = df['typeofcancer'].map({0: 'Malignant', 1: 'Benign'})

# Task 3: Generate Figure 2 with scatter plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Scatter plot of 'mean radius' vs 'mean concavity'
sns.scatterplot(data=df, x='mean radius', y='mean concavity', hue='typeofcancer', ax=axes[0])
axes[0].set_title('Mean Radius vs Mean Concavity')

# Scatter plot of 'mean radius' vs 'mean concave points'
sns.scatterplot(data=df, x='mean radius', y='mean concave points', hue='typeofcancer', ax=axes[1])
axes[1].set_title('Mean Radius vs Mean Concave Points')

# Scatter plot of 'mean radius' vs 'mean symmetry'
sns.scatterplot(data=df, x='mean radius', y='mean symmetry', hue='typeofcancer', ax=axes[2])
axes[2].set_title('Mean Radius vs Mean Symmetry')

plt.tight_layout()
plt.show()
