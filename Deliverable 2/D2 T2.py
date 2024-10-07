# Fadi Haider
# 30214496
# D2 T2.py

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

# Task 2: Generate Figure 1 with subplots (adjusted to 2x2 grid since there are only 5 plots)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # 2 rows, 3 columns for the plots

# Histogram of 'mean radius' for each class
sns.histplot(data=df, x='mean radius', hue='typeofcancer', stat='frequency', multiple='layer', ax=axes[0, 0])
axes[0, 0].set_title('Histogram of Mean Radius')

# Scatter plot of 'mean radius' vs 'mean perimeter'
sns.scatterplot(data=df, x='mean radius', y='mean perimeter', hue='typeofcancer', ax=axes[0, 1])
axes[0, 1].set_title('Mean Radius vs Mean Perimeter')

# Scatter plot of 'mean radius' vs 'mean area'
sns.scatterplot(data=df, x='mean radius', y='mean area', hue='typeofcancer', ax=axes[0, 2])
axes[0, 2].set_title('Mean Radius vs Mean Area')

# Scatter plot of 'mean radius' vs 'mean texture'
sns.scatterplot(data=df, x='mean radius', y='mean texture', hue='typeofcancer', ax=axes[1, 0])
axes[1, 0].set_title('Mean Radius vs Mean Texture')

# Boxplot of 'mean radius' for each class
sns.boxplot(data=df, x='typeofcancer', y='mean radius', ax=axes[1, 1])
axes[1, 1].set_title('Box plot of Mean Radius by Cancer Type')

# Empty plot to fill the grid 
axes[1, 2].axis('off')  # Turn off the unused subplots

plt.tight_layout()
plt.show()