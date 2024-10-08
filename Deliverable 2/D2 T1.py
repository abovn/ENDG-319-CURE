# Fadi Haider
# Kaleb Chhoa
# D2 T1.py

import pandas as pd
from sklearn.datasets import load_breast_cancer

# Load the breast cancer dataset
data = load_breast_cancer()

# Task 1(a): Create a dataframe with 569 instances and 30 features
a = pd.DataFrame(data.data, columns=data.feature_names)
a['typeofcancer'] = data.target

# T1 part a
print(a.shape)

# Task 1(b): Slice the dataframe to only include specific features
df = a[['mean radius', 'mean perimeter', 'mean area', 'typeofcancer']]

# Show the first two rows of the sliced dataframe
print("\nFirst two rows of 'df':")
print(df.head(2))

# Show the rows with indexes 17, 18, 19, 20, 21
print("\nRows 17 to 21:")
print(df.iloc[17:22])
