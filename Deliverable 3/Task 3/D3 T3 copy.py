# Fadi Haider
# Kaleb Chhoa
# D3 T3.py

import matplotlib.pyplot as plt
import seaborn as sns

# The data set chosen is Fisher's Iris data set.
# The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant.

# Load the 'iris' dataset using seaborn and store it in the variable `df`
df = sns.load_dataset('iris')

# Create a pairplot to visualize relationships between different features, with different colors for each species
sns.pairplot(df, hue="species")

# Separate the features (X) from the target variable (y) by dropping the 'species' column
X = df.drop('species', axis=1)
y = df['species']

# Display the plots
plt.show()