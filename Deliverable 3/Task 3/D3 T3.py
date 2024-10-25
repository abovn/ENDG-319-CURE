# Fadi Haider
# Kaleb Chhoa
# D3 T2.py

import matplotlib.pyplot as plt
import seaborn as sns

# The data set chosen is Fisher's Iris data set.
# The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant.

# Load the 'iris' dataset using seaborn and store it in the variable `df`
df = sns.load_dataset('iris')

# Create a new column 'iris' in the dataframe `df` by mapping the values of the 'species' column to corresponding flower names
df['iris'] = df['species'].map({0: 'Iris-Setosa', 1: 'Iris-versicolor', 2: 'Iris-virgina'})

# Create a figure with 2 rows and 2 columns of subplots with a specific size
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.histplot(data=df, x='sepal_length', hue='species', stat='frequency', multiple='layer', ax=axes[0, 0])
axes[0, 0].set_title('Histogram of Sepal Length')

# Scatter plot of 'sepal_length' vs 'sepal_width' with different colors based on 'species' on the first subplot
sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='species', ax=axes[0,1])
axes[0,1].set_title('Sepal Length vs Sepal Width \n(a)')

# Scatter plot of 'sepal_length' vs 'petal_length' with different colors based on 'species' on the second subplot
sns.scatterplot(data=df, x='sepal_length', y='petal_length', hue='species', ax=axes[1,0])
axes[1,0].set_title('Sepal Length vs Petal Length \n(b)')

# Scatter plot of 'sepal_length' vs 'petal_width' with different colors based on 'species' on the third subplot
sns.scatterplot(data=df, x='sepal_length', y='petal_width', hue='species', ax=axes[1,1])
axes[1,1].set_title('Sepal Length vs Petal Width \n(c)')

# Adjust the subplots to fit into the figure properly
plt.tight_layout()

# Display the plots
plt.show()