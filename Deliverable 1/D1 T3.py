# Task 3 - Fadi Haider 
# 30214496
# 

import pandas as pd
import matplotlib.pyplot as plt

data1 = {
    'x': [10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0],
    'y': [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
}
dataset1 = pd.DataFrame(data1)
print("Summary statistics for dataset1:")
print(dataset1.describe().round(2))

data2 = {
    'x': [10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0],
    'y': [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]
}
dataset2 = pd.DataFrame(data2)
print("\nSummary statistics for dataset2:")
print(dataset2.describe().round(2))

# Create scatterplots for both datasets on the same axes
plt.figure(figsize=(8, 6))

# Plot Dataset 1
plt.scatter(data1['x'], data1['y'], color='blue', label='Dataset 1')

# Plot Dataset 2
plt.scatter(data2['x'], data2['y'], color='red', label='Dataset 2')

# Add labels and title
plt.title('Scatterplot of Dataset 1 and Dataset 2')
plt.xlabel('X values')
plt.ylabel('Y values')

# Add legend to distinguish between the two datasets
plt.legend()

# Display the plot
plt.show()