# Task 3 - Fadi Haider 
# 30214496
# 

import numpy as np
import pandas as pd

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

