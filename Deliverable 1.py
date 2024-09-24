# Delivarable 1.py 
# ENDG 319 CURE Project Delivarable 1
# Group 158
# September 22, 2024
# Group Members:
# Fadi Haider - 30214496
# Kaleb Chhoa 
# Yahya Elmadhoun
# Youssef Abdalla

import pandas as pd


column_names = ['Gasoline(MPG)', 'Diesel(MPG)']
data = np.genfromtxt(r'C:\Users\haide\OneDrive\Desktop\University\Fall 2024\ENDG-319-CURE\ENDG 319 Deliverable 1 Data - Task 1.csv', delimiter = ',', skip_header = True, dtype = str)