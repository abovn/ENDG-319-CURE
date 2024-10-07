# Fadi Haider
# 30214496
# D2 T4.py

import pandas as pd

# Task 4: Create a custom dataset for vehicle classification
data_vehicleclass = {
    'engine_size': [1.5, 2.0, 3.0, 1.8, 2.5, 4.0, 1.6, 2.3],
    'fuel_efficiency': [35, 30, 20, 32, 25, 15, 34, 28],
    'vehicle_type': ['sedan', 'sedan', 'truck', 'sedan', 'SUV', 'truck', 'sedan', 'SUV']
}

# Convert the data into a dataframe
df_vehicleclass = pd.DataFrame(data_vehicleclass)

# Display the custom dataframe
print("Custom Vehicle Classification Dataframe:")
print(df_vehicleclass)
