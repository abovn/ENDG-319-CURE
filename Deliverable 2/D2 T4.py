# Fadi Haider
# Kaleb Chhoa
# D2 T4.py

import pandas as pd

# Task 4: Create a custom dataset for vehicle classification
data_vehicleclass = {
    'engine_size': [3.5, 2.4, 2.4, 2, 3.3, 3.5, 3.0],
    'fuel_efficiency': [10.5, 8.7, 9.8, 7.5, 11.9, 8.1, 12.4],
    'vehicle_type': ['SUV','Sedan','Hatchback','Sedan','Minivan','Sedan','SUV']
}

# Convert the data into a dataframe
df_vehicleclass = pd.DataFrame(data_vehicleclass)

# Display the custom dataframe
print("Custom Vehicle Classification Dataframe:")
print(df_vehicleclass)
