# Fadi Haider
# Kaleb Chhoa
# D2 T4.py

import pandas as pd

# Task 4: Create a custom dataset for vehicle classification
data_vehicleclass = {
    'engine_size': [2.4,1.6,2.4,4.8,2,2.4,3.5,2.7,3.8,3.6,4,4.8,2.7,6.5,3.5],
    'fuel_efficiency': [9.9,9.4,10,14,11.8,13.5,12.7,10.5,14.3,16.4,15.3,13.3,11.9,19,10.5],
    'vehicle_type': ['Sedan','Sedan','Sedan','Sedan','Sedan','Minivan','Minivan','Minivan','Minivan','Minivan','SUV','SUV','SUV','SUV','SUV',]
}

# Convert the data into a dataframe
df_vehicleclass = pd.DataFrame(data_vehicleclass)

# Display the custom dataframe
print("Custom Vehicle Classification Dataframe:")
print(df_vehicleclass)
