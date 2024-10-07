import pandas as pd
import matplotlib.pyplot as plt
table_data = {
    'Gasoline (MPG)': [22, 25, 28, 30, 27, 26, 24, 29, 23, 31, 22, 25, 28, 32, 26],
    'Diesel (MPG)': [35, 38, 40, 37, 36, 39, 34, 38, 33, 37, 35, 39, 40, 36, 33]
}
pd.options.display.float_format = '{:.2f}'.format
efficiency = pd.DataFrame(table_data)
summary = efficiency.describe()
summary.loc['median'] = efficiency.median()
print(summary)
 
efficiency.boxplot(column=['Gasoline (MPG)', 'Diesel (MPG)'])
plt.title('Comparative Boxplot of Gasoline and Diesel MPG')
plt.ylabel('Miles Per Gallon (MPG)')
plt.show()