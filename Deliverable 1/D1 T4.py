# Task 4 - Kaleb Chhoa
from sklearn.datasets import load_digits, load_breast_cancer
digit_dataset = load_digits()
breast_dataset = load_breast_cancer()

# print(digit_dataset.keys())
# print(digit_dataset.DESCR)

digit_dataset_lines = digit_dataset.DESCR.splitlines()
selected_lines = digit_dataset_lines[7:13]

print("**digits Dataset Info**:")

for line in selected_lines:
    print(line)

selected_lines = digit_dataset_lines[17:20]

for line in selected_lines:
     print(line)
    
# print(breast_dataset.keys())
# print(breast_dataset.DESCR)

# number of instances, numbewr of attributes, attribute information
breast_dataset_lines = breast_dataset.DESCR.splitlines()
selected_lines = breast_dataset_lines[6:22]

print("**breast_cancer Dataset Info**:")

for line in selected_lines:
     print(line)

# missing attribute values 
selected_lines = breast_dataset_lines[68:71]

for line in selected_lines:
      print(line)


# Creator
selected_lines = breast_dataset_lines[73:75]

for line in selected_lines:
      print(line)

# Date
selected_lines = breast_dataset_lines[77:78]

for line in selected_lines:
      print(line)

# class
selected_lines = breast_dataset_lines[28:32]

for line in selected_lines:
      print(line)
