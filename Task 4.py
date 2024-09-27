# Task 4 - Kaleb Chhoa
from sklearn.datasets import load_digits, load_breast_cancer
digit_dataset = load_digits()
breast_dataset = load_breast_cancer()

print(digit_dataset.keys())
# print(digit_dataset.DESCR)

digit_dataset_lines = digit_dataset.DESCR.splitlines()
selected_lines = digit_dataset_lines[7:13]

for line in selected_lines:
    print(line)

selected_lines = digit_dataset_lines[17:20]

for line in selected_lines:
     print(line)
    