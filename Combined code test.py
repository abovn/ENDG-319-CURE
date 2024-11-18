# Import required libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Function to preprocess data and build KNN model
def knn_model(k, scaler, X_train, X_test, y_train, y_test):
    # Scale the data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build the model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    
    # Calculate accuracy
    train_accuracy = knn.score(X_train_scaled, y_train)
    test_accuracy = knn.score(X_test_scaled, y_test)
    return train_accuracy, test_accuracy, knn

# Task 2: Analyze accuracy with different k values
k_values = range(1, 40)
train_accuracies_minmax = []
test_accuracies_minmax = []

scaler_minmax = MinMaxScaler()

for k in k_values:
    train_acc, test_acc, _ = knn_model(k, scaler_minmax, X_train, X_test, y_train, y_test)
    train_accuracies_minmax.append(train_acc)
    test_accuracies_minmax.append(test_acc)

# Plot accuracy vs k for MinMaxScaler
plt.figure(figsize=(10, 5))
plt.plot(k_values, train_accuracies_minmax, label="Training Accuracy (MinMaxScaler)")
plt.plot(k_values, test_accuracies_minmax, label="Testing Accuracy (MinMaxScaler)")
plt.xlabel("k (Number of Neighbors)")
plt.ylabel("Accuracy")
plt.title("Model Accuracy vs k (MinMaxScaler)")
plt.legend()
plt.grid()
plt.show()

# Repeat for StandardScaler
train_accuracies_standard = []
test_accuracies_standard = []
scaler_standard = StandardScaler()

for k in k_values:
    train_acc, test_acc, _ = knn_model(k, scaler_standard, X_train, X_test, y_train, y_test)
    train_accuracies_standard.append(train_acc)
    test_accuracies_standard.append(test_acc)

# Plot accuracy vs k for StandardScaler
plt.figure(figsize=(10, 5))
plt.plot(k_values, train_accuracies_standard, label="Training Accuracy (StandardScaler)")
plt.plot(k_values, test_accuracies_standard, label="Testing Accuracy (StandardScaler)")
plt.xlabel("k (Number of Neighbors)")
plt.ylabel("Accuracy")
plt.title("Model Accuracy vs k (StandardScaler)")
plt.legend()
plt.grid()
plt.show()

# Task 3: Identify the best k and scaler
best_k_minmax = k_values[np.argmax(test_accuracies_minmax)]
best_k_standard = k_values[np.argmax(test_accuracies_standard)]

print(f"Best k for MinMaxScaler: {best_k_minmax}")
print(f"Best k for StandardScaler: {best_k_standard}")

best_scaler = scaler_minmax if max(test_accuracies_minmax) > max(test_accuracies_standard) else scaler_standard
best_k = best_k_minmax if best_scaler == scaler_minmax else best_k_standard
print(f"Best Scaler: {'MinMaxScaler' if best_scaler == scaler_minmax else 'StandardScaler'}")
print(f"Best k: {best_k}")

# Final Model and Confusion Matrix
_, _, final_model = knn_model(best_k, best_scaler, X_train, X_test, y_train, y_test)
y_pred = final_model.predict(best_scaler.transform(X_test))
conf_matrix = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix ')
plt.show()

# Task 4: Predict for a new instance
new_instance = [[5.0, 3.5, 1.5, 0.2]]  # Example values
predicted_class = final_model.predict(best_scaler.transform(new_instance))
print(f"Predicted class for the new instance: {iris.target_names[predicted_class[0]]}")
