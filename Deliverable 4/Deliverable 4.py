import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

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
    return train_accuracy, test_accuracy, knn, X_test_scaled, y_test

# Randomly split the data each time
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Task 2a: Analyze accuracy with MinMaxScaler
k_values = range(1, 50)
train_accuracies_minmax = []
test_accuracies_minmax = []

scaler_minmax = MinMaxScaler()

for k in k_values:
    train_acc, test_acc, _, _, _ = knn_model(k, scaler_minmax, X_train, X_test, y_train, y_test)
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

# Task 2b: Analyze accuracy with StandardScaler
train_accuracies_standard = []
test_accuracies_standard = []

scaler_standard = StandardScaler()

for k in k_values:
    train_acc, test_acc, _, _, _ = knn_model(k, scaler_standard, X_train, X_test, y_train, y_test)
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

# Final Model and Confusion Matrix with Randomized Test Set
best_k = k_values[np.argmax(test_accuracies_minmax)]
train_acc, test_acc, final_model, X_test_scaled, y_test_random = knn_model(best_k, scaler_minmax, X_train, X_test, y_train, y_test)

y_pred_random = final_model.predict(X_test_scaled)
conf_matrix_random = confusion_matrix(y_test_random, y_pred_random)

# Display the Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_random, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Print Final Model Parameters, Scaler, and Accuracy
print(f"Final model parameters and scaler and accuracy:")
print(f"k = {best_k}")
print(f"Scaler used for attribute preprocessing: {scaler_minmax}")
print(f"Model accuracy in the training set = {train_acc:.2f}")
print(f"Model accuracy in the test set = {test_acc:.2f}")

# Task 4: Predict for a randomly generated new instance
new_instance_random = np.random.uniform(X.min(axis=0), X.max(axis=0), size=(1, X.shape[1]))
predicted_class_random = final_model.predict(scaler_minmax.transform(new_instance_random))
print(f"Randomly Generated New Instance: {new_instance_random}")
print(f"Predicted class for the new instance: {iris.target_names[predicted_class_random[0]]}")
