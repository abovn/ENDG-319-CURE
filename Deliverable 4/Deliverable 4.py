# Fadi Haider
# Kaleb Chhoa
# Yahya Elmadhoun

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
X = iris.data  # Features of the dataset
y = iris.target  # Target labels of the dataset

# Function to preprocess data and build KNN model
def knn_model(k, scaler, X_train, X_test, y_train, y_test):
    # Scale the training data using the provided scaler
    X_train_scaled = scaler.fit_transform(X_train)
    # Scale the test data using the same scaler
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize the K-Nearest Neighbors classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit the KNN model using the scaled training data and labels
    knn.fit(X_train_scaled, y_train)
    
    # Calculate the accuracy of the model on the scaled training data
    train_accuracy = knn.score(X_train_scaled, y_train)
    # Calculate the accuracy of the model on the scaled test data
    test_accuracy = knn.score(X_test_scaled, y_test)
    
    # Return the training accuracy, test accuracy, the trained KNN model, scaled test data, and test labels
    return train_accuracy, test_accuracy, knn, X_test_scaled, y_test

# Randomly split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Task 2a: Analyze accuracy with MinMaxScaler
k_values = range(1, 50)  # Range of k values to test
train_accuracies_minmax = []  # List to store training accuracies
test_accuracies_minmax = []  # List to store test accuracies

scaler_minmax = MinMaxScaler()  # Initialize MinMaxScaler

# Iterate over each k value
for k in k_values:
    # Train the model and get accuracies
    train_acc, test_acc, _, _, _ = knn_model(k, scaler_minmax, X_train, X_test, y_train, y_test)
    # Append the accuracies to the respective lists
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
train_accuracies_standard = []  # List to store training accuracies
test_accuracies_standard = []  # List to store test accuracies

scaler_standard = StandardScaler()  # Initialize StandardScaler

# Iterate over each k value
for k in k_values:
    # Train the model and get accuracies
    train_acc, test_acc, _, _, _ = knn_model(k, scaler_standard, X_train, X_test, y_train, y_test)
    # Append the accuracies to the respective lists
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
# Find the best k value based on the highest test accuracy with MinMaxScaler
best_k = k_values[np.argmax(test_accuracies_minmax)]
# Train the final model using the best k value
train_acc, test_acc, final_model, X_test_scaled, y_test_random = knn_model(best_k, scaler_minmax, X_train, X_test, y_train, y_test)

# Predict the labels for the scaled test data
y_pred_random = final_model.predict(X_test_scaled)
# Compute the confusion matrix
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
# Generate a new random instance within the range of the dataset
new_instance_random = np.random.uniform(X.min(axis=0), X.max(axis=0), size=(1, X.shape[1]))
# Predict the class for the new instance
predicted_class_random = final_model.predict(scaler_minmax.transform(new_instance_random))
print(f"Randomly Generated New Instance: {new_instance_random}")
print(f"Predicted class for the new instance: {iris.target_names[predicted_class_random[0]]}")