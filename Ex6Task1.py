import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import Neural_Networks as NN  # Import custom neural network module



# Load the MNIST dataset
mnist = fetch_openml('mnist_784', parser='auto')
X, y = mnist['data'], mnist['target'].astype(np.int32)

# Normalize the input data
X = X / 255.0

# Encode the labels
y = np.eye(10)[y]

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 6000,test_size = 1000, random_state=42)

print(f'Training set size: {X_train.shape[0]} samples')
print(f'Test set size: {X_test.shape[0]} samples')
