from sklearn.datasets import load_iris
import numpy as np

# Load dataset
data = load_iris()
X = data.data

K = 3

# Initialize centroids by randomly selecting K data points from the dataset
np.random.seed(123)  # for reproducibility
initial_centroids = X[np.random.choice(X.shape[0], K, replace=False)]
print("Initial centroids:\n", initial_centroids)