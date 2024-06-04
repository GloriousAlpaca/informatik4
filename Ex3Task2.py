from sklearn.datasets import load_iris
import numpy as np
import info4 as inf4

# Load dataset
data = load_iris()
X = data.data

K = 3
iterations = 100

# Initialize centroids by randomly selecting K data points from the dataset
np.random.seed(13)  # for reproducibility
centroids = X[np.random.choice(X.shape[0], K, replace=False)]
print("Initial centroids:\n", centroids)

for i in range(iterations):
    assignments = inf4.centroid_assign_data(centroids, X)
    inf4.recalculate_centroids(centroids, X)

print("Final Centroids: \n", centroids, "Final Assignments: \n", assignments)