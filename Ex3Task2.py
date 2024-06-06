from sklearn.datasets import load_iris
import numpy as np
import info4 as inf4
import matplotlib.pyplot as plt

# Load dataset
data = load_iris()
X = data.data

K = 3
max_iterations = 1000

# Initialize centroids by randomly selecting K data points from the dataset
np.random.seed(13)  # for reproducibility
centroids = X[np.random.choice(X.shape[0], K, replace=False)]
change_threshold = 0.01


print("Initial centroids:\n", centroids)

change = float('inf')  # Initialize change to a large number
iterations = 0

while change > change_threshold and iterations < max_iterations:
    assignments = inf4.centroid_assign_data(centroids, X)
    new_centroids = inf4.recalculate_centroids(centroids, assignments)
    change = np.mean(np.linalg.norm(new_centroids - centroids, axis=1))
    centroids = new_centroids  # Update centroids for the next iteration
    iterations += 1
    sse = inf4.sum_squared_error(centroids, assignments)
    print("Centroid Change: ", change, " SSE: ", sse)

sse = inf4.sum_squared_error(centroids, assignments)
print("Final Centroids: \n", centroids,  "\nFinal SSE: ", sse)

# Plotting the results in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['r', 'g', 'b']

for idx, cluster in enumerate(assignments):
    cluster = np.array(cluster)
    ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], s=30, color=colors[idx], label=f'Cluster {idx+1}')

ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s=100, color='y', marker='X', edgecolor='k', linewidth=2, label='Centroids')

ax.set_xlabel('Sepal length')
ax.set_ylabel('Sepal width')
ax.set_zlabel('Petal length')
ax.legend()
ax.set_title('K-Means Clustering of Iris Dataset in 3D')

plt.show()