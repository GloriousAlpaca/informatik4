import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

import info4 as inf4

# Load dataset
data = load_iris()
X = data.data

# Parameters
K = 3
max_iterations = 1000
change_threshold = 0.01
num_runs = 100

# Initialize variables for the best run
best_sse = float('inf')
best_centroids = None
best_assignments = None
best_sse_history = []

for run in range(num_runs):
    # Initialize centroids randomly
    centroids = X[np.random.choice(X.shape[0], K, replace=False)]
    assignments = [[] for _ in range(K)]
    print(f"Run {run + 1} - Initial centroids:\n", centroids)

    # Reset change and iteration counter for each run
    change = float('inf')
    iterations = 0
    sse_history = []
    sse = float('inf')

    while change > change_threshold and iterations < max_iterations:
        # Assign data points to centroids
        assignments = inf4.centroid_assign_data(centroids, X)
        # Recalculate centroids
        new_centroids = inf4.recalculate_centroids(centroids, assignments)
        # Calculate change in centroids
        change = np.mean(np.linalg.norm(new_centroids - centroids, axis=1))
        centroids = new_centroids
        iterations += 1
        # Calculate SSE for current iteration
        sse = inf4.sum_squared_error(centroids, assignments)
        sse_history.append(sse)
        print(f"Run {run + 1}, Iteration {iterations}, Centroid Change: {change}, SSE: {sse}")

    # Update best run if current SSE is lower
    if sse < best_sse:
        best_centroids = centroids
        best_assignments = assignments
        best_sse_history = sse_history
        best_sse = sse
    print(f"Run {run + 1} - Final Centroids:\n{centroids}\nFinal SSE: {sse}")

# Print best centroids and SSE
print("Best Centroids:\n", best_centroids)
print("Best SSE:", best_sse)

# Plot SSE convergence
plt.figure()
plt.plot(range(len(best_sse_history)), best_sse_history, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Convergence Plot of SSE over Iterations (Best Run)')
plt.grid(True)
plt.show()

# Plot 3D clustering results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['r', 'g', 'b']

for idx, cluster in enumerate(best_assignments):
    cluster = np.array(cluster)
    ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], s=30, color=colors[idx], label=f'Cluster {idx + 1}')

ax.scatter(best_centroids[:, 0], best_centroids[:, 1], best_centroids[:, 2], s=100, color='y', marker='X',
           edgecolor='k', linewidth=2, label='Centroids')

ax.set_xlabel('Sepal length')
ax.set_ylabel('Sepal width')
ax.set_zlabel('Petal length')
ax.legend()
ax.set_title('K-Means Clustering of Iris Dataset in 3D (Best Run)')

plt.show()
