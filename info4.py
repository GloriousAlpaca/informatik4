import numpy as np
import copy


# Function to predict y value for an array of x values and a given model using theta values
def model_predict(xf, thetaf):
    X = xf
    # Add a column of ones to the beginning of xf for theta_0
    if xf.shape[1] != len(thetaf):
        if xf.ndim == 1:
            xf = xf.reshape(-1, 1)
        X = np.insert(xf, 0, 1, axis=1)
    # Calculate and return the prediction
    predict = np.dot(X, thetaf)
    return predict


# Function to calculate the mean squared error between predicted and actual values
def mean_squared_error(xf, yf, thetaf):
    m = len(xf)  # Total number of data points
    s = np.sum(np.square(model_predict(xf, thetaf) - yf))  # Sum of squared differences
    return 1 / m * s  # Return mean squared error


# Function to perform gradient descent
def gradient_descent(xf, yf, thetaf, learning_rate, iterations, reg):
    loss = np.zeros(iterations)  # Array to keep track of loss at each iteration
    m = len(yf)  # Number of data points
    X = xf
    # Add a column of ones to the beginning of xf for theta_0
    if xf.shape[1] != len(thetaf):
        if xf.ndim == 1:
            xf = xf.reshape(-1, 1)
        X = np.insert(xf, 0, 1, axis=1)
    for it in range(iterations):
        prediction = model_predict(X, thetaf)
        errors = prediction - yf
        gradients = np.zeros_like(thetaf)
        for j in range(len(thetaf)):
            gradients[j] = np.sum(errors * X[:, j]) / m  # Gradient calculation
        thetaf = thetaf * (1 - learning_rate * reg / m) - learning_rate * gradients  # Update parameters
        loss[it] = mean_squared_error(X, yf, thetaf)  # Record the loss
    return thetaf, loss


def tt_split(original_data_array, test_ratio):
    data_array = copy.deepcopy(original_data_array)
    np.random.shuffle(data_array)
    split = int(len(data_array) * (1 - test_ratio))
    train_set = data_array[:split]
    test_set = data_array[split:]
    return train_set, test_set


def centroid_assign_data(centroids, data):
    k = len(centroids)
    assignments = [[] for _ in range(k)]

    for i in range(len(data)):
        sign = 0
        dist = np.sqrt(np.sum((centroids[0] - data[i]) ** 2))
        for j in range(1, k):
            cdist = np.sqrt(np.sum((centroids[j] - data[i]) ** 2))
            if dist > cdist:
                dist = cdist
                sign = j
        assignments[sign].append(data[i])
    return assignments


def recalculate_centroids(centroids, data):
    for i in range(len(centroids)):
        centroids[i] = np.sum(data[i]) / len(data[i])
    return centroids