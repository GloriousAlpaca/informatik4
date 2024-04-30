import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
def gradient_descent(xf, yf, thetaf, learning_rate, iterations):
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
            gradients[j] = np.sum(errors * X[:, j]) * 2 / m  # Gradient calculation
        thetaf -= learning_rate * gradients  # Update parameters
        loss[it] = mean_squared_error(X, yf, thetaf)  # Record the loss
    return thetaf, loss