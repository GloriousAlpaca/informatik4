import numpy as np
import pandas as pd
import info4
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('Real_estate.csv')
data_points = data.values

# Data preprocessing
train_set, test_set = info4.tt_split(data_points, 0.20)
x_train = train_set[:, :-1]
y_train = train_set[:, -1]
x_test = test_set[:, :-1]
y_test = test_set[:, -1]

# Initialize model parameters
theta_init = np.random.rand(x_train.shape[1] + 1)
learning_rate = 0.01
iterations = 100
regularization = np.array([20, 10, 5, 0.5, 0])
theta_mult = np.tile(theta_init, (len(regularization), 1))
loss = np.zeros((iterations, len(theta_mult)))  # Array to store loss values to graph later

# Perform gradient descent for each set of initial theta values and regularization
for it in range(len(theta_mult)):
    theta = theta_mult[it, :]
    print("Durchlauf: ", it + 1, "Regularization: ", regularization[it])
    theta, loss[:, it] = info4.gradient_descent(x_train, y_train, theta, learning_rate, iterations, regularization[it])

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y_test)), y_test, color='blue', label='Real Y')
    predicted_y = info4.model_predict(np.insert(x_test, 0, 1, axis=1), theta)  # Predict all x-values
    plt.plot(range(len(predicted_y)), predicted_y, color='red', label='Predicted Y', marker='x')
    plt.title(f"Real Y vs Predicted Y - Regularization: {regularization[it]} - Iterations: {iterations}")
    plt.xlabel('Index')
    plt.ylabel('Y-Wert')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)
    model_equation = "h = " + " + ".join([f"{theta[i]}*x{i}" if i > 0 else f"{theta[i]}" for i in range(len(theta))])
    print("Model equation:")
    print(model_equation)
    print("MSE: ")
    print(info4.mean_squared_error(x_test, y_test, theta))

# Plot loss functions for each learning rate
plt.figure(figsize=(12, 6))
for it in range(len(regularization)):
    plt.plot(loss[:, it], label=f"Regularization: {regularization[it]}")
plt.title("Loss Functions")
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()