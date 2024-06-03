import matplotlib.pyplot as plt
import numpy as np
import info4
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


# Set up the data
data_amount = 1000
modifier = [2, 6, 8, 2, 4, 6]
independents = len(modifier) - 1
noise_amount = 1
x = np.random.rand(data_amount, independents) * data_amount  # Random independent variables

# Scale data set
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)  # Scale using StandardScaler
noise = np.random.uniform(-noise_amount, noise_amount, data_amount)  # Random noise generation
y = np.dot(np.insert(x_scaled, 0, 1, axis=1), modifier) + noise  # Calculate dependent variable

# Display the equation used to generate the data
real_equation = "y = " + " + ".join(
    [f"{modifier[i]}*x{i}" if i > 0 else f"{modifier[i]}" for i in range(len(modifier))])
print("Real equation:")
print(real_equation)
print("Noise Amount: ")
print(noise_amount)

# Data preprocessing
data_points = np.column_stack((x_scaled, y))
train_set, test_set = train_test_split(data_points, test_size=0.20, random_state=None)
x_train = train_set[:, :-1]
y_train = train_set[:, -1]
x_test = test_set[:, :-1]
y_test = test_set[:, -1]

# Initialize model parameters
theta_init = np.random.rand(x_train.shape[1] + 1)
learning_rates = np.array([0.018, 0.001])
theta_mult = np.tile(theta_init, (len(learning_rates), 1))
iterations = 20000
loss = np.zeros((iterations, len(theta_mult)))  # Array to store loss values to graph later

# Perform gradient descent for each set of initial theta values and learning rates
for it in range(len(theta_mult)):
    theta = theta_mult[it, :]
    print("Durchlauf: ", it + 1, "Theta: ", theta, "Learning Rate: ", learning_rates[it])
    theta, loss[:, it] = info4.gradient_descent(x_train, y_train, theta, learning_rates[it], iterations)
    # Plot graph depending on the amount of thetas
    if len(theta) == 2:
        plt.figure(figsize=(12, 6))
        plt.scatter(x_train, y_train, color='blue', marker='o', label='Train Set')
        plt.scatter(x_test, y_test, color='red', marker='o', label='Test Set')
        plt.plot(np.array([[0], [data_amount]]), model_predict(np.array([[0], [data_amount]]), theta), color='green',
                 label='Model Prediction')
        plt.title(f"Regression Plot - Learning Rate: {learning_rates[it]} - Iterations: {iterations}")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show(block=False)
    else:
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(y_test)), y_test, color='blue', label='Real Y')
        predicted_y = model_predict(np.insert(x_test, 0, 1, axis=1), theta)  # Predict all x-values
        plt.plot(range(len(predicted_y)), predicted_y, color='red', label='Predicted Y', marker='x')
        plt.title(f"Real Y vs Predicted Y - Learning Rate: {learning_rates[it]} - Iterations: {iterations}")
        plt.xlabel('Index')
        plt.ylabel('Y-Wert')
        plt.legend()
        plt.grid(True)
        plt.show(block=False)
    model_equation = "h = " + " + ".join([f"{theta[i]}*x{i}" if i > 0 else f"{theta[i]}" for i in range(len(theta))])
    print("Model equation:")
    print(model_equation)
    print("MSE: ")
    print(mean_squared_error(x_test, y_test, theta))

# Plot loss functions for each learning rate
plt.figure(figsize=(12, 6))
for it in range(len(learning_rates)):
    plt.plot(loss[:, it], label=f"Learning Rate: {learning_rates[it]}")
plt.title("Loss Functions")
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
