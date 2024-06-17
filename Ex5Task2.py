import matplotlib.pyplot as plt
import numpy as np

import Neural_Networks as NN  # Import custom neural network module

# Define initial data parameters
inputs = []
outputs = []

# Updated data parameters
data_amount = 1000  # Total number of data points
min_val = -1  # Minimum value for input data
max_val = 1  # Maximum value for input data

# Generate random inputs uniformly distributed between min_val and max_val
inputs = np.random.uniform(low=min_val, high=max_val, size=(data_amount, 1))
# Compute corresponding outputs using the polynomial
outputs = 3 * inputs ** 5 + 1.5 * inputs ** 4 + 2 * inputs ** 3 + 7 * inputs + 0.5

# Activation function and its derivative
hid_func = [NN.relu, NN.relu_derivative]
out_func = [NN.linear, NN.linear_derivative]
# Initialize neural network parameters
neural_net = NN.create_NN(1, 2, 8, 1, hid_func, out_func, 14, -0.1, 0.1)
lr = 0.5  # Learning rate
iter = 100  # Number of training iterations

# Train the neural network
trained_net, loss = NN.learn(inputs, outputs, neural_net, lr, iter,
          NN.mse_loss, NN.mse_loss_derivative)

# Visualization functions from the neural network module
NN.visualize_thetas(trained_net.thetas)
NN.visualize_loss(loss)

predictions = []
# Predict outputs for the inputs using the trained neural network
for input_data in inputs:
    nn_values = NN.forward_prop(input_data, trained_net)
    predicted = nn_values[1][-1]
    predictions.append(predicted)

predictions = np.array(predictions).reshape(-1, 1)

# Plotting
plt.figure()
# Plot original data points
plt.scatter(inputs, outputs, label='Original Data', color='blue')
# Plot predicted data points
plt.scatter(inputs, predictions, label='Predicted Data', color='red')
# Labeling
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Original vs Predicted Data')
plt.legend()
plt.show()
