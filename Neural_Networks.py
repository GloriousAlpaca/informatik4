import matplotlib.pyplot as plt
import numpy as np


# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # Logistic function


# Derivative of the sigmoid function for backpropagation
def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)  # Derivative of sigmoid


# Rectified Linear Unit (ReLU) activation function
def relu(x):
    return np.maximum(0, x)  # Returns x when x > 0, otherwise 0


# Derivative of the ReLU function
def relu_derivative(x):
    return np.where(x > 0, 1, 0)  # 1 where x is positive, otherwise 0


# Mean squared error loss function
def mse_loss(expected, output):
    return 0.5 * np.mean((expected - output) ** 2)  # Mean of squared differences


# Derivative of the mean squared error loss function
def mse_loss_derivative(expected, output):
    return output - expected  # Gradient of MSE loss


# Quadratic loss function
def quadratic_loss(expected, output):
    return 0.5 * np.sum((expected - output) ** 2)  # Sum of squared differences

# Function to create a neural network with specified topology
def create_NN(inputs, hlayers, hnodes, outputs, seed=None, weight_min=-0.1, weight_max=0.1):
    if seed is not None:
        np.random.seed(seed)  # Set random seed for reproducibility

    thetas = []
    # Input layer to first hidden layer
    theta_input_hidden = np.random.uniform(low=weight_min, high=weight_max, size=(hnodes, inputs + 1))
    thetas.append(theta_input_hidden)

    # Hidden layers
    for i in range(1, hlayers):
        theta_hidden = np.random.uniform(low=weight_min, high=weight_max, size=(hnodes, hnodes + 1))
        thetas.append(theta_hidden)

    # Last hidden layer to output layer
    theta_hidden_output = np.random.uniform(low=weight_min, high=weight_max, size=(outputs, hnodes + 1))
    thetas.append(theta_hidden_output)

    return thetas


# Forward propagation function
def forward_prop(inputs, thetas, activation_func):
    nn_a = [inputs]  # Activation values
    nn_z = [inputs]  # Linear combinations
    for theta in thetas:
        layer_a = []
        layer_z = []
        for t in theta:
            nn_w_bias = np.insert(nn_a[-1], 0, 1)  # Add bias term to input
            dot = np.dot(nn_w_bias, t)  # Calculate dot product
            node_value = activation_func(dot)  # Apply activation function
            layer_a.append(node_value)
            layer_z.append(dot)
        nn_a.append(layer_a)
        nn_z.append(layer_z)
    return nn_a, nn_z


# Backpropagation function to calculate gradients
def back_prop(expected_out, nn_z, thetas, activation_derivative, loss_derivative):
    deltas = []  # Error terms
    # Calculate output error
    output_error = loss_derivative(expected_out, nn_z[-1])
    deltas.append(output_error)
    # Propagate errors back through network
    for i in reversed(range(1, len(nn_z) - 1)):
        layer = []
        for j in range(len(nn_z[i])):
            delta_sum = sum(thetas[i][k][j + 1] * deltas[0][k] for k in range(len(nn_z[i + 1])))
            delta = delta_sum * activation_derivative(nn_z[i][j])
            layer.append(delta)
        deltas.insert(0, layer)
    return deltas


# Update weights of the network based on error gradients
def update_weights(learning_rate, nn_values, thetas, deltas):
    new_thetas = []
    for i in range(len(thetas)):
        layer = []
        for j in range(len(thetas[i])):
            node_thetas = [thetas[i][j][0] - learning_rate * deltas[i][j]]  # Bias Update
            for k in range(1, len(thetas[i][j])):
                # i: Current Layer j:Current Source node k:Current Destination
                new_theta = thetas[i][j][k] - learning_rate * nn_values[i][k-1] * deltas[i][j]
                node_thetas.append(new_theta)
            layer.append(node_thetas)
        new_thetas.append(layer)
    return new_thetas


# Main training function for the neural network
def learn(inputs, expected_outputs, thetas, learning_rate, iterations, activation_func, activation_derivative,
          loss_func, loss_derivative):
    loss = []
    for i in range(iterations):
        iteration_loss = 0
        for input_data, expected_output in zip(inputs, expected_outputs):
            # Perform forward and backward propagation
            nn_a, nn_z = forward_prop(input_data, thetas, activation_func)
            deltas = back_prop(expected_output, nn_z, thetas, activation_derivative, loss_derivative)
            thetas = update_weights(learning_rate, nn_a, thetas, deltas)
            # Compute and accumulate loss
            iteration_loss += loss_func(expected_output, nn_z[-1])
        loss.append(iteration_loss / len(inputs))  # Average loss for this iteration
        print(f'Iteration {i + 1}/{iterations} completed. Loss: {iteration_loss}')
        # Early stopping condition
        if len(loss) >= 10 and len(set(loss[-10:])) == 1:
            print("Stopping early: Loss has stabilized.")
            break

    return thetas, loss


# Function to test the neural network
def test_nn(inputs, expected_outputs, thetas):
    print("Testing Neural Network")
    for input_data, expected_output in zip(inputs, expected_outputs):
        nn_values, _ = forward_prop(input_data, thetas, sigmoid)  # assuming sigmoid used for forward prop
        predicted = np.array(nn_values[-1])
        diff = expected_output - predicted
        print(
            f"Input: {input_data.tolist()}, Predicted: {predicted.tolist()}, Actual: {expected_output.tolist()}, Difference: {diff.tolist()}")


# Function to visualize weights of the neural network
def visualize_thetas(thetas):
    num_matrices = len(thetas)
    fig, axes = plt.subplots(1, num_matrices, figsize=(15, 5))
    for i, theta in enumerate(thetas):
        ax = axes[i]
        cax = ax.matshow(theta, cmap='viridis')
        fig.colorbar(cax, ax=ax)
        ax.set_title(f'Layer {i} to {i + 1}')
        ax.set_xlabel('Inputs')
        ax.set_ylabel('Neurons')
    plt.tight_layout()
    plt.show(block=False)


# Function to visualize activation values across layers
def visualize_nn_values(nn_values):
    num_layers = len(nn_values)
    fig, axes = plt.subplots(1, num_layers, figsize=(15, 5))
    for i, layer in enumerate(nn_values):
        ax = axes[i]
        layer = np.array(layer).reshape(-1, 1)
        cax = ax.matshow(layer, cmap='viridis')
        fig.colorbar(cax, ax=ax)
        ax.set_title(f'Layer {i} values')
        ax.set_ylabel('Neuron')
    plt.tight_layout()
    plt.show(block=False)


# Function to visualize the loss over iterations
def visualize_loss(loss):
    plt.figure()
    plt.plot(loss)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss over iterations')
    plt.show(block=False)
