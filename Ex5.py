import matplotlib.pyplot as plt
import numpy as np


def create_NN(inputs, hlayers, hnodes, outputs):
    thetas = []

    # Input layer to first hidden layer
    theta_input_hidden = np.random.rand(hnodes, inputs + 1)  # +1 for bias term
    thetas.append(theta_input_hidden)

    for i in range(1, hlayers):
        # Between hidden layers
        theta_hidden = np.random.rand(hnodes, hnodes + 1)  # +1 for bias term
        thetas.append(theta_hidden)

    # Last hidden layer to output layer
    theta_hidden_output = np.random.rand(outputs, hnodes + 1)  # +1 for bias term
    thetas.append(theta_hidden_output)

    return thetas


def parametric_forward_prop(inputs, thetas):
    nn_values = []
    nn_values.append(inputs)
    for i in range(len(thetas)):
        layer = []
        for j in range(len(thetas[i])):
            nn_w_bias = np.insert(nn_values[i], 0, 1)  # Add bias term
            dot = np.dot(nn_w_bias, thetas[i][j])
            node_value = 1 / (1 + np.exp(-dot))
            layer.append(node_value)
        nn_values.append(layer)
    return nn_values


def back_prop(expected_out, nn_values, thetas):
    deltas = []
    # Calculate Output error
    deltas.append(np.subtract(nn_values[-1], expected_out))
    for i in reversed(range(1, len(nn_values) - 1)):
        layer = []
        for j in range(len(nn_values[i])):
            delta_sum = 0
            for k in range(len(nn_values[i + 1])):
                delta_sum += thetas[i][k][j + 1] * deltas[0][k]
            delta = delta_sum * nn_values[i][j] * (1 - nn_values[i][j])
            layer.append(delta)
        deltas.insert(0, layer)
    return deltas


def update_weights(learning_rate, nn_values, thetas, deltas):
    new_thetas = []
    for i in range(len(thetas)):
        layer = []
        for j in range(len(thetas[i])):
            node_thetas = []
            # Bias update
            node_thetas.append(thetas[i][j][0] - learning_rate * deltas[i][j])
            for k in range(1, len(thetas[i][j])):
                # i: Current Layer j:Current Source node k:Current Destination
                new_theta = thetas[i][j][k] - learning_rate * nn_values[i][k - 1] * deltas[i][j]
                node_thetas.append(new_theta)
            layer.append(node_thetas)
        new_thetas.append(layer)
    return new_thetas


def learn(inputs, expected_outputs, hlayers, hnodes, outputs, learning_rate, iterations):
    # Initialize network
    thetas = create_NN(len(inputs[0]), hlayers, hnodes, outputs)

    for i in range(iterations):
        for input_data, expected_output in zip(inputs, expected_outputs):
            # Forward propagation
            nn_values = parametric_forward_prop(input_data, thetas)

            # Backpropagation
            deltas = back_prop(expected_output, nn_values, thetas)

            # Update weights
            thetas = update_weights(learning_rate, nn_values, thetas, deltas)

        print(f'Iteration {i + 1}/{iterations} completed.')

    return thetas

def visualize_thetas(thetas):
    num_matrices = len(thetas)
    fig, axes = plt.subplots(1, num_matrices, figsize=(15, 5))

    for i, theta in enumerate(thetas):
        ax = axes[i]
        cax = ax.matshow(theta, cmap='viridis')
        fig.colorbar(cax, ax=ax)
        ax.set_title(f'Theta Matrix {i + 1}')
        ax.set_xlabel('Inputs')
        ax.set_ylabel('Neurons')

    plt.tight_layout()
    plt.show()


def visualize_nn_values(nn_values):
    num_layers = len(nn_values)
    fig, axes = plt.subplots(1, num_layers, figsize=(15, 5))

    for i, layer in enumerate(nn_values):
        ax = axes[i]
        if isinstance(layer, np.ndarray):
            if layer.ndim == 1:
                layer = layer.reshape(-1, 1)
            cax = ax.matshow(layer, cmap='viridis')
        else:
            cax = ax.matshow(np.array(layer).reshape(-1, 1), cmap='viridis')
        fig.colorbar(cax, ax=ax)
        ax.set_title(f'Layer {i} values')
        ax.set_xlabel('Neurons')
        ax.set_ylabel('Samples')

    plt.tight_layout()
    plt.show()


# Example Usage for XOR
inputs = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
expected_outputs = np.array([[0], [0], [1], [0]])
hlayers = 1
nodes = 3
outputs = 1
learning_rate = 0.1
epochs = 10000

# Training durchführen
trained_thetas = learn(inputs, expected_outputs, hlayers, nodes, outputs, learning_rate, epochs)

# Visualisierung der trainierten Gewichte
visualize_thetas(trained_thetas)

# Vorwärtsausbreitung und Visualisierung der Netzwerkausgaben
nn_values = parametric_forward_prop([1, 1], trained_thetas)
visualize_nn_values(nn_values)