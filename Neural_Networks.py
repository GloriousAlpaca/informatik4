import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def quadratic_loss(expected, output):
    loss = 0.5 * np.sum((expected - output) ** 2)
    return loss


def mse_loss(expected, output):
    return 0.5 * np.mean((expected - output) ** 2)


def mse_loss_derivative(expected, output):
    return output - expected


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


def forward_prop(inputs, thetas, activation_func):
    nn_values = []
    nn_values.append(inputs)
    for i in range(len(thetas)):
        layer = []
        for j in range(len(thetas[i])):
            nn_w_bias = np.insert(nn_values[i], 0, 1)  # Add bias term
            dot = np.dot(nn_w_bias, thetas[i][j])
            node_value = activation_func(dot)
            layer.append(node_value)
        nn_values.append(layer)
    return nn_values


def back_prop(expected_out, nn_values, thetas, activation_derivative, loss_derivative):
    deltas = []
    # Calculate Output error
    output_error = loss_derivative(expected_out, nn_values[-1])
    deltas.append(output_error)
    for i in reversed(range(1, len(nn_values) - 1)):
        layer = []
        for j in range(len(nn_values[i])):
            delta_sum = 0
            for k in range(len(nn_values[i+1])):
                delta_sum += thetas[i][k][j+1] * deltas[0][k]
            delta = delta_sum * activation_derivative(nn_values[i][j])
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


def learn(inputs, expected_outputs, thetas, learning_rate, iterations, activation_func, activation_derivative, loss_func, loss_derivative):
    loss = []
    for i in range(iterations):
        iteration_loss = 0
        for input_data, expected_output in zip(inputs, expected_outputs):
            # Forward propagation
            nn_values = forward_prop(input_data, thetas, activation_func)

            # Back propagation
            deltas = back_prop(expected_output, nn_values, thetas, activation_derivative, loss_derivative)

            # Update weights
            thetas = update_weights(learning_rate, nn_values, thetas, deltas)

            # Calculate Loss for this input and accumulate
            iteration_loss += loss_func(expected_output, nn_values[-1])
        # Average loss over all inputs
        loss.append(iteration_loss)
        print(f'Iteration {i + 1}/{iterations} completed.')

    return thetas, loss


def test_nn(inputs, expected_outputs, thetas):
    print("Testing Neural Network")
    for input_data, expected_output in zip(inputs, expected_outputs):
        nn_values = forward_prop(input_data, thetas)
        predicted = np.array(nn_values[-1]) > 0.5  # Threshold for classification
        print(f"Input: {input_data.tolist()}, Predicted: {predicted.tolist()}, Actual: {expected_output.tolist()}")


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
    plt.show(block=False)


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
        ax.set_ylabel('Neuron')

    plt.tight_layout()
    plt.show(block=False)


def visualize_loss(loss):
    plt.figure()
    plt.plot(loss)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss over iterations')
    plt.show(block=False)
