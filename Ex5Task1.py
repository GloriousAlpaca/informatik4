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


def forward_prop(inputs, thetas):
    nn_values = []
    nn_values.append(inputs)
    for i in range(len(thetas)):
        layer = []
        for j in range(len(thetas[i])):
            nn_w_bias = np.insert(nn_values[i], 0, 1)  # Add bias term
            dot = np.dot(nn_w_bias, thetas[i][j])
            node_value = 1 / (1 + np.exp(-dot))
            #print("Layer: ", i+1, "Node: ",j+1 ,"Thetas: ", thetas[i][j], "Neural net Values: ", nn_w_bias, "Node Value: ", node_value)
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


def learn(inputs, expected_outputs, thetas, learning_rate, iterations):
    loss = []
    for i in range(iterations):
        iteration_loss = 0
        for input_data, expected_output in zip(inputs, expected_outputs):
            # Forward propagation
            nn_values = forward_prop(input_data, thetas)

            # Back propagation
            deltas = back_prop(expected_output, nn_values, thetas)

            # Update weights
            thetas = update_weights(learning_rate, nn_values, thetas, deltas)

            # Calculate Loss for this input and accumulate
            iteration_loss += calculate_loss(expected_output, nn_values[-1])
        # Average loss over all inputs
        loss.append(iteration_loss)
        print(f'Iteration {i + 1}/{iterations} completed.')

    return thetas, loss


def calculate_loss(expected, output):
    loss = 0.5 * np.sum((expected - output) ** 2)
    return loss


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

inputs = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
expected_outputs = np.array([[1], [1], [0], [0]])
inodes = 2
hlayers = 1
hnodes = 2
outputs = 1
learning_rate = 0.1
epochs = 10000
thetas = create_NN(inodes, hlayers, hnodes, outputs)

trained_thetas, loss = learn(inputs, expected_outputs, thetas, learning_rate, epochs)

visualize_thetas(trained_thetas)

nn_values = forward_prop([1, 1], trained_thetas)
visualize_nn_values(nn_values)
visualize_loss(loss)
test_nn(inputs, expected_outputs, trained_thetas)
plt.show()
