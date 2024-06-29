import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import Neural_Networks as NN  # Import custom neural network module



# Load the MNIST dataset
mnist = fetch_openml('mnist_784', parser='auto')
X, y = mnist['data'].to_numpy(), mnist['target'].astype(np.int32)

# Normalize the input data
X = X / 255.0

# Encode the labels
y = np.eye(10)[y]

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 6000, test_size = 1000, random_state=42)

print(f'Training set size: {X_train.shape[0]} samples')
print(f'Test set size: {X_test.shape[0]} samples')

# Activation function and its derivative
hid_func = [NN.relu, NN.relu_derivative]
out_func = [NN.softmax, NN.linear_derivative]
inputs = X_train.shape[1]
hnodes = 64
hlayers = 2
outputs = 10

neural_net = NN.create_NN(inputs, hlayers, hnodes, outputs, hid_func, out_func, 12, -0.1, 0.1)
lr = 0.01 # Learning rate
iter = 50  # Number of training iterations

# Train the neural network
trained_net, loss = NN.learn(X_train, y_train, neural_net, lr, iter,
          NN.cross_entropy_loss, NN.cross_entropy_loss_derivative)

# Visualization functions from the neural network module
NN.visualize_thetas(trained_net.thetas)
NN.visualize_loss(loss)
NN.class_test_nn(X_test, y_test, trained_net)
# Test the neural network and get predictions
NN.visualize_confusion_matrix(X_test, y_test, 10, trained_net)
plt.show()

# Function to visualize predictions vs actuals
NN.visualize_predictions(X_test, y_test, trained_net)