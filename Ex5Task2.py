import matplotlib.pyplot as plt
import numpy as np

import Neural_Networks as NN

data_amount = 200
inputs = []
outputs = []

inputs = np.random.rand(data_amount).reshape(-1, 1)
outputs = 3 * inputs
# 3 * inputs ** 5 + 1.5 * inputs ** 4 + 2 * inputs ** 3 + 7 * inputs + 0.5
thetas = NN.create_NN(1, 2, 8, 1)
lr = 0.1
iter = 1000
trained_thetas, loss = NN.learn(inputs, outputs, thetas, lr, iter, NN.relu, NN.relu_derivative, NN.mse_loss,
                                NN.mse_loss_derivative)

NN.visualize_thetas(trained_thetas)
NN.visualize_loss(loss)

predictions = []
for input_data in inputs:
    nn_values = NN.forward_prop(input_data, thetas, NN.relu)
    predicted = nn_values[-1]
    predictions.append(predicted)

predictions = np.array(predictions).reshape(-1, 1)

plt.figure()
plt.scatter(inputs, outputs, label='Original Data', color='blue')
plt.scatter(inputs, predictions, label='Predicted Data', color='red')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Original vs Predicted Data')
plt.legend()
plt.show()
