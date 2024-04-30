import numpy as np
import pandas as pd
import info4


# Data preprocessing
data_points = np.column_stack((x, y))
train_set, test_set = info4.tt_split(data_points, 0.20)
x_train = train_set[:, :-1]
y_train = train_set[:, -1]
x_test = test_set[:, :-1]
y_test = test_set[:, -1]

print("X Array: ", x, " Y Array:", y)
print("X Train: ", x_train, " X Test:", x_test)
print("X Train: ", len(x_train), " X Test:", len(x_test))
