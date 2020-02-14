import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# a random guess:random value
w = 1.0
w_list = []
mse_list = []
# def model for the forward pass


def forward(x):
    return x * w

# loss function


def loss(x, y):
    y_predict = forward(x)
    return (y_predict - y) * (y_predict - y)


# compute loss for w
for w in np.arange(0.0, 4.1, 0.1):
    print('w=', w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        l = loss(x_val, y_val)
        l_sum += l
        print("\t", x_val, y_val, y_pred_val, l)
    print('MSE=', l_sum / 3)
    w_list.append(w)
    mse_list.append(l_sum/3)

# draw a nice graph


plt.plot(w_list, mse_list)
# plt.ylable('lable')
# plt.xlable('w')
plt.show()
