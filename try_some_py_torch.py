import torch
from torch.autograd import Variable

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
b = 2.0
w1 = Variable(torch.Tensor([1.0]), requires_grad=True)  # any random value
w2 = Variable(torch.Tensor([2.0]), requires_grad=True)

# our model forward pass


def forward(x):
    return x * x * w2 + x * w1 + b

# loss function:


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# before training
print('predict (before training)', 4, forward(4).data[0])

# train loop
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x)
        l.backward()
        print('\tgrad: ', x_val, y_val, w1.grad.data[0])
        print('\tgrad: ', x_val, y_val, w2.grad.data[0])
        w1.data = w1.data - 0.01 * w1.grad.data
        w2.data = w2.data - 0.01 * w2.grad.data
        print('w1:d', w1.grad.data)
        print('w2:d', w2.grad.data)
        # manually zero the gradients after updating weights
        w1.grad.data.zero_()
        w2.grad.data.zero_()
    print('progress:', epoch, l.data[0])


# after traffic_engineering
print('predict (after training)', 4, forward(4).data[0])
