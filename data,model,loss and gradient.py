x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w1 = 1.0  # a random guess value
w2 = 2.0
b = 2.0
# model forward pass


def forward(x):
    return x * x * w1 + x * w1 + b

# loss function


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

# compute gradient


# calculate w1's gradient
def gradient1(x, y):  # d_loss/d_w1
    return x * x * (2 * x * x * w1 + 2 * x * w2 + 2 * b - y)


def gradient1(x, y):  # d_loss/d_w1
    return x * (2 * x * x * w1 + 2 * x * w2 + 2 * b - y)


# before training
print('predict (before training)', 4, forward(4))

# train loop
for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        grad1 = gradient1(x_val, y_val)
        grad2 = gradient1(x_val, y_val)
        w1 = w1 - 0.01 * grad1
        w2 = w2 - 0.01 * grad2
        print('\tgrad1: ', x_val, y_val, grad1)
        print('\tgrad2: ', x_val, y_val, grad2)
        l = loss(x_val, y_val)

    print('progress:', epoch, 'w1=', w1, 'loss=', l)
    print('progress epoch, 'w2=', w2, 'loss=', l)

# after training
print('predict (after training)', '4 hours', forward(4))
