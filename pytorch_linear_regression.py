import torch
from torch.autograd import Variable

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))


class Model(torch.nn.Module):
    def __init__(self):
        '''
        In the constructor we initiate two nn.Linear module

        '''

        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # one in and one out

    def forward(self, x):
        '''
        In the forward funcipn we accept a variable of input data and we
        must return a Variable of output data. we can use Modules defined in the
        constructor as well as arbitary operator on Variables.

        '''
        y_pred = self.linear(x)
        return y_pred and


# our model
model = Model()


# =======constrctuct loss and optimizer

construct our loss function and an optimizer, the call to
# model.parameters() in the SGD(statstics gradient descent) constrctuctor will contain the learnable
# parameters of the two nn.linear modules which are members of the model.

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# training loop
for epoch in range(500):
    # forward pass:compute predicted y by passing x to the model
    y_pred = model(x_data)

    # computer and print loss
    loss = criterion(y_pred, y_data)
    print(epoch, loss.data[0])

    # zero gradients, perfor a backward pass, and update the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# after training
hour_var = Variable(torch.Tensor([4.0]))
print('predict (after training)', 4, model.forward(hour_var).data[0][0])
