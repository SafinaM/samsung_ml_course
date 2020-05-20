#predict sin(y)
import torch
import matplotlib.pyplot as plt


class RegressionNet(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(RegressionNet, self).__init__()
        self.fc = torch.nn.Linear(1, n_hidden_neurons)
        self.act1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hidden_neurons, 1)

    def forward(self, x):
        x = self.fc(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x


net = RegressionNet(20)


def target_function(x):
    return 2**x * torch.sin(2**-x)


# ------Dataset preparation start--------:
x_train = torch.linspace(-10, 5, 100)
y_train = target_function(x_train)
noise = torch.randn(y_train.shape) / 20.
y_train = y_train + noise

x_train.unsqueeze_(1)
y_train.unsqueeze_(1)

x_validation = torch.linspace(-10, 5, 100)
y_validation = target_function(x_validation)

plt.plot(x_validation.numpy(), y_validation.numpy(), 'o')
x_validation.unsqueeze_(1)
y_validation.unsqueeze_(1)

# plt.plot(x_train.numpy(), y_train.numpy(), 'o', label='Train')
# plt.plot(x_validation.numpy(), y_validation.data.numpy(), 'o', c='b', label='validation')
# plt.legend(loc='upper left')
# plt.xlabel('$x$')
# plt.ylabel('$y$')
# plt.show()
# ------Dataset preparation end--------:


def metric(pred, target):
    return (pred - target).abs().mean()


def loss(pred, target):
    return (pred - target).abs().mean()
    # squares = (pred - target) ** 2
    # return squares.mean()


def predict(net, x, y):
    y_pred = net.forward(x)

    plt.plot(x.numpy(), y.numpy(), 'o', label='Ground truth')
    plt.plot(x.numpy(), y_pred.data.numpy(), 'x', c='b', label='Prediction')
    plt.legend(loc='upper left')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.show()


# predict(net, x_validation, y_validation)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

for epoch_index in range(3000):
    optimizer.zero_grad()
    y_pred = net.forward(x_train)
    loss_value = loss(y_pred, y_train)
    loss_value.backward()
    optimizer.step()

print(metric(net.forward(x_validation), y_validation).item())

predict(net, x_validation, y_validation)