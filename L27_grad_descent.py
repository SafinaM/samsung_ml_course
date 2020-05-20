import numpy as np
import torch
import math
import matplotlib.pyplot as plt

x = torch.tensor(
    [[1., 2., 3., 4.],
     [5., 6., 7., 8.],
     [9., 10., 11., 12.]], requires_grad=True)

function = 10 * (x ** 2).sum()

function.backward()
print(x.grad, "<- grad")
x.data -= 0.001 * x.grad
x.grad.zero_()

w = torch.tensor([[5., 10.], [1., 2.]], requires_grad=True)

f = torch.prod(torch.log(torch.log(w + 7)))

f.backward()
print(w.grad, "<- grad")

x = torch.tensor([0., 1.], requires_grad=True)

alpha = 1.11

optimizer = torch.optim.SGD([x], lr=alpha)


def function_parabola(variable):
    return (variable ** 2).sum()


def make_gradient_step(function, variable):
    function_result = function(variable)
    function_result.backward()
    optimizer.step()
    optimizer.zero_grad()


for i in range(500):
    make_gradient_step(function_parabola, x)

print("grad", x)


def show_contours(objective,
                  x_lims=[-10.0, 10.0],
                  y_lims=[-10.0, 10.0],
                  x_ticks=100,
                  y_ticks=100):
    x_step = (x_lims[1] - x_lims[0]) / x_ticks
    y_step = (y_lims[1] - y_lims[0]) / y_ticks
    X, Y = np.mgrid[x_lims[0]:x_lims[1]:x_step, y_lims[0]:y_lims[1]:y_step]
    res = []
    for x_index in range(X.shape[0]):
        res.append([])
        for y_index in range(X.shape[1]):
            x_val = X[x_index, y_index]
            y_val = Y[x_index, y_index]
            res[-1].append(objective(np.array([[x_val, y_val]]).T))
    res = np.array(res)
    plt.figure(figsize=(7, 7))
    plt.contour(X, Y, res, 100)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')


var_history = []
fn_history = []

for i in range(500):
    var_history.append(x.data.numpy().copy())
    fn_history.append(function_parabola(x).data.cpu().numpy().copy())
    make_gradient_step(function_parabola, x)


show_contours(function_parabola)
plt.scatter(np.array(var_history)[:,0], np.array(var_history)[:,1], s=10, c='r')

plt.figure(figsize=(7,7))
plt.plot(fn_history)
plt.xlabel('step')
plt.ylabel('function value')
plt.show()

w = torch.tensor([[0., 1.]], requires_grad=True)

f = (x ** 2)

f.backward()
print(w.grad, "<- grad")