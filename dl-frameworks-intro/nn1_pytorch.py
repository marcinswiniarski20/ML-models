import numpy as np
import torch

from matplotlib import pyplot as plt

dtype = torch.FloatTensor
N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in).type(dtype)
y = torch.randn(N, D_out).type(dtype)
w1 = torch.randn(D_in, H).type(dtype)
w2 = torch.randn(H, D_out).type(dtype)

learning_rate = 1e-6
nb_iteration = 500

loss_history = []
for i in range(nb_iteration):
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)
    loss = (y_pred - y).pow(2).sum()

    grad_y_pred = 2.0 *(y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    w1 -= learning_rate*grad_w1
    w2 -= learning_rate*grad_w2

    loss_history.append(loss)

time = np.arange(0, len(loss_history), step=1)
plt.title(f"Loss history over {nb_iteration} iterations")
plt.plot(time, loss_history, 'x')
plt.show()