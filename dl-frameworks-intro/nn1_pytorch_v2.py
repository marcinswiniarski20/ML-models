import numpy as np
import torch
from torch.autograd import Variable

from matplotlib import pyplot as plt

N, D_in, H, D_out = 64, 1000, 100, 10
x = Variable(torch.randn(N, D_in), requires_grad=False)
y = Variable(torch.randn(N, D_out), requires_grad=False)
w1 = Variable(torch.randn(D_in, H), requires_grad=True)
w2 = Variable(torch.randn(H, D_out), requires_grad=True)

learning_rate = 1e-6
nb_iterations = 500
loss_history = []
for i in range(nb_iterations):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred - y).pow(2).sum()
    loss_history.append(loss)
    loss.backward()

    with torch.no_grad():
        w1.data -= learning_rate*w1.grad.data
        w2.data -= learning_rate*w2.grad.data
        w1.grad.zero_()
        w2.grad.zero_()

x = np.arange(0, len(loss_history), step=1)
plt.title(f"Loss history over {nb_iterations} iterations")
plt.plot(x, loss_history, 'o')
plt.show()