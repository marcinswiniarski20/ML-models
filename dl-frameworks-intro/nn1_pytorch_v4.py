import numpy as np

import torch
from torch.autograd import Variable

from matplotlib import pyplot as plt

N, D_in, H, D_out = 64, 1000, 100, 10
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out))
loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
nb_epoch = 500
loss_history = []
for i in range(nb_epoch):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss_history.append(loss)

    model.zero_grad()
    loss.backward()
    optimizer.step()

time = np.arange(0, len(loss_history), step=1)

plt.title(f"Loss history over {nb_epoch} epochs")
plt.plot(time, loss_history, 'o')
plt.show()