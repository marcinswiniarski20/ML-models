import numpy as np

import torch
from torch.autograd import Variable

from matplotlib import pyplot as plt


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


N, D_in, H, D_out = 64, 1000, 100, 10
learning_rate = 1e-4
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

model = TwoLayerNet(D_in, H, D_out)

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

nb_epoch = 500
loss_history = []
for i in range(nb_epoch):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss_history.append(loss)

    model.zero_grad()
    loss.backward()
    optimizer.step()

time = np.arange(0, len(loss_history), step=1)

plt.title(f"Loss history over {nb_epoch} epochs")
plt.plot(time, loss_history, 'o')
plt.show()