import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

from matplotlib import pyplot as plt

N, D, H = 64, 1000, 100
nb_epoch = 100
learning_rate = 1e0

model = Sequential()
model.add(Dense(input_dim=D, output_dim=H))
model.add(Activation('relu'))
model.add(Dense(input_dim=H, output_dim=D))
optimizer = SGD(lr=learning_rate)

model.compile(loss='mean_squared_error', optimizer=optimizer)

x = np.random.randn(N, D)
y = np.random.randn(N, D)
history = model.fit(x, y, nb_epoch=nb_epoch, batch_size=N, verbose=0)

loss_history = history.history['loss']
x = np.arange(0, len(loss_history), 1)

plt.title(f'Loss history over {nb_epoch} epochs')
plt.plot(x, loss_history, 'o')
plt.show()