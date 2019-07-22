import numpy as np

import theano
import theano.tensor as T

from matplotlib import pyplot as plt

#Batch size, input dim, hidden dim, num classes
N, D, H, C = 64, 1000, 100, 10

x = T.matrix('x')
y = T.vector('y', dtype='int64')
w1 = T.matrix('w1')
w2 = T.matrix('w2')

# Forward pass: Compute scores
a = x.dot(w1)
a_relu = T.nnet.relu(a)
scores = a_relu.dot(w2)

# Forward pass: compute softmax loss
probs = T.nnet.softmax(scores)
loss = T.nnet.categorical_crossentropy(probs, y).mean()

# Backward pass: compute gradients
dw1, dw2 = T.grad(loss, [w1, w2])

f = theano.function(inputs = [x, y, w1, w2],
                    outputs = [loss, scores, dw1, dw2])

# Run the funciton
xx = np.random.randn(N, D)
yy = np.random.randint(C, size=N)

# Scale just for smaller values
scale = 1e-2
ww1 = scale*np.random.randn(D, H)
ww2 = scale*np.random.randn(H, C)

loss_history = []
nb_iterations = 50
learning_rate = 1e-1
for i in range(nb_iterations):
    loss, scores, dww1, dww2 = f(xx, yy, ww1, ww2)
    ww1 -= learning_rate * dww1
    ww2 -= learning_rate * dww2
    loss_history.append(loss)

time = np.arange(0, len(loss_history), step=1)
plt.title(f"Loss history over {nb_iterations} iterations")
plt.plot(time, loss_history, 'o')
plt.show()