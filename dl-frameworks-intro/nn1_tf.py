import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

N, D, H = 64, 1000, 100
x = tf.placeholder(tf.float32, shape=(N, D))
y = tf.placeholder(tf.float32, shape=(N, D))
w1 = tf.Variable(tf.random_normal((D, H)))
w2 = tf.Variable(tf.random_normal((H, D)))

# Relu and activation of neurons
h = tf.maximum(tf.matmul(x, w1), 0)
y_pred = tf.matmul(h, w2)
diff = y_pred - y
loss = tf.reduce_mean(tf.reduce_sum(diff**2, axis=1))
# loss = tf.losses.mean_squared_error(y_pred, y)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
updates = optimizer.minimize(loss)

loss_history = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    values = {x: np.random.randn(N, D),
              y: np.random.randn(N, D)}
    learning_rate = 1e-5
    for i in range(50):
        loss_val, _ = sess.run([loss, updates], feed_dict=values)
        loss_history.append(loss_val)

time = np.arange(0, len(loss_history), step=1)

plt.title("Loss history over the backpropagation, 50 iterations")
plt.plot(time, loss_history, 'o')
plt.show()