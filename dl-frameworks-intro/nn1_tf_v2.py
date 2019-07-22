import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

N, D, H = 64, 1000, 100
x = tf.placeholder(tf.float32, shape=(N, D))
y = tf.placeholder(tf.float32, shape=(N, D))

init = tf.contrib.layers.xavier_initializer()
h = tf.layers.dense(inputs=x, units=H, activation=tf.nn.relu, kernel_initializer=init)
y_pred = tf.layers.dense(inputs=h, units=D, kernel_initializer=init)

loss = tf.losses.mean_squared_error(y_pred, y)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e0)
updates = optimizer.minimize(loss)

loss_history = []
iters = 500
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    values = {x: np.random.randn(N, D),
              y: np.random.randn(N, D)}
    for i in range(iters):
        loss_val, _ = sess.run([loss, updates], feed_dict=values)
        loss_history.append(loss_val)

time = np.arange(0, len(loss_history), step=1)

plt.title(f"Loss history over the backpropagation, {iters} iterations")
plt.plot(time, loss_history, 'o')
plt.show()