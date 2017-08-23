import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

sess = tf.InteractiveSession()

x = tf.linspace(-3.0, 3.0, 100)

# Гауссын муруй
mean = 0.0
sigma = 1.0
z = (tf.exp(tf.negative(tf.pow(x - mean, 2.0) /
        (2.0 * tf.pow(sigma, 2.0)))) *
        (1.0 / (sigma * tf.sqrt(2.0 * 3.1415))))

plt.plot(z.eval())
plt.show()
