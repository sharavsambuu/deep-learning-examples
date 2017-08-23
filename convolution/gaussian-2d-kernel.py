import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

sess = tf.InteractiveSession()

# 100 ширхэг -3, 3 ийн хооронд тархсан тоо үүсгэх
x = tf.linspace(-3.0, 3.0, 100)

# Гауссын муруй
mean = 0.0
sigma = 1.0
z = (tf.exp(tf.negative(tf.pow(x - mean, 2.0) /
        (2.0 * tf.pow(sigma, 2.0)))) *
        (1.0 / (sigma * tf.sqrt(2.0 * 3.1415))))

# Гауссын муруй доторхи элементүүдийн тоо
ksize = z.get_shape().as_list()[0]

# 2 хэмжээст Гауссын кернел гаргаж авах
z_2d = tf.matmul(tf.reshape(z, [ksize, 1]), tf.reshape(z, [1, ksize]))

# Гауссын кернелийг зурж харах
plt.imshow(z_2d.eval())
plt.show()
