from skimage import data
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()

# зураг ачаалах
img = data.camera().astype(np.float32)

# зургийг 4-н хэмжээст болгох
# [#Images x H x W x #C]
img_4d = tf.reshape(img, [1, img.shape[0], img.shape[1], 1])

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

# Кернелийг convolution хийхийн тулд 4-н хэмжээсрүү оруулах
# [H x W x #Input channels x #Output channels]
z_4d = tf.reshape(z_2d, [ksize, ksize, 1, 1])

# convolution шүүлтүүр хийх 
convolved = tf.nn.conv2d(img_4d, z_4d, strides=[1, 1, 1, 1], padding='SAME')
# тооцооллын графыг Tensorflow дээр ажиллуулах
res = convolved.eval()

# зурж харах
plt.imshow(np.squeeze(res), cmap='gray')
plt.show()



