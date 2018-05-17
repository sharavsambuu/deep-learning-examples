# reference : 
#   - https://towardsdatascience.com/autoencoders-introduction-and-implementation-3f40483b0a85

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', validation_size=0)

#img = mnist.train.images[2]
#plt.imshow(img.reshape((28, 28)), cmap='Greys_r')

#plt.show()

learning_rate = 0.001

inputs_  = tf.placeholder(tf.float32, (None, 28, 28, 1), name="input")
targets_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name="target")

# Encoder

# 28x28x1 -> 28x28x16
conv1    = tf.layers.conv2d(inputs=inputs_, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# 28x28x16 -> 14x14x16
maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same') 
# 1x14x16 -> 14x14x8
conv2    = tf.layers.conv2d(inputs=maxpool1, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# 14x14x8 -> 7x7x8
maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
# 7x7x8 -> 7x7x8
conv3    = tf.layers.conv2d(inputs=maxpool2, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)

# 7x7x8 -> 4x4x8
encoded  = tf.layers.max_pooling2d(conv3, pool_size=(2,2), strides=(2,2), padding='same')


# Decoder
# 4x4x8 -> 7x7x8
upsample1 = tf.image.resize_images(encoded, size=(7,7), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# 7x7x8 -> 7x7x8
conv4 = tf.layers.conv2d(inputs=upsample1, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# 7x7x8 -> 14x14x8
upsample2 = tf.image.resize_images(conv4, size=(14, 14), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# 14x14x8 -> 14x14x8
conv5 = tf.layers.conv2d(inputs=upsample2, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# 14x14x8 -> 28x28x8
upsample3 = tf.image.resize_images(conv5, size=(28,28), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# 28x28x8 -> 28x28x16
conv6 = tf.layers.conv2d(inputs=upsample3, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)

# 28x28x16 -> 28x28x1
logits = tf.layers.conv2d(inputs=conv6, filters=1, kernel_size=(3,3), padding='same', activation=tf.nn.relu)


decoded = tf.nn.sigmoid(logits)

loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)
cost = tf.reduce_mean(loss)
opt  = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.Session()

epochs       = 20
batch_size   = 200
noise_factor = 0.5

sess.run(tf.global_variables_initializer())
for e in range(epochs):
    for ii in range(mnist.train.num_examples//batch_size):
        batch = mnist.train.next_batch(batch_size)
        imgs  = batch[0].reshape((-1,28,28,1))
        noisy_imgs = imgs + noise_factor*np.random.randn(*imgs.shape)
        noisy_imgs = np.clip(noisy_imgs, 0., 1.)
        batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: noisy_imgs, targets_: imgs})
        print("Epochs: {}/{}".format(e+1, epochs), " loss: {:.4f}".format(batch_cost))
