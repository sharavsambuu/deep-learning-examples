from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def draw_plot(data):
    plt.xlabel('Epoch')
    plt.ylabel('Нарийвчлал')
    plt.plot(data[:, 0], data[:, 1])
    plt.show()
    plt.pause(0.01)
    plt.gcf().clear()

mnist      = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data", one_hot=True)
batch_size = 100
epochs     = 100

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(x_input, weight_hidden, weight_output):
    hidden = tf.nn.sigmoid(tf.matmul(x_input, weight_hidden))
    return tf.matmul(hidden, weight_output)

x             = tf.placeholder("float", [None, 784])
weight_hidden = init_weights([784, 625])
weight_output = init_weights([625, 10])
y             = model(x, weight_hidden, weight_output)
y_            = tf.placeholder("float", [None, 10])

loss       = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
train_op   = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
predict_op = tf.argmax(y, 1)

plt.ion()

history = []
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(epochs):
        for start, end in zip(range(0, len(mnist.train.images), 128), range(128, len(mnist.train.images), 128)):
            sess.run(train_op, feed_dict={x: mnist.train.images[start:end], y_: mnist.train.labels[start:end]})
        accuracy = np.mean(np.argmax(mnist.test.labels, axis=1) == sess.run(predict_op, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
        history.append((i, accuracy*100.0))
        draw_plot(np.asarray(history))
        print("epoch : %d, accuracy : %d,"%(i, accuracy*100))
