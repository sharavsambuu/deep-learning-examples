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
epochs     = 100

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(
        x_input                , 
        weight_hidden1         , 
        weight_hidden2         , 
        weight_output          , 
        placeholder_keep_input , 
        placeholder_keep_hidden
        ):
    hidden1 = tf.nn.relu(tf.matmul(x_input, weight_hidden1))
    hidden1 = tf.nn.dropout(hidden1, placeholder_keep_input)

    hidden2 = tf.nn.relu(tf.matmul(hidden1, weight_hidden2))
    hidden2 = tf.nn.dropout(hidden2, placeholder_keep_hidden)

    return tf.matmul(hidden2, weight_output)

x             = tf.placeholder("float", [None, 784])
y_            = tf.placeholder("float", [None, 10])

weight_hidden1 = init_weights([784, 625])
weight_hidden2 = init_weights([625, 625])
weight_output  = init_weights([625,  10])

placeholder_keep_input  = tf.placeholder("float")
placeholder_keep_hidden = tf.placeholder("float")

y = model(
    x                      , 
    weight_hidden1         , 
    weight_hidden2         , 
    weight_output          , 
    placeholder_keep_input , 
    placeholder_keep_hidden
    )

loss       = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
train_op   = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(loss)
predict_op = tf.argmax(y, 1)

plt.ion()

history = []
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(epochs):
        for start, end in zip(
                range(0, len(mnist.train.images), 128), 
                range(128, len(mnist.train.images), 128)
                ):
            sess.run(
                    train_op, 
                    feed_dict = {
                        x  : mnist.train.images[start:end], 
                        y_ : mnist.train.labels[start:end],
                        placeholder_keep_input  : 0.8,
                        placeholder_keep_hidden : 0.5
                    }
                )
        accuracy = np.mean(
                np.argmax(
                    mnist.test.labels, 
                    axis=1
                ) == sess.run(
                    predict_op, 
                    feed_dict={
                        x  : mnist.test.images, 
                        y_ : mnist.test.labels,
                        placeholder_keep_input  : 1.0,
                        placeholder_keep_hidden : 1.0
                    }
                )
            )
        history.append((i, accuracy*100.0))
        draw_plot(np.asarray(history))
        print("epoch : %d, accuracy : %d,"%(i, accuracy*100))
