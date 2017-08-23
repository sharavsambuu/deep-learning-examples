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

mnist  = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data", one_hot=True)
train_x, train_y, test_x, test_y = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

epochs     = 100
batch_size = 128
test_size  = 256

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(
        x_input                , 
        w1                     ,
        w2                     ,
        w3                     ,
        w4                     ,
        w_out                  ,
        placeholder_keep_conv  , 
        placeholder_keep_hidden
        ):
    # 1-р давхарга
    l1_conv2d     = tf.nn.conv2d(x_input, w1, strides=[1, 1, 1, 1], padding='SAME')
    l1_activation = tf.nn.relu(l1_conv2d)
    l1_maxpool    = tf.nn.max_pool(l1_activation, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l1            = tf.nn.dropout(l1_maxpool, placeholder_keep_conv)
    # 2-р давхарга
    l2_conv2d     = tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME')
    l2_activation = tf.nn.relu(l2_conv2d)
    l2_maxpool    = tf.nn.max_pool(l2_activation, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l2            = tf.nn.dropout(l2_maxpool, placeholder_keep_conv)
    # 3-р давхарга
    l3_conv2d     = tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME')
    l3_activation = tf.nn.relu(l3_conv2d)
    l3_maxpool    = tf.nn.max_pool(l3_activation, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l3_reshaped   = tf.reshape(l3_maxpool, [-1, w4.get_shape().as_list()[0]])
    l3            = tf.nn.dropout(l3_reshaped, placeholder_keep_conv)
    # 4-р давхарга
    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, placeholder_keep_hidden)

    return tf.matmul(l4, w_out)

train_x = train_x.reshape(-1, 28, 28, 1)
test_x  = test_x.reshape (-1, 28, 28, 1)

x       = tf.placeholder("float", [None, 28, 28, 1])
y_      = tf.placeholder("float", [None, 10])

w1      = init_weights([3, 3, 1 , 32 ]) # 3x3x1  conv, 32  outputs
w2      = init_weights([3, 3, 32, 64 ]) # 3x3x32 conv, 64  outputs
w3      = init_weights([3, 3, 64, 128]) # 3x3x32 conv, 128 outputs
w4      = init_weights([128*4*4 , 625]) # fully connected 128*4*4 inputs, 625 outputs
w_out   = init_weights([625     ,  10]) # fully connected 625 inputs, 10 output labels

placeholder_keep_conv   = tf.placeholder("float")
placeholder_keep_hidden = tf.placeholder("float")

y = model(
    x                     , 
    w1                    , 
    w2                    , 
    w3                    ,
    w4                    ,
    w_out                 , 
    placeholder_keep_conv , 
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
        training_batch = zip(
            range(0         , len(train_x), batch_size), 
            range(batch_size, len(train_x), batch_size)
        )
        for start, end in training_batch:
            sess.run(
                    train_op, 
                    feed_dict = {
                        x  : train_x[start:end], 
                        y_ : train_y[start:end],
                        placeholder_keep_conv   : 0.8,
                        placeholder_keep_hidden : 0.5
                    }
                )
        test_indices = np.arange(len(test_x))
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]
        accuracy = np.mean(
                np.argmax(
                    test_y[test_indices], 
                    axis=1
                ) == sess.run(
                    predict_op, 
                    feed_dict={
                        x  : test_x[test_indices], 
                        y_ : test_y[test_indices],
                        placeholder_keep_conv   : 1.0,
                        placeholder_keep_hidden : 1.0
                    }
                )
            )
        history.append((i, accuracy*100.0))
        draw_plot(np.asarray(history))
        print("epoch : %d, accuracy : %f,"%(i, accuracy*100.0))
