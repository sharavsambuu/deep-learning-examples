"""
CGAN, Conditional Generative Adversarial Network
MNIST шиг харагддаг зурагнуудыг үүсгэж сургах, гэхдээ нэмэлтээр label авах
тухайн label-тэй тохирох зурагнуудыг үүсгэж сургах
"""

import pickle as pkl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import animation
import argparse
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--show', help='foo help')
args = parser.parse_args()


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')

def model_inputs(real_dim, z_dim):
    inputs_real = tf.placeholder(tf.float32, (None, real_dim), name="inputs_real")
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name="inputs_z")
    return inputs_real, inputs_z

def generator(
    z,           # generator-т авах оролтын тензор
    out_dim,     # generator-ийн гаралтын утгын хэлбэр
    n_units=128, # далд давхарга дахь нуугдмал юнитуудын тоо
    reuse=False, # tf.variable_scope доторхи хувьсагчийг дахин хэрэглэх эсэх
    alpha=0.01   # leaky ReLU-ийн авах утга
    ):
    with tf.variable_scope('generator', reuse=reuse):
        # Далд давхарга
        h1 = tf.layers.dense(z, n_units, activation=None)
        # Leaky ReLU
        h1 = tf.maximum(h1, alpha*h1) 
        # Logit болон tanh гаралт
        logits = tf.layers.dense(h1, out_dim, activation=None)
        out = tf.nn.tanh(logits)
        return out, logits

def discriminator(
    x,           # disriminator-т авах оролтын тензор
    n_units=128, # далд давхаргын юнитуудын тоо
    reuse=False, # дахин хэрэглэх эсэх
    alpha=0.01   # leaky ReLU-д авах утга
    ):
    with tf.variable_scope('discriminator', reuse=reuse):
        # Далд давхарга
        h1 = tf.layers.dense(x, n_units, activation=None)
        # Leaky ReLU
        h1 = tf.maximum(h1, alpha*h1)
        # Logit-ууд болон sigmoid
        logits = tf.layers.dense(h1, 1, activation=None)
        out = tf.nn.sigmoid(logits)
        return out, logits

# discriminator-т орж ирэх оролтын зургийн хэмжээ
input_size    = 784  # 28x28 хэмжээтэй MNIST зургийг нэг мөрөнд оруулсан байдал
z_size        = 100  # generator-т орж ирэх latent векторын хэмжээ
g_hidden_size = 128  # generator доторхи далд давхаргуудын хэмжээ
d_hidden_size = 128  # discriminator доторхи далд давхаргуудын хэмжээ
alpha         = 0.01 # Leak factor for leaky ReLU
smooth        = 0.1  # Label smoothing 

tf.reset_default_graph()
# Графын оролтын laceholder-ууд
input_real, input_z = model_inputs(input_size, z_size)

# Generator сүлжээ
# g_model нь generator-ийн гаралт
g_model, g_logits   = generator(input_z, input_size, g_hidden_size, reuse=False,  alpha=alpha)


# Disriminator сүлжээ
d_model_real, d_logits_real = discriminator(input_real, d_hidden_size, reuse=False, alpha=alpha)
d_model_fake, d_logits_fake = discriminator(g_model, d_hidden_size, reuse=True, alpha=alpha)


# loss-ууд тооцох
d_labels_real = tf.ones_like(d_logits_real) * (1 - smooth)
d_labels_fake = tf.zeros_like(d_logits_fake)

d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(labels=d_labels_real, logits=d_logits_real)
d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(labels=d_labels_fake, logits=d_logits_fake)

d_loss = tf.reduce_mean(d_loss_real + d_loss_fake)

g_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(d_logits_fake), 
        logits=d_logits_fake))

# Optimizer
learning_rate = 0.002

# Хувьсагчдыг trainable_variables-ээс аваад G болон D хэсгүүдэд хувааж авах
t_vars = tf.trainable_variables()
g_vars = [var for var in t_vars if var.name.startswith("generator")]
d_vars = [var for var in t_vars if var.name.startswith("discriminator")]

d_train_opt = tf.train.AdamOptimizer().minimize(d_loss, var_list=d_vars)
g_train_opt = tf.train.AdamOptimizer().minimize(g_loss, var_list=g_vars)


def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(7,7), nrows=4, ncols=4, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch][0]):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28,28)), cmap='Greys_r')
    return fig, axes
    
if args.show:
	saver = tf.train.Saver(var_list=g_vars)
	with tf.Session() as sess:
	    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
	    for i in range(3):
	        sample_z = np.random.uniform(-1, 1, size=(16, z_size))
	        gen_samples = sess.run(
	                   generator(input_z, input_size, reuse=True),
	                   feed_dict={input_z: sample_z})
	        _ = view_samples(0, [gen_samples])
	        plt.show()
	sys.exit()


batch_size = 100
epochs     = 500
samples    = []
losses     = []
saver      = tf.train.Saver(var_list = g_vars)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for ii in range(mnist.train.num_examples//batch_size):
            batch        = mnist.train.next_batch(batch_size)
            
            # Зурагнууд аваад reshape rescale хийгээд D рүү дамжуулах
            batch_images = batch[0].reshape((batch_size, 784))
            batch_images = batch_images*2 - 1
            
            # G-д зориулж random noise дээжлэн авах
            batch_z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            
            # optimizer-уудыг ажиллуулах
            _ = sess.run(d_train_opt, feed_dict={input_real: batch_images, input_z: batch_z})
            _ = sess.run(g_train_opt, feed_dict={input_z: batch_z})
        
        # epoch болгоны дараа loss утгуудыг авч хэвлэж харах
        train_loss_d = sess.run(d_loss, {input_z: batch_z, input_real: batch_images})
        train_loss_g = g_loss.eval({input_z: batch_z})
            
        print("Epoch {}/{}...".format(e+1, epochs),
              "Discriminator Loss: {:.4f}...".format(train_loss_d),
              "Generator Loss: {:.4f}".format(train_loss_g))    
        # Сургасны дараа харах зорилгоор loss-уудыг хадгалж авах
        losses.append((train_loss_d, train_loss_g))
        
        # Сургасны дараа хадгалах зорилгоор generator-оос дээж авч хадгалах 
        sample_z = np.random.uniform(-1, 1, size=(16, z_size))
        gen_samples = sess.run(
                       generator(input_z, input_size, reuse=True),
                       feed_dict={input_z: sample_z})
        samples.append(gen_samples)
        saver.save(sess, './checkpoints/generator.ckpt')

# Сургалтаар үүсгэсэн дээжүүийг хадгалж авах
with open('train_samples.pkl', 'wb') as f:
    pkl.dump(samples, f)


