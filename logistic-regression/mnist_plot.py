from matplotlib import pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot = True)
def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest')
    return plt

batch_xs, batch_ys = mnist.test.next_batch(2)

gen_image(batch_xs[0]).show()
print(batch_ys[0])
gen_image(batch_xs[1]).show()
print(batch_ys[1])
