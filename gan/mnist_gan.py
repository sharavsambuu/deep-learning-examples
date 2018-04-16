"""
GAN, Generative Adversarial Network
MNIST шиг харагддаг зурагнуудыг үүсгэж сургах.
"""

import pickle as pkl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')

