#encoding=utf-8

import numpy as np
import tensorflow as tf
import os

from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../data/MNIST_data/', one_hot=True)

N = 10000
D = 784
path = './mnist_embedding/'

labels = np.argmax(mnist.test.labels, axis=1)

f = open(os.path.join(path, 'metadata.tsv'), 'w')

for i in range(N):
    f.write(str(labels[i]) + '\n')

embedding_var = tf.Variable(mnist.test.images, name='mnist_embedding')

saver = tf.train.Saver()
sess = tf.Session()

sess.run(tf.global_variables_initializer())

saver.save(sess, os.path.join(path, 'model.ckpt'))

config = projector.ProjectorConfig()

embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name

embedding.metadata_path = os.path.join(path, 'metadata.tsv')
embedding.sprite.image_path = '../data/MNIST_data/mnist_10k_sprite.png'
embedding.sprite.single_image_dim.extend([28,28])

summary_writer = tf.summary.FileWriter(path)

projector.visualize_embeddings(summary_writer, config)
