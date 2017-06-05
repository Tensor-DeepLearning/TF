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

#获得每张图片的label
labels = np.argmax(mnist.test.labels, axis=1)
#新建metadata文件，必须为tsv格式
f = open(os.path.join(path, 'metadata.tsv'), 'w')

#将每张图片的label输出为metadata信息
#格式为
#[标题，当只有一列label时省去]label_0 \t label_1 \t ... \label_m
#[内容] ...
for i in range(N):
    f.write(str(labels[i]) + '\n')

#创建变量，用来表示样本数据，一共10000个样本，每个样本784维
embedding_var = tf.Variable(mnist.test.images, name='mnist_embedding')

saver = tf.train.Saver()
sess = tf.Session()

sess.run(tf.global_variables_initializer())
#保存样本变量
saver.save(sess, os.path.join(path, 'model.ckpt'))

config = projector.ProjectorConfig()

embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
#将元数据路径指定为刚才保存的文件
embedding.metadata_path = os.path.join(path, 'metadata.tsv')
#指定sprite image路径
embedding.sprite.image_path = '../data/MNIST_data/mnist_10k_sprite.png'
#设置单张图片大小
embedding.sprite.single_image_dim.extend([28,28])

summary_writer = tf.summary.FileWriter(path)
#保存embedding
projector.visualize_embeddings(summary_writer, config)
