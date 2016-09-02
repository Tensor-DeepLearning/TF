#! /usr/bin/python3
#encoding=utf-8

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

import tensorflow as tf

#避免建立模型时反复初始化，定义两个函数用于初始化
#初始化权重
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
#初始化偏倚
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#卷积，1步长（stride size），0边距（padding size）
#注意：padding='SAME'表示在图像周围填0，保证输入和输出大小一致
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#池化，2x2大小的模板做max pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder("float", shape=[None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])
#卷积的权重张量形状是[5, 5, 1, 32]，前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目。 对于每一个输出通道都有一个对应的偏置量
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
#第一层卷积和池化
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
#第二层卷积和池化
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
#卷积网络结束时图片大小为7*7*64
#加入一1024个神经元的全连接层
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#减少过拟合，输出层之前增加dropout
keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#输出层，softmax分类器
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
y_ = tf.placeholder('float', [None, 10])

#交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
tf.scalar_summary('cross entropy', cross_entropy) #统计交叉熵
#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) #使用ADAM优化器

summary_op = tf.merge_all_summaries()

sess = tf.Session()
summary_writer = tf.train.SummaryWriter('logs/', sess.graph)
sess.run(tf.initialize_all_variables())

for i in range(5000):
    batch = mnist.train.next_batch(50)
    feed_dict = {x:batch[0], y_:batch[1], keep_prob:0.5}
    sess.run(train_step, feed_dict=feed_dict)
    if i % 100 == 0: #每100次迭代做一次图标汇总
        print('Train step', i)
        summary_str = sess.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, i)
        summary_writer.flush()


correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
#打印正确率
print('The accuracy rate is', sess.run(accuracy,feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))
