#encoding=utf-8

import tensorflow as tf
import numpy as np

learning_rate = 0.001
iter_num = 5000
batch_size = 64
disp_step = 100

n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

W = tf.Variable(tf.random_normal([n_hidden, n_classes]), name='W')
b = tf.Variable(tf.zeros([n_classes]), name='b')

x_list = tf.unstack(value=x, axis=1) #将3维tensor分割成2维list，list中一个元素为一个seq

lstm_cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=n_hidden,
    forget_bias=1.0,
    input_size=n_input
)

output, state = tf.contrib.rnn.static_rnn(lstm_cell, x_list, dtype=tf.float32)

prediction = tf.add(tf.matmul(output[-1], W), b)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
tf.summary.scalar('cross_entropy', loss) #统计损失函数，交叉熵

opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy) #统计正确率

init = tf.global_variables_initializer()
summary_merge = tf.summary.merge_all() #对统计量做merge的op

sess = tf.Session()
summary_writer = tf.summary.FileWriter('./summaries/', sess.graph) #定义写统计，第一个参数为文件夹地址，第二个可选，我们同时将模型结构图写入

sess.run(init)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../data/MNIST_data/', one_hot=True)

test_x = mnist.test.images.reshape([-1, n_steps, n_input])
test_y = mnist.test.labels

for i in range(iter_num):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    batch_x = np.reshape(batch_x, [-1, n_steps, n_input])

    _, l, s = sess.run([opt, loss, summary_merge], feed_dict={x:batch_x, y:batch_y})
    if i % disp_step == 0: #每disp_step写入一次统计量
        summary_writer.add_summary(s, i)
        s, acc = sess.run([summary_merge, accuracy], feed_dict={x:test_x, y:test_y})
        summary_writer.add_summary(s, i)
        print('step:', '{}'.format(i), 'loss =', '{:.6f}'.format(l))
print('*****Traing compelted!*****')

test_acc = sess.run(accuracy, feed_dict={x:test_x, y:test_y})

print('The accuracy of test data is', test_acc)

sess.close()
