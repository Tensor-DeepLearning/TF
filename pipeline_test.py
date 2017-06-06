#encoding=utf-8

import tensorflow as tf
import numpy as np

#生成三个样本文件，每个文件包含5列，假设前4列为特征，最后1列为标签
data = np.zeros([20,5])
np.savetxt('file0.csv', data, fmt='%d', delimiter=',')
data += 1
np.savetxt('file1.csv', data, fmt='%d', delimiter=',')
data += 1
np.savetxt('file2.csv', data, fmt='%d', delimiter=',')

#定义FilenameQueue
filename_queue = tf.train.string_input_producer(["file%d.csv"%i for i in range(3)], num_epochs=6)

#定义ExampleQueue
example_queue = tf.RandomShuffleQueue(
    capacity=1000,
    min_after_dequeue=0,
    dtypes=[tf.int32,tf.int32],
    shapes=[[4],[1]]
)

#读取CSV文件，每次读一行
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

#对一行数据进行解码
record_defaults = [[1], [1], [1], [1], [1]]
col1, col2, col3, col4, col5 = tf.decode_csv(
    value, record_defaults=record_defaults)
features = tf.stack([col1, col2, col3, col4])

#将特征和标签push进ExampleQueue
enq_op = example_queue.enqueue([features, [col5]])

#使用QueueRunner创建两个进程加载数据到ExampleQueue
qr = tf.train.QueueRunner(example_queue, [enq_op]*2)
#使用此方法方便后面tf.train.start_queue_runner统一开始进程
tf.train.add_queue_runner(qr)

xs = example_queue.dequeue()

with tf.Session() as sess:
    sess.run(tf.initialize_local_variables()) #必须加上这句话，否则报错！
    coord = tf.train.Coordinator()
#开始所有进程
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        while not coord.should_stop():
            x = sess.run(xs)
            print(x)
    except tf.errors.OutOfRangeError:
        print('Done training -- epch limit reached')
    finally:
        coord.request_stop()


    coord.request_stop()
    coord.join(threads)
