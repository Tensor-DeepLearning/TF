#encoding=utf-8

import threading
import numpy as np
import tensorflow as tf


#创建一个函数实现多线程，参数为Coordinater和线程号
def func(coord, t_id):
    count = 0
    while not coord.should_stop(): #不应该停止时计数
        print('thread ID:',t_id, 'count =', count)
        count += 1
        if(count == 5): #计到5时请求终止
            coord.request_stop()

coord = tf.train.Coordinator()

threads = [threading.Thread(target=func, args=(coord, i)) for i in range(4)]

#开始所有线程
for t in threads:
    t.start()

coord.join(threads) #等待所有线程结束
