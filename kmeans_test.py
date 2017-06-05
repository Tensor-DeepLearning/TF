import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import tensorflow as tf
num_puntos = 2000
conjunto_puntos = []

#随机生成样本
for i in range(num_puntos):
    if np.random.random() > 0.5:
        conjunto_puntos.append([np.random.normal(0.0, 0.9), np.random.normal(0.0, 0.9)])
    else:
        conjunto_puntos.append([np.random.normal(3.0, 0.5), np.random.normal(1.0, 0.5)])

df = pd.DataFrame({"x": [v[0] for v in conjunto_puntos],
        "y": [v[1] for v in conjunto_puntos]})
sns.lmplot("x", "y", data=df, fit_reg=False, size=6)
plt.show()
vectors = tf.constant(conjunto_puntos)
#vectors shape(2000, 2)
k = 2
#随机取k个点作为中心
centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors),[0,0],[k,-1]))
#centroides shape(k, 2)

#扩展维度以方便计算距离
expanded_vectors = tf.expand_dims(vectors, 0)
#expanded_vectors shape(1,2000,2)
expanded_centroides = tf.expand_dims(centroides, 1)
#expanded_vectors shape(k,1,2)

#assignments为每个点到哪个中心距离最短
assignments = tf.argmin(tf.reduce_sum(tf.square(tf.sub(expanded_vectors, expanded_centroides)), 2), 0)
#(k,2000,2) -> (k,2000) -> (2000,)

#means为k个新的中心点
means = tf.concat(0, [tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where(tf.equal(assignments, c)),[1,-1])), reduction_indices=[1]) for c in range(k)])
#means shape(k,2)

update_centroides = tf.assign(centroides, means)

init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)

#迭代100次
for step in range(100):
    _, centroid_values, assignment_values = sess.run([update_centroides, centroides, assignments])
    if step % 5 == 0:
        print('step %d, new centroides is'%step, centroid_values)

#绘制聚类结果
data = {"x": [], "y": [], "cluster": []}

for i in range(len(assignment_values)):
  data["x"].append(conjunto_puntos[i][0])
  data["y"].append(conjunto_puntos[i][1])
  data["cluster"].append(assignment_values[i])

df = pd.DataFrame(data)
sns.lmplot("x", "y", data=df, fit_reg=False, size=6, hue="cluster", legend=False)
plt.show()
