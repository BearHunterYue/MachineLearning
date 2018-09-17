import input_data # 自己写的库。用于以特殊的形式读入mnist数据
import tensorflow as tf

# 读入数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 建立模型
# 训练模型的自变量
x = tf.placeholder("float", [None, 784])

#要训练的参数
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# 预测值
y = tf.nn.softmax(tf.matmul(x,W) + b)

#优化的对象
y_ = tf.placeholder("float", [None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# 使用优化器优化优化对象
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 初始化
init = tf.global_variables_initializer()

#启动模型
sess = tf.Session()
sess.run(init)

#训练模型100次
for i in range(1001):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  if i % 50 == 0:
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
