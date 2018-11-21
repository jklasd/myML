import numpy as np
import tensorflow as tf
from tensorflow import keras

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
if __name__ == '__main__':
    path = '../testdata/blod.data'
    data = np.loadtxt(path, delimiter=' ',skiprows=1)
    # print(data[0])
    train_data = data[200:];
    test_data = data[:200];
    x_data, y_data = np.split(train_data, (10,), axis=1)
    x_test_data, y_test_data = np.split(test_data, (10,), axis=1)

    image_size = 1 * 10
    num_type = 1*2

    x = tf.placeholder("float", [None, image_size])
    W = tf.Variable(tf.zeros([image_size, num_type]))
    b = tf.Variable(tf.zeros([num_type]))

    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder("float", [None, num_type])

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)

    # saver = tf.train.Saver()
    for i in range(1000):
        train_step.run(feed_dict={x: x_data, y_: y_data})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    tmp = accuracy.eval(feed_dict={x: x_test_data, y_: y_test_data})
    print(tmp)

    # x = tf.placeholder("float", shape=[None, image_size])
    # y_ = tf.placeholder("float", shape=[None, num_type])
    # W = tf.Variable(tf.zeros([image_size, num_type]))
    # b = tf.Variable(tf.zeros([num_type]))

    # W_conv1 = weight_variable([5, 5, 1, 32])
    # b_conv1 = bias_variable([32])
    # x_image = tf.reshape(x, [-1, 28, 28, 1])
    #
    # h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # h_pool1 = max_pool_2x2(h_conv1)
    #
    # W_conv2 = weight_variable([5, 5, 32, 64])
    # b_conv2 = bias_variable([64])
    #
    # h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # h_pool2 = max_pool_2x2(h_conv2)
    #
    # W_fc1 = weight_variable([7 * 7 * 64, 1024])
    # b_fc1 = bias_variable([1024])
    #
    # h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    # h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    #
    # keep_prob = tf.placeholder("float")
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    #
    # W_fc2 = weight_variable([1024, 10])
    # b_fc2 = bias_variable([10])
    #
    # y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    #
    # cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    # train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    #
    # correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #
    # sess = tf.InteractiveSession()
    # sess.run(tf.global_variables_initializer())
    #
    # for i in range(20000):
    #     if i % 100 == 0:
    #         train_accuracy = accuracy.eval(feed_dict={x: x_data, y_: y_data, keep_prob: 1.0})
    #         # train_accuracy = sess.run(accuracy,feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
    #         print("step %d, training accuracy %g" % (i, train_accuracy))
    #
    #     train_step.run(feed_dict={x: x_data, y_: y_data, keep_prob: 0.5})
    #     # sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    #
    # # train_accuracy2 = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    # train_accuracy2 = accuracy.eval(feed_dict={x: x_test_data, y_: y_test_data, keep_prob: 1.0})
    # print("test accuracy %g" % train_accuracy2)