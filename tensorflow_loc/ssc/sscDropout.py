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

def merge(arr):
    new_y = []
    for el in arr:
        if el[0] == 0 :
            if el[1] == 0 :
                new_y.append(1)
            else:
                new_y.append(2)
        else:
            if el[1] == 0 :
                new_y.append(3)
            else:
                new_y.append(4)
    #
    #     print(new_y[i])
    # print(np.array([0.,0.]) == arr[0])
    print(new_y)
    return new_y
if __name__ == '__main__':
    path = '../testdata/blod.data'
    data = np.loadtxt(path, delimiter=' ',skiprows=1)
    # print(data[0])
    train_data = data[100:];
    test_data = data[:100];
    x_data, y_data = np.split(train_data, (10,), axis=1)

    y_data = merge(y_data)

    x_test_data, y_test_data = np.split(test_data, (10,), axis=1)

    y_test_data = merge(y_test_data)
    trait = 1 * 10
    num_type = 4

    x = tf.placeholder("float", shape=[None, trait])
    y_ = tf.placeholder("float", shape=[None, num_type])
    # 784个特征和4个输出值
    W = tf.Variable(tf.zeros([trait, num_type]))
    b = tf.Variable(tf.zeros([num_type]))
