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
                new_y.append(0)
            else:
                new_y.append(1)
        else:
            if el[1] == 0 :
                new_y.append(2)
            else:
                new_y.append(3)
    return np.array(new_y)
class Datasets(object):
    def __init__(self,size):
        self.size = size
        self.x = []
        self.y = []
        self.n = 0

    def subsection(self,x,y):
        length = len(x)
        for k in range(int(len(x)/self.size)):
            if (k+1)*self.size == length:
                self.x.append(x[:length-self.size])
                self.y.append(y[:length-self.size])
            else:
                self.x.append(x[(k+1) * self.size:])
                self.y.append(y[(k+1) * self.size:])

    def next(self):
        if self.n < len(self.x):
            x_t,y_t = self.x[self.n], self.y[self.n]
            self.n = self.n + 1
            return x_t,y_t
        else:
            self.n = 0
            return self.x[self.n], self.y[self.n]

def oneToTwo(tmp_data):
    tmp_x = []
    for x_ in tmp_data:
        tmp_x_ = []
        for x__ in x_:
            tmp_x_.append([x__])
        tmp_x.append(np.array(tmp_x_))
    return np.array(tmp_x)
if __name__ == '__main__':
    path = '../testdata/blod.data'
    data = np.loadtxt(path, delimiter=' ',skiprows=1)
    # print(data[0])
    train_data = data[90*7:];
    test_data = data[:90*7];
    x_data, y_data = np.split(train_data, (10,), axis=1)

    # x_data = x_data/30.




    # print(tmp_x)
    x_data = oneToTwo(x_data)
    y_data = merge(y_data)

    x_test_data, y_test_data = np.split(test_data, (10,), axis=1)
    x_test_data = oneToTwo(x_test_data)
    y_test_data = merge(y_test_data)
    trait = 1 * 10
    num_type = 4

    # x = tf.placeholder("float", shape=[None, trait])
    # y_ = tf.placeholder("float", shape=[None, num_type])
    # # 10个特征和4个输出值
    # W = tf.Variable(tf.zeros([trait, num_type]))
    # b = tf.Variable(tf.zeros([num_type]))

    # 2、构建模型
    # a、设置层
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(10, 1)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(num_type, activation=tf.nn.softmax)
    ])
    # b、编译模型
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # c、训练模型
    model.fit(x_data, y_data, epochs=5)

    # d、评估
    test_loss, test_acc = model.evaluate(x_test_data, y_test_data)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)

    predictions = model.predict(x_test_data)

    print('data:',x_test_data[0],'predictions:', np.argmax(predictions[0]))
    print('data:',x_test_data[1],'predictions:', np.argmax(predictions[1]))