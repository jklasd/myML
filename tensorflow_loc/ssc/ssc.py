import numpy as np
import tensorflow as tf
from tensorflow import keras

def merge(arr):
    new_y = []
    for el in arr:
        if el[0] == 0 :
            if el[1] == 0 :
                new_y.append(np.array([1,0,0,0]))
            else:
                new_y.append(np.array([0,1,0,0]))
        else:
            if el[1] == 0 :
                new_y.append(np.array([0,0,1,0]))
            else:
                new_y.append(np.array([0,0,0,1]))
    #
    #     print(new_y[i])
    # print(np.array([0.,0.]) == arr[0])
    return new_y
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
def oneToTwo(tmp_data,w,h):
    result = []
    for row in tmp_data:
        tmp_x = []
        for i in range(w):
            tmp_x_ = []
            for j in range(h):
                tmp_x_.append(row[i*h+j])
            tmp_x.append(np.array(tmp_x_))
        result.append(np.array(tmp_x))
    return result
if __name__ == '__main__':
    path = '../testdata/blod.data'
    data = np.loadtxt(path, delimiter=' ',skiprows=1)
    # print(data[0])
    train_data = data[90 * 7:];
    test_data = data[:90 * 7];
    x_data, y_data = np.split(train_data, (10,), axis=1)
    x_test_data, y_test_data = np.split(test_data, (10,), axis=1)

    image_size = 2 * 5
    # x_data = oneToTwo(x_data,2,5)
    # x_test_data = oneToTwo(x_test_data, 2, 5)
    # for x in x_test_data:
    #     x.shape = 2,5
    print(x_data[0])

    y_data = np.split(y_data, (1,), axis=1)[0]
    y_test_data = np.split(y_test_data, (1,), axis=1)[0]
    # y_data = merge(y_data)
    # y_test_data = merge(y_test_data)
    num_type = 1

    x = tf.placeholder("float", [None, image_size])
    W = tf.Variable(tf.zeros([image_size, num_type]))
    b = tf.Variable(tf.zeros([num_type]))

    pro = tf.matmul(x, W)
    eq = pro + b

    y = tf.nn.softmax(eq)
    y_ = tf.placeholder("float", [None, num_type])

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)

    # saver = tf.train.Saver()
    # datautil = Datasets(size=90);
    # datautil.subsection(x_data,y_data)
    for i in range(5):
        # x_t,y_t = datautil.next()
        # print(x_t)
        train_step.run(feed_dict={x: x_data, y_: y_data})
    #
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    tmp = accuracy.eval(feed_dict={x: x_test_data, y_: y_test_data})
    print(tmp)
