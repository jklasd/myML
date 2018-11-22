import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.contrib.rnn as rnn

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

BATCH_SIZE=90
HIDDEN_UNITS1=30
HIDDEN_UNITS=10
TRAIN_EXAMPLES=7470
TEST_EXAMPLES=630
if __name__ == '__main__':
    path = '../testdata/blod.data'
    data = np.loadtxt(path, delimiter=' ',skiprows=1)
    # print(data[0])
    train_data = data[90 * 7:];
    test_data = data[:90 * 7];
    x_data, y_data = np.split(train_data, (10,), axis=1)
    x_test_data, y_test_data = np.split(test_data, (10,), axis=1)
    y_data = merge(y_data)

    y_test_data = merge(y_test_data)
    # --------------------------------------Define Graph---------------------------------------------------#
    graph = tf.Graph()
    with graph.as_default():

        # ------------------------------------construct LSTM------------------------------------------#
        # place hoder
        X_p = tf.placeholder(dtype=tf.float32, shape=(None, 1, 10), name="input_placeholder")
        y_p = tf.placeholder(dtype=tf.float32, shape=(None, 4), name="pred_placeholder")

        # lstm instance
        lstm_cell1 = rnn.BasicLSTMCell(num_units=1)
        lstm_cell = rnn.BasicLSTMCell(num_units=10)

        multi_lstm = rnn.MultiRNNCell(cells=[lstm_cell1, lstm_cell])

        # initialize to zero
        init_state = multi_lstm.zero_state(batch_size=BATCH_SIZE, dtype=tf.float32)

        # dynamic rnn
        outputs, states = tf.nn.dynamic_rnn(cell=multi_lstm, inputs=X_p, initial_state=init_state, dtype=tf.float32)
        # print(outputs.shape)
        h = outputs[:, -1, :]
        # print(h.shape)
        # --------------------------------------------------------------------------------------------#

        # ---------------------------------define loss and optimizer----------------------------------#
        cross_loss = tf.losses.softmax_cross_entropy(onehot_labels=y_p, logits=h)
        # print(loss.shape)

        correct_prediction = tf.equal(tf.argmax(h, 1), tf.argmax(y_p, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        optimizer = tf.train.AdamOptimizer(0.001).minimize(loss=cross_loss)

        init = tf.global_variables_initializer()

    # -------------------------------------------Define Session---------------------------------------#
    with tf.Session(graph=graph) as sess:
        sess.run(init)
        for epoch in range(1, 5):
            # results = np.zeros(shape=(TEST_EXAMPLES, 10))
            train_losses = []
            accus = []
            # test_losses=[]
            print("epoch:", epoch)
            for j in range(TRAIN_EXAMPLES // BATCH_SIZE):
                _, train_loss, accu = sess.run(
                    fetches=(optimizer, cross_loss, accuracy),
                    feed_dict={
                        X_p: x_data[j * BATCH_SIZE:(j + 1) * BATCH_SIZE],
                        y_p: y_data[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
                    }
                )
                train_losses.append(train_loss)
                accus.append(accu)
            print("average training loss:", sum(train_losses) / len(train_losses))
            print("accuracy:", sum(accus) / len(accus))
