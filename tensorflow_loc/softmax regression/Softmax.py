import tensorflow_loc.readMNIST.input_data as input_data
import tensorflow as tf
import os

if __name__ == '__main__':
    dirpath = '../testdata/'
    if os.path.exists(dirpath):
        mnist = input_data.read_data_sets(dirpath,one_hot=True);

        image_size = 28 * 28;
        num_type = 10;

        x = tf.placeholder("float", [None, image_size])
        W = tf.Variable(tf.zeros([image_size, num_type]))
        b = tf.Variable(tf.zeros([num_type]))

        y = tf.nn.softmax(tf.matmul(x, W) + b)
        y_ = tf.placeholder("float", [None, num_type])

        cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        for i in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # print(accuracy.eval())
        tmp = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        print(tmp)
    else:
        print("不存在")
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)