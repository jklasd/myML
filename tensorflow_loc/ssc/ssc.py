import numpy as np
import tensorflow as tf
from tensorflow import keras

if __name__ == '__main__':
    path = '../testdata/blod.data'
    data = np.loadtxt(path, delimiter=' ',skiprows=1)
    # print(data[0])
    train_data = data[200:];
    test_data = data[:200];
    x_data, y_data = np.split(train_data, (10,), axis=1)
    x_test_data, y_test_data = np.split(test_data, (10,), axis=1)

    # print(len(x_data))
    # print(len(x_test_data))
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # c、训练模型
    model.fit(x_data, y_data, epochs=5)

    test_loss, test_acc = model.evaluate(x_test_data, y_test_data)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)