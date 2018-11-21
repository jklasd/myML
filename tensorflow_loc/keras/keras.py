import tensorflow as tf
from tensorflow import keras
import tensorflow_loc.readMNIST.input_data_old as input_data

import numpy as np
import matplotlib.pyplot as plt
import gzip

def readLables(path):
    with gzip.open(path, 'rb') as lbpath:
        lables = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    return lables

def readImages(path):
    with gzip.open(path, 'rb') as imgpath:
        images = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(60000, 28, 28)
    return images

def readDemoImag():
    with gzip.open('../testdata/train-images-idx3-ubyte.gz', 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(60000, 28, 28)
    for i in range(10,20):
        plt.figure()
        plt.imshow(x_train[i])
        plt.colorbar()
        plt.grid(False)
        plt.show()
    return

def showImag(imageData):
    plt.figure()
    plt.imshow(imageData)
    plt.colorbar()
    plt.grid(False)
    plt.show()
    return

def show25():
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(train_labels[i])
    plt.show()

paths = [
      'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
      't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
  ]
if __name__ == '__main__':
    # readDemoImag()
    dirpath = '../testdata/'

    # 1、读取到数据
    with gzip.open(dirpath+paths[0], 'rb') as lbpath:
        train_labels = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(dirpath+paths[1], 'rb') as imgpath:
        train_images = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(train_labels), 28, 28)

    with gzip.open(dirpath+paths[2], 'rb') as lbpath:
        test_labels = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(dirpath+paths[3], 'rb') as imgpath:
        test_images = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(test_labels), 28, 28)

    # showImag(train_images[0])
    # print(train_labels[0])

    # show25()
    #预处理
    train_images = train_images / 255.0

    test_images = test_images / 255.0

    # 2、构建模型
    # a、设置层
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    # b、编译模型
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # c、训练模型
    model.fit(train_images, train_labels, epochs=5)

    # d、评估
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)

    # e、预测
    predictions = model.predict(test_images)
    # showImag(test_images[0])
    # print()
    #
    # showImag(test_images[5])
    # print(np.argmax(predictions[5]))
    #
    # showImag(test_images[13])
    # print(np.argmax(predictions[13]))

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_images[i], cmap=plt.cm.binary)
        plt.xlabel(np.argmax(predictions[i]))
    plt.show()