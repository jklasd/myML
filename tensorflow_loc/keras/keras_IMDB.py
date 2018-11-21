import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras_preprocessing import sequence
import json
import matplotlib.pyplot as plt

pad_sequences = sequence.pad_sequences
make_sampling_table = sequence.make_sampling_table
skipgrams = sequence.skipgrams
# TODO(fchollet): consider making `_remove_long_seq` public.
_remove_long_seq = sequence._remove_long_seq  # pylint: disable=protected-access

def loadFile(path='imdb.npz',
              num_words=None,
              skip_top=0,
              maxlen=None,
              seed=113,
              start_char=1,
              oov_char=2,
              index_from=3,
              **kwargs):


    # if 'nb_words' in kwargs:
    #     logging.warning('The `nb_words` argument in `load_data` '
    #                     'has been renamed `num_words`.')
    #     num_words = kwargs.pop('nb_words')
    # if kwargs:
    #     raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))

    with np.load('../testdata/imdb.npz') as f:
        x_train, labels_train = f['x_train'], f['y_train']
        x_test, labels_test = f['x_test'], f['y_test']

    np.random.seed(seed)
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    labels_train = labels_train[indices]

    indices = np.arange(len(x_test))
    np.random.shuffle(indices)
    x_test = x_test[indices]
    labels_test = labels_test[indices]

    xs = np.concatenate([x_train, x_test])
    labels = np.concatenate([labels_train, labels_test])

    if start_char is not None:
        xs = [[start_char] + [w + index_from for w in x] for x in xs]
    elif index_from:
        xs = [[w + index_from for w in x] for x in xs]

    if maxlen:
        xs, labels = _remove_long_seq(maxlen, xs, labels)
        if not xs:
            raise ValueError('After filtering for sequences shorter than maxlen=' +
                             str(maxlen) + ', no sequence was kept. '
                                           'Increase maxlen.')
    if not num_words:
        num_words = max([max(x) for x in xs])

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters:
    # 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        xs = [
            [w if (skip_top <= w < num_words) else oov_char for w in x] for x in xs
        ]
    else:
        xs = [[w for w in x if skip_top <= w < num_words] for x in xs]

    idx = len(x_train)
    x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
    x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])

    return (x_train, y_train), (x_test, y_test)

def get_word_index(path='imdb_word_index.json'):

  with open(path + 'imdb_word_index.json') as f:
    return json.load(f)

word_index = get_word_index(path='../testdata/')
# The first indices are reserved
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


if __name__ == '__main__':
    (train_data, train_labels), (test_data, test_labels) = loadFile(num_words=10000)
    # imdb = keras.datasets.imdb
    #
    # (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


    #将整数转换回字词
    # A dictionary mapping words to an integer index
    # str = decode_review(train_data[0])
    # print(str)

    #准备数据====标准化数据
    train_data = pad_sequences(train_data,
                               value=word_index["<PAD>"],
                               padding='post',
                               maxlen=256)


    test_data = pad_sequences(test_data,
                              value=word_index["<PAD>"],
                              padding='post',
                              maxlen=256)

    # print(train_data[0])
    # str = decode_review(train_data[0])
    # print(str)

    #构建模型
    vocab_size = 10000

    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    model.summary()
    #编译模型
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    #创建征集
    x_val = train_data[:vocab_size]
    partial_x_train = train_data[vocab_size:]

    y_val = train_labels[:vocab_size]
    partial_y_train = train_labels[vocab_size:]

    print(len(x_val))
    print(len(partial_x_train))

    print(len(y_val))
    print(len(partial_y_train))

    #训练模型
    history = model.fit(partial_x_train,partial_y_train,epochs=40,batch_size=512,validation_data=(x_val, y_val),verbose=1)
    # #评估模型
    results = model.evaluate(test_data, test_labels)
    print(results)

    # predictions = model.predict(test_data)
    # print(predictions[1],test_labels[1])

    # history_dict = history.history
    # history_dict.keys()
    # acc = history.history['acc']
    # val_acc = history.history['val_acc']
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    #
    # epochs = range(1, len(acc) + 1)
    #
    # # "bo" is for "blue dot"
    # plt.plot(epochs, loss, 'bo', label='Training loss')
    # # b is for "solid blue line"
    # plt.plot(epochs, val_loss, 'b', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    #
    # plt.show()