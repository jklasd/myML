import numpy as np
from sklearn import tree


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
    path = './blod.data'
    data = np.loadtxt(path, delimiter=' ',skiprows=1)
    # print(data[0])
    train_data = data[90*20:];
    test_data = data[:90*20];
    # test_data = test_data[90:];
    x_data, y_data = np.split(train_data, (10,), axis=1)
    # x_data = oneToTwo(x_data)
    x_test_data, y_test_data = np.split(test_data, (10,), axis=1)
    # x_test_data = oneToTwo(x_test_data)
    # y_data = np.split(y_data, (1,), axis=1)
    # y_test_data = np.split(y_test_data, (1,), axis=1)

    # print(x_data)
    # print(y_data)
    # print(x_test_data)
    # print(y_test_data.T[0])

    model = tree.DecisionTreeClassifier()
    model.fit(x_data,y_data)
    diabetes_y_pred = model.predict(x_test_data)
    # print('真实值：', y_test_data)
    # print('预测值：',diabetes_y_pred)
    result = y_test_data == diabetes_y_pred
    acc = np.mean(result)
    print('二元类准确度: %.2f%%' % (100 * acc))

    # print(y_data)
    y_data2 = y_data[:, :-1]
    # print(y_data2)
    y_test_data2 = y_test_data[:,:-1]

    model2 = tree.DecisionTreeClassifier()
    model2.fit(x_data, y_data2)
    diabetes_y_pred2 = model2.predict(x_test_data)
    # print('真实值：', y_test_data)
    # print('预测值：',diabetes_y_pred)
    result2 = y_test_data2 == diabetes_y_pred2
    acc2 = np.mean(result2)
    print('一元分类准确度: %.2f%%' % (100 * acc2))