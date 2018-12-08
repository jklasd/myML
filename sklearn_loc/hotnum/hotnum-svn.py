import pandas as pd
import numpy as np
from sklearn import svm

if __name__ == '__main__':
    path = './hotnum.data'
    data = pd.read_csv(path, header=0)
    # print(data[0])
    train_data = data.values[90*7:]
    test_data = data.values[:90*7]

    x = train_data[:,:-3]
    y = train_data[:, -2]
    yn = train_data[:, -1]

    x_t = test_data[:, :-3]
    y_t = test_data[:, -2]
    yn_t = test_data[:, -1]

    i = 2
    print(x[i],'=',y[i],'=>',yn[i])


    model = svm.SVC()
    #
    model.fit(x,yn)
    #
    diabetes_y_pred = model.predict(x_t)
    # print(diabetes_y_pred)
    result = yn_t == diabetes_y_pred

    print('预测值:',diabetes_y_pred[:20])
    print('真实值:', yn_t[:20])
    acc = np.mean(result)
    print('准确度: %.2f%%' % (100 * acc))