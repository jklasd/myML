import numpy as np
import pandas as pd
from sklearn import preprocessing,linear_model

def iris_type(s):
    s = str(s,'utf-8')
    # print(type(s))
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]
if __name__ == "__main__":
    path = u'8.iris.data'  # 数据文件路径
    df = pd.read_csv(path, header=0)
    x = df.values[:, :-1]
    y = df.values[:, -1]
    # print(y)

    le = preprocessing.LabelEncoder()
    le.fit(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    y = le.transform(y)
    # print(y)
    #线性模型公式：
    # y = XW
    model = linear_model.LogisticRegression()
    model.fit(x,y)
    # print(model.coef_)
    # print(model.intercept_)
    # print(model.alpha_)
    # print(model)
    print("==========================MODEl==========================")
    # diabetes_y_pred = model.predict(diabetes_X_test)
    lang = np.random.randint(0,150,30)
    diabetes_y_pred = model.predict(x[lang])
    print('原值：',y[lang])
    print('预测值：',diabetes_y_pred)
    result = y[lang] == diabetes_y_pred
    acc = np.mean(result)
    print('准确度: %.2f%%' % (100 * acc))