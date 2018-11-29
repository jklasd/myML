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
    x = df.values[:130, :-1]
    x_test = df.values[129:, :-1]
    y = df.values[:130, -1]
    y_test = df.values[129:, -1]
    print('length=',len(y_test),';y_test=',y_test)

    le = preprocessing.LabelEncoder()
    le.fit(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    y = le.transform(y)
    y_test = le.transform(y_test)
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
    diabetes_y_pred = model.predict(x_test)
    # lang = np.random.randint(0,150,30)#生成 0~150 ，30个随机整数
    # diabetes_y_pred = model.predict(x[lang])
    print('原值：',y_test)
    print('预测值：',diabetes_y_pred)
    result = y_test == diabetes_y_pred
    acc = np.mean(result)
    print('准确度: %.2f%%' % (100 * acc))