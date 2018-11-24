import numpy as np
import pandas as pd
from sklearn import preprocessing

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
    # print(iris_type(y))

    le = preprocessing.LabelEncoder()
    le.fit(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    y = le.transform(y)
    # print(y)
    #线性模型公式：
    # y = XW
    print("==========================begin==========================")