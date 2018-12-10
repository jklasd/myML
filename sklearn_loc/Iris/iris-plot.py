from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt

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

    le = preprocessing.LabelEncoder()
    le.fit(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    y = le.transform(y)

    x1 = x[:,:1]
    x2 = x[:, 1:2]
    col = ['black','blue','red']
    marks = ['o', 'v', '+']
    # print(len(y),'=',len(x1),'=',len(x2))
    plt.grid()
    for k in range(len(y)):
        plt.plot(x1[k], x2[k],marker=marks[y[k]],color=col[y[k]])
    plt.show()
    x3 = x[:, 2:3]
    x4 = x[:, -1]

    # print(x[0],'<<',x3[0],',',x4[0])
    plt.grid()
    for k in range(len(y)):
        plt.plot(x3[k], x4[k],marker=marks[y[k]],color=col[y[k]])
    plt.show()