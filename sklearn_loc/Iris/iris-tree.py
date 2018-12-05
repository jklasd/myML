from sklearn import tree,preprocessing
import pandas as pd
import numpy as np

if __name__ == "__main__":
    path = u'8.iris.data'  # 数据文件路径
    df = pd.read_csv(path, header=0)
    x = df.values[:130, :-1]
    x_test = df.values[129:, :-1]
    y = df.values[:130, -1]
    y_test = df.values[129:, -1]
    print('length=', len(y_test), ';y_test=', y_test)

    le = preprocessing.LabelEncoder()
    le.fit(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    y = le.transform(y)
    y_test = le.transform(y_test)

    print("=====================Tree Model========================")
    model = tree.DecisionTreeClassifier()
    model.fit(x,y)

    predict_value = model.predict(x_test)
    print("真实值：",y_test)
    print("预测值：",predict_value)

    success = np.mean(predict_value==y_test)
    print("命中：%.2f%%"%(success*100))