from sklearn import gaussian_process,preprocessing
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
    print("==================GAUSSIAN Model===================")

    gs = gaussian_process.GaussianProcessClassifier()
    gs.fit(x,y)
    # print(svc)
    predict_value = gs.predict(x_test)
    print(predict_value)
    result = predict_value == y_test
    success = np.mean(result)
    print('准确度: %.2f%%' % (100 * success))