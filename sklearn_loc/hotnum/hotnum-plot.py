import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d
import pandas as pd

if __name__ == '__main__':
    path = './hotnum.data'
    data = pd.read_csv(path, header=0)
    # print(data[0])
    train_data = data.values[:90*10]

    x = train_data[:, :-3]

    print(x)