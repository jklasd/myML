import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d

if __name__ == '__main__':
    path = './blod.data'
    data = np.loadtxt(path, delimiter=' ',skiprows=1)
    data = data[90*7:]
    x_data, y_data = np.split(data, (10,), axis=1)

    x = x_data[:,[1]]
    y = x_data[:, [2]]
    z = x_data[:, [3]]

    m = y_data[:,:-1]
    mv = ['+','v']
    c = ['r','black']
    ax = plt.subplot(111, projection='3d')
    for i in range(len(m)):
        ax.scatter(x[i], y[i], z[i], c=c[int(m[i][0])])
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()
    # plt.show()
