import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_classification
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    X, y = make_classification(n_samples=1000, n_features=3, n_redundant=0, n_classes=3, n_informative=2,
                               n_clusters_per_class=1, class_sep=0.5, random_state=10)

    # print(X)
    # print(y)
    fig = plt.figure('data')
    ax = Axes3D(fig)

    # print(len(X[:, 0]))

    ax.scatter(X[:, 0], X[:, 1], X[:, 2],c=y, marker='o')
    # ax.plot(X[:, 0], X[:, 1], X[:, 2],'o')

    fig = plt.figure('PCA')
    pca = PCA(n_components=2)
    pca.fit(X)
    print("各主成分的方差值:" + str(pca.explained_variance_))
    print("各主成分的方差值比:" + str(pca.explained_variance_ratio_))
    X_new = pca.transform(X)
    plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y, alpha=0.5)
    fig = plt.figure('LDA')
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(X, y)
    X_new = lda.transform(X)
    plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y, alpha=0.5)
    plt.show()