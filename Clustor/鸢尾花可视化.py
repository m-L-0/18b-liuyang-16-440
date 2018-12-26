# -*- coding: utf-8 -*-



import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris = load_iris()

X = iris.data
y = iris.target
X=pd.read_csv('Clustering_data.csv',index_col=0)

'''
kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
X_kpca = kpca.fit_transform(X)
X_back = kpca.inverse_transform(X_kpca)
'''
n_components = 2
pca = PCA(n_components=n_components)
X_transformed = pca.fit_transform(X)

colors = ['navy', 'turquoise', 'darkorange']
title="PCA"
plt.figure(figsize=(8, 8))
for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
    plt.scatter(X_transformed[y == i, 0], X_transformed[y == i, 1],
                color=color, lw=2, label=target_name)
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title(title + " of iris dataset")
#plt.axis([-4, 4, -1.5, 1.5])
plt.show()
d=pd.DataFrame(X_transformed)
d.to_csv('PCA-data.csv')