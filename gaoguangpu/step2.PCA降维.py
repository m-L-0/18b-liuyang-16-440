# -*- coding: utf-8 -*-

import pandas as pd

from sklearn.decomposition import PCA
data=pd.read_csv('x.csv',index_col=0)
test=pd.read_csv('test-x.csv',index_col=0)



pca = PCA(n_components=40)
D=pca.fit_transform(data)
T=pca.fit_transform(test)


