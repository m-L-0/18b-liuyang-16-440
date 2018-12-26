from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
data = load_iris()
x=data.data
y=data.target
#先算相似度矩阵
similarity=np.zeros([150,150])
for i in range(150):
    a=x[i]
    for j in range(i+1,150):
        b=x[j]
        A=a-b
        similarity[i][j]=similarity[j][i]=np.exp(-np.matmul(A,A))        
#计算相似度矩阵的均值和标准差
similarity_mean=np.mean(similarity)#0.190
similarity_std=np.std(similarity)#0.2833



ttt=0.777
similarity[similarity<ttt]=0


#计算度数矩阵
Degree=np.zeros([150,150])
for i in range(150):
    Degree[i][i]=np.sum(similarity[i])
#计算拉普拉斯矩阵
Laplace=Degree-similarity

#用归一割
#计算非归一化对称拉普拉斯矩阵
Degree_sqrt=1/Degree
Degree_sqrt[Degree_sqrt==np.Inf]=0
toone_Laplace=np.matmul(Degree_sqrt,Laplace)
#toone_Laplace=np.matmul(Laplace,Degree_sqrt)

#求归一化对称拉普拉斯矩阵的特征值和特征向量
eigenvalue,eigenvector=np.linalg.eig(toone_Laplace)
temp=pd.Series(eigenvalue)


mineigvector=eigenvector[:,[5,6,7,9,51,101,109,100,95,110,148,144,145,146,147,149]]

'''
149    0.000000e+00
147    0.000000e+00
146    0.000000e+00
145    0.000000e+00
144    0.000000e+00
148    0.000000e+00
110    8.904646e-17
95     5.727343e-16
100    7.320959e-16
109    8.453972e-16
101    8.832556e-16
51     1.396314e-15
9      1.691650e-02
7      5.620822e-02
5      1.063135e-01
6
'''
for i in range(150):
    T=np.sqrt(np.matmul(mineigvector[i],mineigvector[i]))
    mineigvector[i]=mineigvector[i]/T
#mineigvector[]=0.1
###mineigvector[144:150]=mineigvector[138:144]
####
#####存储为文件
Clustering_data=pd.DataFrame(mineigvector)
Clustering_data.to_csv('Clustering_data.csv')
