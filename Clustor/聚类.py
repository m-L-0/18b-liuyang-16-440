import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
#随机产生三个0-150的整数，返回列表
def randomindex():
    return [random.randint(51,150) for i in range(3)]

#穿入质心，返回分类
def cluster(conter):
    Distance=[]
    for i in range(150):
        temp=[]
        for j in range(3):
            a=np.subtract(data.iloc[i],conter.iloc[j])
            temp.append(np.matmul(a,a))
        Distance.append(temp)
    #分簇
    cluster=[]
    for i in range(150):
        cluster.append(Distance[i].index(min(Distance[i])))
    return np.array(cluster),Distance
#更新质心
def ChangeCenter(cluster):
    conter_next=np.zeros([3,2])
    conter_next[0]=np.sum(data[cluster==0])
    conter_next[1]=np.sum(data[cluster==1])
    conter_next[2]= np.sum(data[cluster==2])
    conter_next[0]/=np.sum(cluster==0) if np.sum(cluster==0)!=0 else 1
    conter_next[1]/=np.sum(cluster==1) if np.sum(cluster==1)!=0 else 1
    conter_next[2]/=np.sum(cluster==2) if np.sum(cluster==2)!=0 else 1
    return pd.DataFrame(conter_next)

#比较质心是否相等
data=pd.read_csv('PCA-data.csv',index_col=0)
index=randomindex()
#随机质心
conter=data.iloc[index]
#分类


while True:
    cluster_frist,D1=cluster(conter)
    conter_next=ChangeCenter(cluster_frist)
    cluster_second,D2=cluster(conter_next)
    conter=conter_next
    if np.sum(cluster_frist==cluster_second)==150:
        break
plt.figure(1)
plt.clf()
color=['r','b','g']
for i,c in zip(range(3),color):
    labels=cluster_frist==i
    plt.scatter(data.iloc[labels,0], data.iloc[labels,1], marker='o',c=c)
plt.show()

    
