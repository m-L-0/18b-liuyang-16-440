# -*- coding: utf-8 -*-

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=150, centers=[(1,14),(7.5,1),(14,14)], n_features=2,cluster_std=[3.0,3.0,3.0],
                                          random_state=0,shuffle=False)
x=np.array(X)
pos={}
for i in range(1,151):
    pos[i]=x[i-1]


data=pd.read_csv('similarity.csv',index_col=0)
data=np.array(data)
G = nx.Graph()
for i in range(1,151):
    G.add_node(i)
for i in range(1,150):
    for j  in range(i,150):
        if data[i][j] != 0:
            print(i,j)
            G.add_edge(i,j,weight=data[i][j])



nodes1=[i for i in range(1,51)]
nodes2=[i for i in range(51,101)]
nodes3=[i for i in range(101,151)]

edge1 = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] >= 0.9274]
edge2 = [(u, v) for (u, v, d) in G.edges(data=True) if (d['weight'] < 0.9274) & (d['weight'] > 0.8)]
edge3 = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.8]


# nodes
nx.draw_networkx_nodes(G, pos, node_size=20,nodelist=nodes1,node_shape='s')
nx.draw_networkx_nodes(G, pos, node_size=20,nodelist=nodes2,node_shape='o')
nx.draw_networkx_nodes(G, pos, node_size=20,nodelist=nodes3,node_shape='>')

nx.draw_networkx_edges(G, pos, edgelist=edge1,alpha=0.8,
                       width=2)
nx.draw_networkx_edges(G, pos, edgelist=edge2,
                       width=1, alpha=0.5, edge_color='g')
nx.draw_networkx_edges(G, pos, edgelist=edge3,
                       width=1, alpha=0.5, edge_color='b', style='dashed')
#plt.axis('off')
plt.figure(50)
plt.show()






