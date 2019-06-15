#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from torch.utils.data import Dataset,DataLoader


print(__doc__)

import time as time
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_swiss_roll

# #############################################################################
# Generate data (swiss roll dataset)
n_samples = 1500
noise = 0.00
X, _ = make_swiss_roll(n_samples, noise)
# Make it thinner
#X[:, 1] *= .5

# #############################################################################
# Compute clustering,for data point (x,y,z) y>10 label = 1
label = X[:,1]>10

# #############################################################################
# Plot result
fig = plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -80)
for l in np.unique(label):
    ax.scatter(X[label == l, 0], X[label == l, 1], X[label == l, 2],
               color=plt.cm.jet(np.float(l) / np.max(label + 1)),
               s=20, edgecolor='k')
plt.title('labeled data')


class SwissRoll(Dataset):
    
    def __init__(self,X,label):
        self.x = X.astype(np.float32)
        self.y = np.array([label.astype(np.float32)]).T  #bollean to float 
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,idx):
        x = self.x[idx]
        y = self.y[idx]
        x = torch.from_numpy(x)
          
        y = torch.from_numpy(y)
        return (x,y)
dataset = SwissRoll(X,label)
train_loader = DataLoader(dataset,batch_size=64,shuffle=True)
    


