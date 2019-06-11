#!/usr/bin/env python3
# -*- coding: utf-8 -*-




print(__doc__)

import time as time
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_swiss_roll
XX = X
YY = label
XX=XX.astype(np.float32)
YY = np.array([YY.astype(np.float32)]).T
XX = torch.from_numpy(XX)
YY = torch.from_numpy(YY)

z, w, mu_z, mu_w_0, mu_w_1, sigma_z, sigma_w_0, sigma_w_1=csvae.encode(XX,YY)

z=z.detach().numpy()
w=w.detach().numpy()

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -80)
for l in np.unique(label):
    ax.scatter(w[label == l, 0], w[label == l, 1],
               color=plt.cm.jet(np.float(l) / np.max(label + 1)),
               s=20, edgecolor='k')
plt.title('labeled data')