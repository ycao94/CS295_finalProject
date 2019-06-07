# -*- coding: utf-8 -*-
import os

net_config = {
    'in_features1': 3, # x1, x2, x3
    'in_layer_num1': 3, # number of layers
    'in_layers1': [1000, 1000, 1000], # num of nodes for each layer 
    'in_features2': 4, # x1, x2, x3, y
    'in_layer_num2': 3, # number of layers
    'in_layers2': [1000, 1000, 1000], # num of nodes for each layer 
    'out_features': 4, # z1, z2, w1, w2
    'out_layer_num': 3, # number of layers
    'out_layers': [1000, 1000, 1000], # num of nodes for each layer 
    'z_dim': 2,
    'w_dim': 2,
    'MU_1': 0,
    'MU_2_0': 0,
    'MU_2_1': 3,
    'SIG_1': [1, 1],
    'SIG_2_0': [0.1, 0.1],
    'SIG_2_1': [1, 1]
    }

train_config = {
        'epoches': 1000
        }