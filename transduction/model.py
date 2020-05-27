'''
Model for transduction task

----
ACM CHIL 2020 paper

Neal G. Ravindra, 200228
'''

import os,sys,pickle,time,random,glob
import numpy as np
import pandas as pd

from typing import List
import copy
import os.path as osp
import torch
import torch.utils.data
from torch_sparse import SparseTensor, cat
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from sklearn.model_selection import train_test_split


################################################################################
# hyperparams
################################################################################
LR = 0.001 # learning rate
WeightDecay=5e-4
fastmode = True # if `fastmode=False`, report validation
nHiddenUnits = 8
nHeads = 8 # number of attention heads
nEpochs = 5000
dropout = 0.4 # applied to all GAT layers
alpha = 0.2 # alpha for leaky_relu
patience = 100 # epochs to beat
clip = None # set `clip=1` to turn on gradient clipping
rs=random.randint(1,1000000) # random_seed
n_nodes=d.y.unique().size()[0] # set manually if d not loaded
################################################################################

## model
class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.gat1 = GATConv(d.num_node_features, out_channels=nHiddenUnits,
                            heads=nHeads, concat=True, negative_slope=alpha,
                            dropout=dropout, bias=True)
        self.gat2 = GATConv(nHiddenUnits*nHeads, n_nodes,
                            heads=nHeads, concat=False, negative_slope=alpha,
                            dropout=dropout, bias=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)
