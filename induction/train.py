'''
trainer for GAT model

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


## utils
def scipysparse2torchsparse(x) :
    '''
    Input: scipy csr_matrix
    Returns: torch tensor in experimental sparse format

    REF: Code adatped from [PyTorch discussion forum](https://discuss.pytorch.org/t/better-way-to-forward-sparse-matrix/21915>)
    '''
    samples=x.shape[0]
    features=x.shape[1]
    values=x.data
    coo_data=x.tocoo()
    indices=torch.LongTensor([coo_data.row,coo_data.col]) # OR transpose list of index tuples
    t=torch.sparse.FloatTensor(indices,torch.from_numpy(values).float(),[samples,features])
    return indices,t

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def main() :
    
    ################################################################################
    # hyperparams
    ################################################################################
    pdfp = '/home/ngr4/project/scgraph/data/processed/'
    data_train_pkl = 'induction_50pData_train.pkl'
    data_val_pkl = 'induction_50pData_val.pkl'
    data_test_pkl = 'induction_50pData_test.pkl'
    replicate=sys.argv[1] # for pkl file saved

    BatchSize = 256
    NumParts = 4000 # num sub-graphs
    Device = 'cuda' # if no gpu, `Device='cpu'`
    LR = 0.001 # learning rate
    WeightDecay=5e-4
    fastmode = False # if `fastmode=False`, report validation
    nHiddenUnits = 8
    nHeads = 8 # number of attention heads
    nEpochs = 5000
    dropout = 0.4 # applied to all GAT layers
    alpha = 0.2 # alpha for leaky_relu
    patience = 100 # epochs to beat
    clip = None # set `clip=1` to turn on gradient clipping
    rs=random.randint(1,1000000) # random_seed
    ################################################################################

    ## data
    with open(os.path.join(pdfp,data_train_pkl),'rb') as f :
        datapkl = pickle.load(f)
        f.close()

    node_features = torch.from_numpy(datapkl['features'].todense()).float()
    # _,node_features = scipysparse2torchsparse(features)
    labels = torch.LongTensor(datapkl['labels'])
    edge_index,_ = scipysparse2torchsparse(datapkl['adj'])
    del datapkl

    d = Data(x=node_features, edge_index=edge_index, y=labels)
    del node_features,edge_index,labels

    cd = ClusterData(d,num_parts=NumParts)
    cl = ClusterLoader(cd,batch_size=BatchSize,shuffle=True)

    if not fastmode :
        with open(os.path.join(pdfp,data_val_pkl),'rb') as f :
            datapkl = pickle.load(f)
            f.close()

        features_val = torch.from_numpy(datapkl['features'].todense()).float()
        labels_val = torch.LongTensor(datapkl['labels'])
        edge_index_val,_ = scipysparse2torchsparse(datapkl['adj'])
        del datapkl

    ## model
    class GAT(torch.nn.Module):
        def __init__(self):
            super(GAT, self).__init__()
            self.gat1 = GATConv(d.num_node_features, out_channels=nHiddenUnits,
                                heads=nHeads, concat=True, negative_slope=alpha,
                                dropout=dropout, bias=True)
            self.gat2 = GATConv(nHiddenUnits*nHeads, d.y.unique().size()[0],
                                heads=nHeads, concat=False, negative_slope=alpha,
                                dropout=dropout, bias=True)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = self.gat1(x, edge_index)
            x = F.elu(x)
            x = self.gat2(x, edge_index)
            return F.log_softmax(x, dim=1)


    ## train
    if False :
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # don't let user make decisions
    else :
        device = torch.device(Device)

    random.seed(rs)
    np.random.seed(rs)
    torch.manual_seed(rs)
    if Device == 'cuda' :
        torch.cuda.manual_seed(rs)

    model = GAT().to(device)
    optimizer = torch.optim.Adagrad(model.parameters(),
                                    lr=LR,
                                    weight_decay=WeightDecay)

    # features, adj, labels = Variable(features), Variable(adj), Variable(labels)

    def train(epoch):
        t = time.time()
        epoch_loss = []
        epoch_acc = []
        epoch_acc_val = []
        epoch_loss_val = []

        model.train()
        for batch in cl :
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            # y_true = batch.y.to(device)
            loss = F.nll_loss(output, batch.y)
            loss.backward()
            if clip is not None :
                torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
            optimizer.step()
            epoch_loss.append(loss.item())
            epoch_acc.append(accuracy(output, batch.y).item())
        if not fastmode :
            d_val = Data(x=features_val,edge_index=edge_index_val,y=labels_val)
            d_val = d_val.to(device)
            model.eval()
            output = model(d_val)
            loss_val = F.nll_loss(output, d_val.y)
            acc_val = accuracy(output,d_val.y).item()
            print('Epoch {}\t<loss>={:.4f}\t<acc>={:.4f}\tloss_val={:.4f}\tacc_val={:.4f}\tin {:.2f}-s'.format(epoch,np.mean(epoch_loss),np.mean(epoch_acc),loss_val.item(),acc_val,time.time() - t))
            return loss_val.item()
        else :
            print('Epoch {}\t<loss>={:.4f}\t<acc>={:.4f}\tin {:.2f}-s'.format(epoch,np.mean(epoch_loss),np.mean(epoch_acc),time.time()-t))
            return np.mean(epoch_loss)


    def compute_test():
        with open(os.path.join(pdfp,data_test_pkl),'rb') as f :
            datapkl = pickle.load(f)
            f.close()

        features_test = torch.from_numpy(datapkl['features'].todense()).float()
        labels_test = torch.LongTensor(datapkl['labels'])
        edge_index_test,_ = scipysparse2torchsparse(datapkl['adj'])
        del datapkl

        d_test = Data(x=features_test,edge_index=edge_index_test,y=labels_test)
        del features_test,edge_index_test,labels_test

        model.eval()
        d_test=d_test.to(device)
        output = model(d_test)
        loss_test = F.nll_loss(output, d_test.y)
    #     loss_test = nn.BCEWithLogitsLoss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output, d_test.y).item()
        print("Test set results:",
              "\n    loss={:.4f}".format(loss_test.item()),
              "\n    accuracy={:.4f}".format(acc_test))

    ## call trainer
    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = nEpochs + 1
    best_epoch = 0
    for epoch in range(nEpochs):
        loss_values.append(train(epoch))

        torch.save(model.state_dict(), '{}-'.format(epoch)+replicate+'.pkl')
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == patience:
            break

        files = glob.glob('*-'+replicate+'.pkl')
        for file in files:
            epoch_nb = int(file.split('-'+replicate+'.pkl')[0])
            if epoch_nb < best_epoch:
                os.remove(file)

    files = glob.glob('*-'+replicate+'.pkl')
    for file in files:
        epoch_nb = int(file.split('-'+replicate+'.pkl')[0])
        if epoch_nb > best_epoch:
            os.remove(file)

    print('Optimization Finished!')
    print('Total time elapsed: {:.2f}-min'.format((time.time() - t_total)/60))

    # Restore best model
    print('Loading epoch #{}'.format(best_epoch))
    model.load_state_dict(torch.load('{}-'.format(best_epoch)+replicate+'.pkl'))

    # Testing
    compute_test()
