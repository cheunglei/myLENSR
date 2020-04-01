import sys
import torch.nn as nn
import torch.nn.functional as F
from layers import ChildSumTreeLSTM
import torch


class MLP(nn.Module):
    def __init__(self, ninput=200, nhidden=150, nclass=2, dropout=0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(ninput, nhidden)
        self.fc2 = nn.Linear(nhidden, nclass)
        self.dropout = dropout

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.dropout(out, self.dropout)
        out = self.fc2(out)
        return out

class TreeLSTM(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0):
        super(TreeLSTM, self).__init__()
        self.tlstm = ChildSumTreeLSTM(nfeat,nhid)
        self.ln = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, tree,features):
        c, h = self.tlstm(tree,features)
        x = F.relu(h)
        x = F.dropout(x, self.dropout)
        x = self.ln(x)
        return x
