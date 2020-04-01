import sys
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import torch


class GCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0, indep_weights=True):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(input_size, hidden_size, indep_weights=indep_weights)
        self.gc2 = GraphConvolution(hidden_size, hidden_size, indep_weights=indep_weights)
        self.gc3 = GraphConvolution(hidden_size, output_size, indep_weights=indep_weights)
        self.dropout = dropout

    def forward(self, x, adj, labels):
        x = F.relu(self.gc1(x, adj, labels))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.gc2(x, adj, labels))
        x = F.dropout(x, self.dropout)
        x = self.gc3(x, adj, labels)

        return x


class MLP(nn.Module):
    def __init__(self, input_size=200, hidden_size=150, output_size=2, dropout=0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = dropout

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.dropout(out, self.dropout)
        out = self.fc2(out)
        return out
