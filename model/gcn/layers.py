import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):

    def __init__(self, input_size, output_size, bias=True,indep_weights=True):
        super(GraphConvolution, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.indep_weights = indep_weights

        self.weight = Parameter(torch.FloatTensor(input_size, output_size))

        if self.indep_weights:
            self.weight_global = Parameter(torch.FloatTensor(input_size, output_size))
            self.weight_leaf = Parameter(torch.FloatTensor(input_size, output_size))
            self.weight_or = Parameter(torch.FloatTensor(input_size, output_size))
            self.weight_and = Parameter(torch.FloatTensor(input_size, output_size))
            self.weight_not = Parameter(torch.FloatTensor(input_size, output_size))
        else:
            self.register_parameter('weight_global', None)
            self.register_parameter('weight_leaf', None)
            self.register_parameter('weight_or', None)
            self.register_parameter('weight_and', None)
            self.register_parameter('weight_not', None)

        if bias:
            self.bias = Parameter(torch.FloatTensor(output_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        if self.indep_weights:
            self.weight_global.data.uniform_(-stdv, stdv)
            self.weight_leaf.data.uniform_(-stdv, stdv)
            self.weight_or.data.uniform_(-stdv, stdv)
            self.weight_and.data.uniform_(-stdv, stdv)
            self.weight_not.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj,labels):
        # if self.indep_weights:
        #     # global node
        #     support = torch.mm(input[0].unsqueeze(0), self.weight_global)
        #     for i in range(1,len(labels)):
        #         if labels[i] == 1: # Leaf
        #             temp = torch.mm(input[i].unsqueeze(0), self.weight_leaf)
        #         elif labels[i] == 2: # Or
        #             temp = torch.mm(input[i].unsqueeze(0), self.weight_or)
        #         elif labels[i] == 3: # And
        #             temp = torch.mm(input[i].unsqueeze(0), self.weight_and)
        #         elif labels[i] == 4: # Not
        #             temp = torch.mm(input[i].unsqueeze(0), self.weight_not)
        #
        #         support = torch.cat((support,temp),0)
        # else:
        #     support = torch.mm(input, self.weight)
        if self.indep_weights:
            support = None
            for i in range(len(labels)):
                if labels[i] == 0: # global node
                    temp = torch.mm(input[i].unsqueeze(0), self.weight_global)
                elif labels[i] == 1: #leaf node
                    temp = torch.mm(input[i].unsqueeze(0), self.weight_leaf)
                elif labels[i] == 2: # OR
                    temp = torch.mm(input[i].unsqueeze(0), self.weight_or)
                elif labels[i] == 3: # and
                    temp = torch.mm(input[i].unsqueeze(0), self.weight_and)
                elif labels[i] == 4: # not
                    temp = torch.mm(input[i].unsqueeze(0), self.weight_not)

                if support is None:
                    support = temp
                else:
                    support = torch.cat((support,temp),0)
        else:
            support = torch.mm(input, self.weight)
        # output = torch.spmm(adj, support)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
