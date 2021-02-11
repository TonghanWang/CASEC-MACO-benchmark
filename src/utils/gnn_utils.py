'''
Author: zengliang
Date: 2020-11-11 14:54:35
LastEditTime: 2020-11-23 11:18:27
Description: Utils for Graph Neural Networks
FilePath: /gnn_marl/gnn_marl/src/utils/gnn_utils.py
'''
import numpy as np
import pandas as pd
import os
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data
from scipy.sparse import coo_matrix
from torch_scatter import scatter


# set random number
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed(seed)  #gpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    os.environ['PYTHONHASHSEED'] = str(seed)


'''
description: The PyTorch and Pyg library implementation of GraphNorm: A Principled Approach to Accelerating Graph Neural Network Training.
reference to https://github.com/lsj2408/GraphNorm/blob/master/GraphNorm_ws/ogbg_ws/src/dgl_model/norm.py
param {type} 
return {type} 
'''
class GraphNorm(nn.Module):
    def __init__(self, norm_type, hidden_dim=300):
        super(GraphNorm, self).__init__()
        assert norm_type in ['bn', 'gn', None]
        self.norm = None
        if norm_type == 'bn':
            self.norm = pyg_nn.BatchNorm(hidden_dim)
        elif norm_type == 'gn':
            self.norm = norm_type
            self.alpha = nn.Parameter(torch.ones(hidden_dim))
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x, batch=None):
        """"""
        if self.norm is not None and type(self.norm) != str:
            return self.norm(x)
        elif self.norm is None:
            return x
            
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        graph_mean = scatter(x, batch, dim=0, reduce="mean")
        x = x - self.alpha * graph_mean[batch]
        var = scatter(x*x, batch, dim=0, reduce="mean") + 1e-6
        return self.weight * x / var[batch].sqrt() + self.bias


class GNNStack(torch.nn.Module):
    def __init__(self, num_layers=30, dropout=0.5, task='node', norm_type=None):
        super(GNNStack, self).__init__()
        self.convs = nn.ModuleList()
        assert (num_layers >= 1), 'Number of layers is not >=1'
        for l in range(num_layers-1):
            self.convs.append(self.build_max_sum())
        self.convs.append(self.build_max_sum_final())

        if not (task == 'node' or task == 'graph'):
            raise RuntimeError('Unknown task.')
        self.task = task
        self.dropout = dropout
        self.num_layers = num_layers
        
        for m in self.modules():
            self.weights_init(m)
     
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
            
    def dump(self, filename): #保存模型参数
        torch.save(self.state_dict(), filename)

    def load(self, filename, device): #map_location为改变设备（gpu0,gpu1,cpu…）
        state_dict=torch.load(open(filename,"rb"), map_location=device)
        self.load_state_dict(state_dict, strict=True)

    def build_max_sum(self):
        return MaxSumGNN()

    def build_max_sum_final(self):
        return MaxSumGNNFinal()

    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr

        if data.num_node_features == 0:
            x = torch.ones(data.num_nodes, 1).to('cuda')
        
        for i in range(self.num_layers-1):
            x = self.convs[i](x, edge_index, edge_attr)

        x = self.convs[self.num_layers-1](x, edge_index)

        if self.task == 'graph':
            x = pyg_nn.global_mean_pool(x, batch)       

        # x = self.post_mp(x)
        return x, F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)  ##negative log likelyhood，和 log_softmax配套使用


class MaxSumGNN(pyg_nn.MessagePassing):
    def __init__(self):
        super(MaxSumGNN, self).__init__(aggr='add')  #'add' aggragation
        
    def forward(self, x, edge_index, edge_attr):
        # x: [N, in_channels]  (n,|A|)
        # edge_index: [2, E]  (2,E)
        # edge_attr: [E, edge_dim]  (E,|A|,|A|)
        # 'max' here
        edge_attr_val = torch.max(edge_attr, dim=-1)[0]
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr_val)
        out = out - out.mean(dim=-1, keepdim=True)
        return out
    
    def message(self, x_i, x_j, edge_attr):
        out = x_j + edge_attr
        return out


class MaxSumGNNFinal(pyg_nn.MessagePassing):
    def __init__(self):
        super(MaxSumGNNFinal, self).__init__(aggr='add')  # 'add' aggragation

    def forward(self, x, edge_index):
        # x: [N, in_channels]  (n,|A|)
        # edge_index: [2, E]  (2,E)
        out = self.propagate(edge_index, x=x)
        return out


class GraphSage(pyg_nn.MessagePassing):
    """Non-minibatch version of GraphSage."""
    def __init__(self, in_channels, out_channels, reducer='mean', 
                 normalize_embedding=True):
        super(GraphSage, self).__init__(aggr='mean')

        self.lin = nn.Linear(in_channels, out_channels)
        self.agg_lin = nn.Linear(in_channels, out_channels)

        if normalize_embedding:
            self.normalize_emb = True

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        edge_index, _ = pyg_utils.add_self_loops(edge_index, num_nodes=num_nodes)
        self_x = self.lin(x)
        out = F.relu(self.agg_lin(x))

        return self_x + self.propagate(edge_index, size=(num_nodes, num_nodes), x=out)

    def message(self, x_j, edge_index_i, edge_index_j, size_i, size_j):  ##现在要分开节点i和j来写
        # x_j has shape [E, out_channels]   
        #        print (size_i, size_j)
        edge_index = torch.stack([edge_index_j, edge_index_i], dim=0)
        row, col = edge_index
        deg = pyg_utils.degree(row, size_j, dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.normalize_emb:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)

        return aggr_out


class GAT(pyg_nn.MessagePassing):

    def __init__(self, in_channels, out_channels, num_heads=1, concat=True,
                 dropout=0, bias=True, **kwargs):
        super(GAT, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = num_heads
        self.concat = concat 
        self.dropout = dropout

        self.lin = nn.Linear(in_channels, self.heads * out_channels)
        self.att = nn.Parameter(torch.Tensor(1, self.heads, 2 * out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(self.heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, size=None):    
        x = self.lin(x)
        return self.propagate(edge_index, size=size, x=x)

    def message(self, edge_index_i, x_i, x_j, size_i):
        #  Constructs messages to node i for each edge (j, i).
        
        x_i, x_j = x_i.view(-1, self.heads, self.out_channels), x_j.view(-1, self.heads, self.out_channels)
        x_concat = torch.cat((x_i, x_j), dim=-1)
        alpha_out = (x_concat * self.att).sum(dim=-1)
        alpha_out = F.leaky_relu(alpha_out, negative_slope=0.2)
        
        alpha = pyg_utils.softmax(alpha_out, edge_index_i, size_i)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        # Updates node embedings.
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out


def get_data(x, adj, edge_attr, use_cuda):
    # (b,n,|A|), (b,n,n), (b,E,|A|,|A|)
    dataset = []
    x_split, adj_split, edge_attr_split = torch.split(x, 1, dim=0), torch.split(adj, 1, dim=0), torch.split(edge_attr, 1, dim=0)
    for idx in range(len(x_split)):
        x_split_item = x_split[idx].squeeze(dim=0)
        adj_split_item = adj_split[idx].squeeze(dim=0)
        edge_attr_split_item = edge_attr_split[idx].squeeze(dim=0)
        if not use_cuda:
            adj_split_item = adj_split_item
            coo_A = coo_matrix(adj_split_item)
            edge_index = [coo_A.row, coo_A.col]
            edge_list = torch.tensor(edge_index, dtype=torch.long)
        else:
            adj_split_item = adj_split_item.cpu()
            coo_A = coo_matrix(adj_split_item)
            edge_index = [coo_A.row, coo_A.col]
            edge_list = torch.tensor(edge_index, dtype=torch.long).cuda()
        
        data = Data(x=x_split_item, edge_index=edge_list, edge_attr=edge_attr_split_item)
        dataset.append(data)
    return dataset