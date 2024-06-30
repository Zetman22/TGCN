import numpy as np
import torch
from torch import nn
from .torch_nn import BasicConv, batched_index_select, act_layer
from .torch_edge import DenseDilatedKnnGraph
import torch.nn.functional as F


class MRConv1d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MRConv1d, self).__init__()
        self.nn = BasicConv([in_channels*2, out_channels], act, norm)

    def forward(self, x, edge_index):
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, dim=-1, keepdim=False)
       
        x = torch.cat([x, x_j], dim=1)
        return self.nn(x)
    
def cal_dis(idx, idx_max):
    B, C, T = idx_max.shape
    B, T, k = idx.shape
    idx = idx.unsqueeze(1).expand(B, C, T, k)
    valmax = torch.gather(idx, 3, idx_max.unsqueeze(-1))
    return valmax.float().mean()


class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GraphConv2d, self).__init__()
        self.gconv = MRConv1d(in_channels, out_channels, act, norm, bias)

    def forward(self, x, edge_index):
        return self.gconv(x, edge_index)


class DyGraphConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, act='relu',
                 norm=None, bias=True):
        super(DyGraphConv2d, self).__init__(in_channels, out_channels, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation)

    def forward(self, x):
        edge_index = self.dilated_knn_graph(x)
        x = super(DyGraphConv2d, self).forward(x, edge_index)
        return x


class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, act='relu', norm=None,
                 bias=True):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.fc1 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm1d(in_channels),
        )
        self.graph_conv = DyGraphConv2d(in_channels, in_channels*2, kernel_size, dilation,
                              act, norm, bias)
        self.fc2 = nn.Sequential(
            nn.Conv1d(in_channels*2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm1d(in_channels)
        )

    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)
        x = x + _tmp
        return x

