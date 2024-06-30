import math
import torch
from torch import nn
import torch.nn.functional as F


def pairwise_distance(x):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        x_inner = -2*torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)

    
def dense_knn_matrix(x, k=16): 
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        max_pos_value = torch.finfo(x.dtype).max
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        dist = pairwise_distance(x.detach())

        # causal distance matrix implementation
        l_b = 90 # number of frames to process (1 + K-1 look back)  
        assert(l_b >= k)

        mask_u = torch.ones(n_points, n_points, device=x.device).triu_(1).bool()
        mask_d = ~torch.ones(n_points, n_points, device=x.device).triu_(-l_b+1).bool()
        dist.masked_fill_(mask_u, max_pos_value)
        dist.masked_fill_(mask_d, max_pos_value)

        _, nn_idx = torch.topk(-dist, k=k) # b, n, k
        temp_idx = nn_idx[:, :l_b, 0].clone().unsqueeze(-1)
        for j in range(l_b):
            nn_idx[:, j, j+1:] = temp_idx[:, j]

        ######
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


class DenseDilated(nn.Module):
    """
    Find dilated neighbor from neighbor list
    edge_index: (2, batch_size, num_points, k)
    """
    def __init__(self, k=9, dilation=1):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.k = k
        self.l_b = 90

    def forward(self, edge_index):
        _edge_index = edge_index[:, :, :, ::self.dilation]
        _edge_index[:, :, :self.l_b] = edge_index[:, :, :self.l_b, :self.k]
        return _edge_index


class DenseDilatedKnnGraph(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """
    def __init__(self, k=9, dilation=1):
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.k = k
        self._dilated = DenseDilated(k, dilation)

    def forward(self, x):
        #### normalize
        x = F.normalize(x, p=2.0, dim=1)
        ####
        edge_index = dense_knn_matrix(x, self.k * self.dilation)
        return self._dilated(edge_index)
    