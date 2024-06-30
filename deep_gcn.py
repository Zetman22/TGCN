import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from gcn_lib_1D import Grapher, act_layer


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu'):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv1d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm1d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv1d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm1d(out_features)
        )

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = x + shortcut
        return x


class DeepGCN_1D(nn.Module):
    def __init__(self, channels=192, n_blocks=12, num_knn=17, pos_embed=376):
        super(DeepGCN_1D, self).__init__()
        self.channels = channels
        act = 'prelu'
        norm = 'batch'
        bias = True
        self.n_blocks = n_blocks
        k = num_knn

        self.backbone = Seq(*[Seq(Grapher(self.channels, num_knn, i//4+1, act, norm,
                                            bias),
                                    FFN(self.channels, self.channels*4, act=act)
                                    ) for i in range(self.n_blocks)])

    def forward(self, x):
        
        for i in range(self.n_blocks):
            x = self.backbone[i](x)

        return x

