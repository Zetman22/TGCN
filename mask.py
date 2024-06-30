import torch.nn as nn
import torch
import math
from complexnn import complex_cat, ComplexConv2d, ComplexConvTranspose2d, ComplexBatchNorm2d


class MaskTGCN(torch.nn.Module):

    def __init__(self, embed_class):
        super().__init__()
        self.n_filters = [64, 64, 64, 64, 64, 64, 64]
        self.kernel_size = 3
        self.stride = 2
        chin = 8 * 2

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        self.gcn = embed_class(channels=192, n_blocks=12, num_knn=30)

        for idx, n_filter in enumerate(self.n_filters):
            encode = [
                ComplexConv2d(
                    chin,
                    self.n_filters[idx],
                    kernel_size=(5, 2),
                    stride=(2, 1),
                    padding=(2, 1),
                    causal=True
                ),
                ComplexBatchNorm2d(num_features=n_filter),
                nn.PReLU(num_parameters=1)
            ]
            self.encoder.append(nn.Sequential(*encode))
            decode = [
                ComplexConvTranspose2d(
                    n_filter * 2,
                    chin,
                    kernel_size=(5, 2),
                    stride=(2, 1),
                    padding=(2, 0),
                    output_padding=(0, 0)
                ),
                ComplexBatchNorm2d(num_features=chin),
                nn.PReLU(num_parameters=1)
            ]
            self.decoder.insert(0, nn.Sequential(*decode))
            
            chin = n_filter
        
        self.encoder.append(nn.Sequential(
            ComplexConv2d(
                    self.n_filters[-1],
                    self.n_filters[-1],
                    kernel_size=(5, 2),
                    stride=(1, 1),
                    padding=(2, 1),
                    causal=True
                ),
            ComplexBatchNorm2d(self.n_filters[-1])
        ))

        self.mask_out = nn.Sequential(
            ComplexConv2d(
                16,
                2,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 2),
                causal=True
            ),
            ComplexBatchNorm2d(2)
        )
        

    def forward(self, x):
        B, C, F, T = x.shape

        # encoding
        skips = []
        for idx, encode in enumerate(self.encoder):
            x = encode(x)
            if idx < len(self.encoder) - 1:
                skips.append(x)
        _, c, f, t = x.shape
        x = x.contiguous().view(B, c*f, t)

        # gcn
        # [B, c, f, t]
        x = self.gcn(x)
        x = x.contiguous().view(B, c, f, t)

        # decoding
        for idx, decode in enumerate(self.decoder):
            skip = skips.pop(-1)
            x = complex_cat([skip, x], 1)
            x = decode(x)[..., 1:]
        
        # B, C, T, F
        x = self.mask_out(x)
        return torch.tanh(x)

