import torch.nn as nn
import torch
import torch.nn.functional as F

from deep_gcn import DeepGCN_1D
from conv_stft import ConvSTFT, ConviSTFT
from mask import *

eps = 1e-10

class TGCN(torch.nn.Module):

    def __init__(self, win_len, hop_len, fft_len):
        super().__init__()

        self.win_len = win_len
        self.win_inc = hop_len
        self.fft_len = fft_len

        self.stft = ConvSTFT(win_len=win_len, win_inc=hop_len, fft_len=fft_len, win_type='hann', feature_type='complex')
        self.istft = ConviSTFT(win_len=win_len, win_inc=hop_len, fft_len=fft_len, win_type='hann', feature_type='complex')

        self.mask_gen = MaskTGCN(DeepGCN_1D)

    def forward(self, inputs):

        # (Batch, Sample, Channel)
        B, L, C = inputs.shape
        inputs = inputs.transpose(2, 1)
        inputs = inputs.contiguous().view(B*C, L)

        # (Batch*Channel, Sample)
        specs = self.stft(inputs)
        F, T = specs.shape[1:]
        specs = specs.contiguous().view(B, C, F, T)

        real = specs[:,:,:self.fft_len//2+1]
        imag = specs[:,:,self.fft_len//2+1:]
        cspecs = torch.cat([real,imag],1)

        ################################ Mask ################################
        mask = self.mask_gen(cspecs)

        # (Batch, Channel, Frame, Frequency)
        r_mask, i_mask = torch.chunk(mask, 2, dim=1) 
        r_mask, i_mask = r_mask.squeeze(1), i_mask.squeeze(1)
        # r_mask, i_mask = r_mask.squeeze(1), i_mask.squeeze(1)
        cmp_mask = torch.stack([r_mask, i_mask], dim=-1)

        # reference mic 0
        r_out_spec = r_mask * real[:, 0] - i_mask * imag[:, 0]
        i_out_spec = r_mask * imag[:, 0] + i_mask * real[:, 0]

        enhance = torch.cat([r_out_spec, i_out_spec], dim=1)
        # (Batch, Frame, Frequency, 2) 

        wav = self.istft(enhance)
        # (Batch, Sample)

        return wav, enhance, cmp_mask

    def compute_loss(self, mix, clean):
        enhanced_signal, enhanced_spec, cmp_mask = self(mix)

        B, t, f, _ = cmp_mask.size()
        S = self.stft(clean)
        Sr, Si = S[:, :self.fft_len//2+1], S[:, self.fft_len//2+1:]
        Y = self.stft(mix[..., 0])
        Yr, Yi = Y[:, :self.fft_len//2+1], Y[:, self.fft_len//2+1:]
        Y_pow = Yr**2 + Yi**2
        gth_mask = torch.stack([(Sr * Yr + Si * Yi) / (Y_pow + 1e-8),
                            (Si * Yr - Sr * Yi) / (Y_pow + 1e-8)], -1)
        gth_mask[gth_mask > 2] = 1
        gth_mask[gth_mask < -2] = -1
        amp_loss = F.mse_loss(gth_mask[..., 0], cmp_mask[..., 0])
        pha_loss = F.mse_loss(gth_mask[..., 1], cmp_mask[..., 1])
        mask_loss = amp_loss + pha_loss

        loss_sisnr = self.si_snr_loss(enhanced_signal, clean)
        return torch.log(mask_loss) + loss_sisnr

    def si_snr_loss(self, inf, ref):
        """si-snr loss
            :param ref: (Batch, samples)
            :param inf: (Batch, samples)
            :return: (Batch)
        """
        ref = ref / torch.norm(ref, p=2, dim=1, keepdim=True)
        inf = inf / torch.norm(inf, p=2, dim=1, keepdim=True)

        s_target = (ref * inf).sum(dim=1, keepdims=True) * ref
        e_noise = inf - s_target

        si_snr = 20 * torch.log10(
            torch.norm(s_target, p=2, dim=1) / torch.norm(e_noise, p=2, dim=1)
        )
        return -si_snr.mean()


if __name__ == '__main__':
    nfft = 512
    win_len = 32*16
    hop_len = 16*16
    model = TGCN(win_len, hop_len, nfft)
    x = torch.rand(2, 96000, 8)
    y = torch.rand(2, 96000)
    print(model.compute_loss(x, y))
    from torchinfo import summary
    summary(model, input_size=(2, 96000, 8), depth=3, col_names=["input_size", "output_size", "num_params", "mult_adds"])