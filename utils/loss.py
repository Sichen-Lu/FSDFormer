# Loss for FSDFormer 
import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.pytorch_ssim import msssim

# version adaptation for PyTorch > 1.7.1
IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft


class FocalFrequencyLoss(nn.Module):
    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        y = torch.stack(patch_list, 1)

        if IS_HIGH_VERSION:
            freq = torch.fft.fft2(y, norm='ortho')
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(y, 2, onesided=False, normalized=True)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # 频谱权重矩阵
        if matrix is not None:
            weight_matrix = matrix.detach()
        else:
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1]')

        # 频率距离
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # 动态频谱加权
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None, **kwargs):
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight

class CharbonnierLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(CharbonnierLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = 1e-3

    def forward(self, pred, target):
        return self.loss_weight * torch.mean(torch.sqrt(torch.square(target - pred) + self.eps * self.eps))

class Loss(nn.Module):
    def __init__(self, device=None):
        super(Loss, self).__init__()
        self.perceptual_cri = CharbonnierLoss(loss_weight=1.0)
        self.beta = 1.0
        self.FocalFrequencyLoss = FocalFrequencyLoss()
        self.alpha = 0.1 
        
        self.gamma = 1.0
        self.device = device

    def forward(self, prediction, target, is_ds=False):
        if not is_ds:
            loss = self.beta * self.perceptual_cri(prediction, target) + \
                   self.gamma * (1 - msssim(prediction, target, val_range=2., normalize='relu')) +\
                   self.alpha * self.FocalFrequencyLoss(prediction, target)
        else:
            loss = self.perceptual_cri(prediction, target) + \
                   self.gamma * (1 - msssim(prediction, target, val_range=2., normalize='relu'))

        return loss