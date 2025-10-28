import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs - label
        sqrt_error = torch.pow(error, 2)
        rmse = torch.sqrt(torch.mean(sqrt_error.view(-1)))
        return rmse


class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=1):
        N = im_true.size()[0]
        C = im_true.size()[1]
        H = im_true.size()[2]
        W = im_true.size()[3]
        # Itrue = im_true.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        # Ifake = im_fake.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        Itrue = im_true.reshape(N, C * H * W)
        Ifake = im_fake.reshape(N, C * H * W)
        mse = nn.MSELoss(reduce=False)
        err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)
        psnr = 10. * torch.log((data_range ** 2) / err) / np.log(10.)
        return torch.mean(psnr)
    
class Loss_SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(Loss_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def gaussian_window(self, window_size, sigma):
        # 使用torch.linspace生成序列，并确保所有操作都在Tensor上执行
        gauss = torch.exp(-torch.pow(torch.linspace(-window_size // 2 + 1, window_size // 2, window_size), 2) / (2 * sigma ** 2))
        return gauss / gauss.sum()

    def create_window(self, window_size, channel, device):
        _1D_window = self.gaussian_window(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window.to(device)  # 确保窗口在正确的设备上

    def forward(self, img1, img2):
        channel = img1.size(1)
        self.window = self.create_window(self.window_size, channel, img1.device)  # 确保窗口与输入图像在同一设备

        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=channel)
        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=channel) - mu1.pow(2)
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=channel) - mu2.pow(2)
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=channel) - mu1 * mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1.pow(2) + mu2.pow(2) + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean() if self.size_average else ssim_map.mean(1).mean(1).mean(1)

class Loss_ERGAS(nn.Module):
    def __init__(self, scale=4):
        super(Loss_ERGAS, self).__init__()
        self.scale = scale  # 可以设置比例因子，以适应不同的图像分辨率

    def forward(self, output, label):
        # 假设 output 和 label 的形状是 [B, C, H, W]

        # 将 output 和 label 扁平化，形状从 [B, C, H, W] 变为 [C, N] 其中 N = H * W
        output_flat = output.view(output.size(1), -1)  # 变为 [166, 92*92]
        label_flat = label.view(label.size(1), -1)  # 变为 [166, 92*92]

        # 计算 RMSE
        mse = torch.mean((output_flat - label_flat) ** 2, dim=1)
        rmse = torch.sqrt(mse)

        # 计算平均值
        mean = torch.mean(label_flat, dim=1)

        # 计算 ERGAS
        ergas = torch.mean((rmse / mean) ** 2)
        ergas = 100 / self.scale * torch.sqrt(ergas)  # 使用 scale 因子调整最终结果

        return ergas
    
class Metrics:
    def __init__(self) -> None:
        super(Metrics, self).__init__()
        self.criterion_ergas = Loss_ERGAS().cuda()
        self.criterion_ssim = Loss_SSIM().cuda()
        self.criterion_psnr = Loss_PSNR().cuda()
        self.criterion_rmse = Loss_RMSE().cuda()

        self.cur_result = {}
        self.cur_result['rmse'] = []
        self.cur_result['ssim'] = []
        # self.cur_result['cc'] = []
        # self.cur_result['uiqi'] = []
        self.cur_result['ergas'] = []
        self.cur_result['psnr'] = []
        # self.cur_result['sam'] = []
       
    def update(self, real_predict: Tensor, real_im: Tensor) -> None:
        # pred = pred.argmax(dim=1)
        self.real_predict = real_predict # .squeeze().cpu().numpy()
        self.real_im = real_im # .squeeze().cpu().numpy()
        
    def compute(self) -> Tuple[float, float]:
        self.cur_result['rmse'].append(self.criterion_rmse(self.real_predict, self.real_im).cpu().numpy())
        self.cur_result['ssim'].append(self.criterion_ssim(self.real_predict, self.real_im).cpu().numpy())
        # self.cur_result['cc'].append(cc)
        # self.cur_result['uiqi'].append(uiqi)
        self.cur_result['ergas'].append(self.criterion_ergas(self.real_predict, self.real_im).cpu().numpy())
        self.cur_result['psnr'].append(self.criterion_psnr(self.real_predict, self.real_im).cpu().numpy())
        # self.cur_result['sam'] = sam
      
        return self.cur_result
    
if __name__ == '__main__':
    image = torch.randn(10, 6, 256, 256).cuda()
    label = torch.randn(10, 6, 256, 256).cuda()
    criterion_ergas = Loss_ERGAS().cuda()
    criterion_ssim = Loss_SSIM().cuda()
    criterion_psnr = Loss_PSNR().cuda()
    criterion_rmse = Loss_RMSE().cuda()
    psnr = criterion_psnr(image, label)
    print(psnr)
    rmse = criterion_rmse(image, label)
    print(rmse)
    ssim = criterion_ssim(image, label)
    print(ssim)
    ergas = criterion_ergas(image, label)
    print(ergas)