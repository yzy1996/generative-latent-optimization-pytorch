import torch
from torch import nn
import torch.nn.functional as F

import numpy as np


def gaussian_conv(img, k_size=5, sigma=1, n_channels=3):

    # 生成高斯核
    grid = np.mgrid[:k_size, :k_size].T
    mu = k_size // 2
    kernel = np.exp(-(grid - mu)**2 / (2 * sigma**2))
    kernel = np.sum(kernel, axis=2)
    kernel /= kernel.sum()

    # 将高斯核转化为卷积核权重
    n_channels = img.size()[1]
    weight = torch.as_tensor(kernel).float()
    weight = weight.expand(n_channels, 1, -1, -1).to(img.device)

    # 对图片进行 padding
    img = F.pad(img, (k_size//2, k_size//2, k_size//2, k_size//2), mode='replicate')

    return F.conv2d(img, weight, groups=n_channels)


def laplacian_pyramid(img, max_levels=5, k_size=5, sigma=1, n_channels=3, downscale=2):
    
    current = img
    pyr = []

    for _ in range(max_levels):
        filtered = gaussian_conv(current, k_size, sigma, n_channels)
        diff = current - filtered
        pyr.append(diff)
        current = F.avg_pool2d(filtered, downscale)

    pyr.append(current)

    return pyr

class LapLoss(nn.Module):
    def __init__(self, max_levels=5, k_size=5, sigma=1, n_channels=3, downscale=2, **kwargs):
        super().__init__()
        self.max_levels = max_levels
        self.k_size = k_size
        self.sigma = sigma
        self.n_channels = n_channels
        self.downscale = downscale

    def forward(self, input, target):

        pyr_input = laplacian_pyramid(input, self.max_levels, self.k_size, self.sigma, self.n_channels, self.downscale)
        pyr_target = laplacian_pyramid(target, self.max_levels, self.k_size, self.sigma, self.n_channels, self.downscale)

        loss = 0
        for j, (i, t) in enumerate(zip(pyr_input, pyr_target), 1): 
            wt = 2 ** (-2 * j)
            loss += wt * torch.mean(torch.abs(i - t))

        return loss


if __name__ == '__main__':

    input = torch.rand((2, 3, 64, 64))
    target = torch.rand((2, 3, 64, 64))

    loss_fn = LapLoss(max_levels=3)
    loss = loss_fn(input, target)

    print(loss)
