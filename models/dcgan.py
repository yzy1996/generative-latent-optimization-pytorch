from torch import nn


class Generator(nn.Module):
    def __init__(self, dim, n_filter=64, out_channels=3):
        super(Generator, self).__init__()
        self.dim = dim
        nf = n_filter
        self.dcnn = nn.Sequential(
            nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0, bias=False),  # 2x2
            nn.BatchNorm2d(nf * 8), nn.ReLU(True),
            nn.ConvTranspose2d(nf * 8, nf * 4, 4, 2, 1, bias=False),  # 4x4
            nn.BatchNorm2d(nf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1, bias=False),  # 8x8
            nn.BatchNorm2d(nf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(nf * 2, nf, 4, 2, 1, bias=False),  # 16x16
            nn.BatchNorm2d(nf), nn.ReLU(True),
            nn.ConvTranspose2d(nf, out_channels, 4, 2, 1, bias=False),  # 32x32
            nn.Tanh(),
        )

    def forward(self, x):
        return self.dcnn(x.view(x.size(0), self.dim, 1, 1))
