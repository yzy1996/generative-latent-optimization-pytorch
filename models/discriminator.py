import torch
from torch import nn
import functools
from torchsummary import summary


SN = torch.nn.utils.spectral_norm

def _get_norm_layer_2d(norm):
    if norm == 'none':
        return nn.Identity
    elif norm == 'batch_norm':
        return nn.BatchNorm2d
    elif norm == 'instance_norm':
        return functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm == 'layer_norm':
        return lambda num_features: nn.GroupNorm(1, num_features)
    else:
        raise NotImplementedError


class Discriminator(nn.Module):

    def __init__(self,
                 input_channels=3,
                 last_kernel_size=4,
                 dim=64,
                 depth=3,
                 norm='batch_norm'):

        super(Discriminator, self).__init__()

        Norm = _get_norm_layer_2d(norm)

        def conv_norm_lrelu(in_dim, out_dim, kernel_size=4, stride=2, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride,
                          padding=padding, bias=False),
                Norm(out_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3)
            )

        layers = []

        # first layer
        out_dim = dim
        layers.append(nn.Conv2d(input_channels, out_dim,
                                kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Dropout(0.3))

        # middle layer
        for _ in range(depth):
            in_dim = out_dim
            out_dim = in_dim * 2
            layers.append(conv_norm_lrelu(in_dim, out_dim))

        # last layer
        layers.append(nn.Conv2d(out_dim, 1, kernel_size=last_kernel_size, stride=1, padding=0, bias=False))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        return y

if __name__ == '__main__':

    D = Discriminator()

    summary(D, input_size=(3, 1, 1))

