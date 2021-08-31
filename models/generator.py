from torch import nn
import functools
from torchsummary import summary

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


class Generator(nn.Module):
    def __init__(self,
                 input_channels=128,
                 output_channels=3,
                 first_kernel_size=4,
                 dim=64,
                 depth=3,
                 norm='batch_norm'):

        super().__init__()

        Norm = _get_norm_layer_2d(norm)

        def conv_norm_relu(in_dim, out_dim, kernel_size=4, stride=2, padding=1):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride=stride,
                                   padding=padding, bias=False),
                Norm(out_dim),
                nn.LeakyReLU(0.2, inplace=True),
            )

        layers = []

        # first layer
        out_dim = dim * (2 ** depth)
        layers.append(conv_norm_relu(input_channels, out_dim, kernel_size=first_kernel_size, stride=1, padding=0))

        # middle layer
        for _ in range(depth):
            in_dim = out_dim
            out_dim = in_dim // 2
            layers.append(conv_norm_relu(in_dim, out_dim))

        # last layer
        layers.append(nn.ConvTranspose2d(out_dim, output_channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input.reshape(input.size(0), -1, 1, 1))


if __name__ == '__main__':

    g_kwargs = {'input_channels': 128,
                'output_channels': 1,
                'first_kernel_size': 7,
                'dim': 64,
                'depth': 1}

    G = Generator(**g_kwargs)
    # G = Generator().to(device)

    # noise = torch.randn(1, 128, 1, 1).to(device)
    # print(G(noise))
    summary(G, input_size=(128, 1, 1))
