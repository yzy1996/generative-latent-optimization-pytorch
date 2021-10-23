import torch
from torch import nn
from torch.nn import functional as F


class PixelwiseLoss(nn.Module):
    def forward(self, inputs, targets):
        return F.smooth_l1_loss(inputs, targets)


def mmgan_loss_fn():

    bce = nn.BCEWithLogitsLoss()

    def g_loss_fn(fake):
        g_loss = - bce(fake, torch.zeros_like(fake))
        return g_loss

    def d_loss_fn(real, fake):
        d_loss = bce(real, torch.ones_like(real)) + \
            bce(fake, torch.zeros_like(fake))
        return d_loss

    return g_loss_fn, d_loss_fn


class MinMaxGANLoss(nn.Module):
    def forward(self, real, fake):

        bce = nn.BCEWithLogitsLoss()

        g_loss = - bce(fake, torch.zeros_like(fake))
        d_loss = bce(real, torch.ones_like(real)) + \
            bce(fake, torch.zeros_like(fake))

        return g_loss, d_loss


def nsgan_loss_fn():

    bce = nn.BCEWithLogitsLoss()

    def g_loss_fn(fake):
        g_loss = bce(fake, torch.ones_like(fake))
        return g_loss

    def d_loss_fn(real, fake):
        d_loss = bce(real, torch.ones_like(real)) + \
            bce(fake, torch.zeros_like(fake))
        return d_loss

    return g_loss_fn, d_loss_fn


class NonSaturatingGANLoss(nn.Module):
    def forward(self, real, fake):

        bce = nn.BCEWithLogitsLoss()

        g_loss = - bce(fake, torch.zeros_like(fake))
        d_loss = bce(real, torch.ones_like(real)) + \
            bce(fake, torch.zeros_like(fake))

        return g_loss, d_loss


def lsgan_loss_fn():

    mse = nn.MSELoss()

    def g_loss_fn(fake):
        g_loss = mse(fake, torch.ones_like(fake))
        return g_loss

    def d_loss_fn(real, fake):
        d_loss = mse(real, torch.ones_like(real)) + \
            mse(fake, torch.zeros_like(fake))
        return d_loss

    return g_loss_fn, d_loss_fn


def wgan_loss_fn():

    def g_loss_fn(fake):
        g_loss = - fake.mean()
        return g_loss

    def d_loss_fn(real, fake):
        d_loss = - real.mean() + fake.mean()
        return d_loss

    return g_loss_fn, d_loss_fn


def hinge_v1_loss_fn():

    def g_loss_fn(fake):
        g_loss = F.relu(1 - fake).mean()
        return g_loss

    def d_loss_fn(real, fake):
        d_loss = (F.relu(1 - real) + F.relu(1 + fake)).mean()
        return d_loss

    return g_loss_fn, d_loss_fn


def hinge_v2_loss_fn():

    def g_loss_fn(fake):
        g_loss = - fake.mean()
        return g_loss

    def d_loss_fn(real, fake):
        d_loss = (F.relu(1 - real) + F.relu(1 + fake)).mean()
        return d_loss

    return g_loss_fn, d_loss_fn


def gan_losses_fn(mode='nsgan'):

    if mode == 'mmgan':
        return mmgan_loss_fn()

    elif mode == 'nsgan':
        return nsgan_loss_fn()

    elif mode == 'lsgan':
        return lsgan_loss_fn()

    elif mode == 'wgan':
        return wgan_loss_fn()

    elif mode == 'hinge_v1':
        return hinge_v1_loss_fn()

    elif mode == 'hinge_v2':
        return hinge_v2_loss_fn()

    else:
        raise RuntimeError("Wrong gan_type")

import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.models import vgg16_bn

class FeatureLoss(nn.Module):
    def __init__(self, loss, blocks, weights, device):
        super().__init__()
        self.feature_loss = loss
        assert all(isinstance(w, (int, float)) for w in weights)
        assert len(weights) == len(blocks)

        self.weights = torch.tensor(weights).to(device)
        #VGG16 contains 5 blocks - 3 convolutions per block and 3 dense layers towards the end
        assert len(blocks) <= 5
        assert all(i in range(5) for i in blocks)
        assert sorted(blocks) == blocks

        # freeze the feature weight
        vgg = vgg16_bn(pretrained=True).features
        vgg.eval()

        for param in vgg.parameters():
            param.requires_grad = False

        vgg = vgg.to(device)

        # 
        bns = [i - 2 for i, m in enumerate(vgg) if isinstance(m, nn.MaxPool2d)]
        assert all(isinstance(vgg[bn], nn.BatchNorm2d) for bn in bns)

        self.hooks = [FeatureHook(vgg[bns[i]]) for i in blocks]
        self.features = vgg[0: bns[blocks[-1]] + 1]

    def forward(self, inputs, targets):

        # normalize foreground pixels to ImageNet statistics for pre-trained VGG
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        inputs = F.normalize(inputs, mean, std)
        targets = F.normalize(targets, mean, std)

        # extract feature maps
        self.features(inputs)
        input_features = [hook.features.clone() for hook in self.hooks]

        self.features(targets)
        target_features = [hook.features for hook in self.hooks]

        loss = 0.0
        
        # compare their weighted loss
        for lhs, rhs, w in zip(input_features, target_features, self.weights):
            lhs = lhs.view(lhs.size(0), -1)
            rhs = rhs.view(rhs.size(0), -1)
            loss += self.feature_loss(lhs, rhs) * w

        return loss

class FeatureHook:
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.on)

    def on(self, module, inputs, outputs):
        self.features = outputs

    def close(self):
        self.hook.remove()

features = []
def hook(module, input, output):
    features.append(output.clone().detach())

def perceptual_loss(x, y):
    F.mse_loss(x, y)
    
def PerceptualLoss(blocks, weights, device):
    return FeatureLoss(perceptual_loss, blocks, weights, device)

if __name__ == '__main__':

    d_real = torch.randn([1, 5])
    d_fake = torch.randn([1, 5])

    g_loss_fn, d_loss_fn = gan_losses_fn('wgan')
    g_loss = g_loss_fn(d_fake)
    d_loss = d_loss_fn(d_fake, d_real)

    print(g_loss, d_loss)
