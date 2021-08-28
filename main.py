import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

import numpy as np
from pathlib import Path
from tqdm import tqdm
import addict
import yaml
import argparse

from dataio import get_data
from models.dcgan import Generator
from losses.laplacian_loss import LapLoss

import utils


class IndexedDataset(Dataset):
    """ 
    produce (mages, lables, idx)
    """

    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return (img, label, idx)


def project_to_sphera(x):

    if isinstance(x, np.ndarray):
        return x / np.linalg.norm(x, axis=1)[:, np.newaxis]

    elif isinstance(x, torch.Tensor):
        return x / torch.norm(x, dim=1).unsqueeze(1)

    else:
        raise ValueError("cannot recognized x type!")


def initialize_latent(train_loader, z_dim, mode='random'):

    print("Starting latent initialization!")

    z_num = len(train_loader.dataset)

    if mode == 'pca':

        from sklearn.decomposition import PCA

        images, _, _ = zip(*[(images, _, _) for images, _, _ in train_loader])
        images_all = torch.cat(images)
        images_all = images_all.reshape(z_num, -1)

        z = np.empty((z_num, z_dim))

        pca = PCA(n_components=z_dim)
        z[range(z_num)] = pca.fit_transform(images_all)

    else:
        z = np.random.randn(z_num, z_dim)

    print("Finishing initialization!")

    return project_to_sphera(z)


def main(args):

    # 一些初始设置
    log_dir = Path(f'logs/{args.expname}')

    monitor_dir = log_dir / 'events'
    img_dir = log_dir / 'images'
    ckpt_dir = log_dir / 'ckpts'
    backup_dir = log_dir / 'backup'

    for dir in [monitor_dir, img_dir, ckpt_dir, backup_dir]:
        utils.cond_mkdir(dir)

    epoch = 0
    iter = 0
    batch_size = args.training.batch_size
    z_dim = args.model.z_dim

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(monitor_dir)
    utils.save_config(args.to_dict(), backup_dir / 'config.yaml')

    # 加载数据
    train_dataset = get_data()
    train_dataset = IndexedDataset(train_dataset)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=0,
                              pin_memory=True,
                              shuffle=True,
                              drop_last=True)

    # 根据数据加载模型
    generator = Generator(z_dim).to(device)

    z = initialize_latent(train_loader, z_dim)
    zi = torch.zeros((batch_size, z_dim)).to(device).requires_grad_()

    optimizer = optim.SGD([{'params': generator.parameters(), 'lr': args.training.lr_model},
                           {'params': zi, 'lr': args.training.lr_z}])

    # loss function
    loss_fn = LapLoss(**configs.model) if args.model.loss_type == 'laplacian' else nn.MSELoss()

    # 加载 checkpoints
    ckpt_file = ckpt_dir / 'latest.pt'
    if os.path.exists(ckpt_file):
        print('=> Loading checkpoint from local file', ckpt_file)
        ckpt = torch.load(ckpt_file)

        epoch = ckpt['epoch']
        iter = ckpt['iter']
        loss = ckpt['loss']
        z = ckpt['z']
        generator.load_state_dict(ckpt['g_state_dict'])
        optimizer.load_state_dict(ckpt['opt_state_dict'])

    # train
    epoch = 0

    while epoch < args.training.epoch:

        epoch += 1
        loss_total = []

        with tqdm(train_loader, desc=f'Epoch {epoch}/{args.training.epoch}') as pbar:
            for i, (images, _, idx) in enumerate(pbar):

                iter += 1
                idx = idx.numpy()

                optimizer.zero_grad()

                real_images = images.to(device)

                zi.data = torch.as_tensor(z[idx]).float().to(device)
                fake_images = generator(zi)

                loss = loss_fn(fake_images, real_images)
                loss_total.append(loss.item())

                loss.backward()
                optimizer.step()

                z[idx] = project_to_sphera(zi.detach().cpu().numpy())

                pbar.set_postfix(loss=loss.item())

                writer.add_scalar('Loss/loss_iteration', loss, iter)

        writer.add_scalar('Loss/loss_epoch', np.mean(loss_total), epoch)

        # 保存中断点
        print('Saving checkpoint...')
        torch.save({
            'epoch': epoch,
            'iter': iter,
            'loss': loss,
            'z': z,
            'g_state_dict': generator.state_dict(),
            'opt_state_dict': optimizer.state_dict(),
        }, ckpt_file)

    torch.cuda.empty_cache()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='configs/default.yaml', help='Path to config file.')
    args, unknown = parser.parse_known_args()

    with open(args.config, encoding='utf8') as yaml_file:
        configs_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
        configs = addict.Dict(configs_dict)

    main(configs)
