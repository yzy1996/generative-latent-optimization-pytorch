import os

import torch

from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import addict
import yaml
import argparse
from pathlib import Path
import numpy as np

def main(args):

    ckpt_file = Path(f'logs/{args.expname}') / 'ckpts' / 'latest.pt'

    print('=> Loading checkpoint from local file', ckpt_file)

    ckpt = torch.load(ckpt_file)

    sample_num = 10000

    z = ckpt['z'][0:2]

    print(np.linalg.norm(z, axis=1))

    return

    #------------ TSNE plot
    # TODO: !!! 需要调参。
    tsne = TSNE(verbose=True, n_components=2, n_iter=50000, learning_rate=20, n_iter_without_progress=3000)
    tsne_results = tsne.fit_transform(z)
    # print(tsne_results.shape)
    # # 2D
    sns.scatterplot(x=tsne_results[:,0], y=tsne_results[:,1]) # [tSNE results]
    # sns.scatterplot(x=z[:,0], y=z[:,1]) # [random sub dimensions]

    # 每一维直方图
    # sns.displot(z, kind="kde", legend=False)
    # plt.show()

    # 3D
    # fig = plt.figure(figsize=(6,6))
    # ax = Axes3D(fig)
    # # g = ax.scatter(tsne_results[:,0], tsne_results[:,1], tsne_results[:,2], marker='o', depthshade=False)  # [tSNE results]
    # g = ax.scatter(z[:,0], z[:,1], z[:,2], marker='o', depthshade=False)   # [random sub dimensions]
    # ax.set_aspect('equal')
    plt.show()
    # pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, default='configs/default.yaml', help='Path to config file.')
    args, unknown = parser.parse_known_args()

    with open(args.configs, encoding='utf8') as yaml_file:
        configs_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
        configs = addict.Dict(configs_dict)

    main(configs)