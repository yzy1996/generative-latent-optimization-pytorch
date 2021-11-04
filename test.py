from numpy.lib.npyio import recfromcsv
import torch

import addict
import yaml
import argparse
from pathlib import Path
import numpy as np

from models.generator import Generator
from torchvision.utils import make_grid, save_image
from PIL import Image

def imsave(filename, array):
    im = Image.fromarray((array * 255).astype(np.uint8))
    im.save(filename)

def project_to_sphera(x):

    if isinstance(x, np.ndarray):
        return x / np.linalg.norm(x, axis=1)[:, np.newaxis]

    elif isinstance(x, torch.Tensor):
        return x / torch.norm(x, dim=1).unsqueeze(1)

    else:
        raise ValueError("cannot recognized x type!")

    # return x

def main(args):
    ckpt_file = Path(f'logs/{args.expname}') / 'ckpts' / 'latest.pt'
    ckpt = torch.load(ckpt_file)
    z = ckpt['z']

    generator = Generator(args.model.z_dim)
    generator.load_state_dict(ckpt['g_state_dict'])


    generator.eval()

    new_sample_z = torch.randn(128, args.model.z_dim, 1, 1)
    inter_z = (z[0:127] + z[1000:1127]) / 2 
    inter_z = torch.as_tensor(inter_z).float()
    new_sample_z = project_to_sphera(new_sample_z)
    inter_z = project_to_sphera(inter_z)

    with torch.no_grad():
        rec1 = generator(new_sample_z)
        rec2 = generator(inter_z)

    iiimge = (rec1 / 2. + 0.5)
    save_image(iiimge, f'test.png')
    # imsave(f'{args.expname}-re-new_sample.png', make_grid(rec1.data.cpu() / 2. + 0.5, nrow=8).numpy().transpose(1, 2, 0))
    # imsave(f'{args.expname}-re-inter.png', make_grid(rec2.data.cpu() / 2. + 0.5, nrow=8).numpy().transpose(1, 2, 0))

    # imsave(f'new_sample-reproject2.png', make_grid(rec.data.cpu() / 2. + 0.5, nrow=1).numpy().transpose(1, 2, 0))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, default='configs/default.yaml', help='Path to config file.')
    args, unknown = parser.parse_known_args()

    with open(args.configs, encoding='utf8') as yaml_file:
        configs_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
        configs = addict.Dict(configs_dict)

    main(configs)