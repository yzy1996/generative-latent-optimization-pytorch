# (GLO) Optimizing the Latent Space of Generative Networks

This is an **unofficial** pytorch reimplementation of the paper [(GLO) Optimizing the Latent Space of Generative Networks](https://arxiv.org/pdf/1707.05776.pdf) by Piotr Bojanowski, Armand Joulin, David Lopez-Paz, Arthur Szlam.

## Prerequisites

- pytorch related
- others packages

```bash
pip install numpy, scikit-learn, tqdm, addict, argparse, pyyaml, lmdb
```

## Datasets

[LSUN dataset](https://github.com/fyu/lsun)

CelebA

## Usage

```python
python main.py

# or 

python main.py --config configs/default.yaml
```

If you need to set new configs,  just modify the `configs` folder. You can change the `default.yaml` or add new `xx.yaml` files.

## Tricks

- Initialization of the latent vectors with PCA is very crucial.
- For Laplacian pyramid, we use 3 levels.
- For latent space dimension, we use 128.

## Example Results

![fake_200](https://raw.githubusercontent.com/yzy1996/Image-Hosting/master/20211023211240.png)

## To Do

- [ ] add more datasets and models
- [ ] add results images

## Related

Thanks [tneumann](tneumann/minimal_glo), [nathanaelbosch](https://github.com/nathanaelbosch/generative-latent-optimization), [Hyeokreal](https://github.com/Hyeokreal/Generative-Latent-Optimization-GLO--Pytorch-MNINST), [clvrai](https://github.com/clvrai/Generative-Latent-Optimization-Tensorflow)

If you are interested in other GAN projects, you can ref my `Awesome-Generative-Adversarial-Models`.
