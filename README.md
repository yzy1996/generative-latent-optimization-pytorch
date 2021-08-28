# (GLO) Optimizing the Latent Space of Generative Networks

This is an **unofficial** pytorch reimplementation of the paper [(GLO) Optimizing the Latent Space of Generative Networks](https://arxiv.org/pdf/1707.05776.pdf) by Piotr Bojanowski, Armand Joulin, David Lopez-Paz, Arthur Szlam.



## Prerequisites

- pytorch related
- others packages

```bash
pip install numpy, scikit-learn, tqdm, addict, argparse, pyyaml, lmdb
```

- [LSUN dataset](https://github.com/fyu/lsun)



## Usage

```python
python main.py

# or 

python main.py --config configs/default.yaml
```

If you need to set new configs,  just modify the `configs` folder. You can change the `default.yaml` or add new `xx.yaml` files.



## Example Results



## To Do

- [ ] add more datasets and models
- [ ] add results images



## Related 

Thanks [tneumann](tneumann/minimal_glo), [nathanaelbosch](https://github.com/nathanaelbosch/generative-latent-optimization), [Hyeokreal](https://github.com/Hyeokreal/Generative-Latent-Optimization-GLO--Pytorch-MNINST), [clvrai](https://github.com/clvrai/Generative-Latent-Optimization-Tensorflow)