import torch

__all__ = ['torch_dev']

__author__ = 'Haoyan Huo'
__email__ = 'haoyan.huo@lbl.gov'
__maintainer__ = 'Haoyan Huo'


def torch_dev():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    return dev
