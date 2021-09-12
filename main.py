import os
import numpy as np
import torch
from torchaudio.datasets import GTZAN
from torchaudio.datasets.utils import download_url

from dataset import MusicDataset, get_dataset
from args import get_args

if __name__ == '__main__':
    # get and print arguments
    args = get_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))
    # set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # read in data and dataloaders
    dataloaders = get_dataset(args)
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for audio, label in dataloaders['train']:
        print(label)
        print(audio)

    print("Using device: ", device)


