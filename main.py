import os
import numpy as np
import torch
from torchaudio.datasets import GTZAN
from torchaudio.datasets.utils import download_url

from dataset import MusicDataset
from args import get_args


def get_dataset(args):
    train_dataset = MusicDataset(args.root)
    valid_dataset = MusicDataset(args.root)
    num_train = .9
    len_dataset = len(train_dataset)
    indices = list(range(len_dataset))
    split = int(np.floor(.9 * num_train))

    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

    # load dataset
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, sampler=valid_sampler,
        num_workers=args.num_workers)

    return {'train': train_loader, 'valid':valid_loader}

if __name__ == '__main__':
    args = get_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    datasets = get_dataset(args)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)


