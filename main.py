import os
import numpy as np
import torch
from torchaudio.datasets import GTZAN
from torchaudio.datasets.utils import download_url

from dataset import MusicDataset, get_dataset
from args import get_args


def get_dataset(args):

    # Create the function that will transform our audio to mel_spectograms. 
    # We will pass this to the dataset class.
    mel_spectograms = torchaudio.transforms.mel_spectograms(
        sample_rate = SAMPLE_RATE,
        n_fft       = 1024,
        hop_length  = 512,
        n_mels      = 64)
    
    train_dataset = MusicDataset(args.root, mel_spectograms)
    valid_dataset = MusicDataset(args.root, mel_spectograms)
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
