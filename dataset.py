import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
from torchvision import transforms

class MusicDataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.data = []
        self.labels = []
        self.classes = []
        self.transform = transform
        for path, dir_names, files in os.walk(root):
            if not self.classes:
                self.classes = dir_names
            for f in files:
                full_path = os.path.join(path, f)
                label = self.classes.index(f.split('_')[0])
                self.labels.append(label)
                self.data.append(full_path)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        fname = self.data[i]
        audio = cv2.imread(fname)
        if self.transform: 
            audio = self.transform(audio)
        class_idx = self.labels[i]
        return audio, class_idx

def get_transforms(train=True):
    if train:
        transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize((128, 128)),
                 transforms.RandomHorizontalFlip(p=.5),
                 transforms.RandomVerticalFlip(p=.5),
                 transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
                    ])
    else:
        transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize((128, 128)),
                 transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
                    ])
    return transform


# padding audio files to ensure equal length
# create dataloaders to train CNN
def get_dataset(args):
    # load dataset
    train_dataset = MusicDataset(args.root+"/train", transform=get_transforms())
    valid_dataset = MusicDataset(args.root+"/test", transform=get_transforms(train=False))

    # load dataset
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers)

    # return dataloaders in dict
    return {'train': train_loader, 'valid':valid_loader}
