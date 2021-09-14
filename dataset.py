import os
import cv2
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
import torchaudio
from torchvision import transforms

class MusicDataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.data = []
        self.labels = []
        self.classes = []
        self.transform = transform
        for path, dir_names, files in os.walk(root):
            if not self.classes:
                self.classes = dir_names
            for f in files:
                full_path = os.path.join(path, f)
                label = self.classes.index(f.split('0')[0])
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

def get_transforms():
    transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((432, 288))
                ])
    return transform


# padding audio files to ensure equal length
# create dataloaders to train CNN
def get_dataset(args):
    # load dataset
    train_dataset = MusicDataset(args.root, transform=get_transforms())
    valid_dataset = MusicDataset(args.root, transform=get_transforms())

    # Split dataset in validation and train dataset using sampler
    len_dataset = len(train_dataset)
    indices = list(range(len_dataset))
    split = int(np.floor(.9 * len_dataset))
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

    # return dataloaders in dict
    return {'train': train_loader, 'valid':valid_loader}
