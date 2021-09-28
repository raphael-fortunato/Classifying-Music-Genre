import os
import numpy as np
import json
import torch
from torch.utils.data import Dataset
import torchaudio
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler

class MusicDataset(Dataset):
    def __init__(self, X, Y, args, transform=None):
        super().__init__()
        self.data = X
        self.labels = Y
        self.class_names = os.listdir(args.root+"/train")
        self.transform = transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        audio = self.data[i].unsqueeze(0)
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


# loading data from json file
def load_data(data_path):
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """
    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = torch.tensor(data["mfcc"])
    y = torch.tensor(data["labels"])
    z = np.array(data['mapping'])
    return X, y, z

# padding audio files to ensure equal length
# create dataloaders to train CNN
def get_dataset(args):
    X_train, Y_train, _ = load_data("train.json")
    X_test, Y_test, _ = load_data("test.json")
    max = X_train.max()
    min = X_train.min()
    X_std = (X_train - min) / (max - min)
    X_train = X_std * (1 - -1)  -1
    X_std = (X_test - min) / (max - min)
    X_test = X_std * (1 - -1)  -1
    # load dataset
    train_dataset = MusicDataset(X_train, Y_train, args, transform=None)
    valid_dataset = MusicDataset(X_test, Y_test, args, transform=None)

    # load dataset
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers)

    # return dataloaders in dict
    return {'train': train_loader, 'valid':valid_loader}
