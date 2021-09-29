import os
import numpy as np
import json
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pylab as plt


class MusicDS(Dataset):
    def __init__(self, root):
        self.classes = []
        self.labels = []
        self.song_paths = []
        self.labels_to_idx = {}
        for path, dir_names, files in os.walk(root):
            if not self.classes:
                self.classes = dir_names
                self.labels_to_idx = {k:v for v, k in enumerate(self.classes)}
                print(self.labels_to_idx)
            for f in files:
                full_path = os.path.join(path, f)
                class_name = [l for l in self.classes if l in f][0]
                self.labels.append(self.labels_to_idx[class_name])
                self.song_paths.append(full_path)

        
    def plot_specgram(self, waveform, sample_rate, title="Spectrogram", xlim=None):
        waveform = waveform.numpy()

        num_channels, _ = waveform.shape

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].specgram(waveform[c], Fs=sample_rate)
            if num_channels > 1:
                axes[c].set_ylabel(f'Channel {c+1}')
            if xlim:
                axes[c].set_xlim(xlim)
        figure.suptitle(title)
        plt.show(block=False)
        
    def print_stats(self, waveform, sample_rate=None, src=None):
        if src:
            print("-" * 10)
            print("Source:", src)
            print("-" * 10)
        if sample_rate:
            print("Sample Rate:", sample_rate)
        print("Shape:", tuple(waveform.shape))
        print("Dtype:", waveform.dtype)
        print(f" - Max:     {waveform.max().item():6.3f}")
        print(f" - Min:     {waveform.min().item():6.3f}")
        print(f" - Mean:    {waveform.mean().item():6.3f}")
        print(f" - Std Dev: {waveform.std().item():6.3f}")
        print()
        print(waveform)
        print()
        
    def get_sample(self, path, sample_rate=4000):
        effects = [
          ["lowpass", "-1", "150"], # apply single-pole lowpass filter
          ["speed", "0.9"],  # reduce the speed
                             # This only changes sample rate, so it is necessary to
                             # add `rate` effect with original sample rate after this.
          ["rate", f"{sample_rate}"],
          ["reverb", "-w"],  # Reverbration gives some dramatic feeling
        ]
        return torchaudio.sox_effects.apply_effects_file(path, effects=effects)
    
    def __len__(self):
        return len(self.song_paths)
    
    def __getitem__(self, idx):
        song_path = self.song_paths[idx]
        try:
            waveform, frame_num = self.get_sample(song_path)
        except:
            idx += 1
            song_path = self.song_paths[idx]
            waveform, frame_num = self.get_sample(song_path)
        waveform = torch.unsqueeze(waveform, 0)
        waveform = F.interpolate(waveform, size=(300134))
        waveform = torch.squeeze(waveform, 0)
        print(waveform.shape)
        return waveform, self.labels[idx]

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

# create dataloaders to train CNN
def get_dataset(args):
    train_dataset = MusicDS(args.root+"/train")
    test_dataset = MusicDS(args.root+"/test")

    # load dataset
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers)
    valid_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers)

    # return dataloaders in dict
    return {'train': train_loader, 'valid':valid_loader}

