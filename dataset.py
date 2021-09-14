import os
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
import torchaudio

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
                label = self.classes.index(f.split('.')[0])
                self.labels.append(label)
                self.data.append(full_path)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        fname = self.data[i]
        audio = torchaudio.load(fname)[0]
        if self.transform: 
            audio = self.transform(audio)
        class_idx = self.labels[i]
        return audio.unsqueeze(0), class_idx

# padding audio files to ensure equal length
def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.squeeze().t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

# used to create test/train batches
def collate_fn(batch):
    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number
    tensors, targets = [], []
    # Gather in lists, and encode labels as indices
    for waveform, label in batch:
        tensors += [waveform]
        targets += [torch.tensor(label)]
    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)
    return tensors, targets

# create dataloaders to train CNN
def get_dataset(args):
    # Create the function that will transform our audio to mel_spectograms. 
    # We will pass this to the dataset class.
    mel_spectrograms = torchaudio.transforms.MelSpectrogram(
        n_fft       = 1024,
        hop_length  = 512,
        n_mels      = 64)
    
    # load dataset
    train_dataset = MusicDataset(args.root, transform=mel_spectrograms)
    valid_dataset = MusicDataset(args.root, transform=mel_spectrograms)

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
        collate_fn=collate_fn, num_workers=args.num_workers)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, sampler=valid_sampler,
        collate_fn=collate_fn, num_workers=args.num_workers)

    # return dataloaders in dict
    return {'train': train_loader, 'valid':valid_loader}
