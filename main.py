import os
import numpy as np
import torch
from torchaudio.datasets import GTZAN
from torchaudio.datasets.utils import download_url

from dataset import MusicDataset


if __name__ == '__main__':
    dataset = MusicDataset('dataset/gtzan/genres_original')
    print(len(dataset))



