import torch.nn as nn
import torch.nn.functional as F

class AudioModel(nn.Module):
    def __init__(self, args):
        super(AudioModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), padding=(1,1))
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d((4,1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), padding=(1,1))
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d((4,1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3,3), padding=(1,1))
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d((4,3))
        self.fc1 = nn.Linear(2048, 256)
        self.relu4 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d((4,3))
        self.dropout1 = nn.Dropout(p=.5)
        self.out = nn.Linear(256, args.num_classes)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = x.view(-1, 128*4*4)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout1(x)
        x = self.out(x)
        return x

