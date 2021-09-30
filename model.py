import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class AudioModel(nn.Module):
    def __init__(self, args):
        super(AudioModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3,3), padding=(1,1))
        self.maxpool1 = nn.MaxPool2d((4,1))
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,3), padding=(1,1))
        self.maxpool2 = nn.MaxPool2d((4,1))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,3), padding=(1,1))
        self.maxpool3 = nn.MaxPool2d((4,3))
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(1024, 256)
        self.dropout1 = nn.Dropout(p=.5)
        self.out = nn.Linear(256, args.num_classes)

    def __call__(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool3(x)
        x = self.bn3(x)
        x = x.view(-1, 64*4*4)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.out(x)
        return x


