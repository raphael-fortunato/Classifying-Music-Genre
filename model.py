import torch.nn as nn
import torchvision.models as models


class GenreClassifier(nn.Module):
    def __init__(self, num_classes):
        super(GenreClassifier, self).__init__()

        self.CNN_Block = nn.Sequential(
            nn.Conv1d(64, 64, 5, 4, 2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, 5, 4, 2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 256, 5, 4, 2),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.Fully_Conn = nn.Sequential(nn.Linear(5376, num_classes), nn.Softmax())

    def __call__(self, tensor):
        batch_size = tensor.size(0)
        hidden = self.CNN_Block(tensor)
        return self.Fully_Conn(hidden.view(batch_size, -1))


class ResNet(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(ResNet, self).__init__()
        self.model = models.resnet18(pretrained)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
		
    def forward(self, x):
        output = self.model(x)
        return output