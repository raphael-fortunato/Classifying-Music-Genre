import sys
import time
import copy
import math
import time
import numpy as np
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

from args import get_args
from dataset import get_dataset
from model import AudioModel, ResNet

def test_model(args, model, dataloader, criterion, device):
    model.eval()
    running_loss = 0
    running_corrects = 0
    y_true, y_pred = [], []

    # Iterate over data.
    for inputs, labels in dataloader:
        # transfer labels and input to the correct device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(outputs.argmax(1).detach().cpu().tolist())

    epoch_loss = running_loss / len(dataloader.dataset)
    # balanced_accuracy_score to correct for the nonuniform class distribution
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    conf_matrix = confusion_matrix(y_true, y_pred)
    heatmap = sns.heatmap(
        conf_matrix / np.repeat(np.sum(conf_matrix, axis=1),args.num_classes).reshape((args.num_classes,args.num_classes)),
        annot=True,
        fmt=".0%",
        cbar=False,
        xticklabels=dataloader.dataset.class_names,
        yticklabels=dataloader.dataset.class_names)
    plt.figure(figsize=(20, 15))
    figure = heatmap.get_figure()
    figure.savefig(
            f'heatmap_{time.time()}.png',
            dpi=700,
            bbox_inches='tight')

def train(dataloader, model, optim, criterion, args, device):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloader[phase]:
                # load data to correct device
                inputs = inputs.to(device)
                labels = labels.to(device)
                # clear optimizers' history
                optim.zero_grad()

                # set grad if training
                with torch.set_grad_enabled(phase == 'train'):
                    # generate output
                    outputs = model(inputs) 
                    _, preds = torch.max(outputs, 1)
                    # calculate loss
                    loss = criterion(outputs, labels)
                    # update model
                    if phase == 'train':
                        loss.backward()
                        optim.step()
                # if loss is infinite end training
                if not math.isfinite(loss):
                    print("Loss is {}, stopping training".format(loss))
                    sys.exit(1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(dataloader[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloader[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
                # deep copy the model
            if phase == 'valid':
                scheduler.step()
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


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
    print("Using device: ", device)
    model = AudioModel(args)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma,
            verbose=True)
    model = train(dataloaders, model, optimizer, criterion, args, device)
    test = test_model(args, model, dataloaders['valid'], criterion, device)
    torch.save(model, f"models/model{time.time()}.pt")

