import copy 
import time
import numpy as np
import torch
import torchaudio

from dataset import get_dataset
from args import get_args
from model import GenreClassifier

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
                    import pdb; pdb.set_trace()
                    loss = criterion(outputs, labels)
                    # update model
                    loss.backward()
                    optim.step()
                # if loss is infinite end training
                if not math.isfinite(loss):
                    print("Loss is {}, stopping training".format(loss))
                    sys.exit(1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects = torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(dataloader[phase].dataset)
            epoch_acc = running_corrects.double() / dataloader[phase].dataset
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
                # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
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
    model = GenreClassifier(10)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    train(dataloaders, model, optimizer, criterion, args, device)

