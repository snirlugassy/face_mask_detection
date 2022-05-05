import torch
from torch.utils.data import DataLoader


def calc_accuracy(data_loader: DataLoader, model: torch.nn.Module, device, limit=None):
    num_correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device=device)
            labels = labels.to(device=device)

            predictions = model(data)
            predicted = torch.softmax(predictions, dim=1).argmax(dim=1)
            total += labels.size(0)
            num_correct += (predicted == labels).sum()

            if limit is not None and total >= limit:
                break

    return float(num_correct)/float(total)

def calc_confusion_mat(data_loader: DataLoader, model: torch.nn.Module, device):
    tp, fp, fn, tn = 0, 0, 0, 0
    model.eval()

    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device=device)
            labels = labels.to(device=device)

            predictions = model(data)
            predicted = torch.softmax(predictions, dim=1).argmax(dim=1)

            if len(labels) != len(predicted):
                print(labels, predicted)
                n = min(len(labels), len(predicted))
                labels = labels[:n]
                predicted = predicted[:n]

            tp += torch.logical_and(labels==1, predicted==1)
            fp += torch.logical_and(labels==0, predicted==1)
            fn += torch.logical_and(labels==1, predicted==0)
            tn += torch.logical_and(labels==0, predicted==0)

    return tp, fp, fn, tn
