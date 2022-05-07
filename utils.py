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

            tp += int(torch.logical_and(labels==1, predicted==1).sum())
            fp += int(torch.logical_and(labels==0, predicted==1).sum())
            fn += int(torch.logical_and(labels==1, predicted==0).sum())
            tn += int(torch.logical_and(labels==0, predicted==0).sum())

    return tp, fp, fn, tn

def calc_scores(data_loader: DataLoader, model: torch.nn.Module, device):
    y_true = None
    y_score = None
    model.eval()

    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device=device)
            labels = labels.to(device=device)

            predictions = model(data)
            y_prob = torch.softmax(predictions, dim=1)

            if y_score is not None:
                y_score = torch.cat([y_score, y_prob])
            else:
                y_score = y_prob

            if y_true is not None:
                y_true = torch.cat([y_true, labels])
            else:
                y_true = labels

    return y_true, y_score
