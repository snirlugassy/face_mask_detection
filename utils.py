import torch

def calc_accuracy(classifier, loader, device, limit=None):
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = classifier(images)
            predicted = torch.softmax(outputs, dim=1).argmax(dim=1)
            total += len(labels)
            correct += (predicted == labels).sum()

            if limit is not None and total >= limit:
                break
    return correct / total
