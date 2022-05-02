import torch

def calc_accuracy(classifier, loader, device, limit=None):
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = classifier(images)
        predicted = torch.argmax(outputs, 1)
        total += len(labels)
        correct += (predicted == labels).sum()

        if total >= limit:
            break
    return correct / total
