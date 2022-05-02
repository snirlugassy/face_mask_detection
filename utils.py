import torch

def calc_accuracy(classifier, loader, limit=None):
    correct = 0
    total = 0
    for images, labels in loader:
        outputs = classifier(images)
        predicted = torch.argmax(outputs, 1)
        total += len(labels)
        correct += (predicted == labels).sum()

        if total >= limit:
            break
    return correct / total
