import os
import sys
from time import time
import torch
from torch.utils.data.dataloader import DataLoader

from transform import mask_image_train_transform
from transform import mask_image_test_transform
from model import MaskDetectionModel
from dataset import MaskImageDataset
from utils import calc_accuracy

BATCH_SIZE = 56
EPOCHS = 30
LEARNING_RATE = 0.1
MOMENTUM = 0.9

if __name__ == '__main__':
    data_path = sys.argv[-1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device =', device)

    print('Loading datasets')
    train_dataset = MaskImageDataset(os.path.join(data_path, 'train'), transform=mask_image_train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    train_size = len(train_dataset)

    test_dataset = MaskImageDataset(os.path.join(data_path, 'test'), transform=mask_image_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    print('Initalizing model')
    model = MaskDetectionModel()
    model.to(device)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}\n---------------------------")
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            y_prob = model(images)
            L = loss(y_prob, labels)

            # Backpropagation
            optimizer.zero_grad()
            L.backward()
            optimizer.step()

            if i % 20 == 0:
                # test_accuracy = calc_accuracy(model, test_loader, device, limit=100)
                print(f'Loss: {L.item():>7f}  [{i * len(labels):>5d}/{train_size:>5d}]')
            break

        # torch.save(model.state_dict(), 'model.state')
