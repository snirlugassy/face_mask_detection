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
EPOCHS = 20
LEARNING_RATE = 1e-4


if __name__ == '__main__':
    data_path = sys.argv[-1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device =', device)

    print('Loading datasets')
    train_dataset = MaskImageDataset(os.path.join(data_path, 'train'), transform=mask_image_train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = MaskImageDataset(os.path.join(data_path, 'test'), transform=mask_image_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    print('Initalizing model')
    model = MaskDetectionModel()
    model.to(device)

    print('Calculating initial accuracy')
    t0 = time()
    acc = 
    print('accuracy = ', acc)
    print('took', time() - t0, 'seconds to calc acc')

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCHS):
        print('Starting epoch', epoch+1)
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            y_prob = model(images)
            L = loss(y_prob, labels)

            # Backpropagation
            L.backward()
            optimizer.step()

            if i % 20 == 0:
                test_accuracy = calc_accuracy(model, test_loader, limit=100)
                print(f'Epoch: [{epoch+1}/{EPOCHS}], Step: {i+1}, Loss: {L}, Test Acc.: {test_accuracy}')

        torch.save(model.state_dict(), 'model.state')
