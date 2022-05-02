import os
import argparse

import torch
from torch.utils.data.dataloader import DataLoader

from transform import mask_image_train_transform2
from transform import mask_image_test_transform
from model import MaskDetectionModel
from dataset import MaskImageDataset
from utils import calc_accuracy


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Train mask detection nerual network')
    argparser.add_argument('--data-path', type=str, required=True, dest='data_path')
    argparser.add_argument('--batch-size', type=int, dest='batch_size')
    argparser.add_argument('--test-limit-size', type=int, dest='test_limit_size')
    argparser.add_argument('--epochs', type=int, dest='epochs')
    argparser.add_argument('--lr', type=float, dest='lr')
    argparser.add_argument('--momentum', type=float, dest='momentum')
    argparser.add_argument('--print-steps', type=int, dest='print_steps')

    args = argparser.parse_args()

    BATCH_SIZE = args.batch_size or 100
    TEST_LIMIT_SIZE = args.test_limit_size
    EPOCHS = args.epochs or 50
    LEARNING_RATE = args.lr or 0.01
    MOMENTUM = args.momentum or 0.7
    PRINT_STEPS = args.print_steps or 20

    data_path = args.data_path

    print('====== TRAIN =======')
    print('batch-size:', BATCH_SIZE)
    print('epochs:', EPOCHS)
    print('l-rate:', LEARNING_RATE)
    print('momentum:', MOMENTUM)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    print('====================')

    print('-> Loading datasets')
    train_dataset = MaskImageDataset(os.path.join(data_path, 'train'), transform=mask_image_train_transform2)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_size = len(train_dataset)

    test_dataset = MaskImageDataset(os.path.join(data_path, 'test'), transform=mask_image_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    print('-> Initalizing model')
    model = MaskDetectionModel()
    model.to(device)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}\n---------------------------")
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

            if i % PRINT_STEPS == 0:
                # test_accuracy = calc_accuracy(model, test_loader, device, limit=100)
                print(f'Loss: {L.item():>7f}  [{i * len(labels):>5d}/{train_size:>5d}]')

        print('Test accuracy = ', calc_accuracy(model, test_loader, device, limit=TEST_LIMIT_SIZE))

        print('-> Saving state')
        torch.save(model.state_dict(), 'model.state')
