import os
import gc
import csv
import argparse

import torch
from torch.utils.data.dataloader import DataLoader

from transform import mask_image_train_transform
from transform import mask_image_test_transform
from model import MaskDetectionModel
from dataset import MaskImageDataset
from utils import calc_confusion_mat

OPTIMIZERS = {
    'adam': torch.optim.Adam,
    'adadelta': torch.optim.Adadelta,
    'sgd': torch.optim.SGD,
    'adagrad': torch.optim.Adagrad
}

if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()

    argparser = argparse.ArgumentParser(description='Train mask detection nerual network')
    argparser.add_argument('--data-path', type=str, required=True, dest='data_path')
    argparser.add_argument('--batch-size', type=int, dest='batch_size')
    argparser.add_argument('--epochs', type=int, dest='epochs')
    argparser.add_argument('--optimizer', type=str, dest='optimizer', choices=OPTIMIZERS.keys())
    argparser.add_argument('--lr', type=float, dest='lr')
    argparser.add_argument('--print-steps', type=int, dest='print_steps')

    args = argparser.parse_args()

    BATCH_SIZE = args.batch_size or 100
    EPOCHS = args.epochs or 50
    LEARNING_RATE = args.lr or 0.01
    PRINT_STEPS = args.print_steps or 20
    OPTIMIZER = args.optimizer or 'adam'

    if OPTIMIZER not in OPTIMIZERS:
        OPTIMIZER = 'adam'

    data_path = args.data_path

    print('====== TRAIN =======')
    print('optimizer:', OPTIMIZER)
    print('batch-size:', BATCH_SIZE)
    print('epochs:', EPOCHS)
    print('l-rate:', LEARNING_RATE)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    print('====================')

    print('-> Loading datasets')
    train_dataset = MaskImageDataset(os.path.join(data_path, 'train'), transform=mask_image_train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_size = len(train_dataset)

    test_dataset = MaskImageDataset(os.path.join(data_path, 'test'), transform=mask_image_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    print('-> Initalizing model')
    model = MaskDetectionModel()
    model.to(device)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = OPTIMIZERS[OPTIMIZER](model.parameters(), lr=LEARNING_RATE)

    output = []
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}\n---------------------------")
        train_loss = 0.0
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            y_prob = model(images)
            L = loss(y_prob, labels)
            train_loss += L.item() * images.size(0)

            # Backpropagation
            L.backward()
            optimizer.step()

            if i % PRINT_STEPS == 0:
                print(f'Loss: {L.item():>7f}  [{i * len(labels):>5d}/{train_size:>5d}]')

        test_loss = 0.0
        model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):

                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                y_prob = model(images)
                L = loss(y_prob, labels)
                test_loss += L.item() * images.size(0)


        train_tp, train_fp, train_fn, train_tn = calc_confusion_mat(train_loader, model, device)
        test_tp, test_fp, test_fn, test_tn = calc_confusion_mat(test_loader, model, device)

        output.append({
            'epoch': epoch,
            'train_loss': train_loss / len(train_dataset),
            'test_loss': test_loss / len(test_dataset),
            'train_tp': train_tp,
            'train_fp': train_fp, 
            'train_fn': train_fn, 
            'train_tn': train_tn,
            'test_tp': test_tp,
            'test_fp': test_fp, 
            'test_fn': test_fn, 
            'test_tn': test_tn
        })
        print('-> Saving state')
        torch.save(model.state_dict(), 'model.state')

    print('Saving output CSV')
    with open('output.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(output[0].keys()))
        writer.writeheader()
        for x in output:
            writer.writerow(x)
