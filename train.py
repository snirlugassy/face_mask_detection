import torch
from torch.utils.data.dataloader import DataLoader

from transform import mask_image_transform, mask_256_image_transform
from model import MaskDetectionModel
from dataset import MaskImageDataset

BATCH_SIZE = 10
EPOCHS = 1
LEARNING_RATE = 1e-3
INPUT_DIM = (224, 224)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device =', device)

    # With resizing:
    # face_mask_dataset = MaskImageDataset('./data/train', transform=mask_image_transform)

    print('Loading dataset')
    face_mask_dataset = MaskImageDataset('./data/train256', transform=mask_256_image_transform)
    train_loader = DataLoader(face_mask_dataset, batch_size=BATCH_SIZE, shuffle=True)
    N = len(train_loader)

    print('Initalizing model')
    model = MaskDetectionModel()
    model.to(device)

    loss = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCHS):
        print('Starting epoch', epoch+1)
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

            if i % 10 == 0:
                M = N//BATCH_SIZE
                print(f'Epoch: [{epoch+1}/{EPOCHS}], Step: [{i+1}/{M}], Loss: {L}')
        
        torch.save(model.state_dict(), 'model.state')
