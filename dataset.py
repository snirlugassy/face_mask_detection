import os
import torch
from torch.utils.data.dataset import Dataset
from torchvision.io import read_image

class MaskImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.dir = img_dir
        self.image_files = [os.path.join(img_dir, i) for i in os.listdir(img_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = read_image(self.image_files[idx])
        label = int(os.path.splitext(self.image_files[idx])[0].split('_')[1])
        if self.transform:
            image = self.transform(image)
        return image.float(), label
