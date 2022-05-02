import os
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor

class MaskImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.dir = img_dir
        self.image_files = [os.path.join(img_dir, i) for i in os.listdir(img_dir)]
        self.transform = transform
        self.pil2tensor = ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # image = self.pil2tensor(Image.open(self.image_files[idx]))
        image = Image.open(self.image_files[idx])
        label = int(os.path.splitext(self.image_files[idx])[0].split('_')[1])
        if self.transform:
            image = self.transform(image)
        return image, label
