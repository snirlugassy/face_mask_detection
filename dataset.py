import os
from PIL import Image
from torch.utils.data.dataset import Dataset

class MaskImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, limit=None):
        self.dir = img_dir
        self.image_files = [os.path.join(img_dir, i) for i in os.listdir(img_dir)]
        if isinstance(limit, int):
            self.image_files = self.image_files[:limit]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx])
        label = int(os.path.splitext(self.image_files[idx])[0].split('_')[1])
        if self.transform:
            image = self.transform(image)
        return image, label
