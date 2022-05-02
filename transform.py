import torch
from torchvision.transforms import Compose
from torchvision.transforms import InterpolationMode
from torchvision.transforms import Resize
from torchvision.transforms import RandomCrop
from torchvision.transforms import ToTensor
from torchvision.transforms import RandomVerticalFlip
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomGrayscale
from torchvision.transforms import ColorJitter


# Transform with resize to (256, 256)
mask_image_train_transform = Compose([
    Resize((128,128), InterpolationMode.BILINEAR),
    # RandomCrop((128,128)),
    ToTensor(),
    RandomHorizontalFlip(),
    # RandomVerticalFlip(),
    RandomGrayscale(),
    # ColorJitter(brightness=0.5, hue=0.5, contrast=0.5)
])

mask_image_test_transform = Compose([
    Resize((224,224), InterpolationMode.BILINEAR),
    ToTensor()
])
