from torchvision.transforms import Compose
from torchvision.transforms import InterpolationMode
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import Grayscale
from torchvision.transforms import Normalize

TRAIN_PIXEL_MEAN = 0.468
TRAIN_PIXEL_STD = 0.2251

mask_image_train_transform = Compose([
    Resize((128,128), InterpolationMode.BILINEAR),
    Grayscale(),
    ToTensor(),
    Normalize(TRAIN_PIXEL_MEAN, TRAIN_PIXEL_STD),
    RandomHorizontalFlip()
])

mask_image_test_transform = Compose([
    Resize((128,128), InterpolationMode.BILINEAR),
    Grayscale(),
    ToTensor(),
    Normalize(TRAIN_PIXEL_MEAN, TRAIN_PIXEL_STD)
])
