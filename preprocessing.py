import os
from PIL import Image

train_images_dir = './data/train'

for i in os.listdir(train_images_dir):
    im = Image.open(os.path.join(train_images_dir, i))
    im = im.resize((256,256), Image.BILINEAR)
    im.save(os.path.join('./data/train256', i))
