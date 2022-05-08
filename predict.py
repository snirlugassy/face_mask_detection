import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from torch.utils.data.dataloader import DataLoader

from model import MaskDetectionModel
from dataset import MaskImageDataset
from  transform import mask_image_test_transform
from utils import calc_scores

if __name__ == '__main__':
    data_path = sys.argv[-1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MaskDetectionModel()
    model.load_state_dict(torch.load('model.state', map_location=device))

    image_files = os.listdir(data_path)
    results = []
    for img in image_files:
        im = Image.open(os.path.join(data_path, img))
        t = mask_image_test_transform(im).unsqueeze(0)
        predicted = torch.softmax(model(t), dim=1).argmax(dim=1)
        predicted = int(predicted)
        results.append((img, predicted))


    with open('prediction.csv', 'w') as output_file:
        for result in results:
            output_file.write(f'{result[0]},{result[1]}\n')
