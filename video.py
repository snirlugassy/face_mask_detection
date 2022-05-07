import cv2
import sys
import pickle
import time
import os

from PIL import Image
import torch
import pickle
from torch.utils.data.dataloader import DataLoader

from transform import mask_image_test_transform

from model import MaskDetectionModel
from dataset import MaskImageDataset
from utils import calc_accuracy


video_capture = cv2.VideoCapture(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MaskDetectionModel()
model.load_state_dict(torch.load('model.state', map_location=device))

avg_prob = 0.0
T = 0
while True:
    if T >= 30:
        T = 0
        avg_prob = 0.0

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    tensor = mask_image_test_transform(Image.fromarray(frame)).reshape(1,1,128,128)
    model_output = torch.softmax(model(tensor), dim=1)
    has_mask = int(model_output.argmax(dim=1))
    mask_prob = float(model_output[:,1])

    T += 1
    avg_prob -= avg_prob / T
    avg_prob += mask_prob / T
    # avg_prob = (1/T)*mask_prob + (1-1/T)*avg_prob

    if has_mask:
        text = 'WITH MASK'
        c = (0,255,0)
    else:
        text = 'NO MASK'
        c = (0,0,255)

    cv2.putText(frame, text, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)
    cv2.putText(frame, f'MOVING AVG PROB:{round(avg_prob, 5):.5}', (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()