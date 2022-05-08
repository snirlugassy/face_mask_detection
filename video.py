import cv2
from PIL import Image
import torch
from transform import mask_image_test_transform
from model import MaskDetectionModel

BLACK = (0,0,0)
WHITE = (255,255,255)
GREEN = (0,255,0)
RED = (0,0,255)

video_capture = cv2.VideoCapture(0)
ret, frame = video_capture.read()
h,w,_ = frame.shape

assert w > h

block_size = int(h // 1.5)
step = int(block_size // 2)
x_center = int(w // 2)
y_center = int(h // 2)

rect_top_left = (x_center - step, y_center + step)
rect_bottom_right = (x_center + step, y_center - step)

print(rect_top_left, rect_bottom_right)
print('heigh:',h)
print('weight:',w)
print('block:', block_size, 'step:', step)

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

    rect = frame[y_center-step:y_center+step,x_center-step:x_center+step]
    tensor = mask_image_test_transform(Image.fromarray(rect)).reshape(1,1,128,128)
    model_output = torch.softmax(model(tensor), dim=1)
    has_mask = int(model_output.argmax(dim=1))
    mask_prob = float(model_output[:,1])

    T += 1
    avg_prob -= avg_prob / T
    avg_prob += mask_prob / T

    if has_mask:
        text = 'VALID MASK'
        c = GREEN
    else:
        text = 'INVALID MASK'
        c = RED

    cv2.rectangle(frame, rect_top_left, rect_bottom_right, BLACK, 2)
    cv2.putText(frame, text, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)
    cv2.putText(frame, f'AVG PROB:{round(avg_prob, 5):.5}', (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()