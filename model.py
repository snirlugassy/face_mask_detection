import torch
from torch import nn

class MaskDetectionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding='same'),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same'),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same'),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2)
        # )

        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding='same'),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same'),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same'),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2)
        # )

        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same'),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same'),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same'),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2)
        # )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128*32*32, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=2),
        )

    def forward(self, x):
        # Convolutional
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        
        # Fully connected
        x = self.fc(x)
        return x
