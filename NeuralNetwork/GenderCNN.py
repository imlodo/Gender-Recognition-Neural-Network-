#Definizione di una rete neurale convoluzionale
import os

import torch
import torch.nn as nn
from PIL import Image

from NeuralNetwork.Utils import checkGPUAvailable
import torch.functional as F
import torch.optim as optim
from NeuralNetwork.Training import train
from NeuralNetwork.Dataset import train_data_loader
from NeuralNetwork.Dataset import val_data_loader
from NeuralNetwork.Dataset import img_transforms

REBUILD_DATA = True
EPOCHS = 30
# Qui controlliamo se è presente una GPU per eseguire calcoli più veloci
deviceSelected = checkGPUAvailable()

class GenderCNN(nn.Module):
    def __init__(self, num_classes = 2):
        super(GenderCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=11,stride=4,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if REBUILD_DATA:
    genderCNN = GenderCNN().to(deviceSelected)
    # Per eseguire update dei pesi della rete neurale bisogna utilizzare un optimizer
    optimizer = optim.Adam(genderCNN.parameters(), lr=0.001)  # gendernet.parameters() sono i pesi della rete neurale
    # Training
    # Call train function
    train(genderCNN, optimizer, torch.nn.CrossEntropyLoss(), train_data_loader, val_data_loader, epochs=EPOCHS,
          device=deviceSelected)
    # Saving Models
    torch.save(genderCNN, "genderCNN.pth")

genderCNN = torch.load("genderCNN.pth")

# Making predictions
path = "../Dataset/test/"
arr = os.listdir(path)
labels = ['man', 'woman']
for folder in arr:
    for img in os.listdir(path + folder):
        try:
            print("Name: " + img, end=" ------------- ")
            img = Image.open(path + folder + "/" + img)
            img = img_transforms(img).to(deviceSelected)
            prediction = F.softmax(genderCNN(img))
            prediction = prediction.argmax()
            print("Prediction: " + labels[prediction])
        except UserWarning as uw:
            pass