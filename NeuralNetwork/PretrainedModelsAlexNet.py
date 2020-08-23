import torchvision.models as models
import os
import shutil
import torch
import torch.nn as nn
from PIL import Image
from NeuralNetwork.Utils import checkGPUAvailable
import torch.nn.functional as F
import torch.optim as optim
from NeuralNetwork.Training import train
from NeuralNetwork.Dataset import train_data_loader
from NeuralNetwork.Dataset import val_data_loader
from NeuralNetwork.Dataset import img_transforms

REBUILD_DATA = True
# Qui controlliamo se è presente una GPU per eseguire calcoli più veloci
deviceSelected = checkGPUAvailable()

if REBUILD_DATA:
    alexnet = (models.alexnet(num_classes=2)).to(deviceSelected)
    torch.save(alexnet, "ALEXNET.pth")

genderCNN = torch.load("ALEXNET.pth")

 # Making predictions
path = "../Dataset/test/"
arr = os.listdir(path)
labels = ['man', 'woman']
for folder in arr:
    for img in os.listdir(path + folder):
        try:
            print("Name: " , path + folder, "/",img, end=" ------------- ")
            img = Image.open(path + folder + "/" + img)
            img = img_transforms(img).to(deviceSelected)
            img = img.view(-1,3,64,64)
            prediction = F.softmax(genderCNN(img), dim=1)
            prediction = prediction.argmax()
            print("Prediction: " + labels[prediction])
        except Exception as e:
            pass