#Questa architettura riconosce 1000 classi di ImageNet
#https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
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

REBUILD_DATA = False
# Qui controlliamo se è presente una GPU per eseguire calcoli più veloci
deviceSelected = checkGPUAvailable()
model = (torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)).to(deviceSelected)

if REBUILD_DATA:
    genderCNN = model
    torch.save(genderCNN, "ResNet50.pth")

genderCNN = torch.load("ResNet50.pth")

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
            img = img.view(-1,3,64,64)
            with torch.no_grad():
                output = genderCNN(img)
                prediction = torch.nn.functional.softmax(output[0], dim=0)
                prediction = prediction.argmax()
                print(prediction)
        except UserWarning as uw:
            pass
