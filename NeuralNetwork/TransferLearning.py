import os
import torch
from PIL import Image
from torch import nn
import torchvision.models as models
from NeuralNetwork.Dataset import img_transforms, train_data_loader, val_data_loader
from NeuralNetwork.Training import train
from NeuralNetwork.Utils import checkGPUAvailable
import torch.nn.functional as F

#carichiamo il modello pre-addestrato (In questo caso ResNet50)
transfer_model = models.resnet50(pretrained=True)

#Congeliamo tutti i layers cosi da non invalidare il modello pre-addestrato
for name, param in transfer_model.named_parameters():
    param.requires_grad = False

#Aggiungiamo i nostri layers
transfer_model.fc = nn.Sequential(nn.Linear(transfer_model.fc.in_features,500),
                                  nn.ReLU(),
                                  nn.Dropout(),
                                  nn.Linear(500,2))

REBUILD_DATA = True
EPOCHS = 5
# Qui controlliamo se è presente una GPU per eseguire calcoli più veloci
deviceSelected = checkGPUAvailable()

if REBUILD_DATA:
    transfer_model = transfer_model.to(deviceSelected)
    # Per eseguire update dei pesi della rete neurale bisogna utilizzare un optimizer
    optimizer = torch.optim.Adam(transfer_model.parameters(), lr=0.001)  # gendernet.parameters() sono i pesi della rete neurale
    # Training
    # Call train function
    train(transfer_model, optimizer, torch.nn.CrossEntropyLoss(), train_data_loader, val_data_loader, epochs=EPOCHS,
          device=deviceSelected)
    # Saving Models
    torch.save(transfer_model, "transfer_model.pth")

transfer_model = torch.load("transfer_model.pth")

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
            prediction = F.softmax(transfer_model(img))
            prediction = prediction.argmax()
            print("Prediction: " + labels[prediction])
        except UserWarning as uw:
            pass