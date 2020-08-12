# Definizione di una rete neurale fully connected
import os
from NeuralNetwork.Training import train
from NeuralNetwork.Utils import checkGPUAvailable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from NeuralNetwork.Dataset import train_data_loader
from NeuralNetwork.Dataset import val_data_loader
from NeuralNetwork.Dataset import img_transforms

REBUILD_DATA = False
EPOCHS = 30

# Qui controlliamo se è presente una GPU per eseguire calcoli più veloci
deviceSelected = checkGPUAvailable()

# Creating a Neural Network class
class GenderNet(nn.Module):

    def __init__(self):
        super().__init__()
        # Qui creo i layer lineari (ovvero layer fully connected)
        self.fc1 = nn.Linear(12288, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    # la forward è la funzione che ci dice come i dati verranno trasportati attraverso la rete
    def forward(self, x):
        x = x.view(-1, 12288)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if REBUILD_DATA:
    gendernet = GenderNet().to(deviceSelected)
    # Per eseguire update dei pesi della rete neurale bisogna utilizzare un optimizer
    optimizer = optim.Adam(gendernet.parameters(), lr=0.001)  # gendernet.parameters() sono i pesi della rete neurale
    # Training
    # Call train function
    train(gendernet, optimizer, torch.nn.CrossEntropyLoss(), train_data_loader, val_data_loader, epochs=EPOCHS,
          device=deviceSelected)
    # Saving Models
    torch.save(gendernet, "gendernet.pth")

gendernet = torch.load("gendernet.pth")

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
            prediction = F.softmax(gendernet(img))
            prediction = prediction.argmax()
            print("Prediction: " + labels[prediction])
        except UserWarning as uw:
            pass

# path = "../Dataset/not"
# arr = os.listdir(path)
# for k in arr:
#     try:
#         img = Image.open(path + "/" + k)
#         img = img_transforms(img).to(deviceSelected)
#         prediction = F.softmax(gendernet(img))
#         prediction = prediction.argmax()
#         print(labels[prediction], "\n")
#         if labels[prediction] == "man":
#             os.rename(path + "/" + k, "../Dataset/man/" + k)
#             shutil.move(path + "/" + k, "../Dataset/man/" + k)
#             os.replace(path + "/" + k, "../Dataset/man/" + k)
#         else:
#             os.rename(path + "/" + k, "../Dataset/woman/" + k)
#             shutil.move(path + "/" + k, "../Dataset/woman/" + k)
#             os.replace(path + "/" + k, "../Dataset/woman/" + k)
#     except Exception as e:
#         print(e)
#         pass
