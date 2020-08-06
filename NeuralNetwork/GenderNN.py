import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from NeuralNetwork.Dataset import train_data_loader
from NeuralNetwork.Dataset import val_data_loader
from NeuralNetwork.Dataset import test_data_loader
from tqdm import tqdm
from NeuralNetwork.Dataset import img_transforms
import cv2

REBUILD_DATA = False

# Qui controlliamo se è presente una GPU per eseguire calcoli più veloci
if torch.cuda.is_available():
    deviceSelected = torch.device("cuda")
else:
    deviceSelected = torch.device("cpu")


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


# Creiamo la funzione di training della nostra rete neurale
def train(model, opt, lossFn, train_loader, val_loader, epochs=10, device="cpu"):
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        num_correct = 0
        num_examples = 0
        for batch in tqdm(train_loader):
            # azzera i gradienti calcolati per il batch successivo altrimenti, il batch successivo dovrà occuparsi inutilmente anche dei gradienti già calcolati nel batch precedente.
            opt.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)  # previsione rete neurale
            loss = lossFn(output, targets)  # calcolo perdità NN
            loss.backward()  # calcolo dei gradienti
            opt.step()  # ottimizzazione dei pesi per questo step
            correct = torch.eq(torch.max(F.softmax(output), dim=1)[1], targets).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(train_loader.dataset)
        print('\nEpoch: {}, Training Loss: {:.2f}, accuracy = {:.2f}\n'.
              format(epoch, training_loss * 100, num_correct / num_examples * 100))

        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in tqdm(val_loader):
            input, target = batch
            input = input.to(device)
            output = model(input)
            target = target.to(device)
            loss = lossFn(output, target)
            valid_loss += loss.data.item() * input.size(0)
            correct = torch.eq(torch.max(F.softmax(output), dim=1)[1], target).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader.dataset)

        print('\nEpoch: {}, Validation Loss: {:.2f}, accuracy = {:.2f}\n'.
              format(epoch, valid_loss * 100, num_correct / num_examples * 100))


if REBUILD_DATA:
    gendernet = GenderNet().to(deviceSelected)
    # Per eseguire update dei pesi della rete neurale bisogna utilizzare un optimizer
    optimizer = optim.Adam(gendernet.parameters(), lr=0.001)  # gendernet.parameters() sono i pesi della rete neurale
    # Training
    EPOCHS = 15
    # Call train function
    train(gendernet, optimizer, torch.nn.CrossEntropyLoss(), train_data_loader, val_data_loader, epochs=EPOCHS,
          device=deviceSelected)
    # Saving Models
    torch.save("../model/gendernet")

gendernet = torch.load("../model/gendernet")

# Making predictions
labels = ['man', 'woman']
img = Image.open("../Dataset/test/Man/WhatsApp Image 2020-08-03 at 10.40.53.jpeg")
img = img_transforms(img).to(deviceSelected)
prediction = F.softmax(gendernet(img))
prediction = prediction.argmax()
print(labels[prediction])
