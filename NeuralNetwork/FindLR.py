"""This class find Learning rate"""
from numpy import loadtxt
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import nn as nn
from NeuralNetwork.Dataset import train_data_loader
import numpy as np

# noinspection PyAbstractClass
class GenderCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(GenderCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
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


def find_lr(model, loss_fn, optimizer, train_loader, init_value=1e-8, final_value=10.0, device="cpu"):

    number_in_epoch = len(train_loader) - 1
    update_step = (final_value / init_value) ** (1 / number_in_epoch)
    lr = init_value
    optimizer.param_groups[0]["lr"] = lr
    best_loss = 0.0
    batch_num = 0
    lossS = []
    log_lrs = []

    for data in tqdm(train_loader):
        batch_num += 1
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        # Crash out if loss explodes
        if batch_num > 1 and loss > 4 * best_loss:
            if len(log_lrs) > 20:
                return log_lrs[10:-5], lossS[10:-5]
            else:
                return log_lrs, lossS
        # Record the best loss
        if loss < best_loss or batch_num == 1:
            best_loss = loss
        # Store the values
        lossS.append(loss.item())
        log_lrs.append(lr)
        # Do the backward pass and optimize
        loss.backward()
        optimizer.step()
        # Update the lr for the next step and store
        lr *= update_step
        optimizer.param_groups[0]["lr"] = lr

    if len(log_lrs) > 20:
        return log_lrs[10:-5], lossS[10:-5]
    else:
        return log_lrs, lossS


# modello = GenderCNN()
# opt = torch.optim.Adam(modello.parameters(), lr=0.001)  # modello.parameters() sono i pesi della rete neurale
# (lrs, losses) = find_lr(modello, torch.nn.CrossEntropyLoss(), opt, train_data_loader)
# print(lrs, losses)
# plt.plot(lrs, losses)
# plt.grid()
# plt.xscale("log")
# plt.xlabel("Learning rate")
# plt.ylabel("Loss")
# plt.show()
#
# np.savetxt('data.txt', lrs, delimiter=',')   #lrs is an array (x)
# np.savetxt("data2.txt", losses, delimiter=',') #losses is an array (y)

# load array
lrs = loadtxt('data.txt', delimiter=',')
losses = loadtxt('data2.txt', delimiter=',')

"""Calculate Gradient Descent"""
def calculateGradientDescent(x_array,y_array):
    arr_x = [] # questo array contiene coppie di x
    arr_y = [] # questo array contiene coppie di y
    for i in range(0, len(x_array) - 1):
        arr_x += [(x_array[i], x_array[i + 1])]
        arr_y += [(y_array[i], y_array[i + 1])]
    punto_num2 = 0
    min_value = 1000
    for i in range(0, len(arr_y)):
        primo, secondo = arr_y[i]
        primo_x, secondo_x = arr_x[i]
        tmp = secondo - primo
        tmp_x = secondo_x - primo_x
        if tmp < 0 and tmp/tmp_x < min_value:
            min_value = tmp/tmp_x
            punto_num2 = i
    return punto_num2

lrs_float = []
losses_float = []
for item in lrs:
    lrs_float.append(float(item))
for item in losses:
    losses_float.append(float(item))

gradientDescOriginal = calculateGradientDescent(lrs_float,losses_float)
plt.subplot(2,2,1)
plt.title("Original")
plt.grid()
# plt.xscale("log")
plt.xlabel("Learning rate")
found_lr = str(lrs_float[gradientDescOriginal])
plt.xlabel("Learning rate is: " + found_lr)
plt.ylabel("Loss")
plt.plot(lrs_float,losses_float)
# # plt.plot(xx, yyp2)
# # plt.plot(xx, yyp3)
plt.plot(lrs_float[gradientDescOriginal],losses_float[gradientDescOriginal],'o', color="red")

# Interpolazione
plt.grid()
xx = np.linspace(0.0, max(lrs), 1000)
p2 = np.polyfit(lrs_float, losses_float, 2)
yyp2 = np.polyval(p2, xx)
plt.subplot(2,2,3)
plt.title("Interpolated using the method of least squares (deg = 2)")
plt.plot(xx,yyp2)
gradientDescIp2 = calculateGradientDescent(xx,yyp2)
plt.plot(xx[gradientDescIp2],yyp2[gradientDescIp2],'o', color="red")
plt.xlabel("Learning rate")
plt.ylabel("Loss")

plt.grid()
p3 = np.polyfit(lrs_float, losses_float, 3)
yyp3 = np.polyval(p3, xx)
plt.subplot(2,2,4)
plt.plot(xx,yyp3)
gradientDescIp3 = calculateGradientDescent(xx,yyp3)
plt.plot(xx[gradientDescIp3],yyp3[gradientDescIp3],'o', color="red")
plt.title('Interpolated using the method of least squares (deg = 3)')
plt.xlabel("Learning rate")
plt.ylabel("Loss")

plt.grid()
plt.subplot(2,2,2)
plt.plot(lrs, losses, 'o')
plt.plot(lrs_float[gradientDescOriginal],losses_float[gradientDescOriginal],'x',color="red")
plt.title('Point Cloud')
plt.xlabel("Learning rate")
plt.ylabel("Loss")

#Adjust space subplots
plt.subplots_adjust(wspace=None, hspace=1)
#Show plt
plt.show()
