# Find Learning rate
import math
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import nn as nn
from NeuralNetwork.Dataset import train_data_loader
from NeuralNetwork.GenderCNN import GenderCNN

def find_lr(model, loss_fn, optimizer, init_value=1e-8, final_value=10.0):
    number_in_epoch = len(train_data_loader) - 1
    update_step = (final_value / init_value) ** (1 / number_in_epoch)
    lr = init_value
    optimizer.param_groups[0]["lr"] = lr
    best_loss = 0.0
    batch_num = 0
    losses = []
    log_lrs = []
    for data in tqdm(train_data_loader):
        batch_num += 1
        inputs, labels = data
        inputs, labels = inputs, labels
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        # Crash out if loss explodes
        if batch_num > 1 and loss > 4 * best_loss:
            return log_lrs[10:-5], losses[10:-5]
        # Record the best loss
        if loss < best_loss or batch_num == 1:
            best_loss = loss
        # Store the values
        losses.append(loss)
        log_lrs.append(math.log10(lr))
        # Do the backward pass and optimize
        loss.backward()
        optimizer.step()
        # Update the lr for the next step and store
        lr *= update_step
        optimizer.param_groups[0]["lr"] = lr
    return log_lrs[10:-5], losses[10:-5]

modello = GenderCNN()
opt = torch.optim.Adam(modello.parameters(), lr=0.001)  # gendernet.parameters() sono i pesi della rete neurale
logs, lossL = find_lr(modello, torch.nn.CrossEntropyLoss(), opt)
plt.plot(logs, lossL)
found_lr = 1e-2
