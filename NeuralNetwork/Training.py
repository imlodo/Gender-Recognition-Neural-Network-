# Creiamo la funzione di training della nostra rete neurale
import torch
import torch.nn.functional as F
from tqdm import tqdm


def train(model, opt, lossFn, train_loader, val_loader, epochs=10, device="cuda"):
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