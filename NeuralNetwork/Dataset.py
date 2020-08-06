#Load datasets
import torch
import torchvision
from PIL import Image
from torchvision import transforms
train_data_path = "../Dataset/train"
img_transforms = transforms.Compose([
    transforms.Resize((64,64)), #ridimensioniamo le immagini a 64x64 (per aumentare le prestazioni)
    transforms.ToTensor(), #trasforma le immagini ridimensionate in un tensore
    #qui normalizziamo ad un insieme specifico di punti di media e deviazione standard, la normalizzazione è importante
    #per evitare di avere numeri molto grandi quando vengono effettuate le moltiplicazioni passando i dati tra i vari
    # strati della rete neurale, cosi evitiamo il problema di exploding gradient.
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def check_image(path):
    try:
        im = Image.open(path)
        return True
    except:
        return False

#Qui sfruttiamo il metodo ImageFolder che ci permette di avere un datasets con i label corrispondenti alle cartelle
# presenti in Datasets/train/ e le immagini corrispondenti a quei label.
train_data = torchvision.datasets.ImageFolder(root=train_data_path,transform=img_transforms,is_valid_file=check_image)
#train_data è il dataset di training della nostra rete neurale

#qui creiamo il dataset di validazione per capire ad ogni epoca se la rete neurale sta imparando qualcosa
val_data_path = "../Dataset/validate"
val_data = torchvision.datasets.ImageFolder(root=val_data_path, transform=img_transforms, is_valid_file=check_image) #dataset di validazione della nostra rete neurale

#qui creiamo il dataset di test per testare la rete neurale dopo che la fase di training è stata completata.
test_data_path = "../Dataset/test"
test_data = torchvision.datasets.ImageFolder(root=test_data_path,transform=img_transforms, is_valid_file=check_image) #dataset di test della nostra rete neurale

#DataLoaders
batch_size = 64 #numero di immagini passate alla rete neurale ogni iterazione
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_data_loader = torch.utils.data.DataLoader(val_data,batch_size=batch_size, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size, shuffle=False)