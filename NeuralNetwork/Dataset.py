#Load datasets
import torchvision
from torchvision import transforms
train_data_path = "../Dataset/train"
transforms = transforms.Compose([
    transforms.Resize(64), #ridimensioniamo le immagini a 64x64 (per aumentare le prestazioni)
    transforms.ToTensor(), #trasforma le immagini ridimensionate in un tensore
    #qui normalizziamo ad un insieme specifico di punti di media e deviazione standard, la normalizzazione è importante
    #per evitare di avere numeri molto grandi quando vengono effettuate le moltiplicazioni passando i dati tra i vari
    # strati della rete neurale, cosi evitiamo il problema di exploding gradient.
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
#Qui sfruttiamo il metodo ImageFolder che ci permette di avere un datasets con i label corrispondenti alle cartelle
# presenti in Datasets/train/ e le immagini corrispondenti a quei label.
train_data = torchvision.datasets.ImageFolder(root=train_data_path,transform=transforms)
#train_data è il dataset di training della nostra rete neurale

#qui creiamo il dataset di validazione per capire ad ogni epoca se la rete neurale sta imparando qualcosa
val_data_path = "../Dataset/validate"
val_data = torchvision.datasets.ImageFolder(root=val_data_path, transform=transforms) #dataset di validazione della nostra rete neurale

#qui creiamo il dataset di test per testare la rete neurale dopo che la fase di training è stata completata.
test_data_path = "../Dataset/test"
test_data = torchvision.datasets.ImageFolder(root=test_data_path,transform=transforms) #dataset di test della nostra rete neurale


