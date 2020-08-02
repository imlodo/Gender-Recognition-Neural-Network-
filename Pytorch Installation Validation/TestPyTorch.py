# Questo file serve a verificare se Pytorch funziona correttamente.

#Eseguiamo l'import della libreria PyTorch
import torch
#Verifichiamo che la GPU sia abilitata per eseguire calcoli sfruttando la potenza della GPU
print(torch.cuda.is_available())
#Verifichiamo che venga creato un tensore 2x2
print(torch.rand(2,2))

#Nel caso in cui la prima print restituisce False, allora bisogna eseguire il debug dell'installazione di CUDA, in modo
#che PyTorch possa rilevare la scheda grafica.
