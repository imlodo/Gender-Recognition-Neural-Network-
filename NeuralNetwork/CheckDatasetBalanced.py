#Questo file serve per capire se il dataset è bilanciato, infatti dataset non bilanciati, non permettono di avere reti neurali efficienti e si può ricadere in overfitting
from distutils.command.check import check
import NeuralNetwork.Dataset
from NeuralNetwork.Dataset import train_data
from NeuralNetwork.Dataset import val_data
from NeuralNetwork.Dataset import test_data
from tqdm import tqdm
import time

#Funzione per capire se il dataset è bilanciato.
def checkBalanced(dataset):
    # Nel nostro caso abbiamo due label, pertanto creiamo un dizionario di 2 elementi per mantenere il conto di rispettive imamgini di uomo e donna lette
    counter_dict = {i: 0 for i in range(2)}
    count_total = 0
    # tqdm lo utilizziamo per avere una progress bar che ci informa dell'avanzamento dell'operazione (essendo un operazione molto lunga, per via delle numerose immagini)
    for data in tqdm(dataset):
        xs, ys = data
        counter_dict[int(ys)] += 1
        count_total += 1
    # Nella stampa 0:numb indica il numero di immagini di uomini e 1:numb il numero di immagini di donne
    print("\n", counter_dict, "\n")
    for i in counter_dict:
        print(
            f"{i}:{counter_dict[i] / count_total * 100.0}%\n")  # stampiamo in percentuale l'incidenza dei due tipi di immagini
    time.sleep(3)

#__________________Qui testiamo se il dataset di traning è bilanciato______________
checkBalanced(train_data)
#__________________Qui testiamo se il dataset di validazione è bilanciato______________
checkBalanced(val_data)
#__________________Qui testiamo se il dataset di test è bilanciato______________
checkBalanced(test_data)