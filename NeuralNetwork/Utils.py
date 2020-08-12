import torch

def checkGPUAvailable():
    # Qui controlliamo se è presente una GPU per eseguire calcoli più veloci
    if torch.cuda.is_available():
        print("GPU available")
        return torch.device("cuda")
    else:
        print("GPU not available")
        return torch.device("cpu")