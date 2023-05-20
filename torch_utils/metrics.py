import torch.nn.functional as F

def nn_accuracy(y_hat,y):
    soft_y_hat = F.softmax(y_hat).argmax(dim=-1)
    soft_y = y.argmax(dim=-1)
    acc = (soft_y_hat.int() == soft_y.int()).float().mean()

    return acc