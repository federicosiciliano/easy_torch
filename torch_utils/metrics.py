import torch.nn.functional as F
import torchmetrics

def nn_accuracy(y_hat,y):
    soft_y_hat = F.softmax(y_hat).argmax(dim=-1)
    soft_y = y.argmax(dim=-1)
    acc = (soft_y_hat.int() == soft_y.int()).float().mean()

    return acc

#Custom Accuracy to compute accuracy with Soft Labels
class SoftLabelsAccuracy(torchmetrics.Accuracy):
    def init(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.__name__ = "custom_accuracy"

    def forward(self, y_hat, y):
        hard_y_hat = y_hat.argmax(dim=-1)
        hard_y = y.argmax(dim=-1)
        acc = (hard_y_hat.int() == hard_y.int()).float().mean()

        return acc