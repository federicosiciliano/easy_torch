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
        print("NEW ACCURACY")

    def forward(self, y_hat, y):
        hard_y = y.argmax(dim=-1)
        print(hard_y)
        acc = super().forward(y_hat, hard_y)
        return acc