import torch
import pytorch_lightning as pl

class BaseNN(pl.LightningModule):
    def __init__(self, main_module, loss, optimizer, metrics={}, **kwargs):
        super().__init__()

        self.main_module = main_module
        self.loss = loss #loss_name: {log_params}
        self.metrics = metrics
        self.optimizer = optimizer

    def forward(self, x):
        return self.main_module(x)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        return optimizer

    def step(self, batch, batch_idx, split):
        x,y = batch

        y_hat = self(x)

        loss = self.loss(y_hat, y)

        #TODO: create LOG before
        self.log(split+'_loss', loss) #miss log params

        #on_step=False, on_epoch=True, logger=True

        #compute other metrics
        for metric_name,metric_func in self.metrics:
            metric_value = metric_func(y_hat,y)
            self.log(split+'_'+metric_name, metric_value) #miss log params
        
        return loss

    def training_step(self, batch, batch_idx): return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx): return self.step(batch, batch_idx, "val")
        
    def test_step(self,batch,batch_idx): return self.step(batch, batch_idx, "test")

def get_torchvision_model(*args, **kwargs): return torchvision_utils.get_torchvision_model(*args, **kwargs)

def load_torchvision_model(*args, **kwargs): return torchvision_utils.load_torchvision_model(*args, **kwargs)

class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x


# class MLP(BaseNN):
#     def __init__(self, input_size, output_size, neurons_per_layer, activation_function=None, lr=None, loss = None, acc = None, **kwargs):
#         super().__init__()

#         layers = []
#         in_size = input_size
#         for out_size in neurons_per_layer:
#             layers.append(torch.nn.Linear(in_size, out_size))
#             if activation_function is not None:
#                 layers.append(getattr(torch.nn, activation_function)())
#             in_size = out_size
#         layers.append(torch.nn.Linear(in_size, output_size))
#         self.main_module = torch.nn.Sequential(*layers)


from . import torchvision_utils #put here otherwise circular import