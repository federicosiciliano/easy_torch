import torch
import pytorch_lightning as pl
from .model import BaseNN

def create_model(main_module, loss, optimizer, seed=42):
    pl.seed_everything(seed, workers=True) #for weight initialization
    model = BaseNN(main_module, loss, optimizer)
    return model

def train_model(trainer, model, loaders, train_key="train", val_key="val"):
    #pl.seed_everything(42, workers=True) #useless?
    if val_key is not None:
        val_dataloaders = loaders[val_key]
    else: val_dataloaders = None
    trainer.fit(model, loaders[train_key], val_dataloaders)
    
def test_model(trainer, model, loaders, loaders_key = "test"):
    #pl.seed_everything(42, workers=True) #probably useless?
    
    trainer.test(model, loaders[loaders_key])

def shutdown_dataloaders_workers():
    if torch.distributed.is_initialized():
        print("\n\nTORCH DISTR INIT\n\n")
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

def load_model(model_cfg, path):
    model = BaseNN.load_from_checkpoint(path, **model_cfg)
    return model