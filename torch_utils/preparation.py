import multiprocessing

import torch
import pytorch_lightning as pl
import torchmetrics

from .model import BaseNN

def prepare_data_loaders(data, loader_params):
    default_loader_params = {"num_workers": multiprocessing.cpu_count(), "pin_memory": True, "persistent_workers": True, "drop_last": {"all": False, "train": False, "val": False, "test": False}, "splits": ["", "train", "val", "test"]}
    loader_params = dict(list(default_loader_params.items()) + list(loader_params.items()))

    loaders = {}
    for split in loader_params["splits"]:
        if isinstance(split,dict):
            split_key = list(split.keys())[0]
            split_x, split_y = split[split_key]
            split = split_key
        else:
            split_x = "_".join([split,"x"])
            split_y = "_".join([split,"y"])

        if split_x in data:
            split_loader_params = loader_params.copy()
            del split_loader_params["splits"]
            for key,value in split_loader_params.items():
                if isinstance(value, dict):
                    if split in value.keys():
                        split_loader_params[key] = value[split]
                    elif "all" in value.keys():
                        split_loader_params[key] = value["all"]                        
            
            if isinstance(split_loader_params["batch_size"], float):
                split_loader_params["batch_size"] = int(len(data[split_x])*split_loader_params["batch_size"])

            td = torch.utils.data.TensorDataset(torch.Tensor(data[split_x]),torch.Tensor(data[split_y])) #Would need LongTensor if data[split_y] is not one-hot encoded

            #print(split_loader_params)
            loaders[split] = torch.utils.data.DataLoader(td, **split_loader_params)
    return loaders

def prepare_callbacks(trainer_params):
    pl.seed_everything(42, workers=True) #probably useless

    callbacks = []
    if "callbacks" in trainer_params:
        for callback_dict in trainer_params["callbacks"]:
            if isinstance(callback_dict, dict):
                for callback_name,callback_params in callback_dict.items():
                    callbacks.append(getattr(pl.callbacks, callback_name)(**callback_params))
                    # if callback_name=="ModelCheckpoint": #doesn't seem to work; should make ModelCheckpoint overwrite old "best" file
                    #     if os.path.isdir(callbacks[-1].dirpath):
                    #         callbacks[-1].STARTING_VERSION = -1
            else:
                callbacks.append(callback_dict)
    
    return callbacks
    #new_trainer_params = copy.deepcopy(trainer_params)
    #new_trainer_params["callbacks"] = callbacks
    #return new_trainer_params

def prepare_logger():
    print("TO BE IMPLEMENTED") #log type should not be saved as config, since it does not impact the results

def prepare_trainer(experiment_id=0, seed=42, **kwargs):
    pl.seed_everything(seed, workers=True) #useless?

    default_trainer_params = {"enable_checkpointing": False, "logger": pl.loggers.CSVLogger("../out/log", name=str(experiment_id)), "accelerator": "auto", "devices": "auto"}
    trainer_params = dict(list(default_trainer_params.items()) + list(kwargs.items()))

    trainer = pl.Trainer(**trainer_params)

    return trainer

def prepare_loss(loss):
    if isinstance(loss, str):
        loss_name = loss
        loss_params = {}
    elif isinstance(loss, dict):
        loss_name = list(loss.keys())[0] #accepts only one loss
        loss_params = loss[loss_name]
    else:
        raise NotImplementedError
    return getattr(torch.nn, loss_name)(**loss_params)

def prepare_metrics(metrics_info):
    metrics = []
    for metric_name,metric_vals in metrics_info.items():
        metrics[metric_name] = getattr(torchmetrics,metric_name)(**metric_vals)
    return metrics

def prepare_optimizer(name, params):
    return lambda model_params: getattr(torch.optim,name)(model_params,**params)

def prepare_model(model_cfg):
    pl.seed_everything(model_cfg["seed"], workers=True) #for weight initialization
    model = BaseNN(**model_cfg)
    return model

#To solve OSError: [Errno 24] Too many open files?
    #sharing_strategy = "file_system"
# def set_worker_sharing_strategy(worker_id: int) -> None:
#     torch.multiprocessing.set_sharing_strategy(sharing_strategy)
#torch.multiprocessing.set_sharing_strategy(sharing_strategy)


# def add_exp_info_to_ModelCheckpoint(callbacks_dict, add_to_dirpath):
#     #print(callbacks_dict)
#     new_list = copy.deepcopy(callbacks_dict)

#     for MC_index, dc in enumerate(new_list):
#         if any([x=="ModelCheckpoint" for x in new_list]):
#             break

#     new_list[MC_index]["ModelCheckpoint"]["dirpath"] += str(add_to_dirpath)
#     #print(new_list)
#     return new_list

# def express_neuron_per_layers(cfg_model_cfg, model_cfg):
#     #probably not efficient since expressing all possible combinations
#     num_neurons = model_cfg["num_neurons"]
#     num_layers = model_cfg["num_layers"]

#     neurons_per_layer = []

#     for layer in num_layers:
#         neurons_per_layer += list(it.product(num_neurons,repeat=layer))

#     for cfg in [cfg_model_cfg, model_cfg]:
#         cfg.pop('num_neurons', None)
#         cfg.pop('num_layers', None)

#         cfg["neurons_per_layer"] = neurons_per_layer