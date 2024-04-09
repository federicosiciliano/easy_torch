# Import necessary libraries
import multiprocessing
import torch
import pytorch_lightning as pl
import torchmetrics
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from codecarbon import EmissionsTracker
import wandb
import os

# Import modules and functions from local files
from .model import BaseNN

# Function to prepare data loaders
def prepare_data_loaders(data, split_keys={"train": ["train_x", "train_y"], "val": ["val_x", "val_y"], "test": ["test_x", "test_y"]}, dtypes = None, **loader_params):                             
    # Default loader parameters
    default_loader_params = {
        "num_workers": multiprocessing.cpu_count(),
        "pin_memory": True,
        "persistent_workers": True,
        "drop_last": {"train": False, "val": False, "test": False},
        "shuffle": {"train": True, "val": False, "test": False}
    }
    # Combine default and custom loader parameters
    loader_params = dict(list(default_loader_params.items()) + list(loader_params.items()))

    if dtypes is None or isinstance(dtypes, str) or isinstance(dtypes, torch.dtype):
        if isinstance(dtypes, str):
            dtypes = getattr(torch, dtypes)
        dtypes = {split_name: {data_key:dtypes for data_key in data_keys} for split_name, data_keys in split_keys.items()}
    elif isinstance(dtypes, dict):
        new_dtypes = {}
        for split_name, data_keys in split_keys.items():
            new_dtypes[split_name] = {}
            for data_key in data_keys:
                new_dtypes[split_name][data_key] = dtypes[data_key] if data_key in dtypes.keys() else None
                if isinstance(new_dtypes[split_name][data_key], str):
                    new_dtypes[split_name][data_key] = getattr(torch, new_dtypes[split_name][data_key])
        dtypes = new_dtypes
    else:
        raise NotImplementedError(f"Unsupported dtype: {dtypes}")

    loaders = {}
    for split_name, data_keys in split_keys.items():
        split_loader_params = deepcopy(loader_params)
        # Select specific parameters for this split
        for key, value in split_loader_params.items():
            if isinstance(value, dict):
                if split_name in value.keys():
                    split_loader_params[key] = value[split_name]
        
        # Get data and create the TensorDataset
        td = TensorDataset(*[torch.tensor(data[data_key], dtype=dtypes[split_name][data_key]) for data_key in data_keys])

        # Create the DataLoader
        loaders[split_name] = DataLoader(td, **split_loader_params)
    return loaders


# Function to prepare trainer parameters with experiment ID
def prepare_experiment_id(original_trainer_params, experiment_id, cfg=None):
    # Create a deep copy of the original trainer parameters
    trainer_params = deepcopy(original_trainer_params)

    # Check if "callbacks" is in trainer_params
    if "callbacks" in trainer_params:
        for callback_dict in trainer_params["callbacks"]:
            if isinstance(callback_dict, dict):
                for callback_name, callback_params in callback_dict.items():
                    if callback_name == "ModelCheckpoint":
                        # Update the "dirpath" to include the experiment_id
                        callback_params["dirpath"] += experiment_id + "/"
                    else:
                        # Print a warning message for unrecognized callback names
                        print(f"Warning: {callback_name} not recognized for adding experiment_id")
                        pass

    # Check if "logger" is in trainer_params
    if "logger" in trainer_params:
        # Update the "save_dir" in logger parameters to include the experiment_id
        trainer_params["logger"]["params"]["save_dir"] += experiment_id + "/"
        if trainer_params["logger"]["name"] == "WandbLogger":
            trainer_params["logger"]["params"]["id"] = experiment_id
            trainer_params["logger"]["params"]["name"] = experiment_id
            if cfg is not None:
                trainer_params["logger"]["params"]["config"] = cfg #TODO: Clean configuration
    return trainer_params

# Function to prepare callbacks
def prepare_callbacks(trainer_params, seed=42):
    pl.seed_everything(seed) # Seed the random number generator

    # Initialize an empty list for callbacks
    callbacks = []

    # Check if "callbacks" is in trainer_params
    if "callbacks" in trainer_params:
        for callback_dict in trainer_params["callbacks"]:
            if isinstance(callback_dict, dict):
                for callback_name, callback_params in callback_dict.items():
                    # Create callback instances based on callback names and parameters
                    callbacks.append(getattr(pl.callbacks, callback_name)(**callback_params))
            else:
                # If the callback is not a dictionary, add it directly to the callbacks list
                callbacks.append(callback_dict)
    
    return callbacks

def remove_keys_from_dict(input_dict, keys_to_remove):
    """
    Recursively remove keys from a dictionary and all its sub-dictionaries.
    """
    if isinstance(input_dict, dict):
        for key in keys_to_remove:
            if key in input_dict:
                del input_dict[key]
        for value in input_dict.values():
            remove_keys_from_dict(value, keys_to_remove)
    return input_dict

def log_wandb(trainer_params):
    items_to_delete = ['__nosave__', 'emission_tracker', 'metrics',
                       'data_folder', 'log_params', 'step_routing']
    cfg = exp_utils.cfg.load_configuration()
    exp_found, experiment_id = exp_utils.exp.get_set_experiment_id(cfg)
    if not exp_found:
        wandb.login(key=trainer_params["logger"]["key"])
        if trainer_params["logger"]["entity"] is not None:
            wandb.init(entity=trainer_params["logger"]["entity"],
                    project=trainer_params["logger"]["project"],
                    name = cfg['__exp__.name'] + "_" + experiment_id,
                    id = experiment_id,
                    config = remove_keys_from_dict(cfg, items_to_delete))
        else:
            wandb.init(project=trainer_params["logger"]["project"],
                    name = cfg['__exp__.name'] + "_" + experiment_id,
                    id = experiment_id,
                    config = remove_keys_from_dict(cfg, items_to_delete))
    

# Function to prepare a logger based on trainer parameters
def prepare_logger(trainer_params, seed=42):
    pl.seed_everything(seed) # Seed the random number generator
    logger = None
    if "logger" in trainer_params:
        # Get the logger class based on its name and initialize it with parameters
        if not os.path.exists(trainer_params["logger"]["params"]["save_dir"]):
            os.makedirs(trainer_params["logger"]["params"]["save_dir"])
        logger = getattr(pl.loggers, trainer_params["logger"]["name"])(**trainer_params["logger"]["params"])
        #if isinstance(logger, pl.loggers.wandb.WandbLogger):
        #This is the case when the logger is wandb so we check for the entity and the the key
            #log_wandb(trainer_params)
        #TODO: Multiple loggers

    return logger

# Function to prepare a PyTorch Lightning Trainer instance
def prepare_trainer(seed=42, **trainer_kwargs):
    pl.seed_everything(seed) # Seed the random number generator

    # Default trainer parameters
    default_trainer_params = {"enable_checkpointing": False, "accelerator": "auto", "devices": "auto"}

    # Combine default parameters with user-provided kwargs
    trainer_params = dict(list(default_trainer_params.items()) + list(trainer_kwargs.items()))

    # Create a Trainer instance with the specified parameters
    trainer = pl.Trainer(**trainer_params)

    return trainer

# Function to prepare a loss function
def prepare_loss(loss_info, additional_module=None, seed=42):
    pl.seed_everything(seed) # Seed the random number generator
    if isinstance(loss_info, str):
        # If 'loss' is a string, assume it's the name of a loss function
        loss = get_single_loss(loss_info, {}, additional_module)
    elif isinstance(loss_info, dict):
        # If 'loss' is a dictionary, assume it contains loss name and parameters
        loss = {}
        for loss_name, loss_params in sorted(loss_info.items()):
            if loss_name != "__weight__":
                loss[loss_name] = get_single_loss(loss_params["name"], loss_params.get("params",{}), additional_module)
        loss = torch.nn.ModuleDict(loss)
        loss.__weight__ = loss_info.get("__weight__", torch.ones(len(loss)))
    else:
        raise NotImplementedError
    return loss

def get_single_loss(loss_name, loss_params, additional_module=None):
    # Check if the loss_name exists in torch.nn or additional_module
    if hasattr(additional_module, loss_name):
        loss_module = additional_module
    elif hasattr(torch.nn, loss_name):
        loss_module = torch.nn
    else:
        raise NotImplementedError(f"The loss function {loss_name} is not found in torch.nn or additional module")

    # Create the loss function using the name and parameters
    return getattr(loss_module, loss_name)(**loss_params)


def prepare_metrics(metrics_info, additional_module=None, seed=42):
    pl.seed_everything(seed) # Seed the random number generator
    # Initialize an empty dictionary to store metrics
    metrics = {}
    
    for metric_name in metrics_info:
        if isinstance(metrics_info, list): 
            metric_vals = {}  # Initialize an empty dictionary for metric parameters
        elif isinstance(metrics_info, dict): 
            metric_vals = metrics_info[metric_name]  # Get metric parameters from the provided dictionary
        else: 
            raise NotImplementedError  # Raise an error for unsupported input types
        
        # Check if the metric_name exists in torchmetrics or additional_module
        if hasattr(additional_module, metric_name):
            metrics_package = additional_module
        elif hasattr(torchmetrics, metric_name):
            metrics_package = torchmetrics
        else:
            raise NotImplementedError  # Raise an error if the metric_name is not found in any package
        
        # Create a metric object using getattr and store it in the metrics dictionary
        metrics[metric_name] = getattr(metrics_package, metric_name)(**metric_vals)
    
    # Convert the metrics dictionary to a ModuleDict for easy handling
    metrics = torch.nn.ModuleDict(metrics)
    return metrics

def prepare_optimizer(name, params={}, seed=42):
    pl.seed_everything(seed) # Seed the random number generator
    # Return a lambda function that creates an optimizer based on the provided name and parameters
    return lambda model_params: getattr(torch.optim, name)(model_params, **params)

def prepare_model(model_cfg):
    # Seed the random number generator for weight initialization
    pl.seed_everything(model_cfg["seed"]) # Seed the random number generator
    
    # Create a model instance based on the provided configuration
    model = BaseNN(**model_cfg)
    return model

def prepare_emission_tracker(experiment_id, **tracker_kwargs):
    # Update the "output_dir" in tracker parameters to include the experiment_id
    tracker_kwargs["output_dir"] = tracker_kwargs.get("output_dir", "../out/log/") + experiment_id + "/"
    tracker = EmissionsTracker(**tracker_kwargs)
    return tracker
