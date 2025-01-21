# Import necessary libraries
import torch
import pytorch_lightning as pl
import torchmetrics

class BaseNN(pl.LightningModule):
    def __init__(self, main_module, loss, optimizer, metrics={}, log_params={},
                 step_routing = {"model_input_from_batch":[0],
                                 "loss_input_from_batch": [1], "loss_input_from_model_output": None,
                                 "metrics_input_from_batch": [1], "metrics_input_from_model_output": None},
                 **kwargs): #TODO? change step order in computation: first model_output then batch
        super().__init__()

        # Store the main neural network module
        self.main_module = main_module

        # Store the optimizer function
        self.optimizer = optimizer

        # Store the primary loss function
        self.loss = loss
        
        # Define the metrics to be used for evaluation
        self.metrics = metrics

        # Define how batch and model output are routed to model, loss and metrics
        self.step_routing = step_routing

        # Define a custom logging function
        self.log_params = log_params
        #self.custom_log = lambda name, value: self.log(name, value, **log_params)

    def log(self, name, value):
        original_log_function = super().log
        if value is not None:
            if isinstance(value, dict):
                for key in value:
                    self.log(name+'_'+key, value[key])
                    # if to_log.size() != 1 and len(to_log.size()) != 0: #Save metrics in batch; TODO: make this better
                    #     if split_name == "test":
                    #         save_path = os.path.join(self.logger.save_dir, self.logger.name, f'version_{self.logger.version}',f"metrics_per_sample.csv")
                    #         with open(save_path, 'a') as f_object:
                    #             writer_object = csv.writer(f_object)
                    #             writer_object.writerow([log_key,*to_log.cpu().detach().tolist()])
                    #             f_object.close()
                    # else:
            else:
                original_log_function(name, value, **self.log_params)

    # Define the forward pass of the neural network
    def forward(self, *args, **kwargs):
        return self.main_module(*args, **kwargs)

    # Configure the optimizer for training
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())   
        return optimizer

    def on_epoch_end(self):
        # Step through each scheduler
        for scheduler in self.lr_schedulers():
            scheduler.step()

    # Define a step function for processing a batch
    def step(self, batch, batch_idx, dataloader_idx, split_name):
        #TODO: what to do with batch_idx and dataloader_idx?
        model_output = self.compute_model_output(batch, self.step_routing["model_input_from_batch"])
        lightning_module_return = {"model_output": model_output}

        if self.loss is not None:
            lightning_module_return["loss"] = self.compute_loss(batch, self.step_routing["loss_input_from_batch"],
                                     model_output, self.step_routing["loss_input_from_model_output"],
                                     split_name)

        if len(self.metrics)>0:
            lightning_module_return["metric_values"] = self.compute_metrics(batch, self.step_routing["metrics_input_from_batch"],
                                                model_output, self.step_routing["metrics_input_from_model_output"],
                                                split_name, dataloader_idx)

        #TODO: is this return correct?
        return lightning_module_return

    def compute_model_output(self, batch, model_input_from_batch):
        model_input_args, model_input_kwargs = self.get_input_args_kwargs((batch, model_input_from_batch))

        model_output = self(*model_input_args, **model_input_kwargs)
        
        return model_output
    
    def get_input_args_kwargs(self, *args):
        input_args, input_kwargs = [],{}
        for obj,keys in args:
            if isinstance(keys, int) or isinstance(keys, str):
                keys = [keys]
            if isinstance(keys, list):
                input_args += [obj[i] for i in keys]
            elif isinstance(keys, dict):
                for k,i in keys.items():
                    if i is None:
                        input_kwargs[k] = obj
                    else:
                        input_kwargs[k] = obj[i]
            elif keys is None:
                input_args.append(obj)
            else:
                raise NotImplementedError("keys type not recognized")
        return input_args, input_kwargs

    def compute_loss(self, batch, loss_input_from_batch, model_output, loss_input_from_model_output, split_name):
        if isinstance(self.loss, torch.nn.ModuleDict):
            loss = torch.tensor(0.0, device=self.device)
            for i, (loss_name, loss_func) in enumerate(self.loss.items()):
                loss += self.loss.__weight__[i]*self._compute(loss_name, loss_func, batch, loss_input_from_batch, model_output, loss_input_from_model_output, split_name)
            self.log(split_name+'_loss', loss)
        else:
            loss = self._compute("loss", self.loss, batch, loss_input_from_batch, model_output, loss_input_from_model_output, split_name)
        return loss
    
    def compute_metrics(self, batch, metrics_input_from_batch, model_output, metrics_input_from_model_output, split_name, dataloader_idx):
        metric_values = {}
        for metric_name, metric_func in self.metrics[split_name][dataloader_idx].items():
            metric_values[metric_name] = self._compute(metric_name, metric_func, batch, metrics_input_from_batch, model_output, metrics_input_from_model_output, split_name)
        return metric_values
    
    def _compute(self, name, func, batch, input_from_batch, model_output, input_from_model_output, split_name):
        # If metrics_input is a dictionary, routing is different for each metric
        batch_routing = self.get_key_if_dict_and_exists(input_from_batch, name)
        output_routing = self.get_key_if_dict_and_exists(input_from_model_output, name)

        input_args, input_kwargs = self.get_input_args_kwargs((batch, batch_routing), (model_output, output_routing))

        value = func(*input_args,**input_kwargs)

        log_name = split_name+'_'+name
        self.log(log_name, value)

        return value
    
    def get_key_if_dict_and_exists(self, obj, key):
        if isinstance(obj, dict) and key in obj:
            return obj[key]
        else:
            return obj

    # Training step
    def training_step(self, batch, batch_idx, dataloader_idx=0): return self.step(batch, batch_idx, dataloader_idx, "train")

    # Validation step
    # TODO: why dataloader_idx=0?
    def validation_step(self, batch, batch_idx, dataloader_idx=0): return self.step(batch, batch_idx, dataloader_idx, "val")

    # Test step
    # TODO: why dataloader_idx=0?
    def test_step(self, batch, batch_idx, dataloader_idx=0): return self.step(batch, batch_idx, dataloader_idx, "test")
    
    # Predict step
    def predict_step(self, batch, batch_idx, dataloader_idx=0): return self.step(batch, batch_idx, dataloader_idx, "predict")

    def on_train_epoch_end(self) -> None:
        self.on_epoch_end("train")
    
    def on_validation_epoch_end(self) -> None:
        self.on_epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self.on_epoch_end("test")

    def on_epoch_end(self, split_name): #not a lightning method
        for metric_func in self.metrics.values():
            if isinstance(metric_func, torchmetrics.metric.Metric):
                metric_func.reset()

# Define functions for getting and loading torchvision models
def get_torchvision_model(*args, **kwargs): return torchvision_utils.get_torchvision_model(*args, **kwargs)
#TODO: add set seed

def get_torchvision_model_as_decoder(example_datum, *args, **kwargs):
    forward_model = torchvision_utils.get_torchvision_model(*args, **kwargs)
    inverted_model = torchvision_utils.invert_model(forward_model, example_datum)
    return inverted_model

def load_torchvision_model(*args, **kwargs): return torchvision_utils.load_torchvision_model(*args, **kwargs)

# Define an Identity module
class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

# Define a LambdaLayer module
class LambdaLayer(torch.nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

# Class MLP (Multi-Layer Perceptron) (commented out for now)
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

# Import additional libraries
from . import torchvision_utils # put here otherwise circular import
