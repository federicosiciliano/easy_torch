import time
import pytorch_lightning as pl

class TimeCallback(pl.callbacks.Callback):
    def __init__(self, log_params={}):
        self.custom_log = lambda name, value: self.log(name, value, **log_params)

    def on_epoch_start(self):
        self.start_time = time.time()

    def on_epoch_end(self, split_name):
        self.elapsed_time = time.time() - self.start_time
        self.custom_log(split_name+"_time", self.elapsed_time)

    def on_train_epoch_start(self, trainer, pl_module):
        self.on_epoch_start()

    def on_train_epoch_end(self, trainer, pl_module):
        self.on_epoch_end("train")

    def on_validation_epoch_start(self, trainer, pl_module):
        self.on_epoch_start()
        
    def on_validation_epoch_end(self, trainer, pl_module):
        self.on_epoch_end("val")

    def on_test_epoch_start(self, trainer, pl_module):
        self.on_epoch_start()

    def on_test_epoch_end(self, trainer, pl_module):
        self.on_epoch_end("test")