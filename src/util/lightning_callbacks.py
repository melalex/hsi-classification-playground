from pytorch_lightning.callbacks import Callback

class ValidateEveryNEpochs(Callback):
    def __init__(self, validate_every_n_epochs=5):
        self.validate_every_n_epochs = validate_every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.validate_every_n_epochs != 0:
            trainer.should_validate = False
