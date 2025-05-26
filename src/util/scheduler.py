from torch.optim.lr_scheduler import _LRScheduler
from timm.scheduler import CosineLRScheduler


class CosineLRSchedulerWrapper(_LRScheduler):
    def __init__(
        self, optimizer, t_initial, lr_min=0.0, warmup_t=0, warmup_lr_init=0.0, **kwargs
    ):
        self.scheduler = CosineLRScheduler(
            optimizer,
            t_initial=t_initial,
            lr_min=lr_min,
            warmup_t=warmup_t,
            warmup_lr_init=warmup_lr_init,
            **kwargs
        )
        self._step_count = 0
        # Pass a dummy last_epoch to _LRScheduler to avoid auto stepping
        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        # This is unused since we delegate to the internal scheduler
        return [group["lr"] for group in self.optimizer.param_groups]

    def step(self, epoch=None):
        # CosineLRScheduler expects either `epoch` or step count
        if epoch is None:
            self.scheduler.step(self._step_count)
        else:
            self.scheduler.step(epoch)
        self._step_count += 1

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)
