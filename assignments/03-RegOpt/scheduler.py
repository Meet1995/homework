import numpy as np
from typing import List
from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    One step policy implementation

    Args:
        optimizer (_LRScheduler): Optimizer
        ini_lr (float): Initial LR
        max_lr (float): Max LR
        final_lr (float): Final LR
        n_epochs (int): Number of Epochs
        batches_per_epoch (int): Batches per epoch
        cycle_perct (float): Percentage of total steps (epochs * batches_per_epoch),
        considered for cycle
        last_epoch (int, optional): Used to reinitialize scheduler. Defaults to -1.
    """
    def __init__(
        self,
        optimizer: _LRScheduler,
        max_lr: float,
        ini_lr_div: float,
        final_lr_div: float,
        n_epochs: int,
        batches_per_epoch: int,
        cycle_perct: float,
        last_epoch: int = -1,
    ):
        self.max_lr = max_lr
        self.ini_lr = max_lr/ini_lr_div
        self.final_lr = self.ini_lr/final_lr_div
        self.n_epochs = n_epochs
        self.batches_per_epoch = batches_per_epoch
        self.cycle_perct = cycle_perct

        self.total_steps = n_epochs * batches_per_epoch
        self.cycle_end = int(self.total_steps * cycle_perct)
        self.amp = max_lr - self.ini_lr
        self.m = (self.final_lr - self.ini_lr) / (self.total_steps - self.cycle_end)

        max_lrs = [max_lr] * len(optimizer.param_groups)
        if last_epoch == -1:
            for i, group in enumerate(optimizer.param_groups):
                group["initial_lr"] = self.ini_lr
                group["max_lr"] = max_lrs[i]
                group["min_lr"] = self.final_lr
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Returns learning rate0

        Returns:
            List[float]: learning rate for each parameter group
        """
        lrs = []
        step_num = self.last_epoch
        for group in self.optimizer.param_groups:
            if step_num <= self.cycle_end:
                lr = self.ini_lr + self.amp * np.sin(np.pi * step_num / self.cycle_end)
            elif (step_num > self.cycle_end) and (step_num <= self.total_steps):
                lr = self.m * (step_num - self.cycle_end) + self.ini_lr
            else:
                raise ValueError(f"Called .step() more than {self.total_steps}")
            lrs.append(lr)
        return lrs
