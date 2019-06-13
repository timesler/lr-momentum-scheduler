import torch
from torch.optim import SGD, lr_scheduler
import numpy as np


class LRUpdate(object):
    """A callable class used to define an arbitrary lr schedule defined by a list.

    This object is designed to be passed to torch's LambdaLR scheduler to apply the given schedule.
    
    Arguments:
        lrs {Union[list, numpy.array]} -- List or numpy array defining LR schedule.
    """

    def __init__(self, lrs):
        self.lrs = np.hstack([lrs, 0])

    def __call__(self, epoch):
        return self.lrs[epoch] / self.lrs[0]
    

class ListScheduler(lr_scheduler.LambdaLR):
    """LR scheduler that implements an arbitrary schedule.
    
    Arguments:
        optimizer {torch.optim.optimizer.Optimizer} -- Pytorch optimizer.
        lrs {Union[list, numpy.array]} -- Learning rate schedule.
    """

    def __init__(self, optimizer, lrs):
        self.lr_lambda = LRUpdate(lrs)
        super().__init__(optimizer, self.lr_lambda)


class ScheduledOptimizer(ListScheduler):
    """SGD Optimizer class that follows a specific LR schedule.
    
    Arguments:
        optimizer {torch.optim.optimizer.Optimizer} -- Pytorch optimizer.
        lrs {Union[list, numpy.array]} -- Learning rate schedule.
    """
    
    def __init__(self, params, lrs):
        self.optimizer = SGD(params, lr=lrs[0])
        super().__init__(self.optimizer, lrs)

    def step(self, epoch=None):
        self.optimizer.step()

        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
    
    def zero_grad(self):
        self.optimizer.zero_grad()


class RangeFinder(ScheduledOptimizer):
    """SGD Optimizer class that implements the LR range search specified in:

        A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch
        size, momentum, and weight decay. Leslie N. Smith, 2018, arXiv:1803.09820.
    
    Logarithmically spaced learning rates from 1e-7 to 1 are searched. The number of increments in
    that range is determined by 'epochs'.
    
    Arguments:
        params {generator} -- Module parameters to be optimized.
        epochs {int} -- The number of epochs to use during search.
    """

    def __init__(self, params, epochs):
        lrs = np.logspace(-7, 0, epochs)
        super().__init__(params=params, lrs=lrs)


class OneCyclePolicy(ScheduledOptimizer):
    """SGD Optimizer class that implements the 1cycle policy search specified in:

        A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch
        size, momentum, and weight decay. Leslie N. Smith, 2018, arXiv:1803.09820.
    
    Arguments:
        params {generator} -- Module parameters to be optimized.
        lr {float} -- Maximum learning rate in range.
        epochs {int} -- The number of epochs to use during search.
        
    Keyword Arguments:
        phase_ratio {float} -- Fraction of epochs used for the increasing and decreasing phase of
            the schedule. For example, if phase_ratio=0.45 and epochs=100, the learning rate will
            increase from lr/10 to lr over 45 epochs, then decrease back to lr/10 over 45 epochs,
            then decrease to lr/100 over the remaining 10 epochs. (default: {0.45})
    """

    def __init__(self, params, lr, epochs, phase_ratio=0.45):
        phase_epochs = int(phase_ratio * epochs)
        lrs = np.hstack([
            np.linspace(lr * 1e-1, lr, phase_epochs),
            np.linspace(lr, lr * 1e-1, phase_epochs),
            np.linspace(lr * 1e-1, lr * 1e-2, epochs - 2 * phase_epochs),
        ])
        super().__init__(params=params, lrs=lrs)
