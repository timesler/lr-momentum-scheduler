# lr-momentum-scheduler

This repo contains pytorch scheduler classes for implementing the following:

* Arbitrary LR and momentum schedulers
  * Lambda function-based scheduler based on lr_scheduler.LambdaLR
  * List-based scheduler that accepts explicitly defined schedule lists for LR and momentum
* Learning rate range finder for preparing the 1cycle policy
* The 1cycle policy scheduler

These classes inherit from, and and based on, the core learning rate schedulers included in Pytorch, and can be used in an identical manner, with the added ability to schedule momentum.

## Schedulers

See detailed documentation and implementation by running:

```python
import scheduler
help(scheduler.LambdaScheduler)
help(scheduler.ListScheduler)
help(scheduler.RangeFinder)
help(scheduler.OneCyclePolicy)
```

1. `LambdaScheduler`: based on pytorch's `LambdaLR`, but can also (optionally) schedule momentum in the same way. Note that, like LambdaLR, individual schedules can be defined for each parameter group in the optimizer by passing a list of lambdas/functions/callables for LR and momentum.
1. `ListScheduler`: similar to the `LambdaScheduler`, but defines LR and momentum using passed lists. Per-parameter schedules are specified using lists of lists or 2D numpy arrays.
1. `RangeFinder`: a simple predefined schedule that varies LR from 1e-7 to 1 over a certain number of epochs. This is a preparatory step for the One Cycle Policy.
1. `OneCyclePolicy`: The One Cycle Policy scheduler for LR and momentum, see [References](#references).

## The One Cycle Policy

1. Import modules and define some test data:
    ```python
    import torch
    from torch import nn
    from torch import optim
    from scheduler import *
    
    epochs = 100
    x = torch.randn(100, 10)
    ```
1. Instantiate model:
    ```python
    mdl = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
        nn.Sigmoid()
    )
    ```
1. Run range test to find suitable LR:
    ```python
    optimizer = optim.SGD(mdl.parameters(), lr=1e-7)
    range_finder = RangeFinder(optimizer, epochs)
    
    losses = []
    for epoch in range(epochs):
        loss = mdl(x).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        range_finder.step()
        losses.append(loss.item())
    ```
    Based on results above, let's say the max LR is 1e-2
1. Re-instantiate model:
    ```python
    mdl = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
        nn.Sigmoid()
    )
    ```
1. Define 1cycle policy optimizer:
    ```python
    optimizer = optim.SGD(mdl.parameters(), lr=1e-3)
    one_cycle = OneCyclePolicy(optimizer, 1e-2, epochs, momentum_rng=[0.85, 0.95])
    ```
1. Train model:
    ```python
    losses = []
    for epoch in range(epochs):
        loss = mdl(x).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        one_cycle.step()
        losses.append(loss.item())
    print(losses)
    ```

## References

* _A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay_. Leslie N. Smith, 2018, arXiv:1803.09820.
    
