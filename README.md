# lr-momentum-scheduler

This repo contains pytorch scheduler and optimizer classes for implementing the following:

* Arbitrary LR schedules
* Optimizers that follow arbitrary LR schedules (i.e., that don't require a separate scheduler)
* The 1cycle policy optimizer

## Usage (1cycle policy)

1. Import modules and define some test data:
    ```python
    import torch
    from torch import nn
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
    range_finder = RangeFinder(mdl.parameters(), epochs=epochs)
    
    losses = []
    for epoch in range(epochs):
        loss = mdl(x).mean()
        loss.backward()
        range_finder.step()
        range_finder.zero_grad()
        losses.append(loss.item())
    ```
    Based on results above, let's say the best LR is 1e-2
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
    one_cycle = OneCyclePolicy(mdl.parameters(), lr=1e-2, epochs=epochs)
    ```
1. Train model:
    ```python
    losses = []
    for epoch in range(epochs):
        loss = mdl(x).mean()
        loss.backward()
        one_cycle.step()
        one_cycle.zero_grad()
        losses.append(loss.item())
    print(losses)
    ```

## References

* _A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay_. Leslie N. Smith, 2018, arXiv:1803.09820.
    
