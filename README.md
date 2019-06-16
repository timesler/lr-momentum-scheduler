# lr-momentum-scheduler

This repo contains pytorch scheduler classes for implementing the following:

* Arbitrary LR and momentum schedules
* The 1cycle policy optimizer

## Usage (1cycle policy)

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
    
