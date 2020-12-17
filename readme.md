# Auto Checkpoint

** Call For Testing And Contribution.**

## Introduction
This is an automatic checkpoint module for `PyTorch`. 

## Example

```python
from auto_chkpt import registry, SimpleSaver

# Catching Exceptions, e.g., KeyboardInterrupt.
@registry.watch_training_process()
def train():
    model = Mode().cuda()
    optimizer = Optimizer(model.parameters())
    
    saver = SimpleSaver(chkpt_fold="checkpoint", tag="mnist",
         model=model, optimizer=optimizer, chkpt_steps=10)

    registry.attach_saver(saver, "saver_mnist")

    for e in range(saver.epoch, MAX_EPOCH):
        # training ...
        # validation ...
        saver.step()
```