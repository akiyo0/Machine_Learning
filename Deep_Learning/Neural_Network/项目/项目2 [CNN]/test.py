import torch
from torch.optim.lr_scheduler import StepLR
model = torch.nn.Linear(5,10)
optim = torch.optim.SGD(model.parameters(), lr = 1)
scheduler = StepLR(optim,step_size=5 , gamma=0.1)
model.train()
for i in range(25):
    optim.step()
    scheduler.step()
    print(scheduler.get_lr()[0])