import torch
import torch.nn.functional as F

Layer1 = torch.nn.Linear(784, 200)
Layer2 = torch.nn.Linear(200, 200)
Layer3 = torch.nn.Linear(200, 10)

x = torch.randn([1, 784])
x = Layer1(x) #torch.Size([1, 200])
x = F.relu(x, inplace=True)
x = Layer2(x) #torch.Size([1, 200])
x = F.relu(x, inplace=True)
x = Layer3(x) #torch.Size([1, 10])
print(x)
x = F.relu(x, inplace=True)

print(x)



