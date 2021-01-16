import torch
import time

print(torch.__version__)
print(torch.cuda.is_available)

a = torch.randn(1000,1000)
b = torch.randn(1000,200)

t0 = time.time()
c = torch.matmul(a, b)
t1 = time.time()
print(a.device, t1-t0, c.norm(2))

device = torch.device("cuda")
a = a.to(device)
b = b.to(device)
