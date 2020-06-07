import torch as t
import torch.nn.functional as F
from torch.autograd import Variable

x1 = t.randn(1, 1, 1024, 1024)
x2 = t.randn(1, 1, 32, 32)

print(x1.shape) 

diffY = x2.size()[2] - x1.size()[2]
diffX = x2.size()[3] - x1.size()[3]

#x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
x1 = F.pad(x1, [diffX // 2, diffX // 2, diffY // 2, diffY // 2])

print(x1.ndim)



print(t.cat([x2, x1], dim=1).shape)