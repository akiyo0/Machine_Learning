import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

m = nn.Linear(in_features = 64*64*3, out_features = 1)
print(m)
input = t.randn(1,64,64,3).view(1,64*64*3)
output = m(input)

print(input)
print(output)
print(output.size())
#torch.Size([128, 30])