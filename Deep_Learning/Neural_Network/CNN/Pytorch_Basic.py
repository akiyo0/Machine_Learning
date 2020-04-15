import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

x = t.Tensor(5,3)

x = t.rand(5,3)
#print(x.size())
#print(x.size()[0],x.size(1))

y = t.rand(5,3)
result = t.Tensor(5,3)
#print(x + y)
t.add(x,y,out=result)
#print(result)
#print(y.add(x)) #不覆盖加法
#print(y.add_(x)) #覆盖加法

a = t.ones(5).numpy()
#print(a)
a = t.from_numpy(a)
#print(a)

######
# Tensor
a = t.Tensor(2, 3)
print(a.size(), a.numel())
# a.numel()：总元素个数
a = t.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = t.Tensor(a.size())
c = t.arange(1, 6, 2).tolist()
print(c)

# 索引操作
b = t.randn(3, 4)
print(b[0], b[0] > 0)
print(b[:, 0], b[:, 0] > 0)



if t.cuda.is_available():
    print("cuda is available now")
    x = x.cuda()
    y = y.cuda()
    x + y
else:
    print("cuda is not available")

## Pytorch 中 view 的用法
'''
把原先tensor中的数据按照行优先的顺序排成一个一维的数据
（这里应该是因为要求地址是连续存储的），然后按照参数组合成其他维度的tensor。
比如说是不管你原先的数据是[[[1,2,3],[4,5,6]]]还是[1,2,3,4,5,6]，
因为它们排成一维向量都是6个元素，所以只要view函数的参数一致，得到的结果都是一样的。
'''
a=t.Tensor([[[1,2,3],[4,5,6]]])
b=t.Tensor([1,2,3,4,5,6])

#print(a.view(1,6))
#print(b.view(1,6))

#print(a.view(1, -1)) # "-1" 为自适应
#print(b.view(-1, 6))

########
m = nn.Linear(20, 20)
#print(m)
input = t.randn(128, 20)
output = m(input)
#print(output.size())
#print(output,input)
#torch.Size([128, 30])


########
#Pytorch L

