import torch as t
from torch.autograd import Variable
'''
x = Variable(t.ones(2, 2), requires_grad=True)
y = x.sum()
y.grad_fn

y.backward()
print(x.grad)
y.backward()
print(x.grad)

x.grad.data.zero_()
y.backward()
print(x.grad)
'''
########

x = Variable(t.randn(2, 2), requires_grad=True)
y = Variable(t.randn(2, 2))
z = Variable(t.randn(2, 2))
a = x + y
b = a + z
print(x.grad_fn, a.grad_fn)
print(x.requires_grad, y.requires_grad, z.requires_grad)
print(a.requires_grad, b.requires_grad)
print(x)
print(y)
print(a)

########
x = t.Tensor([[1.,2.,3.],[4.,5.,6.]])
x = Variable(x, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()

print(x.grad_fn)
print(y.grad_fn)
print(z.grad_fn)
print(out.grad_fn)

'''
计算图已经构建好了，下面来计算梯度。
所计算的梯度都是结果变量关于创建变量的梯度。
在这个计算图中，能计算的梯度有三个，
分别是out,z和y关于x的梯度，以out关于x的梯度为例：
根据链式求导法则：
out = 1/6 
https://zhuanlan.zhihu.com/p/29904755
https://pytorch-cn.readthedocs.io/zh/latest/notes/autograd/
https://zhuanlan.zhihu.com/p/30830840
'''
print(x)
out.backward()
print(x.grad)

######
x = Variable(t.ones(4, 5))
y = t.cos(x)
x_tensor_cos = t.cos(x.data)
print(x_tensor_cos == t.cos(x.data))
