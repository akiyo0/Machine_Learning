import torch as t

x = t.Tensor(5,3)

x = t.rand(5,3)
print(x.size())
print(x.size()[0],x.size(1))

y = t.rand(5,3)
result = t.Tensor(5,3)
print(x + y)
t.add(x,y,out=result)
print(result)
print(y.add(x)) #不覆盖加法
print(y.add_(x)) #覆盖加法

a = t.ones(5).numpy()
print(a)
a = t.from_numpy(a)
print(a)

if t.cuda.is_available():
    print("cuda is available now")
    x = x.cuda()
    y = y.cuda()
    x + y
else:
    print("cuda is not available")