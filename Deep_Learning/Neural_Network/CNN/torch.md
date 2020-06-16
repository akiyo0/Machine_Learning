[toc]

PyTorch Examples: [link](https://github.com/pytorch/examples)



## torch.utils

### torch.utils.data

#### torch.utils.data.DataLoader
数据加载器。组合数据集和采样器，并在数据集上提供单进程或多进程迭代器。
```python
class torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False)
```
参数：
+ dataset (Dataset) – 加载数据的数据集。
+ batch_size (int, optional) – 每个batch加载多少个样本(默认: 1)。
+ shuffle (bool, optional) – 设置为`True`时会在每个epoch重新打乱数据(默认: `False`).
+ sampler (Sampler, optional) – 定义从数据集中提取样本的策略。如果指定，则忽略shuffle参数。
+ num_workers (int, optional) – 用多少个子进程加载数据。0表示数据将在主进程中加载(默认: 0)
+ collate_fn (callable, optional) –
+ pin_memory (bool, optional) –
+ drop_last (bool, optional) – 如果数据集大小不能被batch size整除，则设置为True后可删除最后一个不完整的batch。如果设为False并且数据集的大小不能被batch size整除，则最后一个batch将更小。(默认: False)

## torch.optim

### torch.optim.lr_scheduler

`torch.optim.lr_scheduler`模块提供了一些根据epoch训练次数来调整学习率（learning rate）的方法。一般情况下我们会设置随着epoch的增大而逐渐减小学习率从而达到更好的训练效果。
而`torch.optim.lr_scheduler.ReduceLROnPlateau`则提供了基于训练中某些测量值使学习率动态下降的方法。

```python
class torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
```



## torch.nn

### torch.nn.Module
https://blog.csdn.net/u012609509/article/details/81203436
#### modules()

```python
#[1]
for name,sub_module in net.named_children():
    if name in ['conv1']:
        print(sub_module)
        
#[2]
params = list(net.parameters())
print(len(params))

for name,parameters in net.named_parameters():
    print(name, ':', parameters.size())

#[3]
for module in model.modules():
    print(module)

#[4]
for name, module in model.named_children():
    if name in ['conv4', 'conv5']:
        print(module)
        
#[5]
for param in model.parameters():
    print(type(param.data), param.size())
```

输出结果
```
<class 'torch.FloatTensor'> (20L,)
<class 'torch.FloatTensor'> (20L, 1L, 5L, 5L)
```


### torch.nn.Conv1d

```python
class torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
```

一维卷积层，输入的尺度是 $\left(N, C_{\mathrm{in}}, L\right)$，输出尺度 $\left(N, C_{\text{out}}, L_{\text {out}}\right)$ 的计算方式：
$$ \operatorname{out}\left(N_{i}, C_{\mathrm{out}_{j}}\right)=\operatorname{bias}\left(C_{\mathrm{out}_{j}}\right)+\sum_{k=0}^{C_{in}-1} \operatorname{weight}\left(C_{\mathrm{out}_{j}}, k\right) \star \operatorname{input}\left(N_{i}, k\right) $$

where $\star$ is the valid cross-correlation operator(有效互相关演算子), $N$ is a batch size, $C$ denotes a number of channels, $L$ is a length of signal sequence.

+ `in_channels` (python:int) – Number of channels in the input image
+ `out_channels` (python:int) – Number of channels produced by the convolution
+ `kernel_size` (python:int or tuple) – Size of the convolving kernel
+ `stride` (python:int or tuple, optional) – Stride of the convolution. Default: 1
+ `padding` (python:int or tuple, optional) – Zero-padding added to both sides of the input. Default: 0
+ `padding_mode` (string, optional) – zeros
+ `dilation` (python:int or tuple, optional) – Spacing between kernel elements. Default: 1
+ `groups` (python:int, optional) – Number of blocked connections from input channels to output channels. Default: 1
+ `bias` (bool, optional) – If `True`, adds a learnable bias to the output. Default: `True`

### torch.nn.Conv2d
[pytorch docs](https://pytorch.org/docs/stable/nn.html#conv2d)
```python
class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
```



```python
import torch

x = torch.randn(2,1,7,3)
conv = torch.nn.Conv2d(1,8,(2,3))
res = conv(x)

print(res.shape)    # shape = (2, 8, 6, 1)
```

这其中，输入函数

```python
torch.randn([batch_size, channels, height, width])
```

| 参数名     | 解释                       | 输入值 |
| ---------- | -------------------------- | ------ |
| batch_size | 一个batch中样例的个数      | 2      |
| channels   | 通道数，也就是当前层的深度 | 1      |
| height *1  | 图片的高                   | 7      |
| width *1   | 图片的宽                   | 3      |

```python
torch.nn.Conv2d([channels, output, height, width])
```

| 参数名    | 解释               | 输入值 |
| --------- | ------------------ | ------ |
| channels  | 通道数(同上的解释) | 1      |
| output    | 输出的深度         | 8      |
| height *2 | 过滤器filter的高   | 2      |
| width *2  | 过滤器filter的宽   | 3      |

得到结果

```python
res -> [batch_size, output, height, width]
```

| 参数名     | 解释                        | 输入值 |
| ---------- | --------------------------- | ------ |
| batch_size | 一个batch中样例的个数，同上 | 2      |
| output     | 输出的深度                  | 8      |
| height *3  | 卷积结果的高度              | 6      |
| width *3   | 卷积结果的宽度              | 1      |

height *3 = height *1 - height *2 + 1 = 7 - 2 + 1
width *3 = width *1 - width *2 + 1 = 3 - 3 + 1 

### torch.nn.Linear

Applies a linear transformation to the incoming data: $y=x A^{T}+b$ 

```python
class torch.nn.Linear(in_features, out_features, bias=True)
```

+ `in_features` – size of each input sample
+ `out_features` – size of each output sample
+ `bias` – If set to `False`, the layer will not learn an additive bias. Default: `True`

### 

```python
class torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
```

Applies a 2D max pooling over an input signal composed of several input planes.
对于输入信号的输入通道，提供2维最大池化（max pooling）操作

In the simplest case, the output value of the layer with input size $(N,C,H,W)$ , output $(N, C, H_{out},W_{out})$ and kernel_size $(kH,kW)$ can be precisely described as:
$$
\begin{aligned}
\operatorname{out}\left(N_{i}, C_{j}, h, w\right)=& \max _{m=0, \ldots, k H-1} \max _{n=0, \ldots, k W-1} \\
& \text { input }\left(N_{i}, C_{j}, \text { stride }[0] \times h+m, \text { stride }[1] \times w+n\right)
\end{aligned}
$$



如果padding不是0，会在输入的每一边添加相应数目0 dilation用于控制内核点之间的距离，详细描述在这里

参数kernel_size，stride, padding，dilation数据类型： 可以是一个int类型的数据，此时卷积height和width值相同; 也可以是一个tuple数组（包含来两个int类型的数据），第一个int数据表示height的数值，tuple的第二个int类型的数据表示width的数值

对几个输入平面组成的输入信号应用2D卷积。

有关详细信息和输出形状，请参见Conv2d。

参数：

+ `input`：输入张量 (`minibatch` x `in_channels` x `iH` x `iW`)
+ `weight`：过滤器张量 (`out_channels`, `in_channels`/`groups`, `kH`, `kW`)
+ `bias`：可选偏置张量 (`out_channels`)
+ `stride`：卷积核的步长，可以是单个数字或一个元组 (`sh ` x `sw`)。默认为`1`
+ `padding`：输入上隐含零填充。可以是单个数字或元组。 默认值：`0`
+ `groups`：将输入分成组，`in_channels`应该被组数除尽

### torch.nn.MaxPool2d

### torch.nn.BatchNorm2d
```python
torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
```

## torch.nn.functional
### F.pad()
https://blog.csdn.net/geter_CS/article/details/88052206
```python
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

print(x1.shape)
```
输出结果：
```
torch.Size([1, 1, 1024, 1024])
torch.Size([1, 1, 32, 32])
```

## torch.squeeze()[^squeeze]
### torch.squeeze()
先看torch.squeeze() 这个函数主要对数据的维度进行压缩，去掉维数为1的的维度，比如是一行或者一列这种，一个一行三列（1,3）的数去掉第一个维数为一的维度之后就变成（3）行。squeeze(a)就是将a中所有为1的维度删掉。不为1的维度没有影响。a.squeeze(N) 就是去掉a中指定的维数为一的维度。还有一种形式就是b=torch.squeeze(a，N) a中去掉指定的定的维数为一的维度。
### torch.unsqueeze()
再看torch.unsqueeze()这个函数主要是对数据维度进行扩充。给指定位置加上维数为一的维度，比如原本有个三行的数据（3），在0的位置加了一维就变成一行三列（1,3）。a.squeeze(N) 就是在a中指定位置N加上一个维数为1的维度。还有一种形式就是b=torch.squeeze(a，N) a就是在a中指定位置N加上一个维数为1的维度
[^squeeze]: [pytorch学习 中 torch.squeeze() 和torch.unsqueeze()的用法](https://blog.csdn.net/xiexu911/article/details/80820028)
## torch.cat()[^cat]

```python
torch.cat((A, B), dim)
```
注意在使用此函数时，除拼接维数dim数值可不同外其余维数数值需相同，方能对齐。

+ dim = 0 时表示按维数0（行）拼接A和B，也就是竖着拼接，A上B下。此时需要注意：列数必须一致，即维数1数值要相同，这里都是3列，方能列对齐。拼接后的C的第0维是两个维数0数值和，即2+4=6。
+ dim = 1 时表示按维数1（列）拼接A和B，也就是横着拼接，A左B右。此时需要注意：行数必须一致，即维数0数值要相同，这里都是2行，方能行对齐。拼接后的C的第1维是两个维数1数值和，即3+4=7。

```python
import torch
A = torch.ones(3, 3)
B = 2 * torch.ones(3, 3)

C = torch.cat((A, B), 0)
print(C.ndim)

C = torch.cat((A, B), 1)
print(C.ndim)
```
输出结果
```
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [2., 2., 2.],
        [2., 2., 2.],
        [2., 2., 2.]])
tensor([[1., 1., 1., 2., 2., 2.],
        [1., 1., 1., 2., 2., 2.],
        [1., 1., 1., 2., 2., 2.]])
```
[^cat]: [PyTorch的torch.cat](https://blog.csdn.net/qq_39709535/article/details/80803003)


## CNN 写法总结
### MNIST_Pytorch
```python
class Net(nn.Module):
    def __init__(self):
            super(Net, self).__init__()
            ...
    def forward(self, x):
    ...
    return output
```

````python
def train(args, model, device, train_loader, optimizer, epoch):
  model.train() # switch to train mode
  for batch_idx, (input, target) in enumerate(train_loader):
		optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
````
```python
def test(model, device, test_loader):
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
```
```python
def main():
    ## train_loader
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,)
                )
            ])
        ),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )
    ## test_loader
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data', 
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,)
                )
            ])
        ),
        batch_size=args.test_batch_size, 
        shuffle=True, 
        **kwargs
    )
    
    model = Net().to(device) # 'cpu' OR 'cuda'
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()
    
    if args.save_model:
        torch.save(model.state_dict(), "xxxxxx.pt")
```
```python
if __name__ == '__main__':
    main()
```

### VGG-16 Pytorch
```python
def main():
    args = parser.parse_args()
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    ...
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=(
            train_sampler is None
        ),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        ),
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.workers, 
        pin_memory=True
    )
```
```python
def train(train_loader, model, criterion, optimizer, epoch, args):
    model.train()
    for batch_idx, (input, target) in enumerate(train_loader):
        output = model(input)
        loss = criterion(output, target)
        
```
```python
def validate(val_loader, model, criterion, args):
    model.eval()
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(val_loader):
            output = model(input)
            loss = criterion(output, target)
```
```python
if __name__ == '__main__':
    main()
```


### FCN
```python
train_data = CityscapesDataset(csv_file=train_file, phase='train')
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
val_data = CityscapesDataset(csv_file=val_file, phase='val', flip_rate=0)
val_loader = DataLoader(val_data, batch_size=1, num_workers=8)

vgg_model = VGGNet(requires_grad=True, remove_fc=True)
fcn_model = FCNs(pretrained_net=vgg_model, n_class=n_class)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma) # decay LR by a factor of 0.5 every 30 epochs
```

```python
def train():
    for epoch in range(epochs):
        scheduler.step()
        for iter, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            outputs = fcn_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        torch.save(fcn_model, model_path)
        val(epoch)
```
```python
def val(epoch):
    fcn_model.eval()
    for iter, batch in enumerate(val_loader):
        output = fcn_model(inputs)
        output = output.data.cpu().numpy()
    # Calculate average IoU
```

## 并行编程
https://www.jiqizhixin.com/articles/2019-04-30-8
https://blog.csdn.net/m0_38008956/article/details/86559432