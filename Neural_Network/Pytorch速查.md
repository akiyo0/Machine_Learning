### MaxPool2d
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

## torch.utils.data
### torch.utils.data.DataLoader
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

### torch.optim.lr_scheduler
`torch.optim.lr_scheduler`模块提供了一些根据epoch训练次数来调整学习率（learning rate）的方法。一般情况下我们会设置随着epoch的增大而逐渐减小学习率从而达到更好的训练效果。
而`torch.optim.lr_scheduler.ReduceLROnPlateau`则提供了基于训练中某些测量值使学习率动态下降的方法。

```python
class torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
```
