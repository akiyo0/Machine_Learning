## MaxPool2d
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
