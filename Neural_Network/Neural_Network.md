## 对 Neural_Network 的若干点补充
### Batch_Size
Batch_Size（批尺寸）是机器学习中一个重要参数。Batch 的选择，首先决定的是下降的方向。
如果数据集比较小，完全可以采用全数据集 (Full Batch Learning) 的形式，这样做至少有 2 个好处：
+ 由全数据集确定的方向能够更好地代表样本总体，从而更准确地朝向极值所在的方向。
+ 由于不同权重的梯度值差别巨大，因此选取一个全局的学习率很困难。 Full Batch Learning 可以使用 Rprop 只基于梯度符号并且针对性单独更新各权值。

对于更大的数据集，以上 2 个好处又变成了 2 个坏处：
+ 随着数据集的海量增长和内存限制，一次性载入所有的数据进来变得越来越不可行。
+ 以 Rprop 的方式迭代，会由于各个 Batch 之间的采样差异性，各次梯度修正值相互抵消，无法修正。这才有了后来 RMSProp 的妥协方案。

既然 Full Batch Learning 并不适用大数据集，那么走向另一个极端怎么样？所谓另一个极端，就是每次只训练一个样本，即 Batch_Size = 1。即在线学习 (Online Learning)。
线性神经元在均方误差代价函数的错误面是一个抛物面，横截面是椭圆。对于多层神经元、非线性网络，在局部依然近似是抛物面。使用在线学习，每次修正方向以各自样本的梯度方向修正，横冲直撞各自为政，难以达到收敛。

可不可以选择一个适中的 Batch_Size 值呢？
当然可以，这就是批梯度下降法（Mini-batches Learning）。因为如果数据集足够充分，那么用一半（甚至少得多）的数据训练算出来的梯度与用全部数据训练出来的梯度是几乎一样的。
引用元：https://blog.csdn.net/ycheng_sjtu/article/details/49804041


### 2
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

| 参数名 | 解释 | 输入值 |
| --- | --- | --- |
| batch_size | 一个batch中样例的个数 | 2 |
| channels | 通道数，也就是当前层的深度 | 1 |
| height *1 | 图片的高 | 7 |
| width *1 | 图片的宽 | 3 |

```python
torch.nn.Conv2d([channels, output, height, width])
```

| 参数名 | 解释 | 输入值 |
| --- | --- | --- |
| channels | 通道数(同上的解释) | 1 |
| output | 输出的深度 | 8 |
| height *2 | 过滤器filter的高 | 2 |
| width *2 | 过滤器filter的宽 | 3 |

得到结果
```python
res -> [batch_size, output, height, width]
```
| 参数名 | 解释 | 输入值 |
| --- | --- | --- |
| batch_size | 一个batch中样例的个数，同上 |2|
| output | 输出的深度 | 8 |
| height *3 | 卷积结果的高度 | 6 | 
| width *3 | 卷积结果的宽度 | 1 |

 height *3 = height *1 - height *2 + 1 = 7 - 2 + 1
width *3 = width *1 - width *2 + 1 = 3 - 3 + 1 

### modules()
```python
for name,sub_module in net.named_children():
    if name in ['conv1']:
        print(sub_module)
```
```python
params = list(net.parameters())
print(len(params))
```
```python
for name,parameters in net.named_parameters():
    print(name, ':', parameters.size())
```
```python
for module in model.modules():
    print(module)
```
```python
for name, module in model.named_children():
    if name in ['conv4', 'conv5']:
        print(module)
```
```python
for param in model.parameters():
    print(type(param.data), param.size())
```
输出结果
`<class 'torch.FloatTensor'> (20L,)
<class 'torch.FloatTensor'> (20L, 1L, 5L, 5L)`

## 卷积
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
Applies a 2D convolution over an input signal composed of several input planes.

In the simplest case, the output value of the layer with input size $\left(N, C_{\mathrm{in}}, H, W\right)$ and output $\left(N, C_{\text{out}}, H_{\text {out}}, W_{\text{out}}\right)$  can be precisely described as:
```python
class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
```
$$\operatorname{out}\left(N_{i}, C_{\text{out}_{j}}\right)=\operatorname{bias}\left(C_{\text{out}_{j}}\right)+\sum_{k=0}^{C_{\text{in}}-1} \text {weight}\left(C_{\text{out}_{j}}, k\right) \star \text{input}\left(N_{i}, k\right)$$

where $\star$ is the valid 2D cross-correlation operator, $N$ is a batch size, $C$ denotes a number of channels, $H$ is a height of input planes in pixels, and $W$ is width in pixels.

## 仿射层/全连接层
### torch.nn.Linear
Applies a linear transformation to the incoming data: $y=x A^{T}+b$ 
```python
class torch.nn.Linear(in_features, out_features, bias=True)
```
+ `in_features` – size of each input sample
+ `out_features` – size of each output sample
+ `bias` – If set to `False`, the layer will not learn an additive bias. Default: `True`

